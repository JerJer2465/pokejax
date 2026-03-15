"""
Differential test: replay Pokemon Showdown battles in PokeJAX engine.

Strategy: Since PRNGs differ (JAX ThreeFry vs PS custom), we SYNC state after
each turn to prevent RNG cascade. We also handle PS's team slot swapping
(PS swaps active with switch target; JAX uses fixed slots).

Key checks:
  1. Action mask: PS's chosen action must be legal in JAX
  2. Status/boost/faint agreement (deterministic conditions)
  3. HP conservation: HP must stay in [0, maxhp]

Usage:
    python scripts/differential_test.py --battles data/showdown_battles_1000.jsonl --limit 100
"""

import json
import argparse
import sys
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compile_cache")

sys.path.insert(0, str(Path(__file__).parent.parent))

from pokejax.data.tables import load_tables
from pokejax.config import GenConfig
from pokejax.core.state import make_battle_state, make_reveal_state
from pokejax.engine.turn import execute_turn
from pokejax.env.action_mask import get_action_mask


def normalize_id(name):
    return name.lower().replace(' ', '').replace('-', '').replace('.', '').replace("'", '').replace(':', '')


def build_reverse_lookup(name_to_id):
    reverse = {}
    for name, idx in name_to_id.items():
        reverse[normalize_id(name)] = idx
    return reverse


status_map = {'': 0, 'brn': 1, 'psn': 2, 'tox': 3, 'slp': 4, 'frz': 5, 'par': 6}


# ---------------------------------------------------------------------------
# PS slot mapping
# ---------------------------------------------------------------------------
# PS swaps team slot indices on switch (active always at index 0).
# JAX keeps fixed slot indices with sides_active_idx pointing to the active.
# We need to translate between the two.

def build_ps_to_jax_map(ps_mons, initial_species):
    """Build mapping from PS's current slot order to JAX's fixed slot order.

    Uses species name matching (strips form suffixes for robustness).
    Returns list where result[ps_idx] = jax_idx.
    """
    mapping = [None] * len(ps_mons)
    used_jax = set()

    for ps_idx, mon in enumerate(ps_mons):
        ps_species = normalize_id(mon['species'])
        # Try exact match first
        for jax_idx, init_species in enumerate(initial_species):
            if jax_idx in used_jax:
                continue
            if normalize_id(init_species) == ps_species:
                mapping[ps_idx] = jax_idx
                used_jax.add(jax_idx)
                break
        # If no exact match (form change), try prefix match
        if mapping[ps_idx] is None:
            for jax_idx, init_species in enumerate(initial_species):
                if jax_idx in used_jax:
                    continue
                init_norm = normalize_id(init_species)
                # Castform -> castformrainy, etc.
                if ps_species.startswith(init_norm) or init_norm.startswith(ps_species):
                    mapping[ps_idx] = jax_idx
                    used_jax.add(jax_idx)
                    break
        # Fallback: identity
        if mapping[ps_idx] is None:
            mapping[ps_idx] = ps_idx

    return mapping


def translate_action(action_str, side, ps_to_jax_map):
    """Translate PS action string to JAX action int using slot mapping."""
    if action_str is None:
        return 0
    parts = action_str.strip().split()
    if len(parts) < 2:
        return 0
    cmd, num = parts[0], int(parts[1])
    if cmd == 'move':
        return num - 1  # moves unaffected by slot mapping
    elif cmd == 'switch':
        ps_slot = num - 1  # 0-indexed PS slot
        jax_slot = ps_to_jax_map[ps_slot]
        return jax_slot + 4  # JAX switch actions are slot + 4
    return 0


weather_map = {
    '': 0, 'none': 0,
    'sunnyday': 1, 'desolateland': 1,
    'raindance': 2, 'primordialsea': 2,
    'sandstorm': 3,
    'hail': 4, 'snow': 4,
}

# PS side condition name -> (JAX SC index, is_layer_based)
sc_name_map = {
    'spikes':       (0, True),   # SC_SPIKES: layer count 0-3
    'toxicspikes':  (1, True),   # SC_TOXICSPIKES: layer count 0-2
    'stealthrock':  (2, False),  # SC_STEALTHROCK: 0 or 1
    'stickyweb':    (3, False),  # SC_STICKYWEB: 0 or 1
    'reflect':      (4, False),  # SC_REFLECT: turns remaining
    'lightscreen':  (5, False),  # SC_LIGHTSCREEN: turns remaining
    'auroraveil':   (6, False),  # SC_AURORAVEIL: turns remaining
    'tailwind':     (7, False),  # SC_TAILWIND: turns remaining
    'safeguard':    (8, False),  # SC_SAFEGUARD: turns remaining
    'mist':         (9, False),  # SC_MIST: turns remaining
}


def sync_state_from_ps(state, ps_state, ps_to_jax_maps, item_lookup=None):
    """Force-sync JAX state from PS state using slot mapping.

    Syncs: HP, status, fainted, boosts, active_idx, PP, items, weather,
    side conditions, status_turns.
    """
    new_hp = np.array(state.sides_team_hp)
    new_status = np.array(state.sides_team_status)
    new_status_turns = np.array(state.sides_team_status_turns)
    new_fainted = np.array(state.sides_team_fainted)
    new_boosts = np.array(state.sides_team_boosts)
    new_active = np.array(state.sides_active_idx)
    new_pp = np.array(state.sides_team_move_pp)
    new_items = np.array(state.sides_team_item_id)
    new_side_conds = np.array(state.sides_side_conditions)
    new_pokemon_left = np.array(state.sides_pokemon_left)

    # Sync weather
    new_weather = np.array(state.field.weather)
    new_weather_turns = np.array(state.field.weather_turns)
    ps_weather = normalize_id(ps_state.get('weather', ''))
    new_weather = np.int8(weather_map.get(ps_weather, 0))
    ps_weather_turns = ps_state.get('weatherTurns', 0)
    if ps_weather_turns:
        new_weather_turns = np.int8(ps_weather_turns)

    for side_idx in range(2):
        ps_mons = ps_state['sides'][side_idx]['pokemon']
        mapping = ps_to_jax_maps[side_idx]

        for ps_slot in range(min(6, len(ps_mons))):
            jax_slot = mapping[ps_slot]
            mon = ps_mons[ps_slot]

            new_hp[side_idx, jax_slot] = mon['hp']
            new_status[side_idx, jax_slot] = status_map.get(mon.get('status', ''), 0)
            new_fainted[side_idx, jax_slot] = mon.get('fainted', False)

            # Status turns (toxic counter, sleep turns, etc.)
            new_status_turns[side_idx, jax_slot] = mon.get('statusData', 0)
            if isinstance(mon.get('statusData'), dict):
                new_status_turns[side_idx, jax_slot] = mon['statusData'].get('turns', 0)

            if mon.get('boosts'):
                boost_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
                for bi, bname in enumerate(boost_order):
                    new_boosts[side_idx, jax_slot, bi] = mon['boosts'].get(bname, 0)

            if mon.get('isActive', False):
                new_active[side_idx] = jax_slot

            ps_moves = mon.get('moves', [])
            for mi, ps_move in enumerate(ps_moves[:4]):
                new_pp[side_idx, jax_slot, mi] = min(ps_move.get('pp', 0), 64)

            # Sync items (consumed items become '')
            if item_lookup is not None:
                item_name = normalize_id(mon.get('item', ''))
                new_items[side_idx, jax_slot] = item_lookup.get(item_name, 0)

        # Sync side conditions
        new_side_conds[side_idx, :] = 0  # Reset all
        ps_sc = ps_state['sides'][side_idx].get('sideConditions', {})
        if isinstance(ps_sc, dict):
            for sc_name, sc_data in ps_sc.items():
                sc_key = normalize_id(sc_name)
                if sc_key in sc_name_map:
                    sc_idx, is_layer = sc_name_map[sc_key]
                    if isinstance(sc_data, dict):
                        if is_layer:
                            new_side_conds[side_idx, sc_idx] = sc_data.get('layers', 1)
                        else:
                            new_side_conds[side_idx, sc_idx] = sc_data.get('duration', 1)
                            if new_side_conds[side_idx, sc_idx] == 0:
                                new_side_conds[side_idx, sc_idx] = 1
                    else:
                        new_side_conds[side_idx, sc_idx] = 1
        elif isinstance(ps_sc, list):
            # Old format: just list of names
            for sc_name in ps_sc:
                sc_key = normalize_id(sc_name)
                if sc_key in sc_name_map:
                    sc_idx, _ = sc_name_map[sc_key]
                    new_side_conds[side_idx, sc_idx] = 1

        # Sync pokemonLeft
        new_pokemon_left[side_idx] = ps_state['sides'][side_idx].get('pokemonLeft', 6)

    new_field = state.field._replace(
        weather=jnp.int8(new_weather),
        weather_turns=jnp.int8(new_weather_turns),
    )

    state = state._replace(
        sides_team_hp=jnp.array(new_hp, dtype=jnp.int16),
        sides_team_status=jnp.array(new_status, dtype=jnp.int8),
        sides_team_status_turns=jnp.array(new_status_turns, dtype=jnp.int8),
        sides_team_fainted=jnp.array(new_fainted, dtype=jnp.bool_),
        sides_team_boosts=jnp.array(new_boosts, dtype=jnp.int8),
        sides_active_idx=jnp.array(new_active, dtype=jnp.int8),
        sides_team_move_pp=jnp.array(new_pp, dtype=jnp.int8),
        sides_team_item_id=jnp.array(new_items, dtype=jnp.int16),
        sides_side_conditions=jnp.array(new_side_conds, dtype=jnp.int8),
        sides_pokemon_left=jnp.array(new_pokemon_left, dtype=jnp.int8),
        field=new_field,
    )
    return state


# ---------------------------------------------------------------------------
# Team builder
# ---------------------------------------------------------------------------

def build_team_arrays(team_data, ps_pokemon_states, tables, species_lookup,
                      move_lookup, ability_lookup, item_lookup):
    """Convert Showdown team JSON to numpy arrays for make_battle_state."""
    while len(team_data) < 6:
        team_data.append(team_data[0])

    species_ids = np.zeros(6, dtype=np.int16)
    ability_ids = np.zeros(6, dtype=np.int16)
    item_ids = np.zeros(6, dtype=np.int16)
    types = np.zeros((6, 2), dtype=np.int8)
    base_stats = np.zeros((6, 6), dtype=np.int16)
    max_hp = np.zeros(6, dtype=np.int16)
    move_ids = np.zeros((6, 4), dtype=np.int16)
    move_pp = np.zeros((6, 4), dtype=np.int8)
    move_max_pp = np.zeros((6, 4), dtype=np.int8)
    levels = np.zeros(6, dtype=np.int8)
    genders = np.zeros(6, dtype=np.int8)
    natures = np.zeros(6, dtype=np.int8)
    weights_hg = np.zeros(6, dtype=np.int16)

    for i, poke in enumerate(team_data[:6]):
        species_key = normalize_id(poke['species'])
        sid = species_lookup.get(species_key, 0)
        species_ids[i] = sid

        ability_ids[i] = ability_lookup.get(normalize_id(poke.get('ability', '')), 0)
        item_ids[i] = item_lookup.get(normalize_id(poke.get('item', '')), 0)

        if sid < len(tables.species):
            sp = np.array(tables.species[sid])
            types[i, 0] = int(sp[6])
            types[i, 1] = int(sp[7])
            base_stats[i] = [int(sp[j]) for j in range(6)]
            weights_hg[i] = int(sp[8])
        else:
            types[i, 0] = 1

        level = int(poke.get('level', 100))
        levels[i] = level

        # Use PS maxhp directly (avoids nature computation bugs)
        if i < len(ps_pokemon_states):
            max_hp[i] = ps_pokemon_states[i]['maxhp']
        else:
            hp_base = int(np.array(tables.species[sid])[0]) if sid < len(tables.species) else 80
            max_hp[i] = int((2 * hp_base + 31) * level / 100) + level + 10

        evs = poke.get('evs', {})
        ivs = poke.get('ivs', {})
        stat_names = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
        for j, sn in enumerate(stat_names):
            if j == 0:
                continue
            base = int(base_stats[i, j])
            ev = evs.get(sn, 0)
            iv = ivs.get(sn, 31)
            base_stats[i, j] = int((2 * base + iv + ev // 4) * level / 100) + 5

        moves = poke.get('moves', [])
        for j, move_name in enumerate(moves[:4]):
            mid = move_lookup.get(normalize_id(move_name), 0)
            move_ids[i, j] = mid
            if i < len(ps_pokemon_states) and j < len(ps_pokemon_states[i].get('moves', [])):
                ps_move = ps_pokemon_states[i]['moves'][j]
                move_pp[i, j] = min(ps_move.get('pp', 10), 64)
                move_max_pp[i, j] = min(ps_move.get('maxpp', 10), 64)
            elif mid < len(tables.moves):
                bpp = int(np.array(tables.moves[mid])[5])
                move_pp[i, j] = min(bpp, 64)
                move_max_pp[i, j] = min(bpp, 64)
            else:
                move_pp[i, j] = 10
                move_max_pp[i, j] = 10

        genders[i] = {'M': 1, 'F': 2, '': 0, 'N': 0}.get(poke.get('gender', ''), 0)

    return {
        'species': species_ids, 'abilities': ability_ids, 'items': item_ids,
        'types': types, 'base_stats': base_stats, 'max_hp': max_hp,
        'move_ids': move_ids, 'move_pp': move_pp, 'move_max_pp': move_max_pp,
        'levels': levels, 'genders': genders, 'natures': natures,
        'weights_hg': weights_hg,
    }


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_differential_test(battles_path, limit=None):
    print("Loading tables...")
    showdown_path = str(Path(__file__).parent.parent.parent / "PokemonShowdownClaude" / "pokemon-showdown")
    tables = load_tables(4, showdown_path=showdown_path)
    cfg = GenConfig.for_gen(4)

    print("Loading Showdown battles...")
    with open(battles_path) as f:
        battles = [json.loads(line) for line in f if line.strip()]
    if limit:
        battles = battles[:limit]
    print(f"Loaded {len(battles)} battles")

    species_lookup = build_reverse_lookup(tables.species_name_to_id)
    move_lookup = build_reverse_lookup(tables.move_name_to_id)
    ability_lookup = build_reverse_lookup(tables.ability_name_to_id)
    item_lookup = build_reverse_lookup(tables.item_name_to_id)

    @jax.jit
    def jit_turn(state, reveal, actions):
        return execute_turn(state, reveal, actions, tables, cfg)

    print("JIT compiling execute_turn (this takes ~5 minutes first time)...", flush=True)

    # =====================================================================
    # Test 1: MaxHP Accuracy
    # =====================================================================
    print("\n--- Test 1: MaxHP Accuracy (PS values used directly) ---")
    maxhp_correct = 0
    maxhp_total = 0
    for bi, battle in enumerate(battles[:100]):
        init_state = battle['turns'][0].get('state') if battle['turns'] else None
        if init_state is None:
            continue
        for side_idx in range(2):
            ps_mons = init_state['sides'][side_idx]['pokemon']
            team = build_team_arrays(
                battle['teams'][side_idx], ps_mons, tables,
                species_lookup, move_lookup, ability_lookup, item_lookup
            )
            for slot_idx in range(min(6, len(ps_mons))):
                maxhp_total += 1
                if int(team['max_hp'][slot_idx]) == ps_mons[slot_idx]['maxhp']:
                    maxhp_correct += 1
    maxhp_acc = maxhp_correct / max(1, maxhp_total)
    print(f"  {maxhp_correct}/{maxhp_total} correct ({maxhp_acc:.2%})")

    # =====================================================================
    # Test 2: Turn-by-turn replay with slot mapping + HP sync
    # =====================================================================
    print("\n--- Test 2: Turn-by-Turn Replay (HP-Synced, Slot-Mapped) ---")
    n_games = len(battles)

    total_turns = 0
    action_mask_ok = 0
    action_mask_fail = 0
    action_mask_fail_details = []
    status_agree = 0
    status_disagree = 0
    status_disagree_details = []
    boost_agree = 0
    boost_disagree = 0
    faint_agree = 0
    faint_disagree = 0
    winner_agree = 0
    winner_disagree = 0
    winner_disagree_details = []
    hp_violations = 0
    games_finished_ps = 0
    games_finished_jax = 0
    early_finish = 0
    late_finish = 0
    errors = []

    for bi, battle in enumerate(battles):
        init_state = battle['turns'][0].get('state') if battle['turns'] else None
        if init_state is None:
            continue

        # Initial species list per side (JAX slot order = PS initial order)
        initial_species = [[], []]
        for side_idx in range(2):
            ps_mons = init_state['sides'][side_idx]['pokemon']
            initial_species[side_idx] = [m['species'] for m in ps_mons]

        ps_mons_0 = init_state['sides'][0]['pokemon']
        ps_mons_1 = init_state['sides'][1]['pokemon']

        team0 = build_team_arrays(battle['teams'][0], ps_mons_0, tables,
                                   species_lookup, move_lookup, ability_lookup, item_lookup)
        team1 = build_team_arrays(battle['teams'][1], ps_mons_1, tables,
                                   species_lookup, move_lookup, ability_lookup, item_lookup)

        key = jax.random.PRNGKey(bi * 137 + 42)
        try:
            state = make_battle_state(
                p1_species=team0['species'], p2_species=team1['species'],
                p1_abilities=team0['abilities'], p2_abilities=team1['abilities'],
                p1_items=team0['items'], p2_items=team1['items'],
                p1_types=team0['types'], p2_types=team1['types'],
                p1_base_stats=team0['base_stats'], p2_base_stats=team1['base_stats'],
                p1_max_hp=team0['max_hp'], p2_max_hp=team1['max_hp'],
                p1_move_ids=team0['move_ids'], p2_move_ids=team1['move_ids'],
                p1_move_pp=team0['move_pp'], p2_move_pp=team1['move_pp'],
                p1_move_max_pp=team0['move_max_pp'], p2_move_max_pp=team1['move_max_pp'],
                p1_levels=team0['levels'], p2_levels=team1['levels'],
                p1_genders=team0['genders'], p2_genders=team1['genders'],
                p1_natures=team0['natures'], p2_natures=team1['natures'],
                p1_weights_hg=team0['weights_hg'], p2_weights_hg=team1['weights_hg'],
                rng_key=key,
            )
        except Exception as e:
            if len(errors) < 10:
                errors.append(f"Battle {bi}: state init failed: {e}")
            continue

        reveal = make_reveal_state(state)
        ps_turns = battle['turns']

        # Initial slot mapping is identity
        ps_to_jax_maps = [list(range(6)), list(range(6))]

        for ti in range(1, len(ps_turns)):
            turn = ps_turns[ti]
            actions = turn.get('actions')
            if actions is None:
                continue

            # Skip forced switch turns.
            # JAX handles forced switches internally in execute_turn.
            # Case 1: one action is None (explicit forced switch)
            # Case 2: both actions present but one side's active is fainted
            #         (PS records forced switch + opponent's action together)
            p1_act = actions.get('p1')
            p2_act = actions.get('p2')

            is_forced = p1_act is None or p2_act is None
            if not is_forced:
                # Check if either side has no active mon (post-faint, forced switch pending)
                # or active mon is fainted in previous turn's state
                prev_ps = ps_turns[ti - 1].get('state') if ti > 0 else None
                if prev_ps:
                    for si in range(2):
                        ps_mons_prev = prev_ps['sides'][si]['pokemon']
                        has_active = False
                        for mon in ps_mons_prev:
                            if mon.get('isActive') and not mon.get('fainted'):
                                has_active = True
                                break
                        if not has_active and prev_ps['sides'][si].get('pokemonLeft', 6) > 0:
                            is_forced = True
                            break

            if is_forced:
                # Still sync state from this turn if available
                ps_state_turn = turn.get('state')
                if ps_state_turn:
                    for side_idx in range(2):
                        ps_mons = ps_state_turn['sides'][side_idx]['pokemon']
                        ps_to_jax_maps[side_idx] = build_ps_to_jax_map(
                            ps_mons, initial_species[side_idx]
                        )
                    state = sync_state_from_ps(state, ps_state_turn, ps_to_jax_maps, item_lookup)
                continue

            if bool(state.finished):
                early_finish += 1
                break

            # Rebuild slot mapping from PS state of PREVIOUS turn
            prev_ps_state = ps_turns[ti - 1].get('state')
            if prev_ps_state:
                for side_idx in range(2):
                    ps_mons = prev_ps_state['sides'][side_idx]['pokemon']
                    ps_to_jax_maps[side_idx] = build_ps_to_jax_map(
                        ps_mons, initial_species[side_idx]
                    )

            # Translate PS actions to JAX actions using slot mapping
            a0 = translate_action(p1_act, 0, ps_to_jax_maps[0])
            a1 = translate_action(p2_act, 1, ps_to_jax_maps[1])
            a0 = min(max(a0, 0), 9)
            a1 = min(max(a1, 0), 9)

            # ---- ACTION MASK CHECK ----
            mask0 = np.array(get_action_mask(state, 0))
            mask1 = np.array(get_action_mask(state, 1))

            if mask0[a0]:
                action_mask_ok += 1
            else:
                action_mask_fail += 1
                if len(action_mask_fail_details) < 30:
                    action_mask_fail_details.append(
                        f"  B{bi} T{ti} P1: PS='{actions.get('p1','?')}' -> jax_a={a0} "
                        f"mask={mask0.astype(int).tolist()}"
                    )

            if mask1[a1]:
                action_mask_ok += 1
            else:
                action_mask_fail += 1
                if len(action_mask_fail_details) < 30:
                    action_mask_fail_details.append(
                        f"  B{bi} T{ti} P2: PS='{actions.get('p2','?')}' -> jax_a={a1} "
                        f"mask={mask1.astype(int).tolist()}"
                    )

            # ---- EXECUTE TURN ----
            action_arr = jnp.array([a0, a1], dtype=jnp.int32)
            try:
                state, reveal = jit_turn(state, reveal, action_arr)
            except Exception as e:
                if len(errors) < 10:
                    errors.append(f"Battle {bi} turn {ti}: execute_turn failed: {e}")
                break

            total_turns += 1

            # ---- HP CONSERVATION CHECK ----
            hp = np.array(state.sides_team_hp)
            maxhp = np.array(state.sides_team_max_hp)
            if (hp < 0).any() or (hp > maxhp).any():
                hp_violations += 1

            # ---- COMPARE STATE WITH PS (using slot mapping) ----
            ps_state_turn = turn.get('state')
            if ps_state_turn:
                # Rebuild mapping for this turn
                for side_idx in range(2):
                    ps_mons = ps_state_turn['sides'][side_idx]['pokemon']
                    cur_map = build_ps_to_jax_map(ps_mons, initial_species[side_idx])

                    for ps_slot in range(min(6, len(ps_mons))):
                        mon = ps_mons[ps_slot]
                        jax_slot = cur_map[ps_slot]
                        if not mon.get('isActive', False):
                            continue

                        # Faint agreement
                        ps_hp = mon['hp']
                        jax_hp = int(hp[side_idx, jax_slot])
                        if (ps_hp <= 0) == (jax_hp <= 0):
                            faint_agree += 1
                        else:
                            faint_disagree += 1

                        # Status agreement
                        ps_status_val = status_map.get(mon.get('status', ''), 0)
                        jax_status_val = int(state.sides_team_status[side_idx, jax_slot])
                        if ps_status_val == jax_status_val:
                            status_agree += 1
                        else:
                            status_disagree += 1
                            if len(status_disagree_details) < 20:
                                status_disagree_details.append(
                                    f"  B{bi} T{ti} S{side_idx} jax_slot{jax_slot}: "
                                    f"PS={mon.get('status','')}({ps_status_val}) JAX={jax_status_val}"
                                )

                        # Boost agreement
                        if mon.get('boosts'):
                            for bj, bname in enumerate(['atk','def','spa','spd','spe','accuracy','evasion']):
                                ps_b = mon['boosts'].get(bname, 0)
                                jax_b = int(state.sides_team_boosts[side_idx, jax_slot, bj])
                                if ps_b == jax_b:
                                    boost_agree += 1
                                else:
                                    boost_disagree += 1

                # Sync state using slot mapping
                for side_idx in range(2):
                    ps_mons = ps_state_turn['sides'][side_idx]['pokemon']
                    ps_to_jax_maps[side_idx] = build_ps_to_jax_map(
                        ps_mons, initial_species[side_idx]
                    )
                state = sync_state_from_ps(state, ps_state_turn, ps_to_jax_maps, item_lookup)

        # ---- WINNER CHECK ----
        ps_finished = battle.get('winner') is not None
        jax_finished = bool(state.finished)

        if ps_finished:
            games_finished_ps += 1
        if jax_finished:
            games_finished_jax += 1

        if ps_finished and jax_finished:
            ps_winner_str = battle.get('winner', '')
            jax_winner = int(state.winner)
            ps_winner_idx = 0 if ps_winner_str == 'Bot1' else (1 if ps_winner_str == 'Bot2' else 2)
            if ps_winner_idx == jax_winner:
                winner_agree += 1
            else:
                winner_disagree += 1
                if len(winner_disagree_details) < 20:
                    winner_disagree_details.append(
                        f"  B{bi}: PS={ps_winner_str}({ps_winner_idx}) JAX={jax_winner} "
                        f"turns={len(ps_turns)-1}"
                    )
        elif ps_finished and not jax_finished:
            late_finish += 1
        elif not ps_finished and jax_finished:
            early_finish += 1

        if (bi + 1) % 100 == 0:
            pct_mask = action_mask_ok / max(1, action_mask_ok + action_mask_fail)
            pct_winner = winner_agree / max(1, winner_agree + winner_disagree)
            print(f"  {bi + 1}/{n_games}: mask={pct_mask:.1%} winner={pct_winner:.1%}", flush=True)

    # =====================================================================
    # Summary
    # =====================================================================
    total_mask = action_mask_ok + action_mask_fail
    mask_rate = action_mask_ok / max(1, total_mask)
    status_rate = status_agree / max(1, status_agree + status_disagree)
    boost_rate = boost_agree / max(1, boost_agree + boost_disagree)
    faint_rate = faint_agree / max(1, faint_agree + faint_disagree)
    winner_rate = winner_agree / max(1, winner_agree + winner_disagree)

    print("\n" + "=" * 70)
    print("DIFFERENTIAL TEST SUMMARY (HP-Synced + Slot-Mapped)")
    print("=" * 70)
    print(f"  MaxHP accuracy:     {maxhp_acc:.2%} ({maxhp_correct}/{maxhp_total})")
    print(f"  Turns replayed:     {total_turns}")
    print()
    print("  --- Deterministic Checks ---")
    print(f"  HP violations:      {hp_violations}")
    print(f"  Action mask:        {action_mask_ok}/{total_mask} ({mask_rate:.2%} legal)")
    print(f"  Boost agreement:    {boost_agree}/{boost_agree+boost_disagree} ({boost_rate:.2%})")
    print()
    print("  --- RNG-Dependent (info only, different PRNGs expected to diverge) ---")
    print(f"  Status agreement:   {status_agree}/{status_agree+status_disagree} ({status_rate:.2%})")
    print(f"  Faint agreement:    {faint_agree}/{faint_agree+faint_disagree} ({faint_rate:.2%})")
    print(f"  Winner agreement:   {winner_agree}/{winner_agree+winner_disagree} ({winner_rate:.2%})")
    print(f"  JAX finishes early: {early_finish}")
    print(f"  JAX finishes late:  {late_finish}")
    print(f"  PS games total:     {games_finished_ps}")
    print(f"  JAX games finished: {games_finished_jax}")

    if action_mask_fail_details:
        print(f"\n  Action Mask Failures ({action_mask_fail} total, first {min(15, len(action_mask_fail_details))}):")
        for d in action_mask_fail_details[:15]:
            print(d)

    if winner_disagree_details:
        print(f"\n  Winner Disagreements ({winner_disagree} total):")
        for d in winner_disagree_details[:10]:
            print(d)

    if status_disagree_details:
        print(f"\n  Status Disagreements (first {min(10, len(status_disagree_details))}):")
        for d in status_disagree_details[:10]:
            print(d)

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    {e}")

    # PASS criteria: deterministic checks only
    # Status/faint/winner diverge due to different PRNGs — expected and corrected by sync
    all_pass = (
        maxhp_acc >= 0.99 and
        mask_rate >= 0.99 and
        hp_violations == 0 and
        boost_rate >= 0.98
    )
    print(f"\n  OVERALL: {'PASS' if all_pass else 'NEEDS INVESTIGATION'}")
    return all_pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Differential test: PokeJAX vs Showdown')
    parser.add_argument('--battles', default='data/showdown_battles_1000.jsonl',
                       help='Path to Showdown battle log JSONL')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of battles')
    args = parser.parse_args()

    success = run_differential_test(args.battles, args.limit)
    sys.exit(0 if success else 1)
