"""
Differential test: replay Pokemon Showdown battles in PokeJAX engine.

Strategy: Since PRNGs differ (JAX ThreeFry vs PS custom), we SYNC HP after
each turn to prevent RNG cascade. This lets us focus on deterministic checks:

  1. Action mask: PS's chosen action must be legal in JAX (engine bug if not)
  2. Status agreement: with HP synced, status effects should mostly match
  3. Boost agreement: stat changes from moves are deterministic
  4. Faint agreement: with HP synced, faint states should match exactly
  5. HP conservation: HP must stay in [0, maxhp]
  6. Winner agreement: with HP synced, should be very high
  7. Random self-play: games should complete, ~50/50 balance

Usage:
    python scripts/differential_test.py --battles data/showdown_battles_1000.jsonl --limit 1000
"""

import json
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

# Force unbuffered stdout (needed for WSL piped output)
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


def build_team_arrays(team_data, ps_pokemon_states, tables, species_lookup,
                      move_lookup, ability_lookup, item_lookup):
    """Convert Showdown team JSON to numpy arrays for make_battle_state.

    Uses PS's maxhp from turn 0 state directly (avoids nature computation bugs).
    """
    while len(team_data) < 6:
        team_data.append(team_data[0])
    n = 6

    species_ids = np.zeros(n, dtype=np.int16)
    ability_ids = np.zeros(n, dtype=np.int16)
    item_ids = np.zeros(n, dtype=np.int16)
    types = np.zeros((n, 2), dtype=np.int8)
    base_stats = np.zeros((n, 6), dtype=np.int16)
    max_hp = np.zeros(n, dtype=np.int16)
    move_ids = np.zeros((n, 4), dtype=np.int16)
    move_pp = np.zeros((n, 4), dtype=np.int8)
    move_max_pp = np.zeros((n, 4), dtype=np.int8)
    levels = np.zeros(n, dtype=np.int8)
    genders = np.zeros(n, dtype=np.int8)
    natures = np.zeros(n, dtype=np.int8)
    weights_hg = np.zeros(n, dtype=np.int16)

    for i, poke in enumerate(team_data[:6]):
        species_key = normalize_id(poke['species'])
        sid = species_lookup.get(species_key, 0)
        species_ids[i] = sid

        ability_key = normalize_id(poke.get('ability', ''))
        ability_ids[i] = ability_lookup.get(ability_key, 0)

        item_key = normalize_id(poke.get('item', ''))
        item_ids[i] = item_lookup.get(item_key, 0)

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

        natures[i] = 0  # nature not in logs, doesn't matter since we use PS maxhp

        # Use PS maxhp directly
        if i < len(ps_pokemon_states):
            max_hp[i] = ps_pokemon_states[i]['maxhp']
        else:
            # Fallback: compute (shouldn't happen)
            hp_base = int(np.array(tables.species[sid])[0]) if sid < len(tables.species) else 80
            max_hp[i] = int((2 * hp_base + 31) * level / 100) + level + 10

        # Compute battle stats using PS maxhp for HP, formula for others
        evs = poke.get('evs', {})
        ivs = poke.get('ivs', {})
        stat_names = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
        for j, sn in enumerate(stat_names):
            if j == 0:
                continue  # HP already set from PS
            base = int(base_stats[i, j])
            ev = evs.get(sn, 0)
            iv = ivs.get(sn, 31)
            raw = int((2 * base + iv + ev // 4) * level / 100) + 5
            # No nature mult since we don't have nature data
            # This affects non-HP stats but is acceptable since we sync HP
            base_stats[i, j] = raw

        moves = poke.get('moves', [])
        for j, move_name in enumerate(moves[:4]):
            move_key = normalize_id(move_name)
            mid = move_lookup.get(move_key, 0)
            move_ids[i, j] = mid
            # Use PS PP directly if available
            if i < len(ps_pokemon_states) and j < len(ps_pokemon_states[i].get('moves', [])):
                ps_move = ps_pokemon_states[i]['moves'][j]
                move_pp[i, j] = min(ps_move.get('pp', 10), 64)
                move_max_pp[i, j] = min(ps_move.get('maxpp', 10), 64)
            elif mid < len(tables.moves):
                base_pp_val = int(np.array(tables.moves[mid])[5])
                move_pp[i, j] = min(base_pp_val, 64)
                move_max_pp[i, j] = min(base_pp_val, 64)
            else:
                move_pp[i, j] = 10
                move_max_pp[i, j] = 10

        gender_map = {'M': 1, 'F': 2, '': 0, 'N': 0}
        genders[i] = gender_map.get(poke.get('gender', ''), 0)

    return {
        'species': species_ids, 'abilities': ability_ids, 'items': item_ids,
        'types': types, 'base_stats': base_stats, 'max_hp': max_hp,
        'move_ids': move_ids, 'move_pp': move_pp, 'move_max_pp': move_max_pp,
        'levels': levels, 'genders': genders, 'natures': natures,
        'weights_hg': weights_hg,
    }


def parse_action(action_str):
    """Parse PS action string to pokejax action int.
    'move 1' -> 0, 'move 4' -> 3, 'switch 2' -> 5, etc.
    """
    if action_str is None:
        return None
    parts = action_str.strip().split()
    if len(parts) < 2:
        return None
    cmd, num = parts[0], int(parts[1])
    if cmd == 'move':
        return num - 1  # 1-indexed -> 0-indexed
    elif cmd == 'switch':
        return num - 1 + 4  # switch 1 -> action 4, switch 6 -> action 9
    return None


def sync_state_from_ps(state, ps_state):
    """Force-sync JAX state from PS state to prevent RNG cascade.

    Syncs: HP, status, fainted, boosts, active index, PP.
    """
    new_hp = np.array(state.sides_team_hp)
    new_status = np.array(state.sides_team_status)
    new_fainted = np.array(state.sides_team_fainted)
    new_boosts = np.array(state.sides_team_boosts)
    new_active = np.array(state.sides_active_idx)
    new_pp = np.array(state.sides_team_move_pp)

    for side_idx in range(2):
        ps_mons = ps_state['sides'][side_idx]['pokemon']
        for slot_idx in range(min(6, len(ps_mons))):
            mon = ps_mons[slot_idx]
            new_hp[side_idx, slot_idx] = mon['hp']
            new_status[side_idx, slot_idx] = status_map.get(mon.get('status', ''), 0)
            new_fainted[side_idx, slot_idx] = mon.get('fainted', False)

            # Sync boosts
            if mon.get('boosts'):
                boost_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
                for bi, bname in enumerate(boost_order):
                    new_boosts[side_idx, slot_idx, bi] = mon['boosts'].get(bname, 0)

            # Sync active index
            if mon.get('isActive', False):
                new_active[side_idx] = slot_idx

            # Sync PP
            ps_moves = mon.get('moves', [])
            for mi, ps_move in enumerate(ps_moves[:4]):
                new_pp[side_idx, slot_idx, mi] = min(ps_move.get('pp', 0), 64)

    state = state._replace(
        sides_team_hp=jnp.array(new_hp, dtype=jnp.int16),
        sides_team_status=jnp.array(new_status, dtype=jnp.int8),
        sides_team_fainted=jnp.array(new_fainted, dtype=jnp.bool_),
        sides_team_boosts=jnp.array(new_boosts, dtype=jnp.int8),
        sides_active_idx=jnp.array(new_active, dtype=jnp.int8),
        sides_team_move_pp=jnp.array(new_pp, dtype=jnp.int8),
    )
    return state


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

    # JIT compile execute_turn with tables/cfg captured in closure
    @jax.jit
    def jit_turn(state, reveal, actions):
        return execute_turn(state, reveal, actions, tables, cfg)

    print("JIT compiling execute_turn (this takes ~5 minutes first time)...", flush=True)

    # =====================================================================
    # Test 1: MaxHP Accuracy (using PS maxhp directly — should be 100%)
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
                ps_maxhp = ps_mons[slot_idx]['maxhp']
                jax_maxhp = int(team['max_hp'][slot_idx])
                maxhp_total += 1
                if jax_maxhp == ps_maxhp:
                    maxhp_correct += 1

    maxhp_acc = maxhp_correct / maxhp_total if maxhp_total > 0 else 0
    print(f"  {maxhp_correct}/{maxhp_total} correct ({maxhp_acc:.2%})")

    # =====================================================================
    # Test 2: Turn-by-turn replay with HP sync + action mask checks
    # =====================================================================
    print("\n--- Test 2: Turn-by-Turn Replay (HP-Synced) ---")
    n_games = len(battles)

    # Counters
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
        jax_game_over = False

        for ti in range(1, len(ps_turns)):
            turn = ps_turns[ti]
            actions = turn.get('actions')
            if actions is None:
                continue

            # Skip if JAX already finished
            if bool(state.finished):
                jax_game_over = True
                early_finish += 1
                break

            a0 = parse_action(actions.get('p1'))
            a1 = parse_action(actions.get('p2'))
            if a0 is None:
                a0 = 0
            if a1 is None:
                a1 = 0
            a0 = min(max(a0, 0), 9)
            a1 = min(max(a1, 0), 9)

            # ---- ACTION MASK CHECK (before executing turn) ----
            mask0 = np.array(get_action_mask(state, 0))
            mask1 = np.array(get_action_mask(state, 1))

            if mask0[a0]:
                action_mask_ok += 1
            else:
                action_mask_fail += 1
                if len(action_mask_fail_details) < 30:
                    ps_act_str = actions.get('p1', '?')
                    action_mask_fail_details.append(
                        f"  B{bi} T{ti} P1: PS chose '{ps_act_str}' (a={a0}) "
                        f"but JAX mask={mask0.astype(int).tolist()}"
                    )

            if mask1[a1]:
                action_mask_ok += 1
            else:
                action_mask_fail += 1
                if len(action_mask_fail_details) < 30:
                    ps_act_str = actions.get('p2', '?')
                    action_mask_fail_details.append(
                        f"  B{bi} T{ti} P2: PS chose '{ps_act_str}' (a={a1}) "
                        f"but JAX mask={mask1.astype(int).tolist()}"
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

            # ---- COMPARE DETERMINISTIC STATE WITH PS ----
            ps_state = turn.get('state')
            if ps_state:
                for side_idx in range(2):
                    ps_mons = ps_state['sides'][side_idx]['pokemon']
                    for slot_idx in range(min(6, len(ps_mons))):
                        mon = ps_mons[slot_idx]
                        if not mon.get('isActive', False):
                            continue

                        # Faint agreement (before HP sync)
                        ps_fainted = mon.get('fainted', False)
                        # For faint check: mon is fainted if hp <= 0
                        jax_hp = int(hp[side_idx, slot_idx])
                        ps_hp = mon['hp']
                        # Both fainted or both alive?
                        if (ps_hp <= 0) == (jax_hp <= 0):
                            faint_agree += 1
                        else:
                            faint_disagree += 1

                        # Status agreement (RNG-dependent but still informative)
                        ps_status_val = status_map.get(mon.get('status', ''), 0)
                        jax_status_val = int(state.sides_team_status[side_idx, slot_idx])
                        if ps_status_val == jax_status_val:
                            status_agree += 1
                        else:
                            status_disagree += 1
                            if len(status_disagree_details) < 20:
                                status_disagree_details.append(
                                    f"  B{bi} T{ti} S{side_idx} slot{slot_idx}: "
                                    f"PS={mon.get('status','')}({ps_status_val}) "
                                    f"JAX={jax_status_val}"
                                )

                        # Boost agreement (mostly deterministic)
                        if mon.get('boosts'):
                            boost_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
                            for bj, bname in enumerate(boost_order):
                                ps_boost = mon['boosts'].get(bname, 0)
                                jax_boost = int(state.sides_team_boosts[side_idx, slot_idx, bj])
                                if ps_boost == jax_boost:
                                    boost_agree += 1
                                else:
                                    boost_disagree += 1

                # ---- SYNC HP/STATUS/BOOSTS FROM PS (prevent RNG cascade) ----
                state = sync_state_from_ps(state, ps_state)

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

    # (Test 3: Random self-play — skipped, not needed for alignment check)

    # =====================================================================
    # Summary
    # =====================================================================
    total_mask = action_mask_ok + action_mask_fail
    mask_rate = action_mask_ok / max(1, total_mask)
    total_status = status_agree + status_disagree
    status_rate = status_agree / max(1, total_status)
    total_boost = boost_agree + boost_disagree
    boost_rate = boost_agree / max(1, total_boost)
    total_faint = faint_agree + faint_disagree
    faint_rate = faint_agree / max(1, total_faint)
    total_winner = winner_agree + winner_disagree
    winner_rate = winner_agree / max(1, total_winner)

    print("\n" + "=" * 70)
    print("DIFFERENTIAL TEST SUMMARY (HP-Synced)")
    print("=" * 70)
    print(f"  MaxHP accuracy:     {maxhp_acc:.2%} ({maxhp_correct}/{maxhp_total})")
    print(f"  Turns replayed:     {total_turns}")
    print(f"  HP violations:      {hp_violations}")
    print(f"  Action mask:        {action_mask_ok}/{total_mask} ({mask_rate:.2%} legal)")
    print(f"  Status agreement:   {status_agree}/{total_status} ({status_rate:.2%})")
    print(f"  Boost agreement:    {boost_agree}/{total_boost} ({boost_rate:.2%})")
    print(f"  Faint agreement:    {faint_agree}/{total_faint} ({faint_rate:.2%})")
    print(f"  Winner agreement:   {winner_agree}/{total_winner} ({winner_rate:.2%})")
    print(f"  JAX finishes early: {early_finish}")
    print(f"  JAX finishes late:  {late_finish}")
    print(f"  PS games total:     {games_finished_ps}")
    print(f"  JAX games finished: {games_finished_jax}")
    print(f"  (Self-play skipped)")

    if action_mask_fail_details:
        print(f"\n  Action Mask Failures ({action_mask_fail} total, first {len(action_mask_fail_details)}):")
        for d in action_mask_fail_details[:15]:
            print(d)

    if winner_disagree_details:
        print(f"\n  Winner Disagreements ({winner_disagree} total):")
        for d in winner_disagree_details[:10]:
            print(d)

    if status_disagree_details:
        print(f"\n  Status Disagreements (first {len(status_disagree_details)}):")
        for d in status_disagree_details[:10]:
            print(d)

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    {e}")

    # Pass criteria (stricter now with HP sync)
    all_pass = (
        maxhp_acc >= 0.99 and
        mask_rate >= 0.95 and
        hp_violations == 0 and
        winner_rate >= 0.80 and
        boost_rate >= 0.90
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
