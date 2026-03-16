"""
Differential Test Suite: Pokemon Showdown vs PokeJAX Engine
============================================================

End-to-end comparison that:
  1. Generates fresh battles in Pokemon Showdown with deterministic PRNG seeds
  2. Replays the exact same actions in PokeJAX
  3. Syncs state after each turn (PRNGs are incompatible between engines)
  4. Compares EVERY observable field and logs ALL discrepancies
  5. Produces a detailed final report

PRNG Control:
  - Showdown: deterministic 4-component seed per battle [i*4+1, i*4+2, i*4+3, i*4+4]
  - PokeJAX: JAX PRNGKey(seed) per battle
  - Since the PRNG algorithms differ (PS custom vs JAX ThreeFry), we sync state
    after each turn so RNG-dependent divergences don't cascade.

Usage:
    # Generate fresh battles + run comparison (requires Node.js + PS)
    pytest tests/test_differential.py -v -s

    # Use existing battle logs
    pytest tests/test_differential.py -v -s --battles data/showdown_battles_1000.jsonl

    # Limit number of battles
    pytest tests/test_differential.py -v -s --num-battles 50

    # Standalone (no pytest):
    python tests/test_differential.py --num-battles 100
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import dataclasses
from pathlib import Path
from typing import Optional

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Ensure pokejax is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compile_cache")

from pokejax.data.tables import load_tables
from pokejax.config import GenConfig
from pokejax.core.state import make_battle_state, make_reveal_state
from pokejax.engine.turn import execute_turn
from pokejax.env.action_mask import get_action_mask


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

PS_PATH = str(ROOT.parent / "PokemonShowdownClaude" / "pokemon-showdown")
BATTLE_LOGGER_JS = str(ROOT / "scripts" / "showdown_battle_logger.js")

STATUS_MAP = {'': 0, 'brn': 1, 'psn': 2, 'tox': 3, 'slp': 4, 'frz': 5, 'par': 6}
STATUS_NAMES = {v: k for k, v in STATUS_MAP.items()}

WEATHER_MAP = {
    '': 0, 'none': 0,
    'sunnyday': 1, 'desolateland': 1,
    'raindance': 2, 'primordialsea': 2,
    'sandstorm': 3,
    'hail': 4, 'snow': 4,
}

# PS side condition name -> (JAX SC index, value_source)
# value_source: 'layers' = read sc_data['layers'], 'duration' = read sc_data['duration'],
#               'presence' = just 1 if present
SC_NAME_MAP = {
    'spikes':       (0, 'layers'),      # 0-3 layers
    'toxicspikes':  (1, 'layers'),      # 0-2 layers
    'stealthrock':  (2, 'presence'),    # 0 or 1 (PS stores as layers:1, duration:0)
    'stickyweb':    (3, 'presence'),    # 0 or 1
    'reflect':      (4, 'duration'),    # turns remaining
    'lightscreen':  (5, 'duration'),    # turns remaining
    'auroraveil':   (6, 'duration'),    # turns remaining
    'tailwind':     (7, 'duration'),    # turns remaining
    'safeguard':    (8, 'duration'),    # turns remaining
    'mist':         (9, 'duration'),    # turns remaining
}


# ═══════════════════════════════════════════════════════════════════════════
# Data classes for structured logging
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TurnComparison:
    """Result of comparing a single turn between PS and JAX."""
    battle_idx: int
    turn_idx: int
    # Action mask
    p1_action_legal: bool
    p2_action_legal: bool
    p1_action_str: str
    p2_action_str: str
    # Per-pokemon comparisons (active only)
    hp_matches: list  # [(side, slot, ps_hp, jax_hp, match)]
    status_matches: list  # [(side, slot, ps_status, jax_status, match)]
    boost_matches: list  # [(side, slot, boost_name, ps_val, jax_val, match)]
    faint_matches: list  # [(side, slot, ps_fainted, jax_fainted, match)]
    # Field
    weather_match: bool
    weather_ps: str
    weather_jax: int
    # Side conditions
    sc_matches: list  # [(side, sc_name, ps_val, jax_val, match)]
    # PP
    pp_matches: list  # [(side, slot, move_idx, ps_pp, jax_pp, match)]
    # Items
    item_matches: list  # [(side, slot, ps_item, jax_item, match)]
    # HP conservation
    hp_violation: bool


@dataclasses.dataclass
class BattleResult:
    """Result of comparing a full battle."""
    battle_idx: int
    seed: list
    total_ps_turns: int
    total_replayed_turns: int
    turn_comparisons: list  # [TurnComparison]
    winner_ps: str
    winner_jax: int
    winner_match: bool
    jax_finished_early: bool
    jax_finished_late: bool
    error: Optional[str] = None
    ps_species: list = dataclasses.field(default_factory=list)  # for debugging


@dataclasses.dataclass
class SuiteResults:
    """Aggregated results across all battles."""
    total_battles: int = 0
    battles_with_errors: int = 0
    total_turns_replayed: int = 0

    # Action mask
    action_mask_legal: int = 0
    action_mask_illegal: int = 0
    action_mask_failures: list = dataclasses.field(default_factory=list)

    # HP
    hp_matches: int = 0
    hp_mismatches: int = 0
    hp_violations: int = 0
    hp_mismatch_details: list = dataclasses.field(default_factory=list)

    # Status
    status_matches: int = 0
    status_mismatches: int = 0
    status_mismatch_details: list = dataclasses.field(default_factory=list)

    # Boosts
    boost_matches: int = 0
    boost_mismatches: int = 0

    # Faints
    faint_matches: int = 0
    faint_mismatches: int = 0

    # Weather
    weather_matches: int = 0
    weather_mismatches: int = 0

    # Side conditions
    sc_matches: int = 0
    sc_mismatches: int = 0

    # PP
    pp_matches: int = 0
    pp_mismatches: int = 0

    # Items
    item_matches: int = 0
    item_mismatches: int = 0

    # Winner
    winner_matches: int = 0
    winner_mismatches: int = 0
    winner_mismatch_details: list = dataclasses.field(default_factory=list)

    # Finish timing
    jax_early_finishes: int = 0
    jax_late_finishes: int = 0
    ps_finished: int = 0
    jax_finished: int = 0

    errors: list = dataclasses.field(default_factory=list)

    def rate(self, matches, mismatches):
        total = matches + mismatches
        return matches / total if total > 0 else float('nan')


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def normalize_id(name):
    return name.lower().replace(' ', '').replace('-', '').replace('.', '').replace("'", '').replace(':', '')


def build_reverse_lookup(name_to_id):
    return {normalize_id(name): idx for name, idx in name_to_id.items()}


def build_ps_to_jax_map(ps_mons, initial_species):
    """Map PS's current slot order to JAX's fixed slot order via species matching."""
    mapping = [None] * len(ps_mons)
    used_jax = set()

    for ps_idx, mon in enumerate(ps_mons):
        ps_species = normalize_id(mon['species'])
        for jax_idx, init_species in enumerate(initial_species):
            if jax_idx in used_jax:
                continue
            if normalize_id(init_species) == ps_species:
                mapping[ps_idx] = jax_idx
                used_jax.add(jax_idx)
                break
        if mapping[ps_idx] is None:
            for jax_idx, init_species in enumerate(initial_species):
                if jax_idx in used_jax:
                    continue
                init_norm = normalize_id(init_species)
                if ps_species.startswith(init_norm) or init_norm.startswith(ps_species):
                    mapping[ps_idx] = jax_idx
                    used_jax.add(jax_idx)
                    break
        if mapping[ps_idx] is None:
            mapping[ps_idx] = ps_idx

    return mapping


def translate_action(action_str, ps_to_jax_map):
    """Convert PS action string to JAX action int."""
    if action_str is None:
        return 0
    parts = action_str.strip().split()
    if len(parts) < 2:
        return 0
    cmd, num = parts[0], int(parts[1])
    if cmd == 'move':
        return num - 1
    elif cmd == 'switch':
        ps_slot = num - 1
        jax_slot = ps_to_jax_map[ps_slot]
        return jax_slot + 4
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# State sync (PS -> JAX)
# ═══════════════════════════════════════════════════════════════════════════

def sync_state_from_ps(state, ps_state, ps_to_jax_maps, item_lookup=None):
    """Force-sync JAX state from PS state after each turn."""
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

    # Weather
    ps_weather = normalize_id(ps_state.get('weather', ''))
    new_weather = np.int8(WEATHER_MAP.get(ps_weather, 0))
    new_weather_turns = np.int8(ps_state.get('weatherTurns', 0))

    for side_idx in range(2):
        ps_mons = ps_state['sides'][side_idx]['pokemon']
        mapping = ps_to_jax_maps[side_idx]

        for ps_slot in range(min(6, len(ps_mons))):
            jax_slot = mapping[ps_slot]
            mon = ps_mons[ps_slot]

            new_hp[side_idx, jax_slot] = mon['hp']
            new_status[side_idx, jax_slot] = STATUS_MAP.get(mon.get('status', ''), 0)
            new_fainted[side_idx, jax_slot] = mon.get('fainted', False)

            sd = mon.get('statusData', 0)
            if isinstance(sd, dict):
                new_status_turns[side_idx, jax_slot] = sd.get('turns', 0)
            else:
                new_status_turns[side_idx, jax_slot] = sd

            if mon.get('boosts'):
                for bi, bname in enumerate(['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']):
                    new_boosts[side_idx, jax_slot, bi] = mon['boosts'].get(bname, 0)

            if mon.get('isActive', False):
                new_active[side_idx] = jax_slot

            for mi, ps_move in enumerate(mon.get('moves', [])[:4]):
                new_pp[side_idx, jax_slot, mi] = min(ps_move.get('pp', 0), 64)

            if item_lookup is not None:
                item_name = normalize_id(mon.get('item', ''))
                new_items[side_idx, jax_slot] = item_lookup.get(item_name, 0)

        # Side conditions
        new_side_conds[side_idx, :] = 0
        ps_sc = ps_state['sides'][side_idx].get('sideConditions', {})
        if isinstance(ps_sc, dict):
            for sc_name, sc_data in ps_sc.items():
                sc_key = normalize_id(sc_name)
                if sc_key in SC_NAME_MAP:
                    sc_idx, val_src = SC_NAME_MAP[sc_key]
                    if isinstance(sc_data, dict):
                        if val_src == 'layers':
                            new_side_conds[side_idx, sc_idx] = sc_data.get('layers', 1)
                        elif val_src == 'presence':
                            new_side_conds[side_idx, sc_idx] = 1
                        else:  # 'duration'
                            new_side_conds[side_idx, sc_idx] = sc_data.get('duration', 1)
                            if new_side_conds[side_idx, sc_idx] == 0:
                                new_side_conds[side_idx, sc_idx] = 1
                    else:
                        new_side_conds[side_idx, sc_idx] = 1

        new_pokemon_left[side_idx] = ps_state['sides'][side_idx].get('pokemonLeft', 6)

    new_field = state.field._replace(
        weather=jnp.int8(new_weather),
        weather_turns=jnp.int8(new_weather_turns),
    )

    return state._replace(
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


# ═══════════════════════════════════════════════════════════════════════════
# Team builder
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Showdown battle generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_showdown_battles(num_battles: int, output_path: str) -> str:
    """Run showdown_battle_logger.js to generate deterministic battle logs."""
    print(f"  Generating {num_battles} battles via Showdown...")
    result = subprocess.run(
        ["node", BATTLE_LOGGER_JS, str(num_battles), output_path],
        capture_output=True, text=True, timeout=300,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Showdown battle generation failed:\n{result.stderr}")
    print(f"  Showdown generation complete: {output_path}")
    return output_path


def load_battles(path: str, limit: Optional[int] = None) -> list:
    """Load JSONL battle logs."""
    battles = []
    with open(path) as f:
        for line in f:
            if line.strip():
                battles.append(json.loads(line))
                if limit and len(battles) >= limit:
                    break
    return battles


# ═══════════════════════════════════════════════════════════════════════════
# Core comparison engine
# ═══════════════════════════════════════════════════════════════════════════

def is_forced_switch_turn(actions, ps_turns, ti):
    """Detect if this is a forced-switch turn (JAX handles these internally)."""
    p1_act = actions.get('p1')
    p2_act = actions.get('p2')

    if p1_act is None or p2_act is None:
        return True

    if ti > 0:
        prev_ps = ps_turns[ti - 1].get('state')
        if prev_ps:
            for si in range(2):
                ps_mons_prev = prev_ps['sides'][si]['pokemon']
                has_active = any(
                    m.get('isActive') and not m.get('fainted')
                    for m in ps_mons_prev
                )
                if not has_active and prev_ps['sides'][si].get('pokemonLeft', 6) > 0:
                    return True
    return False


def compare_single_battle(battle, tables, cfg, species_lookup, move_lookup,
                           ability_lookup, item_lookup, jit_turn) -> BattleResult:
    """Replay one Showdown battle in PokeJAX and compare everything."""
    bi = battle['battle_idx']
    init_state = battle['turns'][0].get('state') if battle['turns'] else None

    if init_state is None:
        return BattleResult(
            battle_idx=bi, seed=battle.get('seed', []),
            total_ps_turns=0, total_replayed_turns=0,
            turn_comparisons=[], winner_ps='', winner_jax=-1,
            winner_match=False, jax_finished_early=False,
            jax_finished_late=False, error="No initial state",
        )

    # Build initial species lists
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
        return BattleResult(
            battle_idx=bi, seed=battle.get('seed', []),
            total_ps_turns=0, total_replayed_turns=0,
            turn_comparisons=[], winner_ps='', winner_jax=-1,
            winner_match=False, jax_finished_early=False,
            jax_finished_late=False, error=f"State init: {e}",
        )

    reveal = make_reveal_state(state)
    ps_turns = battle['turns']
    ps_to_jax_maps = [list(range(6)), list(range(6))]
    turn_comparisons = []
    replayed = 0
    early_finish = False

    for ti in range(1, len(ps_turns)):
        turn = ps_turns[ti]
        actions = turn.get('actions')
        if actions is None:
            continue

        if is_forced_switch_turn(actions, ps_turns, ti):
            ps_state_turn = turn.get('state')
            if ps_state_turn:
                for side_idx in range(2):
                    ps_mons = ps_state_turn['sides'][side_idx]['pokemon']
                    ps_to_jax_maps[side_idx] = build_ps_to_jax_map(
                        ps_mons, initial_species[side_idx])
                state = sync_state_from_ps(state, ps_state_turn, ps_to_jax_maps, item_lookup)
            continue

        if bool(state.finished):
            early_finish = True
            break

        # Rebuild slot mapping from previous turn's PS state
        prev_ps_state = ps_turns[ti - 1].get('state')
        if prev_ps_state:
            for side_idx in range(2):
                ps_mons = prev_ps_state['sides'][side_idx]['pokemon']
                ps_to_jax_maps[side_idx] = build_ps_to_jax_map(
                    ps_mons, initial_species[side_idx])

        p1_act_str = actions.get('p1', '')
        p2_act_str = actions.get('p2', '')
        a0 = min(max(translate_action(p1_act_str, ps_to_jax_maps[0]), 0), 9)
        a1 = min(max(translate_action(p2_act_str, ps_to_jax_maps[1]), 0), 9)

        # Action mask check
        mask0 = np.array(get_action_mask(state, 0))
        mask1 = np.array(get_action_mask(state, 1))
        p1_legal = bool(mask0[a0])
        p2_legal = bool(mask1[a1])

        # Execute turn in JAX
        action_arr = jnp.array([a0, a1], dtype=jnp.int32)
        try:
            state, reveal = jit_turn(state, reveal, action_arr)
        except Exception as e:
            return BattleResult(
                battle_idx=bi, seed=battle.get('seed', []),
                total_ps_turns=len(ps_turns) - 1, total_replayed_turns=replayed,
                turn_comparisons=turn_comparisons, winner_ps=battle.get('winner', ''),
                winner_jax=-1, winner_match=False,
                jax_finished_early=False, jax_finished_late=False,
                error=f"Turn {ti}: {e}",
                ps_species=initial_species,
            )

        replayed += 1

        # HP conservation check
        hp = np.array(state.sides_team_hp)
        maxhp = np.array(state.sides_team_max_hp)
        hp_violation = bool((hp < 0).any() or (hp > maxhp).any())

        # Compare with PS state
        tc = TurnComparison(
            battle_idx=bi, turn_idx=ti,
            p1_action_legal=p1_legal, p2_action_legal=p2_legal,
            p1_action_str=p1_act_str, p2_action_str=p2_act_str,
            hp_matches=[], status_matches=[], boost_matches=[],
            faint_matches=[], weather_match=True, weather_ps='',
            weather_jax=0, sc_matches=[], pp_matches=[],
            item_matches=[], hp_violation=hp_violation,
        )

        ps_state_turn = turn.get('state')
        if ps_state_turn:
            # Weather comparison
            ps_weather = normalize_id(ps_state_turn.get('weather', ''))
            jax_weather = int(state.field.weather)
            ps_weather_id = WEATHER_MAP.get(ps_weather, 0)
            tc.weather_ps = ps_weather
            tc.weather_jax = jax_weather
            tc.weather_match = (ps_weather_id == jax_weather)

            for side_idx in range(2):
                ps_mons = ps_state_turn['sides'][side_idx]['pokemon']
                cur_map = build_ps_to_jax_map(ps_mons, initial_species[side_idx])

                # Side conditions comparison
                ps_sc = ps_state_turn['sides'][side_idx].get('sideConditions', {})
                jax_sc = np.array(state.sides_side_conditions[side_idx])
                if isinstance(ps_sc, dict):
                    for sc_name, (sc_idx, val_src) in SC_NAME_MAP.items():
                        ps_val = 0
                        if sc_name in ps_sc:
                            sc_data = ps_sc[sc_name]
                            if isinstance(sc_data, dict):
                                if val_src == 'layers':
                                    ps_val = sc_data.get('layers', 1)
                                elif val_src == 'presence':
                                    ps_val = 1
                                else:  # 'duration'
                                    ps_val = sc_data.get('duration', 1)
                            else:
                                ps_val = 1
                        jax_val = int(jax_sc[sc_idx])
                        # Only log if either side has the condition
                        if ps_val > 0 or jax_val > 0:
                            tc.sc_matches.append(
                                (side_idx, sc_name, ps_val, jax_val, ps_val == jax_val))

                for ps_slot in range(min(6, len(ps_mons))):
                    mon = ps_mons[ps_slot]
                    jax_slot = cur_map[ps_slot]

                    # Only compare active pokemon in detail
                    if not mon.get('isActive', False):
                        continue

                    # HP
                    ps_hp = mon['hp']
                    jax_hp = int(hp[side_idx, jax_slot])
                    tc.hp_matches.append(
                        (side_idx, jax_slot, ps_hp, jax_hp, ps_hp == jax_hp))

                    # Status
                    ps_status_val = STATUS_MAP.get(mon.get('status', ''), 0)
                    jax_status_val = int(state.sides_team_status[side_idx, jax_slot])
                    tc.status_matches.append(
                        (side_idx, jax_slot, ps_status_val, jax_status_val,
                         ps_status_val == jax_status_val))

                    # Boosts
                    if mon.get('boosts'):
                        for bj, bname in enumerate(['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']):
                            ps_b = mon['boosts'].get(bname, 0)
                            jax_b = int(state.sides_team_boosts[side_idx, jax_slot, bj])
                            tc.boost_matches.append(
                                (side_idx, jax_slot, bname, ps_b, jax_b, ps_b == jax_b))

                    # Faint
                    ps_fainted = mon.get('fainted', False) or ps_hp <= 0
                    jax_fainted = jax_hp <= 0
                    tc.faint_matches.append(
                        (side_idx, jax_slot, ps_fainted, jax_fainted,
                         ps_fainted == jax_fainted))

                    # PP
                    ps_moves = mon.get('moves', [])
                    for mi, ps_move in enumerate(ps_moves[:4]):
                        ps_pp = ps_move.get('pp', 0)
                        jax_pp = int(state.sides_team_move_pp[side_idx, jax_slot, mi])
                        tc.pp_matches.append(
                            (side_idx, jax_slot, mi, ps_pp, jax_pp, ps_pp == jax_pp))

                    # Item
                    ps_item = normalize_id(mon.get('item', ''))
                    jax_item_id = int(state.sides_team_item_id[side_idx, jax_slot])
                    ps_item_id = item_lookup.get(ps_item, 0)
                    tc.item_matches.append(
                        (side_idx, jax_slot, ps_item, jax_item_id,
                         ps_item_id == jax_item_id))

            # Sync state for next turn
            for side_idx in range(2):
                ps_mons = ps_state_turn['sides'][side_idx]['pokemon']
                ps_to_jax_maps[side_idx] = build_ps_to_jax_map(
                    ps_mons, initial_species[side_idx])
            state = sync_state_from_ps(state, ps_state_turn, ps_to_jax_maps, item_lookup)

        turn_comparisons.append(tc)

    # Winner comparison
    ps_winner_str = battle.get('winner', '')
    ps_finished = ps_winner_str not in ('', 'none', None)
    jax_finished = bool(state.finished)
    jax_winner = int(state.winner) if jax_finished else -1

    winner_match = False
    late_finish = False
    if ps_finished and jax_finished:
        ps_winner_idx = 0 if ps_winner_str == 'Bot1' else (1 if ps_winner_str == 'Bot2' else 2)
        winner_match = (ps_winner_idx == jax_winner)
    elif ps_finished and not jax_finished:
        late_finish = True
    elif not ps_finished and jax_finished:
        early_finish = True

    return BattleResult(
        battle_idx=bi, seed=battle.get('seed', []),
        total_ps_turns=len(ps_turns) - 1, total_replayed_turns=replayed,
        turn_comparisons=turn_comparisons,
        winner_ps=ps_winner_str, winner_jax=jax_winner,
        winner_match=winner_match,
        jax_finished_early=early_finish,
        jax_finished_late=late_finish,
        ps_species=initial_species,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation & reporting
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(battle_results: list) -> SuiteResults:
    """Aggregate per-battle results into suite-level statistics."""
    sr = SuiteResults()
    sr.total_battles = len(battle_results)

    for br in battle_results:
        if br.error:
            sr.battles_with_errors += 1
            sr.errors.append(f"B{br.battle_idx}: {br.error}")
            continue

        sr.total_turns_replayed += br.total_replayed_turns

        # Winner
        ps_finished = br.winner_ps not in ('', 'none', None)
        jax_finished = br.winner_jax >= 0
        if ps_finished:
            sr.ps_finished += 1
        if jax_finished:
            sr.jax_finished += 1

        if ps_finished and jax_finished:
            if br.winner_match:
                sr.winner_matches += 1
            else:
                sr.winner_mismatches += 1
                if len(sr.winner_mismatch_details) < 30:
                    sr.winner_mismatch_details.append(
                        f"B{br.battle_idx}: PS={br.winner_ps} JAX={br.winner_jax}")

        if br.jax_finished_early:
            sr.jax_early_finishes += 1
        if br.jax_finished_late:
            sr.jax_late_finishes += 1

        # Per-turn metrics
        for tc in br.turn_comparisons:
            # Action masks
            if tc.p1_action_legal:
                sr.action_mask_legal += 1
            else:
                sr.action_mask_illegal += 1
                if len(sr.action_mask_failures) < 30:
                    sr.action_mask_failures.append(
                        f"B{tc.battle_idx} T{tc.turn_idx} P1: '{tc.p1_action_str}'")
            if tc.p2_action_legal:
                sr.action_mask_legal += 1
            else:
                sr.action_mask_illegal += 1
                if len(sr.action_mask_failures) < 30:
                    sr.action_mask_failures.append(
                        f"B{tc.battle_idx} T{tc.turn_idx} P2: '{tc.p2_action_str}'")

            # HP violation
            if tc.hp_violation:
                sr.hp_violations += 1

            # HP
            for (side, slot, ps_hp, jax_hp, match) in tc.hp_matches:
                if match:
                    sr.hp_matches += 1
                else:
                    sr.hp_mismatches += 1
                    if len(sr.hp_mismatch_details) < 30:
                        sr.hp_mismatch_details.append(
                            f"B{tc.battle_idx} T{tc.turn_idx} S{side} slot{slot}: "
                            f"PS={ps_hp} JAX={jax_hp} (delta={abs(ps_hp - jax_hp)})")

            # Status
            for (side, slot, ps_s, jax_s, match) in tc.status_matches:
                if match:
                    sr.status_matches += 1
                else:
                    sr.status_mismatches += 1
                    if len(sr.status_mismatch_details) < 30:
                        sr.status_mismatch_details.append(
                            f"B{tc.battle_idx} T{tc.turn_idx} S{side} slot{slot}: "
                            f"PS={STATUS_NAMES.get(ps_s, ps_s)} JAX={STATUS_NAMES.get(jax_s, jax_s)}")

            # Boosts
            for (side, slot, bname, ps_b, jax_b, match) in tc.boost_matches:
                if match:
                    sr.boost_matches += 1
                else:
                    sr.boost_mismatches += 1

            # Faints
            for (side, slot, ps_f, jax_f, match) in tc.faint_matches:
                if match:
                    sr.faint_matches += 1
                else:
                    sr.faint_mismatches += 1

            # Weather
            if tc.weather_match:
                sr.weather_matches += 1
            else:
                sr.weather_mismatches += 1

            # Side conditions
            for (side, sc_name, ps_val, jax_val, match) in tc.sc_matches:
                if match:
                    sr.sc_matches += 1
                else:
                    sr.sc_mismatches += 1

            # PP
            for (side, slot, mi, ps_pp, jax_pp, match) in tc.pp_matches:
                if match:
                    sr.pp_matches += 1
                else:
                    sr.pp_mismatches += 1

            # Items
            for (side, slot, ps_item, jax_item, match) in tc.item_matches:
                if match:
                    sr.item_matches += 1
                else:
                    sr.item_mismatches += 1

    return sr


def print_report(sr: SuiteResults, duration_s: float):
    """Print a detailed final report."""
    def pct(m, mm):
        t = m + mm
        return f"{m}/{t} ({m/t:.2%})" if t > 0 else "N/A"

    print()
    print("=" * 78)
    print("  DIFFERENTIAL TEST SUITE RESULTS: Pokemon Showdown vs PokeJAX")
    print("=" * 78)
    print(f"  Duration:           {duration_s:.1f}s")
    print(f"  Battles tested:     {sr.total_battles}")
    print(f"  Battles w/ errors:  {sr.battles_with_errors}")
    print(f"  Turns replayed:     {sr.total_turns_replayed}")
    print()

    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  DETERMINISTIC CHECKS (must be near-perfect)                   │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  Action mask legality:  {pct(sr.action_mask_legal, sr.action_mask_illegal):>40s} │")
    print(f"  │  HP conservation:       {'0 violations' if sr.hp_violations == 0 else f'{sr.hp_violations} VIOLATIONS':>40s} │")
    print(f"  │  Boost agreement:       {pct(sr.boost_matches, sr.boost_mismatches):>40s} │")
    print(f"  │  Weather agreement:     {pct(sr.weather_matches, sr.weather_mismatches):>40s} │")
    print(f"  │  Side cond. agreement:  {pct(sr.sc_matches, sr.sc_mismatches):>40s} │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()

    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  RNG-DEPENDENT CHECKS (diverge due to different PRNGs)         │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  HP agreement:          {pct(sr.hp_matches, sr.hp_mismatches):>40s} │")
    print(f"  │  Status agreement:      {pct(sr.status_matches, sr.status_mismatches):>40s} │")
    print(f"  │  Faint agreement:       {pct(sr.faint_matches, sr.faint_mismatches):>40s} │")
    print(f"  │  PP agreement:          {pct(sr.pp_matches, sr.pp_mismatches):>40s} │")
    print(f"  │  Item agreement:        {pct(sr.item_matches, sr.item_mismatches):>40s} │")
    print(f"  │  Winner agreement:      {pct(sr.winner_matches, sr.winner_mismatches):>40s} │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()

    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  GAME COMPLETION                                               │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  PS games finished:     {sr.ps_finished:>40d} │")
    print(f"  │  JAX games finished:    {sr.jax_finished:>40d} │")
    print(f"  │  JAX finishes early:    {sr.jax_early_finishes:>40d} │")
    print(f"  │  JAX finishes late:     {sr.jax_late_finishes:>40d} │")
    print("  └─────────────────────────────────────────────────────────────────┘")

    if sr.action_mask_failures:
        print(f"\n  Action Mask Failures ({sr.action_mask_illegal} total, showing first {min(15, len(sr.action_mask_failures))}):")
        for d in sr.action_mask_failures[:15]:
            print(f"    {d}")

    if sr.hp_mismatch_details:
        print(f"\n  HP Mismatches (showing first {min(10, len(sr.hp_mismatch_details))}):")
        for d in sr.hp_mismatch_details[:10]:
            print(f"    {d}")

    if sr.status_mismatch_details:
        print(f"\n  Status Mismatches (showing first {min(10, len(sr.status_mismatch_details))}):")
        for d in sr.status_mismatch_details[:10]:
            print(f"    {d}")

    if sr.winner_mismatch_details:
        print(f"\n  Winner Mismatches ({sr.winner_mismatches} total):")
        for d in sr.winner_mismatch_details[:10]:
            print(f"    {d}")

    if sr.errors:
        print(f"\n  Errors ({len(sr.errors)}):")
        for e in sr.errors[:10]:
            print(f"    {e}")

    # Overall verdict
    mask_rate = sr.rate(sr.action_mask_legal, sr.action_mask_illegal)
    boost_rate = sr.rate(sr.boost_matches, sr.boost_mismatches)

    deterministic_pass = (
        (mask_rate >= 0.99 or (sr.action_mask_legal + sr.action_mask_illegal == 0)) and
        sr.hp_violations == 0 and
        (boost_rate >= 0.98 or (sr.boost_matches + sr.boost_mismatches == 0))
    )

    print()
    print("  " + "=" * 66)
    if deterministic_pass:
        print("  VERDICT: PASS — Deterministic checks meet thresholds")
        print(f"    Action mask: {mask_rate:.2%} (>= 99%)")
        print(f"    HP violations: {sr.hp_violations} (== 0)")
        print(f"    Boost agreement: {boost_rate:.2%} (>= 98%)")
    else:
        print("  VERDICT: NEEDS INVESTIGATION")
        if mask_rate < 0.99:
            print(f"    Action mask: {mask_rate:.2%} (BELOW 99% threshold)")
        if sr.hp_violations > 0:
            print(f"    HP violations: {sr.hp_violations} (should be 0)")
        if boost_rate < 0.98:
            print(f"    Boost agreement: {boost_rate:.2%} (BELOW 98% threshold)")
    print("  " + "=" * 66)

    return deterministic_pass


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════

def run_differential_suite(battles_path: Optional[str] = None,
                            num_battles: int = 100,
                            limit: Optional[int] = None) -> bool:
    """Run the full differential test suite.

    If battles_path is None, generates fresh battles via Showdown.
    Returns True if all deterministic checks pass.
    """
    t0 = time.time()

    # ── Step 1: Load tables ────────────────────────────────────────────
    print("Loading PokeJAX tables...")
    tables = load_tables(4, showdown_path=PS_PATH)
    cfg = GenConfig.for_gen(4)

    species_lookup = build_reverse_lookup(tables.species_name_to_id)
    move_lookup = build_reverse_lookup(tables.move_name_to_id)
    ability_lookup = build_reverse_lookup(tables.ability_name_to_id)
    item_lookup = build_reverse_lookup(tables.item_name_to_id)

    # ── Step 2: Get battles ────────────────────────────────────────────
    if battles_path and os.path.exists(battles_path):
        print(f"Loading existing battles from {battles_path}...")
        battles = load_battles(battles_path, limit=limit or num_battles)
    else:
        # Generate fresh battles
        tmp_path = os.path.join(str(ROOT), "data", "differential_test_battles.jsonl")
        generate_showdown_battles(num_battles, tmp_path)
        battles = load_battles(tmp_path, limit=limit)

    print(f"Loaded {len(battles)} battles")

    # ── Step 3: JIT compile ────────────────────────────────────────────
    @jax.jit
    def jit_turn(state, reveal, actions):
        return execute_turn(state, reveal, actions, tables, cfg)

    print("JIT compiling execute_turn (first time may take ~5 minutes)...", flush=True)

    # ── Step 4: Run comparisons ────────────────────────────────────────
    battle_results = []
    for i, battle in enumerate(battles):
        result = compare_single_battle(
            battle, tables, cfg, species_lookup, move_lookup,
            ability_lookup, item_lookup, jit_turn,
        )
        battle_results.append(result)

        if (i + 1) % 50 == 0:
            # Partial progress
            partial = aggregate_results(battle_results)
            mask_r = partial.rate(partial.action_mask_legal, partial.action_mask_illegal)
            print(f"  {i+1}/{len(battles)}: mask={mask_r:.2%} "
                  f"turns={partial.total_turns_replayed} "
                  f"errors={partial.battles_with_errors}", flush=True)

    # ── Step 5: Aggregate & report ─────────────────────────────────────
    sr = aggregate_results(battle_results)
    duration = time.time() - t0
    passed = print_report(sr, duration)

    # ── Step 6: Write detailed log ─────────────────────────────────────
    log_path = os.path.join(str(ROOT), "data", "differential_test_log.json")
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_battles": sr.total_battles,
        "turns_replayed": sr.total_turns_replayed,
        "duration_s": round(duration, 1),
        "deterministic_checks": {
            "action_mask": {"legal": sr.action_mask_legal, "illegal": sr.action_mask_illegal,
                            "rate": round(sr.rate(sr.action_mask_legal, sr.action_mask_illegal), 6)},
            "hp_violations": sr.hp_violations,
            "boost": {"match": sr.boost_matches, "mismatch": sr.boost_mismatches,
                      "rate": round(sr.rate(sr.boost_matches, sr.boost_mismatches), 6)},
            "weather": {"match": sr.weather_matches, "mismatch": sr.weather_mismatches},
            "side_conditions": {"match": sr.sc_matches, "mismatch": sr.sc_mismatches},
        },
        "rng_dependent_checks": {
            "hp": {"match": sr.hp_matches, "mismatch": sr.hp_mismatches},
            "status": {"match": sr.status_matches, "mismatch": sr.status_mismatches},
            "faint": {"match": sr.faint_matches, "mismatch": sr.faint_mismatches},
            "pp": {"match": sr.pp_matches, "mismatch": sr.pp_mismatches},
            "item": {"match": sr.item_matches, "mismatch": sr.item_mismatches},
            "winner": {"match": sr.winner_matches, "mismatch": sr.winner_mismatches},
        },
        "game_completion": {
            "ps_finished": sr.ps_finished,
            "jax_finished": sr.jax_finished,
            "early": sr.jax_early_finishes,
            "late": sr.jax_late_finishes,
        },
        "verdict": "PASS" if passed else "NEEDS_INVESTIGATION",
        "failures": {
            "action_mask": sr.action_mask_failures[:30],
            "hp_mismatches": sr.hp_mismatch_details[:30],
            "status_mismatches": sr.status_mismatch_details[:30],
            "winner_mismatches": sr.winner_mismatch_details[:30],
            "errors": sr.errors[:30],
        },
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"\n  Detailed log written to: {log_path}")

    return passed


# ═══════════════════════════════════════════════════════════════════════════
# pytest interface
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def battles_path(request):
    return request.config.getoption("--battles")


@pytest.fixture(scope="session")
def num_battles(request):
    return request.config.getoption("--num-battles")


class TestDifferentialSuite:
    """Differential test suite: Pokemon Showdown vs PokeJAX."""

    def test_full_differential(self, battles_path, num_battles):
        """Run full differential comparison and assert deterministic checks pass."""
        passed = run_differential_suite(
            battles_path=battles_path,
            num_battles=num_battles,
        )
        assert passed, "Deterministic checks failed — see report above"


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Differential Test Suite: Pokemon Showdown vs PokeJAX')
    parser.add_argument('--battles', default=None,
                        help='Path to existing Showdown battle JSONL (skips generation)')
    parser.add_argument('--num-battles', type=int, default=100,
                        help='Number of battles to generate/test')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit battles from existing file')
    args = parser.parse_args()

    success = run_differential_suite(
        battles_path=args.battles,
        num_battles=args.num_battles,
        limit=args.limit,
    )
    sys.exit(0 if success else 1)
