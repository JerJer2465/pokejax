"""
Comprehensive Pokemon Showdown ↔ PokeJAX parity test suite.

Tests are structured to verify that PokeJAX produces *exactly* the same
results as Pokemon Showdown (PS) for Gen 4 mechanics.  Each test category
has two layers:

  1. **Unit parity**: call a single PokeJAX function and assert the result
     matches the known-correct PS value (computed by hand from the PS source
     or extracted via the showdown_oracle.js bridge).

  2. **Oracle parity** (marked @pytest.mark.oracle): spin up a real PS
     process via showdown_oracle.js, run the same scenario in both engines,
     and diff the resulting state.  These require `POKEMON_SHOWDOWN_PATH`
     and are skipped if PS is unavailable.

Run unit tests only (fast, no PS needed):
    pytest tests/test_showdown_parity.py -v -m "not oracle"

Run all tests (requires PS):
    POKEMON_SHOWDOWN_PATH=/path/to/pokemon-showdown pytest tests/test_showdown_parity.py -v
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import pytest
import numpy as np
import jax
import jax.numpy as jnp

# ── PokeJAX imports ─────────────────────────────────────────────────────────
from pokejax.core.state import (
    make_battle_state, make_reveal_state, make_field,
    set_status, set_volatile, set_volatile_counter,
    set_side_condition, set_boost, add_boost, reset_boosts,
    set_hp, set_weather, set_terrain, set_trick_room,
    has_volatile, has_active_volatile,
    get_active_status, get_active_hp, get_active_boosts,
    get_active_idx, get_active_types, get_side_condition,
    clear_volatiles, set_item, set_active, deduct_pp,
    consume_item, set_fainted,
)
from pokejax.core.damage import (
    base_damage, compute_damage,
    apply_weather_modifier, apply_crit_modifier,
    apply_stab_modifier, apply_type_modifier,
    apply_burn_modifier, apply_screen_modifier,
    apply_random_modifier,
    calc_stat, calc_hp, type_effectiveness,
    fraction_of_max_hp, apply_damage, apply_heal,
    MF_BASE_POWER, MF_ACCURACY, MF_TYPE, MF_CATEGORY, MF_PRIORITY,
)
from pokejax.core.priority import (
    get_effective_speed, sort_two_actions, compute_turn_order,
    ACTION_MOVE, ACTION_SWITCH,
)
from pokejax.engine.turn import execute_turn, decode_action
from pokejax.engine.actions import (
    execute_move_action, execute_switch_action,
    check_fainted, check_win, find_forced_switch_slot,
)
from pokejax.engine.switch import switch_out, switch_in
from pokejax.engine.hit_pipeline import execute_move_hit
from pokejax.mechanics.conditions import (
    apply_burn_residual, apply_poison_residual,
    apply_sleep_residual, check_sleep_before_move,
    check_paralysis_before_move, check_freeze_before_move,
    check_confusion_before_move,
    apply_volatile_residuals, decrement_volatile_timers,
    apply_entry_hazards, tick_side_conditions,
    try_set_status, apply_residual,
)
from pokejax.mechanics.events import (
    run_event_switch_in, run_event_switch_out,
    run_event_residual_state, run_event_speed,
)
from pokejax.types import (
    BattleState, RevealState, FieldState,
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX,
    STATUS_SLP, STATUS_FRZ, STATUS_PAR,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    TERRAIN_NONE,
    SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
    SC_REFLECT, SC_LIGHTSCREEN, SC_AURORAVEIL, SC_TAILWIND, SC_SAFEGUARD, SC_MIST,
    VOL_CONFUSED, VOL_FLINCH, VOL_PARTIALLY_TRAPPED, VOL_SEEDED,
    VOL_SUBSTITUTE, VOL_PROTECT, VOL_ENCORE, VOL_TAUNT,
    VOL_CHARGING, VOL_RECHARGING, VOL_LOCKEDMOVE, VOL_CHOICELOCK,
    VOL_FOCUSENERGY, VOL_DISABLE, VOL_YAWN, VOL_DESTINYBOND,
    VOL_INGRAIN, VOL_PERISH, VOL_CURSE, VOL_NIGHTMARE,
    TYPE_NONE, TYPE_NORMAL, TYPE_FIRE, TYPE_WATER, TYPE_ELECTRIC,
    TYPE_GRASS, TYPE_ICE, TYPE_FIGHTING, TYPE_POISON, TYPE_GROUND,
    TYPE_FLYING, TYPE_PSYCHIC, TYPE_BUG, TYPE_ROCK, TYPE_GHOST,
    TYPE_DRAGON, TYPE_DARK, TYPE_STEEL,
    CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS,
    STAT_HP, STAT_ATK, STAT_DEF, STAT_SPA, STAT_SPD, STAT_SPE,
    BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE, BOOST_ACC, BOOST_EVA,
    MAX_TEAM_SIZE, MAX_MOVES, MAX_VOLATILES,
)
from pokejax.config import GenConfig
from pokejax.data.tables import load_tables

# ── Markers ─────────────────────────────────────────────────────────────────
# Tests that require a live Pokemon Showdown process
oracle_mark = pytest.mark.oracle
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0: Fixtures & helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def tables4():
    return load_tables(4)

@pytest.fixture(scope="session")
def cfg4():
    return GenConfig.for_gen(4)


def _make_state(
    max_hp: int = 160,
    p1_types: tuple = (TYPE_NORMAL, 0),
    p2_types: tuple = (TYPE_NORMAL, 0),
    p1_base_stats: tuple = (80, 80, 80, 80, 80, 80),
    p2_base_stats: tuple = (80, 80, 80, 80, 80, 80),
    level: int = 100,
    rng_seed: int = 42,
    p1_move_ids=None,
    p2_move_ids=None,
    p1_abilities=None, p2_abilities=None,
    p1_items=None, p2_items=None,
    p1_max_hp=None, p2_max_hp=None,
) -> BattleState:
    """Configurable BattleState factory for unit tests."""
    n = 6
    zeros6 = np.zeros(n, dtype=np.int16)
    zeros6i8 = np.zeros(n, dtype=np.int8)

    t1 = np.zeros((n, 2), dtype=np.int8)
    t1[:, 0] = p1_types[0]; t1[:, 1] = p1_types[1]
    t2 = np.zeros((n, 2), dtype=np.int8)
    t2[:, 0] = p2_types[0]; t2[:, 1] = p2_types[1]

    bs1 = np.array([list(p1_base_stats)] * n, dtype=np.int16)
    bs2 = np.array([list(p2_base_stats)] * n, dtype=np.int16)

    mhp1 = np.full(n, p1_max_hp or max_hp, dtype=np.int16) if p1_max_hp is None else np.array(p1_max_hp, dtype=np.int16)
    mhp2 = np.full(n, p2_max_hp or max_hp, dtype=np.int16) if p2_max_hp is None else np.array(p2_max_hp, dtype=np.int16)

    if p1_move_ids is None:
        mid1 = np.tile(np.arange(4, dtype=np.int16), (n, 1))
    else:
        mid1 = np.array(p1_move_ids, dtype=np.int16)
        if mid1.ndim == 1:
            mid1 = np.tile(mid1, (n, 1))

    if p2_move_ids is None:
        mid2 = np.tile(np.arange(4, dtype=np.int16), (n, 1))
    else:
        mid2 = np.array(p2_move_ids, dtype=np.int16)
        if mid2.ndim == 1:
            mid2 = np.tile(mid2, (n, 1))

    move_pp = np.full((n, 4), 35, dtype=np.int8)
    levels = np.full(n, level, dtype=np.int8)

    ab1 = np.full(n, 0, dtype=np.int16) if p1_abilities is None else np.full(n, p1_abilities, dtype=np.int16)
    ab2 = np.full(n, 0, dtype=np.int16) if p2_abilities is None else np.full(n, p2_abilities, dtype=np.int16)
    it1 = np.full(n, 0, dtype=np.int16) if p1_items is None else np.full(n, p1_items, dtype=np.int16)
    it2 = np.full(n, 0, dtype=np.int16) if p2_items is None else np.full(n, p2_items, dtype=np.int16)

    return make_battle_state(
        p1_species=zeros6, p2_species=zeros6,
        p1_abilities=ab1, p2_abilities=ab2,
        p1_items=it1, p2_items=it2,
        p1_types=t1, p2_types=t2,
        p1_base_stats=bs1, p2_base_stats=bs2,
        p1_max_hp=mhp1, p2_max_hp=mhp2,
        p1_move_ids=mid1, p2_move_ids=mid2,
        p1_move_pp=move_pp, p2_move_pp=move_pp,
        p1_move_max_pp=move_pp, p2_move_max_pp=move_pp,
        p1_levels=levels, p2_levels=levels,
        p1_genders=zeros6i8, p2_genders=zeros6i8,
        p1_natures=zeros6i8, p2_natures=zeros6i8,
        p1_weights_hg=np.full(n, 100, dtype=np.int16),
        p2_weights_hg=np.full(n, 100, dtype=np.int16),
        rng_key=jax.random.PRNGKey(rng_seed),
    )


def _get_move_id(tables, name: str) -> int:
    """Look up a move ID by name, or -1 if not found."""
    return tables.move_name_to_id.get(name, -1)


def _get_ability_id(tables, name: str) -> int:
    return tables.ability_name_to_id.get(name, -1)


def _get_item_id(tables, name: str) -> int:
    return tables.item_name_to_id.get(name, -1)


def _get_species_id(tables, name: str) -> int:
    return tables.species_name_to_id.get(name, -1)


def _make_state_with_moves(tables, p1_moves: list, p2_moves: list, **kwargs):
    """Create a state with named moves resolved to IDs."""
    def _resolve(names):
        ids = [_get_move_id(tables, n) for n in names]
        while len(ids) < 4:
            ids.append(-1)
        return ids[:4]
    return _make_state(p1_move_ids=_resolve(p1_moves),
                       p2_move_ids=_resolve(p2_moves), **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Damage calculation parity
# ═══════════════════════════════════════════════════════════════════════════

class TestBaseDamageFormula:
    """Gen 4 base damage formula: floor(floor(floor(2*L/5+2)*P*A/D)/50)+2"""

    @pytest.mark.parametrize("level,power,attack,defense,expected", [
        (100, 100, 200, 100, 170),   # Standard case
        (100, 100, 100, 100, 86),    # Balanced
        (50,  100, 100, 100, 46),    # Level 50
        (100, 0,   100, 100, 2),     # Status move (power=0)
        (100, 120, 300, 150, 203),   # High stats
        (100, 40,  55,  35,  54),    # Low power (Tackle-like)
        (1,   100, 100, 100, 6),     # Level 1
        (100, 250, 400, 100, 842),   # Extreme power/atk
        (100, 80,  100, 200, 35),    # High defense
    ])
    def test_base_damage_exact(self, level, power, attack, defense, expected):
        dmg = base_damage(jnp.int32(level), jnp.int32(power),
                          jnp.int32(attack), jnp.int32(defense))
        assert int(dmg) == expected, f"base_damage({level},{power},{attack},{defense})={int(dmg)}, expected {expected}"

    def test_base_damage_monotonic_in_power(self):
        """Higher base power → higher damage (all else equal)."""
        prev = 0
        for bp in [20, 40, 60, 80, 100, 120, 150, 200]:
            dmg = int(base_damage(jnp.int32(100), jnp.int32(bp),
                                  jnp.int32(100), jnp.int32(100)))
            assert dmg >= prev
            prev = dmg

    def test_base_damage_monotonic_in_attack(self):
        prev = 0
        for atk in [50, 100, 150, 200, 300, 400]:
            dmg = int(base_damage(jnp.int32(100), jnp.int32(80),
                                  jnp.int32(atk), jnp.int32(100)))
            assert dmg >= prev
            prev = dmg

    def test_base_damage_monotonic_decreasing_in_defense(self):
        prev = 9999
        for dfn in [50, 100, 150, 200, 300, 400]:
            dmg = int(base_damage(jnp.int32(100), jnp.int32(80),
                                  jnp.int32(100), jnp.int32(dfn)))
            assert dmg <= prev
            prev = dmg


class TestWeatherDamageModifier:
    """PS: Sun boosts Fire 1.5x, nerfs Water 0.5x. Rain is inverse."""

    @pytest.mark.parametrize("move_type,weather,expected_mult", [
        (TYPE_FIRE,    WEATHER_SUN,  1.5),
        (TYPE_WATER,   WEATHER_SUN,  0.5),
        (TYPE_WATER,   WEATHER_RAIN, 1.5),
        (TYPE_FIRE,    WEATHER_RAIN, 0.5),
        (TYPE_FIRE,    WEATHER_NONE, 1.0),
        (TYPE_GRASS,   WEATHER_SUN,  1.0),
        (TYPE_GRASS,   WEATHER_RAIN, 1.0),
        (TYPE_NORMAL,  WEATHER_SUN,  1.0),
        (TYPE_FIRE,    WEATHER_SAND, 1.0),
        (TYPE_FIRE,    WEATHER_HAIL, 1.0),
        (TYPE_WATER,   WEATHER_SAND, 1.0),
        (TYPE_WATER,   WEATHER_HAIL, 1.0),
        (TYPE_ICE,     WEATHER_HAIL, 1.0),  # Hail doesn't boost Ice moves
    ])
    def test_weather_modifier(self, move_type, weather, expected_mult):
        dmg = jnp.int32(100)
        result = apply_weather_modifier(dmg, jnp.int32(move_type), jnp.int8(weather))
        assert int(result) == int(100 * expected_mult)


class TestCritModifier:
    """Gen 4: crits deal 2x (not 1.5x like Gen 6+)."""

    def test_crit_2x_gen4(self):
        assert int(apply_crit_modifier(jnp.int32(100), jnp.bool_(True))) == 200

    def test_no_crit_1x(self):
        assert int(apply_crit_modifier(jnp.int32(100), jnp.bool_(False))) == 100

    def test_crit_odd_damage(self):
        assert int(apply_crit_modifier(jnp.int32(77), jnp.bool_(True))) == 154


class TestSTABModifier:
    """STAB: 1.5x if move type matches attacker type. 2x with Adaptability."""

    @pytest.mark.parametrize("move_type,atk_types,adaptability,expected", [
        (TYPE_FIRE,  (TYPE_FIRE, TYPE_NONE),   False, 150),
        (TYPE_FIRE,  (TYPE_WATER, TYPE_NONE),  False, 100),
        (TYPE_FIRE,  (TYPE_WATER, TYPE_FIRE),  False, 150),
        (TYPE_FIRE,  (TYPE_FIRE, TYPE_NONE),   True,  200),
        (TYPE_WATER, (TYPE_WATER, TYPE_NONE),  True,  200),
        (TYPE_NORMAL,(TYPE_NORMAL, TYPE_NONE), False, 150),
        (TYPE_GRASS, (TYPE_FIRE, TYPE_WATER),  False, 100),
    ])
    def test_stab(self, move_type, atk_types, adaptability, expected):
        dmg = jnp.int32(100)
        types = jnp.array(list(atk_types), dtype=jnp.int8)
        result = apply_stab_modifier(dmg, jnp.int32(move_type), types,
                                     jnp.bool_(adaptability))
        assert int(result) == expected


class TestTypeEffectiveness:
    """Type chart matchups must match PS exactly."""

    @pytest.mark.parametrize("atk,def1,def2,expected", [
        # Super effective
        (TYPE_FIRE,     TYPE_GRASS,   TYPE_NONE,  2.0),
        (TYPE_WATER,    TYPE_FIRE,    TYPE_NONE,  2.0),
        (TYPE_ELECTRIC, TYPE_WATER,   TYPE_NONE,  2.0),
        (TYPE_ICE,      TYPE_DRAGON,  TYPE_NONE,  2.0),
        (TYPE_FIGHTING, TYPE_NORMAL,  TYPE_NONE,  2.0),
        # Not very effective
        (TYPE_FIRE,     TYPE_WATER,   TYPE_NONE,  0.5),
        (TYPE_GRASS,    TYPE_FIRE,    TYPE_NONE,  0.5),
        (TYPE_ELECTRIC, TYPE_GRASS,   TYPE_NONE,  0.5),
        # Immune
        (TYPE_NORMAL,   TYPE_GHOST,   TYPE_NONE,  0.0),
        (TYPE_ELECTRIC, TYPE_GROUND,  TYPE_NONE,  0.0),
        (TYPE_FIGHTING, TYPE_GHOST,   TYPE_NONE,  0.0),
        (TYPE_GROUND,   TYPE_FLYING,  TYPE_NONE,  0.0),
        (TYPE_PSYCHIC,  TYPE_DARK,    TYPE_NONE,  0.0),
        (TYPE_GHOST,    TYPE_NORMAL,  TYPE_NONE,  0.0),
        (TYPE_POISON,   TYPE_STEEL,   TYPE_NONE,  0.0),
        # Dual-type: 4x
        (TYPE_ICE,      TYPE_GROUND,  TYPE_FLYING, 4.0),
        (TYPE_FIRE,     TYPE_GRASS,   TYPE_BUG,    4.0),
        (TYPE_FIRE,     TYPE_GRASS,   TYPE_ICE,    4.0),
        (TYPE_FIGHTING, TYPE_NORMAL,  TYPE_ROCK,   4.0),
        # Dual-type: 0.25x
        (TYPE_FIRE,     TYPE_WATER,   TYPE_ROCK,   0.25),
        (TYPE_GRASS,    TYPE_FIRE,    TYPE_FLYING,  0.25),
        # Dual-type: immune overrides SE
        (TYPE_ELECTRIC, TYPE_WATER,   TYPE_GROUND, 0.0),
        (TYPE_GROUND,   TYPE_ELECTRIC,TYPE_FLYING,  0.0),
        # Neutral
        (TYPE_NORMAL,   TYPE_NORMAL,  TYPE_NONE,   1.0),
        (TYPE_FIRE,     TYPE_NORMAL,  TYPE_NONE,   1.0),
        # Neutral dual
        (TYPE_FIRE,     TYPE_GRASS,   TYPE_WATER,  1.0),  # 2.0 * 0.5 = 1.0
    ])
    def test_type_effectiveness(self, tables4, atk, def1, def2, expected):
        eff = tables4.get_type_effectiveness(
            jnp.int32(atk), jnp.int32(def1), jnp.int32(def2)
        )
        assert float(eff) == pytest.approx(expected, abs=1e-4)


class TestBurnDamageModifier:
    """PS: Burn halves physical damage; no effect on special; Guts overrides."""

    @pytest.mark.parametrize("category,status,has_guts,expected", [
        (CATEGORY_PHYSICAL, STATUS_BRN,  False, 50),
        (CATEGORY_PHYSICAL, STATUS_BRN,  True,  100),  # Guts
        (CATEGORY_SPECIAL,  STATUS_BRN,  False, 100),
        (CATEGORY_PHYSICAL, STATUS_NONE, False, 100),
        (CATEGORY_STATUS,   STATUS_BRN,  False, 100),  # Status moves unaffected
        (CATEGORY_SPECIAL,  STATUS_BRN,  True,  100),
    ])
    def test_burn_modifier(self, category, status, has_guts, expected):
        result = apply_burn_modifier(jnp.int32(100), jnp.int8(category),
                                     jnp.int8(status), jnp.bool_(has_guts))
        assert int(result) == expected


class TestScreenModifier:
    """PS: Reflect halves physical; Light Screen halves special; crits ignore."""

    @pytest.mark.parametrize("category,reflect,lightscreen,is_crit,expected", [
        (CATEGORY_PHYSICAL, True,  False, False, 50),   # Reflect active
        (CATEGORY_PHYSICAL, True,  False, True,  100),  # Crit ignores screen
        (CATEGORY_PHYSICAL, False, False, False, 100),
        (CATEGORY_SPECIAL,  False, True,  False, 50),   # Light Screen
        (CATEGORY_SPECIAL,  False, True,  True,  100),  # Crit ignores
        (CATEGORY_SPECIAL,  False, False, False, 100),
        (CATEGORY_PHYSICAL, False, True,  False, 100),  # LS doesn't affect physical
        (CATEGORY_SPECIAL,  True,  False, False, 100),  # Reflect doesn't affect special
    ])
    def test_screen_modifier(self, category, reflect, lightscreen, is_crit, expected):
        result = apply_screen_modifier(jnp.int32(100), jnp.int8(category),
                                       jnp.bool_(reflect), jnp.bool_(lightscreen),
                                       jnp.bool_(is_crit))
        assert int(result) == expected


class TestRandomRollModifier:
    """PS: damage * roll where roll ∈ [0.85, 1.00] in 16 tiers."""

    def test_max_roll_no_change(self):
        assert int(apply_random_modifier(jnp.int32(100), jnp.float32(1.0))) == 100

    def test_min_roll_085(self):
        assert int(apply_random_modifier(jnp.int32(100), jnp.float32(0.85))) == 85

    def test_floor_behavior(self):
        # floor(99 * 0.85) = floor(84.15) = 84
        assert int(apply_random_modifier(jnp.int32(99), jnp.float32(0.85))) == 84

    def test_mid_roll(self):
        # floor(100 * 0.93) = 93
        assert int(apply_random_modifier(jnp.int32(100), jnp.float32(0.93))) == 93


class TestStatCalculation:
    """PS stat formulas: Gen 3+ non-HP and HP calculations."""

    @pytest.mark.parametrize("base,level,ev,iv,nature_mult,expected", [
        (100, 100, 0,   31, 1.0, 236),   # Neutral nature
        (100, 100, 0,   31, 1.1, 259),   # Boosting nature
        (100, 100, 0,   31, 0.9, 212),   # Hindering nature
        (100, 100, 252, 31, 1.0, 299),   # Max EVs
        (50,  100, 0,   31, 1.0, 136),   # Low base
        (150, 100, 0,   31, 1.0, 336),   # High base
        (100, 50,  0,   31, 1.0, 120),   # Level 50
        (100, 100, 0,   0,  1.0, 205),   # Zero IVs
    ])
    def test_calc_stat(self, base, level, ev, iv, nature_mult, expected):
        result = calc_stat(jnp.int32(base), jnp.int32(level),
                           jnp.int32(ev), jnp.int32(iv), jnp.float32(nature_mult))
        assert int(result) == expected

    @pytest.mark.parametrize("base_hp,level,ev,iv,expected", [
        (100, 100, 0,   31, 341),
        (100, 100, 252, 31, 404),
        (50,  100, 0,   31, 241),
        (150, 100, 0,   31, 441),
        (100, 50,  0,   31, 175),
        (1,   100, 0,   31, 143),   # Shedinja-like (low base HP)
    ])
    def test_calc_hp(self, base_hp, level, ev, iv, expected):
        result = calc_hp(jnp.int32(base_hp), jnp.int32(level),
                         jnp.int32(ev), jnp.int32(iv))
        assert int(result) == expected


class TestBoostMultipliers:
    """PS boost stages: max(2,2+stage)/max(2,2-stage)."""

    @pytest.mark.parametrize("stage,expected", [
        (-6, 2/8), (-5, 2/7), (-4, 2/6), (-3, 2/5), (-2, 2/4), (-1, 2/3),
        (0, 1.0),
        (1, 3/2), (2, 4/2), (3, 5/2), (4, 6/2), (5, 7/2), (6, 8/2),
    ])
    def test_boost_multiplier(self, tables4, stage, expected):
        mult = tables4.get_boost_multiplier(jnp.int32(stage))
        assert float(mult) == pytest.approx(expected, rel=1e-3)


class TestHPHelpers:
    """apply_damage floors at 0, apply_heal caps at max_hp."""

    def test_damage_reduces_hp(self):
        state = _make_state(max_hp=200)
        state = apply_damage(state, 0, 0, jnp.int32(50))
        assert int(state.sides_team_hp[0, 0]) == 150

    def test_damage_floors_at_zero(self):
        state = _make_state(max_hp=100)
        state = apply_damage(state, 0, 0, jnp.int32(999))
        assert int(state.sides_team_hp[0, 0]) == 0

    def test_heal_increases_hp(self):
        state = _make_state(max_hp=200)
        state = apply_damage(state, 0, 0, jnp.int32(80))
        state = apply_heal(state, 0, 0, jnp.int32(30))
        assert int(state.sides_team_hp[0, 0]) == 150

    def test_heal_caps_at_max(self):
        state = _make_state(max_hp=200)
        state = apply_damage(state, 0, 0, jnp.int32(10))
        state = apply_heal(state, 0, 0, jnp.int32(999))
        assert int(state.sides_team_hp[0, 0]) == 200

    def test_fraction_of_max_hp(self):
        state = _make_state(max_hp=160)
        val = fraction_of_max_hp(state, 0, 0, 1, 8)
        assert int(val) == 20  # 160/8

    def test_fraction_min_1(self):
        state = _make_state(max_hp=7)
        val = fraction_of_max_hp(state, 0, 0, 1, 8)
        assert int(val) == 1  # floor(7/8)=0 → clamped to 1


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Status condition parity
# ═══════════════════════════════════════════════════════════════════════════

class TestBurnResidual:
    """PS Gen 4: Burn deals 1/8 max HP per turn."""

    @pytest.mark.parametrize("max_hp,expected_dmg", [
        (160, 20), (200, 25), (80, 10), (1, 1), (7, 1),
    ])
    def test_burn_damage_amount(self, cfg4, max_hp, expected_dmg):
        state = _make_state(max_hp=max_hp)
        state = set_status(state, 0, 0, jnp.int8(STATUS_BRN))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_burn_residual(state, 0, cfg4)
        assert hp_before - int(state.sides_team_hp[0, 0]) == expected_dmg

    def test_no_burn_no_damage(self, cfg4):
        state = _make_state(max_hp=160)
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_burn_residual(state, 0, cfg4)
        assert int(state.sides_team_hp[0, 0]) == hp_before

    def test_burn_does_not_faint_below_zero(self, cfg4):
        state = _make_state(max_hp=5)
        state = set_status(state, 0, 0, jnp.int8(STATUS_BRN))
        state = apply_burn_residual(state, 0, cfg4)
        assert int(state.sides_team_hp[0, 0]) >= 0


class TestPoisonResidual:
    """PS: Poison 1/8 per turn. Toxic: N/16 escalating, capped at 15/16."""

    def test_poison_one_eighth(self):
        state = _make_state(max_hp=160)
        state = set_status(state, 0, 0, jnp.int8(STATUS_PSN))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_poison_residual(state, 0)
        assert hp_before - int(state.sides_team_hp[0, 0]) == 20

    @pytest.mark.parametrize("counter,expected_dmg", [
        (1, 10), (2, 20), (3, 30), (5, 50), (10, 100), (15, 150),
    ])
    def test_toxic_escalation(self, counter, expected_dmg):
        state = _make_state(max_hp=160)
        state = set_status(state, 0, 0, jnp.int8(STATUS_TOX), jnp.int8(counter))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_poison_residual(state, 0)
        assert hp_before - int(state.sides_team_hp[0, 0]) == expected_dmg

    def test_toxic_counter_increments(self):
        state = _make_state(max_hp=400)
        state = set_status(state, 0, 0, jnp.int8(STATUS_TOX), jnp.int8(1))
        for expected_counter in range(2, 8):
            state = apply_poison_residual(state, 0)
            assert int(state.sides_team_status_turns[0, 0]) == expected_counter

    def test_toxic_counter_capped_at_15(self):
        state = _make_state(max_hp=500)
        state = set_status(state, 0, 0, jnp.int8(STATUS_TOX), jnp.int8(15))
        state = apply_poison_residual(state, 0)
        assert int(state.sides_team_status_turns[0, 0]) == 15

    def test_no_status_no_damage(self):
        state = _make_state(max_hp=160)
        hp = int(state.sides_team_hp[0, 0])
        state = apply_poison_residual(state, 0)
        assert int(state.sides_team_hp[0, 0]) == hp


class TestSleepMechanics:
    """PS Gen 4: Sleep lasts 1-3 turns; counter decrements; wakes at 0."""

    def test_sleep_counter_decrements(self):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_SLP))
        state = state._replace(sides_team_sleep_turns=
            state.sides_team_sleep_turns.at[0, 0].set(jnp.int8(3)))
        state, _ = apply_sleep_residual(state, 0, jax.random.PRNGKey(0))
        assert int(state.sides_team_sleep_turns[0, 0]) == 2

    def test_wakes_at_zero(self):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_SLP))
        state = state._replace(sides_team_sleep_turns=
            state.sides_team_sleep_turns.at[0, 0].set(jnp.int8(1)))
        state, _ = apply_sleep_residual(state, 0, jax.random.PRNGKey(0))
        assert int(state.sides_team_status[0, 0]) == STATUS_NONE

    def test_asleep_cannot_move(self):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_SLP))
        state = state._replace(sides_team_sleep_turns=
            state.sides_team_sleep_turns.at[0, 0].set(jnp.int8(2)))
        can_move, _, _ = check_sleep_before_move(state, 0, jax.random.PRNGKey(0))
        assert not bool(can_move)

    def test_not_asleep_can_move(self):
        state = _make_state()
        can_move, _, _ = check_sleep_before_move(state, 0, jax.random.PRNGKey(0))
        assert bool(can_move)

    def test_sleep_duration_random_1_to_3(self, cfg4):
        durations = set()
        for seed in range(100):
            s, _ = try_set_status(_make_state(), 0, 0, jnp.int8(STATUS_SLP),
                                  jax.random.PRNGKey(seed), cfg4)
            durations.add(int(s.sides_team_sleep_turns[0, 0]))
        assert len(durations) > 1, f"Sleep always same duration: {durations}"
        assert all(1 <= d <= 3 for d in durations), f"Duration out of range: {durations}"


class TestFreezeMechanics:
    """PS Gen 4: Frozen 20% thaw/turn; cannot move while frozen."""

    def test_frozen_blocks_movement(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_FRZ))
        blocked = False
        for seed in range(100):
            can_move, _, s = check_freeze_before_move(state, 0, jax.random.PRNGKey(seed), cfg4)
            if not bool(can_move):
                assert int(s.sides_team_status[0, 0]) == STATUS_FRZ
                blocked = True
                break
        assert blocked, "Freeze never blocked in 100 trials"

    def test_thaw_clears_status(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_FRZ))
        thawed = False
        for seed in range(100):
            can_move, _, s = check_freeze_before_move(state, 0, jax.random.PRNGKey(seed), cfg4)
            if bool(can_move):
                assert int(s.sides_team_status[0, 0]) == STATUS_NONE
                thawed = True
                break
        assert thawed, "Never thawed in 100 trials"

    def test_thaw_rate_approximately_20_percent(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_FRZ))
        thaws = 0
        trials = 500
        for seed in range(trials):
            can_move, _, _ = check_freeze_before_move(state, 0, jax.random.PRNGKey(seed), cfg4)
            if bool(can_move):
                thaws += 1
        rate = thaws / trials
        assert 0.10 < rate < 0.35, f"Thaw rate {rate:.2%} not near 20%"

    def test_not_frozen_always_can_move(self, cfg4):
        state = _make_state()
        can_move, _, _ = check_freeze_before_move(state, 0, jax.random.PRNGKey(0), cfg4)
        assert bool(can_move)


class TestParalysisMechanics:
    """PS Gen 4: 25% full paralysis; speed ÷ 4."""

    def test_full_para_blocks_movement(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        blocked = False
        for seed in range(200):
            can_move, _, _ = check_paralysis_before_move(state, 0, jax.random.PRNGKey(seed), cfg4)
            if not bool(can_move):
                blocked = True
                break
        assert blocked, "Full paralysis never triggered"

    def test_can_still_move_when_paralyzed(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        moved = False
        for seed in range(20):
            can_move, _, _ = check_paralysis_before_move(state, 0, jax.random.PRNGKey(seed), cfg4)
            if bool(can_move):
                moved = True
                break
        assert moved, "Always paralyzed (expected ~75% can move)"

    def test_para_rate_approximately_25_percent(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        blocked = 0
        trials = 500
        for seed in range(trials):
            can_move, _, _ = check_paralysis_before_move(state, 0, jax.random.PRNGKey(seed), cfg4)
            if not bool(can_move):
                blocked += 1
        rate = blocked / trials
        assert 0.15 < rate < 0.40, f"Full para rate {rate:.2%} not near 25%"

    def test_paralysis_speed_divided_by_4_gen4(self, cfg4):
        assert cfg4.paralysis_speed_divisor == 4
        state = _make_state(p1_base_stats=(80, 80, 80, 80, 80, 120))
        speed_normal = get_effective_speed(state, 0, cfg4)
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        speed_para = get_effective_speed(state, 0, cfg4)
        assert int(speed_para) == int(speed_normal) // 4


class TestConfusionMechanics:
    """PS: Confusion 50% self-hit (Gen 4); counter decrements; snaps out at 0."""

    def test_confusion_self_hit_deals_damage(self):
        state = _make_state(max_hp=300)
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, VOL_CONFUSED, True)
        state = set_volatile_counter(state, 0, idx, VOL_CONFUSED, jnp.int8(3))
        hp_before = int(state.sides_team_hp[0, 0])
        hit = False
        for seed in range(200):
            s = state
            can_move, _, s = check_confusion_before_move(s, 0, jax.random.PRNGKey(seed))
            if not bool(can_move):
                assert int(s.sides_team_hp[0, 0]) < hp_before
                hit = True
                break
        assert hit, "Confusion self-hit never triggered in 200 trials"

    def test_confusion_counter_decrements(self):
        state = _make_state()
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, VOL_CONFUSED, True)
        state = set_volatile_counter(state, 0, idx, VOL_CONFUSED, jnp.int8(3))
        for seed in range(200):
            s = state
            can_move, _, s = check_confusion_before_move(s, 0, jax.random.PRNGKey(seed))
            if bool(can_move):
                assert int(s.sides_team_volatile_data[0, idx, VOL_CONFUSED]) == 2
                return
        pytest.skip("No non-self-hit seed found")

    def test_confusion_clears_at_zero(self):
        state = _make_state()
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, VOL_CONFUSED, True)
        state = set_volatile_counter(state, 0, idx, VOL_CONFUSED, jnp.int8(1))
        for seed in range(200):
            s = state
            can_move, _, s = check_confusion_before_move(s, 0, jax.random.PRNGKey(seed))
            if bool(can_move):
                assert not bool(has_volatile(s, 0, idx, VOL_CONFUSED))
                return
        # Even on self-hit, confusion should clear at counter 0
        _, _, s = check_confusion_before_move(state, 0, jax.random.PRNGKey(999))
        assert not bool(has_volatile(s, 0, idx, VOL_CONFUSED))


class TestStatusImmunities:
    """PS: Type-based immunities prevent status application."""

    @pytest.mark.parametrize("types,status,should_apply", [
        ((TYPE_FIRE, 0),    STATUS_BRN,  False),  # Fire immune to burn
        ((TYPE_STEEL, 0),   STATUS_PSN,  False),  # Steel immune to poison
        ((TYPE_STEEL, 0),   STATUS_TOX,  False),  # Steel immune to toxic
        ((TYPE_POISON, 0),  STATUS_PSN,  False),  # Poison immune to poison
        ((TYPE_POISON, 0),  STATUS_TOX,  False),  # Poison immune to toxic
        ((TYPE_ICE, 0),     STATUS_FRZ,  False),  # Ice immune to freeze
        ((TYPE_NORMAL, 0),  STATUS_BRN,  True),   # Normal can be burned
        ((TYPE_NORMAL, 0),  STATUS_PSN,  True),   # Normal can be poisoned
        ((TYPE_WATER, 0),   STATUS_FRZ,  True),   # Water can be frozen
        ((TYPE_FIRE, 0),    STATUS_PAR,  True),   # Fire can be paralyzed
        ((TYPE_FIRE, 0),    STATUS_PSN,  True),   # Fire can be poisoned
        # Dual types
        ((TYPE_FIRE, TYPE_STEEL), STATUS_PSN, False),  # Steel half blocks poison
        ((TYPE_FIRE, TYPE_STEEL), STATUS_BRN, False),  # Fire half blocks burn
    ])
    def test_type_immunity(self, cfg4, types, status, should_apply):
        state = _make_state(p1_types=types)
        state, _ = try_set_status(state, 0, 0, jnp.int8(status),
                                  jax.random.PRNGKey(0), cfg4)
        if should_apply:
            assert int(get_active_status(state, 0)) == status
        else:
            assert int(get_active_status(state, 0)) == STATUS_NONE

    def test_cannot_stack_statuses(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_BRN))
        state, _ = try_set_status(state, 0, 0, jnp.int8(STATUS_PAR),
                                  jax.random.PRNGKey(0), cfg4)
        assert int(get_active_status(state, 0)) == STATUS_BRN

    def test_safeguard_blocks_status(self, cfg4):
        state = _make_state()
        state = set_side_condition(state, 0, SC_SAFEGUARD, jnp.int8(5))
        state, _ = try_set_status(state, 0, 0, jnp.int8(STATUS_BRN),
                                  jax.random.PRNGKey(0), cfg4)
        assert int(get_active_status(state, 0)) == STATUS_NONE


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Turn order / priority parity
# ═══════════════════════════════════════════════════════════════════════════

class TestTurnOrderPriority:
    """PS: Higher priority always moves first. Switches = priority +7."""

    def test_higher_priority_moves_first(self):
        # Priority +1 vs +0: +1 always first
        p0_first, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(1), jnp.int32(50),    # P0: priority +1, speed 50
            ACTION_MOVE, jnp.int8(0), jnp.int32(200),   # P1: priority +0, speed 200
            jnp.bool_(False), jax.random.PRNGKey(0),
        )
        assert bool(p0_first)

    def test_lower_priority_moves_second(self):
        p0_first, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(-1), jnp.int32(200),  # P0: priority -1
            ACTION_MOVE, jnp.int8(0),  jnp.int32(50),   # P1: priority 0
            jnp.bool_(False), jax.random.PRNGKey(0),
        )
        assert not bool(p0_first)

    def test_switch_beats_any_move(self):
        # Switch = effective priority +7, beats any move priority
        p0_first, _, _ = sort_two_actions(
            ACTION_SWITCH, jnp.int8(0), jnp.int32(10),  # P0: switch (slow)
            ACTION_MOVE,   jnp.int8(4), jnp.int32(999), # P1: protect (+4, fast)
            jnp.bool_(False), jax.random.PRNGKey(0),
        )
        assert bool(p0_first)

    def test_both_switch_faster_goes_first(self):
        p0_first, _, _ = sort_two_actions(
            ACTION_SWITCH, jnp.int8(0), jnp.int32(100),
            ACTION_SWITCH, jnp.int8(0), jnp.int32(200),
            jnp.bool_(False), jax.random.PRNGKey(0),
        )
        assert not bool(p0_first)  # P1 faster

    def test_same_priority_faster_goes_first(self):
        p0_first, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(0), jnp.int32(150),
            ACTION_MOVE, jnp.int8(0), jnp.int32(80),
            jnp.bool_(False), jax.random.PRNGKey(0),
        )
        assert bool(p0_first)  # P0 faster

    def test_speed_tie_coin_flip(self):
        # Same priority and speed → random
        results = set()
        for seed in range(50):
            p0_first, _, _ = sort_two_actions(
                ACTION_MOVE, jnp.int8(0), jnp.int32(100),
                ACTION_MOVE, jnp.int8(0), jnp.int32(100),
                jnp.bool_(False), jax.random.PRNGKey(seed),
            )
            results.add(bool(p0_first))
        assert len(results) == 2, "Speed tie not random"


class TestTrickRoom:
    """PS: Trick Room reverses speed order within same priority."""

    def test_trick_room_slower_first(self):
        p0_first, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(0), jnp.int32(50),   # P0: slow
            ACTION_MOVE, jnp.int8(0), jnp.int32(200),  # P1: fast
            jnp.bool_(True), jax.random.PRNGKey(0),     # Trick Room ON
        )
        assert bool(p0_first)  # Slower goes first in TR

    def test_trick_room_priority_unaffected(self):
        # Priority still matters in Trick Room
        p0_first, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(1), jnp.int32(50),   # P0: +1 priority, slow
            ACTION_MOVE, jnp.int8(0), jnp.int32(200),  # P1: +0 priority, fast
            jnp.bool_(True), jax.random.PRNGKey(0),
        )
        assert bool(p0_first)  # Priority still wins


class TestEffectiveSpeed:
    """PS: Speed modified by boosts, paralysis, tailwind, abilities."""

    def test_speed_boost_stages(self, cfg4):
        state = _make_state(p1_base_stats=(80, 80, 80, 80, 80, 100))
        base_speed = int(get_effective_speed(state, 0, cfg4))

        # +1 → 1.5x
        state_b1 = set_boost(state, 0, 0, BOOST_SPE, jnp.int8(1))
        speed_p1 = int(get_effective_speed(state_b1, 0, cfg4))
        assert speed_p1 == int(base_speed * 1.5)

        # +2 → 2.0x
        state_b2 = set_boost(state, 0, 0, BOOST_SPE, jnp.int8(2))
        speed_p2 = int(get_effective_speed(state_b2, 0, cfg4))
        assert speed_p2 == int(base_speed * 2.0)

        # -1 → 2/3x
        state_bm1 = set_boost(state, 0, 0, BOOST_SPE, jnp.int8(-1))
        speed_m1 = int(get_effective_speed(state_bm1, 0, cfg4))
        assert speed_m1 == int(base_speed * 2 / 3)

    def test_tailwind_doubles_speed(self, cfg4):
        state = _make_state(p1_base_stats=(80, 80, 80, 80, 80, 100))
        base_speed = int(get_effective_speed(state, 0, cfg4))
        state = set_side_condition(state, 0, SC_TAILWIND, jnp.int8(4))
        tw_speed = int(get_effective_speed(state, 0, cfg4))
        assert tw_speed == base_speed * 2

    def test_paralysis_then_tailwind(self, cfg4):
        """PS order: paralysis ÷4, then tailwind ×2."""
        state = _make_state(p1_base_stats=(80, 80, 80, 80, 80, 100))
        base_speed = int(get_effective_speed(state, 0, cfg4))
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        state = set_side_condition(state, 0, SC_TAILWIND, jnp.int8(4))
        speed = int(get_effective_speed(state, 0, cfg4))
        assert speed == (base_speed // 4) * 2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Entry hazard parity
# ═══════════════════════════════════════════════════════════════════════════

class TestStealthRock:
    """PS: Stealth Rock deals type-effectiveness-based damage on switch-in."""

    @pytest.mark.parametrize("def_types,max_hp,expected_frac", [
        # Normal type: 1x effectiveness → 1/8 max HP
        ((TYPE_NORMAL, 0), 160, 1/8),
        # Fire (weak to Rock: 2x) → 1/4 max HP
        ((TYPE_FIRE, 0), 160, 1/4),
        # Fire/Flying (4x weak) → 1/2 max HP
        ((TYPE_FIRE, TYPE_FLYING), 160, 1/2),
        # Steel (resists Rock: 0.5x) → 1/16 max HP
        ((TYPE_STEEL, 0), 160, 1/16),
        # Steel/Ground (0.25x) → 1/32 max HP
        ((TYPE_STEEL, TYPE_GROUND), 160, 1/32),
        # Flying (2x) → 1/4
        ((TYPE_FLYING, 0), 160, 1/4),
        # Ice (2x) → 1/4
        ((TYPE_ICE, 0), 160, 1/4),
        # Bug (2x) → 1/4
        ((TYPE_BUG, 0), 160, 1/4),
        # Fighting (resists Rock: 0.5x) → 1/16
        ((TYPE_FIGHTING, 0), 160, 1/16),
        # Ground (resists Rock: 0.5x) → 1/16
        ((TYPE_GROUND, 0), 160, 1/16),
    ])
    def test_stealth_rock_damage(self, tables4, def_types, max_hp, expected_frac):
        state = _make_state(max_hp=max_hp, p1_types=def_types)
        state = set_side_condition(state, 0, SC_STEALTHROCK, jnp.int8(1))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_entry_hazards(state, 0, tables4)
        dmg = hp_before - int(state.sides_team_hp[0, 0])
        expected_dmg = max(1, int(max_hp * expected_frac))
        assert dmg == expected_dmg, f"SR dmg={dmg}, expected={expected_dmg} for types={def_types}"

    def test_no_stealth_rock_no_damage(self, tables4):
        state = _make_state(max_hp=160)
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_hp[0, 0]) == hp_before


class TestSpikes:
    """PS: Spikes deal 1/8, 1/6, 1/4 max HP for 1, 2, 3 layers."""

    @pytest.mark.parametrize("layers,expected_frac", [
        (1, 1/8),
        (2, 1/6),
        (3, 1/4),
    ])
    def test_spikes_damage_by_layer(self, tables4, layers, expected_frac):
        state = _make_state(max_hp=240)  # divisible by 8, 6, 4
        state = set_side_condition(state, 0, SC_SPIKES, jnp.int8(layers))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_entry_hazards(state, 0, tables4)
        dmg = hp_before - int(state.sides_team_hp[0, 0])
        expected_dmg = max(1, int(240 * expected_frac))
        assert dmg == expected_dmg

    def test_spikes_no_damage_to_flying(self, tables4):
        """PS: Flying types are immune to Spikes."""
        state = _make_state(max_hp=160, p1_types=(TYPE_FLYING, 0))
        state = set_side_condition(state, 0, SC_SPIKES, jnp.int8(3))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_hp[0, 0]) == hp_before

    def test_no_spikes_no_damage(self, tables4):
        state = _make_state(max_hp=160)
        hp = int(state.sides_team_hp[0, 0])
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_hp[0, 0]) == hp


class TestToxicSpikes:
    """PS: Toxic Spikes poison (1 layer) or badly poison (2 layers) on switch-in."""

    def test_one_layer_poisons(self, tables4):
        state = _make_state(max_hp=160)
        state = set_side_condition(state, 0, SC_TOXICSPIKES, jnp.int8(1))
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_status[0, 0]) == STATUS_PSN

    def test_two_layers_badly_poisons(self, tables4):
        state = _make_state(max_hp=160)
        state = set_side_condition(state, 0, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_status[0, 0]) == STATUS_TOX

    def test_poison_type_absorbs_toxic_spikes(self, tables4):
        """PS: Poison-type grounded Pokemon absorb Toxic Spikes (remove them)."""
        state = _make_state(max_hp=160, p1_types=(TYPE_POISON, 0))
        state = set_side_condition(state, 0, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_status[0, 0]) == STATUS_NONE
        assert int(state.sides_side_conditions[0, SC_TOXICSPIKES]) == 0

    @pytest.mark.xfail(reason="PokeJAX bug: Steel types not yet immune to Toxic Spikes", strict=False)
    def test_steel_immune_to_toxic_spikes(self, tables4):
        """PS: Steel types are immune to Toxic Spikes."""
        state = _make_state(max_hp=160, p1_types=(TYPE_STEEL, 0))
        state = set_side_condition(state, 0, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_status[0, 0]) == STATUS_NONE

    def test_flying_immune_to_toxic_spikes(self, tables4):
        """PS: Flying types are immune to Toxic Spikes."""
        state = _make_state(max_hp=160, p1_types=(TYPE_FLYING, 0))
        state = set_side_condition(state, 0, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_status[0, 0]) == STATUS_NONE

    def test_already_statused_not_poisoned(self, tables4):
        """PS: Already-statused Pokemon are not affected by Toxic Spikes."""
        state = _make_state(max_hp=160)
        state = set_status(state, 0, 0, jnp.int8(STATUS_BRN))
        state = set_side_condition(state, 0, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_status[0, 0]) == STATUS_BRN  # keeps burn


class TestStickyWeb:
    """PS: Sticky Web lowers Speed by 1 stage on switch-in."""

    def test_sticky_web_lowers_speed(self, tables4):
        state = _make_state(max_hp=160)
        state = set_side_condition(state, 0, SC_STICKYWEB, jnp.int8(1))
        state = apply_entry_hazards(state, 0, tables4)
        boost = int(state.sides_team_boosts[0, 0, BOOST_SPE])
        assert boost == -1

    def test_no_sticky_web_no_speed_drop(self, tables4):
        state = _make_state(max_hp=160)
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_boosts[0, 0, BOOST_SPE]) == 0

    def test_sticky_web_flying_immune(self, tables4):
        """PS: Flying types are immune to Sticky Web."""
        state = _make_state(max_hp=160, p1_types=(TYPE_FLYING, 0))
        state = set_side_condition(state, 0, SC_STICKYWEB, jnp.int8(1))
        state = apply_entry_hazards(state, 0, tables4)
        assert int(state.sides_team_boosts[0, 0, BOOST_SPE]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Weather parity
# ═══════════════════════════════════════════════════════════════════════════

class TestWeatherResidualDamage:
    """PS: Sandstorm/Hail deal 1/16 max HP per turn to non-immune types."""

    def test_sandstorm_damages_non_immune(self, tables4):
        state = _make_state(max_hp=160, p1_types=(TYPE_NORMAL, 0))
        state = set_weather(state, jnp.int8(WEATHER_SAND), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        from pokejax.engine.field import apply_field_residual
        state = apply_field_residual(state)
        dmg = hp_before - int(state.sides_team_hp[0, 0])
        assert dmg == 160 // 16  # 10

    @pytest.mark.parametrize("immune_type", [TYPE_ROCK, TYPE_GROUND, TYPE_STEEL])
    def test_sandstorm_immune_types(self, immune_type):
        state = _make_state(max_hp=160, p1_types=(immune_type, 0))
        state = set_weather(state, jnp.int8(WEATHER_SAND), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        from pokejax.engine.field import apply_field_residual
        state = apply_field_residual(state)
        assert int(state.sides_team_hp[0, 0]) == hp_before

    def test_hail_damages_non_immune(self):
        state = _make_state(max_hp=160, p1_types=(TYPE_NORMAL, 0))
        state = set_weather(state, jnp.int8(WEATHER_HAIL), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        from pokejax.engine.field import apply_field_residual
        state = apply_field_residual(state)
        dmg = hp_before - int(state.sides_team_hp[0, 0])
        assert dmg == 160 // 16

    def test_hail_ice_immune(self):
        state = _make_state(max_hp=160, p1_types=(TYPE_ICE, 0))
        state = set_weather(state, jnp.int8(WEATHER_HAIL), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        from pokejax.engine.field import apply_field_residual
        state = apply_field_residual(state)
        assert int(state.sides_team_hp[0, 0]) == hp_before

    def test_sun_no_residual_damage(self):
        state = _make_state(max_hp=160)
        state = set_weather(state, jnp.int8(WEATHER_SUN), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        from pokejax.engine.field import apply_field_residual
        state = apply_field_residual(state)
        assert int(state.sides_team_hp[0, 0]) == hp_before

    def test_rain_no_residual_damage(self):
        state = _make_state(max_hp=160)
        state = set_weather(state, jnp.int8(WEATHER_RAIN), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        from pokejax.engine.field import apply_field_residual
        state = apply_field_residual(state)
        assert int(state.sides_team_hp[0, 0]) == hp_before


class TestWeatherTimers:
    """PS: Weather lasts N turns and then expires."""

    def test_weather_timer_decrements(self):
        from pokejax.engine.field import tick_all_field_timers
        state = _make_state()
        state = set_weather(state, jnp.int8(WEATHER_SUN), jnp.int8(5))
        state = tick_all_field_timers(state)
        assert int(state.field.weather_turns) == 4

    def test_weather_expires_at_zero(self):
        from pokejax.engine.field import tick_all_field_timers
        state = _make_state()
        state = set_weather(state, jnp.int8(WEATHER_SUN), jnp.int8(1))
        state = tick_all_field_timers(state)
        assert int(state.field.weather) == WEATHER_NONE
        assert int(state.field.weather_turns) == 0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Switching mechanics parity
# ═══════════════════════════════════════════════════════════════════════════

class TestSwitchOut:
    """PS Gen 4: Switch-out clears volatiles, resets boosts, resets toxic counter."""

    def test_switch_out_clears_volatiles(self, cfg4):
        state = _make_state()
        idx = 0
        state = set_volatile(state, 0, idx, VOL_CONFUSED, True)
        state = set_volatile(state, 0, idx, VOL_SEEDED, True)
        state = set_volatile(state, 0, idx, VOL_TAUNT, True)
        state = switch_out(state, 0, cfg4)
        assert int(state.sides_team_volatiles[0, idx]) == 0

    def test_switch_out_resets_boosts(self, cfg4):
        state = _make_state()
        state = set_boost(state, 0, 0, BOOST_ATK, jnp.int8(3))
        state = set_boost(state, 0, 0, BOOST_SPE, jnp.int8(-2))
        state = switch_out(state, 0, cfg4)
        for i in range(7):
            assert int(state.sides_team_boosts[0, 0, i]) == 0

    def test_switch_out_resets_toxic_counter(self, cfg4):
        """PS Gen 4: Toxic counter resets to 1 on switch."""
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_TOX), jnp.int8(5))
        state = switch_out(state, 0, cfg4)
        assert int(state.sides_team_status_turns[0, 0]) == 1

    def test_switch_out_keeps_status(self, cfg4):
        """Status persists through switch (Gen 4)."""
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_BRN))
        state = switch_out(state, 0, cfg4)
        assert int(state.sides_team_status[0, 0]) == STATUS_BRN

    def test_switch_out_clears_choice_lock(self, cfg4):
        state = _make_state()
        state = set_volatile(state, 0, 0, VOL_CHOICELOCK, True)
        state = switch_out(state, 0, cfg4)
        assert not bool(has_volatile(state, 0, 0, VOL_CHOICELOCK))


class TestSwitchIn:
    """PS: Switch-in applies entry hazards and triggers switch-in abilities."""

    def test_switch_in_changes_active(self, tables4, cfg4):
        state = _make_state()
        assert int(state.sides_active_idx[0]) == 0
        state = set_active(state, 0, 2)
        assert int(state.sides_active_idx[0]) == 2

    def test_switch_in_triggers_hazards(self, tables4, cfg4):
        state = _make_state(max_hp=160)
        state = set_side_condition(state, 0, SC_STEALTHROCK, jnp.int8(1))
        hp_before = int(state.sides_team_hp[0, 1])
        state = switch_in(state, 0, 1, tables4, cfg4)
        # Slot 1 is now active and should have taken SR damage
        assert int(state.sides_team_hp[0, 1]) < hp_before


class TestForcedSwitch:
    """PS: Forced switch finds first alive non-active Pokemon."""

    def test_finds_first_alive(self):
        state = _make_state()
        # Faint slots 1, 2; first alive non-active should be 3
        state = state._replace(sides_team_fainted=
            state.sides_team_fainted.at[0, 1].set(True).at[0, 2].set(True))
        slot = find_forced_switch_slot(state, 0)
        assert int(slot) == 3

    def test_no_replacement_returns_minus_one(self):
        state = _make_state()
        # Faint all except slot 0 (active)
        fainted = state.sides_team_fainted.at[0, 1].set(True).at[0, 2].set(True) \
            .at[0, 3].set(True).at[0, 4].set(True).at[0, 5].set(True)
        state = state._replace(sides_team_fainted=fainted)
        slot = find_forced_switch_slot(state, 0)
        assert int(slot) == -1

    def test_skips_active_slot(self):
        state = _make_state()
        slot = find_forced_switch_slot(state, 0)
        assert int(slot) != int(state.sides_active_idx[0])


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Volatile residuals parity
# ═══════════════════════════════════════════════════════════════════════════

class TestLeechSeed:
    """PS: Leech Seed drains 1/8 max HP and heals opponent."""

    def test_drain_amount(self):
        state = _make_state(max_hp=160)
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, VOL_SEEDED, True)
        # Damage P2 to give room to heal
        opp_idx = int(state.sides_active_idx[1])
        state = state._replace(sides_team_hp=
            state.sides_team_hp.at[1, opp_idx].set(jnp.int16(100)))
        hp_p1_before = int(state.sides_team_hp[0, 0])
        hp_p2_before = int(state.sides_team_hp[1, opp_idx])
        state = apply_volatile_residuals(state, 0)
        drain = hp_p1_before - int(state.sides_team_hp[0, 0])
        heal = int(state.sides_team_hp[1, opp_idx]) - hp_p2_before
        assert drain == 20  # 160/8
        assert heal == drain

    def test_no_seed_no_drain(self):
        state = _make_state(max_hp=160)
        hp = int(state.sides_team_hp[0, 0])
        state = apply_volatile_residuals(state, 0)
        assert int(state.sides_team_hp[0, 0]) == hp


class TestIngrain:
    """PS: Ingrain restores 1/16 max HP per turn."""

    def test_ingrain_heals(self):
        state = _make_state(max_hp=160)
        idx = int(state.sides_active_idx[0])
        state = state._replace(sides_team_hp=
            state.sides_team_hp.at[0, idx].set(jnp.int16(100)))
        state = set_volatile(state, 0, idx, VOL_INGRAIN, True)
        state = apply_volatile_residuals(state, 0)
        assert int(state.sides_team_hp[0, 0]) == 110  # +10 (160/16)

    def test_ingrain_no_overheal(self):
        state = _make_state(max_hp=160)
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, VOL_INGRAIN, True)
        state = apply_volatile_residuals(state, 0)
        assert int(state.sides_team_hp[0, 0]) == 160  # capped at max


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Side condition timer parity
# ═══════════════════════════════════════════════════════════════════════════

class TestSideConditionTimers:
    """PS: Reflect/Light Screen/etc. decrement each turn, clear at 0."""

    @pytest.mark.parametrize("sc_idx", [SC_REFLECT, SC_LIGHTSCREEN, SC_TAILWIND, SC_SAFEGUARD, SC_MIST])
    def test_timer_decrements(self, sc_idx):
        state = _make_state()
        state = set_side_condition(state, 0, sc_idx, jnp.int8(5))
        state = tick_side_conditions(state, 0)
        assert int(state.sides_side_conditions[0, sc_idx]) == 4

    @pytest.mark.parametrize("sc_idx", [SC_REFLECT, SC_LIGHTSCREEN, SC_TAILWIND])
    def test_timer_expires_at_zero(self, sc_idx):
        state = _make_state()
        state = set_side_condition(state, 0, sc_idx, jnp.int8(1))
        state = tick_side_conditions(state, 0)
        assert int(state.sides_side_conditions[0, sc_idx]) == 0

    def test_hazards_dont_tick(self):
        """Hazards (Spikes, SR, TSpikes, SWeb) are permanent — no timer."""
        state = _make_state()
        state = set_side_condition(state, 0, SC_SPIKES, jnp.int8(3))
        state = set_side_condition(state, 0, SC_STEALTHROCK, jnp.int8(1))
        state = set_side_condition(state, 0, SC_TOXICSPIKES, jnp.int8(2))
        state = tick_side_conditions(state, 0)
        assert int(state.sides_side_conditions[0, SC_SPIKES]) == 3
        assert int(state.sides_side_conditions[0, SC_STEALTHROCK]) == 1
        assert int(state.sides_side_conditions[0, SC_TOXICSPIKES]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Win condition parity
# ═══════════════════════════════════════════════════════════════════════════

class TestWinConditions:
    """PS: Battle ends when one side has 0 Pokemon left."""

    def test_p0_wins_when_p1_all_fainted(self):
        state = _make_state()
        state = state._replace(sides_pokemon_left=
            state.sides_pokemon_left.at[1].set(jnp.int8(0)))
        state = check_win(state)
        assert bool(state.finished)
        assert int(state.winner) == 0  # P0 wins

    def test_p1_wins_when_p0_all_fainted(self):
        state = _make_state()
        state = state._replace(sides_pokemon_left=
            state.sides_pokemon_left.at[0].set(jnp.int8(0)))
        state = check_win(state)
        assert bool(state.finished)
        assert int(state.winner) == 1  # P1 wins

    def test_draw_when_both_fainted(self):
        state = _make_state()
        state = state._replace(sides_pokemon_left=
            jnp.zeros(2, dtype=jnp.int8))
        state = check_win(state)
        assert bool(state.finished)
        assert int(state.winner) == 2  # draw

    def test_ongoing_when_both_alive(self):
        state = _make_state()
        state = check_win(state)
        assert not bool(state.finished)
        assert int(state.winner) == -1

    def test_check_fainted_marks_zero_hp(self):
        state = _make_state(max_hp=100)
        state = apply_damage(state, 0, 0, jnp.int32(100))
        assert int(state.sides_team_hp[0, 0]) == 0
        state = check_fainted(state, 0)
        assert bool(state.sides_team_fainted[0, 0])
        assert int(state.sides_pokemon_left[0]) == 5  # was 6, now 5

    def test_check_fainted_no_double_count(self):
        state = _make_state(max_hp=100)
        state = apply_damage(state, 0, 0, jnp.int32(100))
        state = check_fainted(state, 0)
        left_after_first = int(state.sides_pokemon_left[0])
        state = check_fainted(state, 0)  # idempotent
        assert int(state.sides_pokemon_left[0]) == left_after_first


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: Action decoding parity
# ═══════════════════════════════════════════════════════════════════════════

class TestActionDecoding:
    """PS action encoding: 0-3 = moves, 4-9 = switches."""

    @pytest.mark.parametrize("action,expected_switch,expected_move_slot,expected_switch_slot", [
        (0, False, 0, -4),
        (1, False, 1, -3),
        (2, False, 2, -2),
        (3, False, 3, -1),
        (4, True,  3, 0),   # switch to slot 0
        (5, True,  3, 1),
        (9, True,  3, 5),
    ])
    def test_decode_action(self, action, expected_switch, expected_move_slot, expected_switch_slot):
        is_switch, move_slot, switch_slot = decode_action(jnp.int32(action))
        assert bool(is_switch) == expected_switch
        if not expected_switch:
            assert int(move_slot) == expected_move_slot


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: Ability parity (integration with event system)
# ═══════════════════════════════════════════════════════════════════════════

class TestIntimidateAbility:
    """PS: Intimidate lowers foe's Attack by 1 on switch-in."""

    def test_intimidate_lowers_foe_attack(self, tables4, cfg4):
        ab_id = _get_ability_id(tables4, "Intimidate")
        if ab_id < 0:
            pytest.skip("Intimidate not in tables")
        state = _make_state(p1_abilities=ab_id)
        # Simulate switch-in of slot 0 (P0 has Intimidate)
        state = run_event_switch_in(state, 0, state.sides_active_idx[0])
        # P1's active should have -1 ATK boost
        p1_atk_boost = int(state.sides_team_boosts[1, int(state.sides_active_idx[1]), BOOST_ATK])
        assert p1_atk_boost == -1

    def test_intimidate_doesnt_lower_own_attack(self, tables4, cfg4):
        ab_id = _get_ability_id(tables4, "Intimidate")
        if ab_id < 0:
            pytest.skip("Intimidate not in tables")
        state = _make_state(p1_abilities=ab_id)
        state = run_event_switch_in(state, 0, state.sides_active_idx[0])
        p0_atk = int(state.sides_team_boosts[0, int(state.sides_active_idx[0]), BOOST_ATK])
        assert p0_atk == 0


class TestWeatherSettingAbilities:
    """PS: Drizzle/Drought/Sand Stream/Snow Warning set weather on switch-in."""

    @pytest.mark.parametrize("ability_name,expected_weather", [
        ("Drizzle",      WEATHER_RAIN),
        ("Drought",      WEATHER_SUN),
        ("Sand Stream",  WEATHER_SAND),
        ("Snow Warning", WEATHER_HAIL),
    ])
    def test_weather_ability_sets_weather(self, tables4, ability_name, expected_weather):
        ab_id = _get_ability_id(tables4, ability_name)
        if ab_id < 0:
            pytest.skip(f"{ability_name} not in tables")
        state = _make_state(p1_abilities=ab_id)
        state = run_event_switch_in(state, 0, state.sides_active_idx[0])
        assert int(state.field.weather) == expected_weather


class TestNaturalCure:
    """PS: Natural Cure clears status on switch-out."""

    def test_natural_cure_clears_status(self, tables4, cfg4):
        ab_id = _get_ability_id(tables4, "Natural Cure")
        if ab_id < 0:
            pytest.skip("Natural Cure not in tables")
        state = _make_state(p1_abilities=ab_id)
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        state = switch_out(state, 0, cfg4)
        assert int(state.sides_team_status[0, 0]) == STATUS_NONE


class TestSpeedBoostAbility:
    """PS: Speed Boost raises Speed by 1 stage at end of each turn."""

    def test_speed_boost_residual(self, tables4, cfg4):
        ab_id = _get_ability_id(tables4, "Speed Boost")
        if ab_id < 0:
            pytest.skip("Speed Boost not in tables")
        state = _make_state(p1_abilities=ab_id)
        slot = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state, _ = run_event_residual_state(state, key, 0, slot)
        spe_boost = int(state.sides_team_boosts[0, int(slot), BOOST_SPE])
        assert spe_boost == 1


class TestSpeedAbilities:
    """PS: Swift Swim/Chlorophyll double speed in weather."""

    @pytest.mark.parametrize("ability_name,weather", [
        ("Swift Swim",   WEATHER_RAIN),
        ("Chlorophyll",  WEATHER_SUN),
    ])
    def test_speed_doubling_ability(self, tables4, cfg4, ability_name, weather):
        ab_id = _get_ability_id(tables4, ability_name)
        if ab_id < 0:
            pytest.skip(f"{ability_name} not in tables")
        state = _make_state(p1_abilities=ab_id, p1_base_stats=(80, 80, 80, 80, 80, 100))
        base_speed = int(get_effective_speed(state, 0, cfg4))
        state = set_weather(state, jnp.int8(weather), jnp.int8(5))
        boosted_speed = int(get_effective_speed(state, 0, cfg4))
        assert boosted_speed == base_speed * 2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: Item parity
# ═══════════════════════════════════════════════════════════════════════════

class TestLeftoversItem:
    """PS: Leftovers restores 1/16 max HP at end of each turn."""

    def test_leftovers_heals(self, tables4, cfg4):
        item_id = _get_item_id(tables4, "Leftovers")
        if item_id < 0:
            pytest.skip("Leftovers not in tables")
        state = _make_state(max_hp=160, p1_items=item_id)
        state = state._replace(sides_team_hp=
            state.sides_team_hp.at[0, 0].set(jnp.int16(100)))
        slot = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state, _ = run_event_residual_state(state, key, 0, slot)
        assert int(state.sides_team_hp[0, 0]) == 110  # +10 (160/16)

    def test_leftovers_no_overheal(self, tables4, cfg4):
        item_id = _get_item_id(tables4, "Leftovers")
        if item_id < 0:
            pytest.skip("Leftovers not in tables")
        state = _make_state(max_hp=160, p1_items=item_id)
        slot = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state, _ = run_event_residual_state(state, key, 0, slot)
        assert int(state.sides_team_hp[0, 0]) == 160


class TestChoiceItems:
    """PS: Choice Band/Specs/Scarf lock to first move used."""

    def test_choice_scarf_speed_boost(self, tables4, cfg4):
        item_id = _get_item_id(tables4, "Choice Scarf")
        if item_id < 0:
            pytest.skip("Choice Scarf not in tables")
        state = _make_state(p1_base_stats=(80, 80, 80, 80, 80, 100))
        base_speed = int(get_effective_speed(state, 0, cfg4))
        state_scarf = _make_state(p1_base_stats=(80, 80, 80, 80, 80, 100), p1_items=item_id)
        scarf_speed = int(get_effective_speed(state_scarf, 0, cfg4))
        assert scarf_speed == int(base_speed * 1.5)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: Full turn execution integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFullTurnExecution:
    """End-to-end turn execution: both players choose, full pipeline runs."""

    def test_turn_increments(self, tables4, cfg4):
        state = _make_state()
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)  # Both use move 0
        state2, reveal2 = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.turn) == 1

    def test_turn_not_finished_after_one_turn(self, tables4, cfg4):
        state = _make_state(max_hp=500)  # high HP so nobody faints
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert not bool(state2.finished)

    def test_pp_deducted_after_move(self, tables4, cfg4):
        state = _make_state(max_hp=500)
        reveal = make_reveal_state(state)
        pp_before = int(state.sides_team_move_pp[0, 0, 0])
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        pp_after = int(state2.sides_team_move_pp[0, 0, 0])
        assert pp_after == pp_before - 1

    def test_switch_action_changes_active(self, tables4, cfg4):
        state = _make_state(max_hp=500)
        reveal = make_reveal_state(state)
        # P0 switches to slot 2 (action=6), P1 uses move 0
        actions = jnp.array([6, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.sides_active_idx[0]) == 2

    def test_multi_turn_hp_tracking(self, tables4, cfg4):
        """Run 3 turns and verify total HP decreases over time."""
        # Use actual damaging moves (Tackle)
        tackle_id = _get_move_id(tables4, "Tackle")
        if tackle_id < 0:
            pytest.skip("Tackle not in tables")
        state = _make_state_with_moves(tables4,
                                       ["Tackle", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"],
                                       max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        initial_total_hp = int(state.sides_team_hp[0, 0]) + int(state.sides_team_hp[1, 0])
        for _ in range(3):
            state, reveal = execute_turn(state, reveal, actions, tables4, cfg4)
        final_total_hp = (int(state.sides_team_hp[0, int(state.sides_active_idx[0])]) +
                          int(state.sides_team_hp[1, int(state.sides_active_idx[1])]))
        # Total HP across both sides should decrease after 3 turns of attacking
        assert final_total_hp < initial_total_hp


class TestRevealState:
    """Verify RevealState tracks information correctly across turns."""

    def test_initial_reveal_shows_leads(self):
        state = _make_state()
        reveal = make_reveal_state(state)
        # P0's lead visible to P1
        assert bool(reveal.revealed_pokemon[1, 0])
        # P1's lead visible to P0
        assert bool(reveal.revealed_pokemon[0, 0])
        # Other slots not revealed
        for s in range(1, 6):
            assert not bool(reveal.revealed_pokemon[0, s])
            assert not bool(reveal.revealed_pokemon[1, s])

    def test_moves_not_revealed_initially(self):
        state = _make_state()
        reveal = make_reveal_state(state)
        assert not reveal.revealed_moves.any()

    def test_switch_reveals_new_pokemon(self, tables4, cfg4):
        state = _make_state(max_hp=500)
        reveal = make_reveal_state(state)
        # P0 switches to slot 2
        actions = jnp.array([6, 0], dtype=jnp.int32)
        state2, reveal2 = execute_turn(state, reveal, actions, tables4, cfg4)
        # P1 should now see P0's slot 2
        assert bool(reveal2.revealed_pokemon[1, 2])


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14: Move-specific effects parity
# ═══════════════════════════════════════════════════════════════════════════

class TestMoveEffects:
    """Verify specific move effects match PS behavior."""

    def test_stealth_rock_move_sets_hazard(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Stealth Rock")
        if mid < 0:
            pytest.skip("Stealth Rock not in tables")
        state = _make_state_with_moves(tables4, ["Stealth Rock", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)  # P0 uses SR
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.sides_side_conditions[1, SC_STEALTHROCK]) == 1

    def test_swords_dance_raises_attack_by_2(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Swords Dance")
        if mid < 0:
            pytest.skip("Swords Dance not in tables")
        state = _make_state_with_moves(tables4, ["Swords Dance", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        p0_atk_boost = int(state2.sides_team_boosts[0, int(state2.sides_active_idx[0]), BOOST_ATK])
        assert p0_atk_boost == 2

    def test_calm_mind_raises_spa_and_spd(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Calm Mind")
        if mid < 0:
            pytest.skip("Calm Mind not in tables")
        state = _make_state_with_moves(tables4, ["Calm Mind", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        idx = int(state2.sides_active_idx[0])
        assert int(state2.sides_team_boosts[0, idx, BOOST_SPA]) == 1
        assert int(state2.sides_team_boosts[0, idx, BOOST_SPD]) == 1

    def test_spikes_adds_layer(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Spikes")
        if mid < 0:
            pytest.skip("Spikes not in tables")
        state = _make_state_with_moves(tables4, ["Spikes", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.sides_side_conditions[1, SC_SPIKES]) == 1

    def test_reflect_sets_screen(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Reflect")
        if mid < 0:
            pytest.skip("Reflect not in tables")
        state = _make_state_with_moves(tables4, ["Reflect", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.sides_side_conditions[0, SC_REFLECT]) > 0

    def test_light_screen_sets_screen(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Light Screen")
        if mid < 0:
            pytest.skip("Light Screen not in tables")
        state = _make_state_with_moves(tables4, ["Light Screen", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.sides_side_conditions[0, SC_LIGHTSCREEN]) > 0

    def test_toxic_spikes_move_adds_layer(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Toxic Spikes")
        if mid < 0:
            pytest.skip("Toxic Spikes not in tables")
        state = _make_state_with_moves(tables4, ["Toxic Spikes", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.sides_side_conditions[1, SC_TOXICSPIKES]) == 1

    def test_trick_room_toggles(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Trick Room")
        if mid < 0:
            pytest.skip("Trick Room not in tables")
        state = _make_state_with_moves(tables4, ["Trick Room", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.field.trick_room) > 0

    def test_sunny_day_sets_sun(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Sunny Day")
        if mid < 0:
            pytest.skip("Sunny Day not in tables")
        state = _make_state_with_moves(tables4, ["Sunny Day", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.field.weather) == WEATHER_SUN

    def test_rain_dance_sets_rain(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Rain Dance")
        if mid < 0:
            pytest.skip("Rain Dance not in tables")
        state = _make_state_with_moves(tables4, ["Rain Dance", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"], max_hp=500)
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        assert int(state2.field.weather) == WEATHER_RAIN

    @pytest.mark.xfail(reason="PokeJAX bug: Protect may not fully block damage in all scenarios", strict=False)
    def test_protect_blocks_damage(self, tables4, cfg4):
        mid = _get_move_id(tables4, "Protect")
        if mid < 0:
            pytest.skip("Protect not in tables")
        state = _make_state_with_moves(tables4,
                                       ["Protect", "Tackle", "Tackle", "Tackle"],
                                       ["Tackle", "Tackle", "Tackle", "Tackle"],
                                       max_hp=500,
                                       # Give P1 much higher speed so it moves first
                                       p2_base_stats=(80, 80, 80, 80, 80, 200))
        # Actually P0 uses Protect (+4 priority), so it always goes first
        reveal = make_reveal_state(state)
        actions = jnp.array([0, 0], dtype=jnp.int32)  # P0: Protect, P1: Tackle
        hp_before = int(state.sides_team_hp[0, 0])
        state2, _ = execute_turn(state, reveal, actions, tables4, cfg4)
        # P0 should have taken no damage (Protect blocks)
        p0_idx = int(state2.sides_active_idx[0])
        hp_after = int(state2.sides_team_hp[0, p0_idx])
        assert hp_after == hp_before, f"Protect should block damage: {hp_before} -> {hp_after}"


class TestBoostClamping:
    """PS: Stat boosts are clamped to [-6, +6]."""

    def test_boost_clamped_at_plus_6(self):
        state = _make_state()
        state = set_boost(state, 0, 0, BOOST_ATK, jnp.int8(10))
        assert int(state.sides_team_boosts[0, 0, BOOST_ATK]) == 6

    def test_boost_clamped_at_minus_6(self):
        state = _make_state()
        state = set_boost(state, 0, 0, BOOST_ATK, jnp.int8(-10))
        assert int(state.sides_team_boosts[0, 0, BOOST_ATK]) == -6

    def test_add_boost_respects_clamp(self):
        state = _make_state()
        state = set_boost(state, 0, 0, BOOST_ATK, jnp.int8(5))
        state = add_boost(state, 0, 0, BOOST_ATK, jnp.int8(3))
        assert int(state.sides_team_boosts[0, 0, BOOST_ATK]) == 6  # 5+3=8 → clamped 6

    def test_add_negative_boost_respects_clamp(self):
        state = _make_state()
        state = set_boost(state, 0, 0, BOOST_DEF, jnp.int8(-5))
        state = add_boost(state, 0, 0, BOOST_DEF, jnp.int8(-3))
        assert int(state.sides_team_boosts[0, 0, BOOST_DEF]) == -6


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 15: Battle state construction invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestBattleStateInvariants:
    """Verify initial state matches PS conventions."""

    def test_initial_hp_equals_max_hp(self):
        state = _make_state(max_hp=200)
        for side in range(2):
            for slot in range(6):
                assert int(state.sides_team_hp[side, slot]) == int(state.sides_team_max_hp[side, slot])

    def test_initial_no_status(self):
        state = _make_state()
        for side in range(2):
            for slot in range(6):
                assert int(state.sides_team_status[side, slot]) == STATUS_NONE

    def test_initial_no_volatiles(self):
        state = _make_state()
        for side in range(2):
            for slot in range(6):
                assert int(state.sides_team_volatiles[side, slot]) == 0

    def test_initial_boosts_zero(self):
        state = _make_state()
        for side in range(2):
            for slot in range(6):
                for b in range(7):
                    assert int(state.sides_team_boosts[side, slot, b]) == 0

    def test_initial_slot_0_active(self):
        state = _make_state()
        for side in range(2):
            assert int(state.sides_active_idx[side]) == 0
            assert bool(state.sides_team_is_active[side, 0])

    def test_initial_6_pokemon_left(self):
        state = _make_state()
        for side in range(2):
            assert int(state.sides_pokemon_left[side]) == 6

    def test_initial_no_weather(self):
        state = _make_state()
        assert int(state.field.weather) == WEATHER_NONE

    def test_initial_not_finished(self):
        state = _make_state()
        assert not bool(state.finished)
        assert int(state.winner) == -1

    def test_initial_turn_zero(self):
        state = _make_state()
        assert int(state.turn) == 0

    def test_initial_no_side_conditions(self):
        state = _make_state()
        for side in range(2):
            for sc in range(10):
                assert int(state.sides_side_conditions[side, sc]) == 0

    def test_initial_no_fainted(self):
        state = _make_state()
        for side in range(2):
            for slot in range(6):
                assert not bool(state.sides_team_fainted[side, slot])
