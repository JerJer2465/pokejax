"""
L1 property tests: damage formula and modifier chain.

All expected values calculated by hand from the Gen 4 formula and
cross-checked against Pokemon damage calculators.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.core.damage import (
    base_damage,
    apply_weather_modifier,
    apply_crit_modifier,
    apply_stab_modifier,
    apply_type_modifier,
    apply_burn_modifier,
    apply_screen_modifier,
    apply_random_modifier,
    calc_stat,
    calc_hp,
    type_effectiveness,
    fraction_of_max_hp,
    apply_damage,
    apply_heal,
)
from pokejax.types import (
    TYPE_FIRE, TYPE_WATER, TYPE_GRASS, TYPE_NORMAL, TYPE_NONE,
    WEATHER_SUN, WEATHER_RAIN, WEATHER_NONE,
    CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS,
    STATUS_BRN, STATUS_NONE,
)
from pokejax.core.state import make_battle_state, make_field
from pokejax.data.extractor import _build_nature_table


# ---------------------------------------------------------------------------
# base_damage formula
# ---------------------------------------------------------------------------

class TestBaseDamage:
    def test_known_value(self):
        # Level 100, 100 bp, 200 atk, 100 def
        # floor(floor(floor(2*100/5+2)*100*200/100)/50)+2
        # = floor(floor(42*100*200/100)/50)+2
        # = floor(floor(8400)/50)+2
        # = floor(168)+2 = 170
        dmg = base_damage(
            level=jnp.int32(100),
            power=jnp.int32(100),
            attack=jnp.int32(200),
            defense=jnp.int32(100),
        )
        assert int(dmg) == 170

    def test_level_scaling(self):
        # Lower level → lower damage
        dmg50 = base_damage(jnp.int32(50), jnp.int32(100), jnp.int32(100), jnp.int32(100))
        dmg100 = base_damage(jnp.int32(100), jnp.int32(100), jnp.int32(100), jnp.int32(100))
        assert int(dmg50) < int(dmg100)

    def test_power_zero(self):
        # Status moves have base power 0
        dmg = base_damage(jnp.int32(100), jnp.int32(0), jnp.int32(100), jnp.int32(100))
        assert int(dmg) == 2  # +2 from formula, power=0 gives 0 + 2

    def test_high_attack(self):
        # Higher attack → more damage
        dmg1 = base_damage(jnp.int32(100), jnp.int32(80), jnp.int32(100), jnp.int32(100))
        dmg2 = base_damage(jnp.int32(100), jnp.int32(80), jnp.int32(200), jnp.int32(100))
        assert int(dmg2) > int(dmg1)

    def test_high_defense_reduces_damage(self):
        dmg1 = base_damage(jnp.int32(100), jnp.int32(80), jnp.int32(100), jnp.int32(100))
        dmg2 = base_damage(jnp.int32(100), jnp.int32(80), jnp.int32(100), jnp.int32(200))
        assert int(dmg2) < int(dmg1)

    def test_tackle_typical(self):
        # Pikachu (level 50) using Tackle (40 bp) on Rattata
        # Pikachu Atk: ~55 base, lv50 ≈ 55; Rattata Def: ~35 base, lv50 ≈ 35
        # Approx expected: reasonable small damage
        dmg = base_damage(jnp.int32(50), jnp.int32(40), jnp.int32(55), jnp.int32(35))
        assert int(dmg) > 0
        assert int(dmg) < 200

    def test_formula_floor_behavior(self):
        # Verify each floor is applied correctly
        # floor(floor(floor(2*50/5+2)*100*100/100)/50)+2
        # = floor(floor(22*100)/50)+2
        # = floor(2200/50)+2 = floor(44)+2 = 46
        dmg = base_damage(jnp.int32(50), jnp.int32(100), jnp.int32(100), jnp.int32(100))
        assert int(dmg) == 46


# ---------------------------------------------------------------------------
# Weather modifiers
# ---------------------------------------------------------------------------

class TestWeatherModifier:
    def test_sun_boosts_fire(self):
        dmg = jnp.int32(100)
        result = apply_weather_modifier(dmg, jnp.int32(TYPE_FIRE), jnp.int8(WEATHER_SUN))
        assert int(result) == 150  # 1.5x

    def test_sun_nerfs_water(self):
        dmg = jnp.int32(100)
        result = apply_weather_modifier(dmg, jnp.int32(TYPE_WATER), jnp.int8(WEATHER_SUN))
        assert int(result) == 50  # 0.5x

    def test_rain_boosts_water(self):
        dmg = jnp.int32(100)
        result = apply_weather_modifier(dmg, jnp.int32(TYPE_WATER), jnp.int8(WEATHER_RAIN))
        assert int(result) == 150

    def test_rain_nerfs_fire(self):
        dmg = jnp.int32(100)
        result = apply_weather_modifier(dmg, jnp.int32(TYPE_FIRE), jnp.int8(WEATHER_RAIN))
        assert int(result) == 50

    def test_no_weather_no_change(self):
        dmg = jnp.int32(100)
        result = apply_weather_modifier(dmg, jnp.int32(TYPE_FIRE), jnp.int8(WEATHER_NONE))
        assert int(result) == 100

    def test_sun_neutral_grass(self):
        dmg = jnp.int32(100)
        result = apply_weather_modifier(dmg, jnp.int32(TYPE_GRASS), jnp.int8(WEATHER_SUN))
        assert int(result) == 100


# ---------------------------------------------------------------------------
# Crit modifier
# ---------------------------------------------------------------------------

class TestCritModifier:
    def test_crit_doubles_gen4(self):
        # Gen 4: crits deal 2x
        dmg = jnp.int32(100)
        result = apply_crit_modifier(dmg, jnp.bool_(True))
        assert int(result) == 200

    def test_no_crit_unchanged(self):
        dmg = jnp.int32(100)
        result = apply_crit_modifier(dmg, jnp.bool_(False))
        assert int(result) == 100


# ---------------------------------------------------------------------------
# STAB modifier
# ---------------------------------------------------------------------------

class TestSTABModifier:
    def test_stab_15x(self):
        dmg = jnp.int32(100)
        atk_types = jnp.array([TYPE_FIRE, TYPE_NONE], dtype=jnp.int8)
        result = apply_stab_modifier(dmg, jnp.int32(TYPE_FIRE), atk_types,
                                      jnp.bool_(False))
        assert int(result) == 150

    def test_no_stab_unchanged(self):
        dmg = jnp.int32(100)
        atk_types = jnp.array([TYPE_WATER, TYPE_NONE], dtype=jnp.int8)
        result = apply_stab_modifier(dmg, jnp.int32(TYPE_FIRE), atk_types,
                                      jnp.bool_(False))
        assert int(result) == 100

    def test_adaptability_stab_2x(self):
        dmg = jnp.int32(100)
        atk_types = jnp.array([TYPE_FIRE, TYPE_NONE], dtype=jnp.int8)
        result = apply_stab_modifier(dmg, jnp.int32(TYPE_FIRE), atk_types,
                                      jnp.bool_(True))
        assert int(result) == 200

    def test_stab_second_type(self):
        dmg = jnp.int32(100)
        atk_types = jnp.array([TYPE_WATER, TYPE_FIRE], dtype=jnp.int8)
        result = apply_stab_modifier(dmg, jnp.int32(TYPE_FIRE), atk_types,
                                      jnp.bool_(False))
        assert int(result) == 150


# ---------------------------------------------------------------------------
# Type modifier
# ---------------------------------------------------------------------------

class TestTypeModifier:
    def test_2x(self):
        assert int(apply_type_modifier(jnp.int32(100), jnp.float32(2.0))) == 200

    def test_05x(self):
        assert int(apply_type_modifier(jnp.int32(100), jnp.float32(0.5))) == 50

    def test_immune(self):
        assert int(apply_type_modifier(jnp.int32(100), jnp.float32(0.0))) == 0

    def test_4x(self):
        assert int(apply_type_modifier(jnp.int32(100), jnp.float32(4.0))) == 400

    def test_025x(self):
        assert int(apply_type_modifier(jnp.int32(100), jnp.float32(0.25))) == 25


# ---------------------------------------------------------------------------
# Burn modifier
# ---------------------------------------------------------------------------

class TestBurnModifier:
    def test_burn_halves_physical(self):
        dmg = jnp.int32(100)
        result = apply_burn_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                      jnp.int8(STATUS_BRN), jnp.bool_(False))
        assert int(result) == 50

    def test_burn_no_effect_on_special(self):
        dmg = jnp.int32(100)
        result = apply_burn_modifier(dmg, jnp.int8(CATEGORY_SPECIAL),
                                      jnp.int8(STATUS_BRN), jnp.bool_(False))
        assert int(result) == 100

    def test_no_burn_unchanged(self):
        dmg = jnp.int32(100)
        result = apply_burn_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                      jnp.int8(STATUS_NONE), jnp.bool_(False))
        assert int(result) == 100

    def test_guts_ignores_burn(self):
        dmg = jnp.int32(100)
        result = apply_burn_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                      jnp.int8(STATUS_BRN), jnp.bool_(True))
        assert int(result) == 100


# ---------------------------------------------------------------------------
# Random roll modifier
# ---------------------------------------------------------------------------

class TestRandomModifier:
    def test_max_roll(self):
        # Roll = 1.0 → no change
        dmg = jnp.int32(100)
        result = apply_random_modifier(dmg, jnp.float32(1.0))
        assert int(result) == 100

    def test_min_roll(self):
        # Roll = 0.85 → floor(100 * 0.85) = 85
        dmg = jnp.int32(100)
        result = apply_random_modifier(dmg, jnp.float32(0.85))
        assert int(result) == 85

    def test_roll_floors(self):
        # floor(101 * 0.85) = floor(85.85) = 85
        dmg = jnp.int32(101)
        result = apply_random_modifier(dmg, jnp.float32(0.85))
        assert int(result) == 85


# ---------------------------------------------------------------------------
# Stat calculation
# ---------------------------------------------------------------------------

class TestStatCalc:
    def test_neutral_nature(self):
        # 100 base, lv100, 0 EV, 31 IV, 1.0 nature
        # floor((floor((200+31+0)*100/100)+5)*1.0) = floor(231+5) = 236
        stat = calc_stat(jnp.int32(100), jnp.int32(100),
                         jnp.int32(0), jnp.int32(31), jnp.float32(1.0))
        assert int(stat) == 236

    def test_boosting_nature(self):
        stat = calc_stat(jnp.int32(100), jnp.int32(100),
                         jnp.int32(0), jnp.int32(31), jnp.float32(1.1))
        # floor(236 * 1.1) = floor(259.6) = 259
        assert int(stat) == 259

    def test_hindering_nature(self):
        stat = calc_stat(jnp.int32(100), jnp.int32(100),
                         jnp.int32(0), jnp.int32(31), jnp.float32(0.9))
        # floor(236 * 0.9) = floor(212.4) = 212
        assert int(stat) == 212

    def test_ev_contribution(self):
        # 252 EVs: +floor(252/4) = +63
        # floor((200+31+63)*100/100)+5 = 294+5 = 299 → 299
        stat_no_ev = calc_stat(jnp.int32(100), jnp.int32(100),
                                jnp.int32(0), jnp.int32(31), jnp.float32(1.0))
        stat_with_ev = calc_stat(jnp.int32(100), jnp.int32(100),
                                  jnp.int32(252), jnp.int32(31), jnp.float32(1.0))
        assert int(stat_with_ev) > int(stat_no_ev)

    def test_hp_calc(self):
        # 100 base HP, lv100, 0 EV, 31 IV
        # floor((200+31+0)*100/100)+100+10 = 231+110 = 341
        hp = calc_hp(jnp.int32(100), jnp.int32(100), jnp.int32(0), jnp.int32(31))
        assert int(hp) == 341

    def test_nature_table_25_natures(self):
        table = _build_nature_table()
        assert table.shape == (25, 5)
        # Neutral natures should have all 1.0
        for i in range(5):  # Hardy, Docile, Serious, Bashful, Quirky are indices 0-4
            assert float(table[0, i]) == pytest.approx(1.0)  # Hardy


# ---------------------------------------------------------------------------
# Boost multiplier table
# ---------------------------------------------------------------------------

class TestBoostTable:
    def test_zero_boost(self, tables4):
        mult = tables4.get_boost_multiplier(jnp.int32(0))
        assert float(mult) == pytest.approx(1.0)

    def test_plus1(self, tables4):
        mult = tables4.get_boost_multiplier(jnp.int32(1))
        assert float(mult) == pytest.approx(1.5)

    def test_plus2(self, tables4):
        mult = tables4.get_boost_multiplier(jnp.int32(2))
        assert float(mult) == pytest.approx(2.0)

    def test_plus6(self, tables4):
        mult = tables4.get_boost_multiplier(jnp.int32(6))
        assert float(mult) == pytest.approx(4.0)

    def test_minus1(self, tables4):
        mult = tables4.get_boost_multiplier(jnp.int32(-1))
        assert float(mult) == pytest.approx(2/3, rel=1e-3)

    def test_minus6(self, tables4):
        mult = tables4.get_boost_multiplier(jnp.int32(-6))
        assert float(mult) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# HP helpers
# ---------------------------------------------------------------------------

def _make_test_state():
    """Create a minimal battle state for HP tests."""
    zeros6   = np.zeros(6, dtype=np.int16)
    zeros6i8 = np.zeros(6, dtype=np.int8)
    z6_4_i16 = np.full((6, 4), -1, dtype=np.int16)
    z6_4_i8  = np.zeros((6, 4), dtype=np.int8)
    z6_2_i8  = np.zeros((6, 2), dtype=np.int8)
    z6_6_i16 = np.zeros((6, 6), dtype=np.int16)

    max_hp = np.array([100, 90, 80, 70, 60, 50], dtype=np.int16)

    return make_battle_state(
        p1_species=zeros6, p2_species=zeros6,
        p1_abilities=zeros6, p2_abilities=zeros6,
        p1_items=zeros6, p2_items=zeros6,
        p1_types=z6_2_i8, p2_types=z6_2_i8,
        p1_base_stats=z6_6_i16, p2_base_stats=z6_6_i16,
        p1_max_hp=max_hp, p2_max_hp=max_hp,
        p1_move_ids=z6_4_i16, p2_move_ids=z6_4_i16,
        p1_move_pp=z6_4_i8, p2_move_pp=z6_4_i8,
        p1_move_max_pp=z6_4_i8, p2_move_max_pp=z6_4_i8,
        p1_levels=zeros6i8+50, p2_levels=zeros6i8+50,
        p1_genders=zeros6i8, p2_genders=zeros6i8,
        p1_natures=zeros6i8, p2_natures=zeros6i8,
        p1_weights_hg=zeros6, p2_weights_hg=zeros6,
        rng_key=jax.random.PRNGKey(0),
    )


import jax


class TestHPHelpers:
    def test_apply_damage(self):
        state = _make_test_state()
        assert int(state.sides_team_hp[0, 0]) == 100
        new_state = apply_damage(state, 0, 0, jnp.int32(30))
        assert int(new_state.sides_team_hp[0, 0]) == 70

    def test_apply_damage_floor_at_zero(self):
        state = _make_test_state()
        new_state = apply_damage(state, 0, 0, jnp.int32(999))
        assert int(new_state.sides_team_hp[0, 0]) == 0

    def test_apply_heal(self):
        state = _make_test_state()
        state = apply_damage(state, 0, 0, jnp.int32(50))
        new_state = apply_heal(state, 0, 0, jnp.int32(20))
        assert int(new_state.sides_team_hp[0, 0]) == 70

    def test_apply_heal_cap_at_max(self):
        state = _make_test_state()
        state = apply_damage(state, 0, 0, jnp.int32(10))
        new_state = apply_heal(state, 0, 0, jnp.int32(999))
        assert int(new_state.sides_team_hp[0, 0]) == 100  # max HP

    def test_fraction_of_max_hp(self):
        state = _make_test_state()
        # slot 0, max_hp=100: 1/8 = 12 (floor, min 1)
        val = fraction_of_max_hp(state, 0, 0, 1, 8)
        assert int(val) == 12

    def test_fraction_min_1(self):
        state = _make_test_state()
        # slot 5, max_hp=50: 1/100 = 0 → clamped to 1
        val = fraction_of_max_hp(state, 0, 5, 1, 100)
        assert int(val) == 1
