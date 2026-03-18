"""
Comprehensive Gen 4 damage formula parity tests.

Verifies that pokejax's damage calculation matches Pokemon Showdown's Gen 4
damage formula exactly, including the Gen 4-specific modifier ordering:

  PS Gen 4 order:
    1. Burn reduction (physical, non-Guts)
    2. Reflect/Light Screen (bypassed by crits)
    3. Spread move penalty (0.75x, doubles only)
    4. Weather modifier
    5. +2 constant
    6. Crit multiplier (2x, not 1.5x)
    7. Random roll (85-100%)
    8. STAB (1.5x or 2.0x with Adaptability)
    9. Type effectiveness
   10. ModifyDamage (Life Orb, Expert Belt, etc.)

All expected values hand-calculated from the PS Gen 4 scripts.ts source.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.core.damage import (
    base_damage,
    apply_weather_modifier,
    apply_crit_modifier,
    apply_stab_modifier,
    apply_type_modifier,
    apply_burn_modifier,
    apply_screen_modifier,
    apply_random_modifier,
    calc_stat, calc_hp,
    type_effectiveness,
    fraction_of_max_hp,
    compute_damage,
    MF_BASE_POWER, MF_ACCURACY, MF_TYPE, MF_CATEGORY, MF_PRIORITY,
)
from pokejax.core.state import (
    make_battle_state, set_status, set_boost, set_side_condition,
    set_weather, set_volatile, set_volatile_counter,
)
from pokejax.types import (
    TYPE_NONE, TYPE_NORMAL, TYPE_FIRE, TYPE_WATER, TYPE_ELECTRIC,
    TYPE_GRASS, TYPE_ICE, TYPE_FIGHTING, TYPE_POISON, TYPE_GROUND,
    TYPE_FLYING, TYPE_PSYCHIC, TYPE_BUG, TYPE_ROCK, TYPE_GHOST,
    TYPE_DRAGON, TYPE_DARK, TYPE_STEEL,
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX, STATUS_PAR,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS,
    SC_REFLECT, SC_LIGHTSCREEN,
    BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE,
    VOL_SUBSTITUTE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    max_hp=300,
    p1_types=(TYPE_NORMAL, 0),
    p2_types=(TYPE_NORMAL, 0),
    p1_base_stats=(100, 100, 100, 100, 100, 100),
    p2_base_stats=(100, 100, 100, 100, 100, 100),
    level=100,
    rng_seed=42,
):
    n = 6
    zeros6 = np.zeros(n, dtype=np.int16)
    zeros6i8 = np.zeros(n, dtype=np.int8)

    t1 = np.zeros((n, 2), dtype=np.int8)
    t1[:, 0] = p1_types[0]
    t1[:, 1] = p1_types[1]
    t2 = np.zeros((n, 2), dtype=np.int8)
    t2[:, 0] = p2_types[0]
    t2[:, 1] = p2_types[1]

    bs1 = np.array([list(p1_base_stats)] * n, dtype=np.int16)
    bs2 = np.array([list(p2_base_stats)] * n, dtype=np.int16)
    mhp = np.full(n, max_hp, dtype=np.int16)
    mid = np.tile(np.arange(4, dtype=np.int16), (n, 1))
    mpp = np.full((n, 4), 35, dtype=np.int8)
    levels = np.full(n, level, dtype=np.int8)

    return make_battle_state(
        p1_species=zeros6, p2_species=zeros6,
        p1_abilities=zeros6, p2_abilities=zeros6,
        p1_items=zeros6, p2_items=zeros6,
        p1_types=t1, p2_types=t2,
        p1_base_stats=bs1, p2_base_stats=bs2,
        p1_max_hp=mhp, p2_max_hp=mhp,
        p1_move_ids=mid, p2_move_ids=mid,
        p1_move_pp=mpp, p2_move_pp=mpp,
        p1_move_max_pp=mpp, p2_move_max_pp=mpp,
        p1_levels=levels, p2_levels=levels,
        p1_genders=zeros6i8, p2_genders=zeros6i8,
        p1_natures=zeros6i8, p2_natures=zeros6i8,
        p1_weights_hg=np.full(n, 100, dtype=np.int16),
        p2_weights_hg=np.full(n, 100, dtype=np.int16),
        rng_key=jax.random.PRNGKey(rng_seed),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Gen 4 base damage formula edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestGen4BaseDamageEdgeCases:
    """Edge cases in the base damage formula floor behavior."""

    @pytest.mark.parametrize("level,power,attack,defense,expected", [
        # Formula: floor(floor(floor(2*L/5+2)*P*A/D)/50)+2
        # Level 1: floor(2*1/5+2)=floor(2.4)=2
        (1,   100, 100, 100, 6),
        # Level 5: floor(2*5/5+2)=4
        (5,   100, 100, 100, 10),
        # Very high power
        (100, 250, 400, 50, 1682),
        # Very low stats
        (100, 10,  10,  10, 10),
        # Power 1 (Struggle-like)
        (100, 1,   100, 100, 2),
        # Extreme defense: floor(floor(42*100*100/500)/50)+2 = floor(840/50)+2 = 18
        (100, 100, 100, 500, 18),
        # All minimum: floor(floor(floor(2*1/5+2)*1*1/1)/50)+2 = floor(2/50)+2 = 2
        (1,   1,   1,   1,   2),
    ])
    def test_edge_case_values(self, level, power, attack, defense, expected):
        dmg = base_damage(
            jnp.int32(level), jnp.int32(power),
            jnp.int32(attack), jnp.int32(defense),
        )
        assert int(dmg) == expected


class TestGen4ModifierChainOrder:
    """
    Verify the Gen 4-specific modifier chain order produces correct results.

    In Gen 4, the order is:
      burn → screens → weather → +2 → crit → random → STAB → type → items
    This differs from Gen 5+ where screens come after weather.
    """

    def test_burn_applied_before_screens(self):
        """Burn halves, then Reflect halves again → 0.25x."""
        dmg = jnp.int32(100)
        dmg = apply_burn_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                  jnp.int8(STATUS_BRN), jnp.bool_(False))
        assert int(dmg) == 50
        dmg = apply_screen_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                    jnp.bool_(True), jnp.bool_(False),
                                    jnp.bool_(False))
        assert int(dmg) == 25

    def test_crit_bypasses_screens(self):
        """Critical hits ignore Reflect/Light Screen in Gen 4."""
        dmg = jnp.int32(100)
        result = apply_screen_modifier(
            dmg, jnp.int8(CATEGORY_PHYSICAL),
            jnp.bool_(True), jnp.bool_(False), jnp.bool_(True)
        )
        assert int(result) == 100

    def test_crit_bypasses_light_screen(self):
        dmg = jnp.int32(100)
        result = apply_screen_modifier(
            dmg, jnp.int8(CATEGORY_SPECIAL),
            jnp.bool_(False), jnp.bool_(True), jnp.bool_(True)
        )
        assert int(result) == 100

    def test_weather_stacks_with_stab(self):
        """Sun + Fire STAB: weather 1.5x then STAB 1.5x = 2.25x total."""
        dmg = jnp.int32(100)
        dmg = apply_weather_modifier(dmg, jnp.int32(TYPE_FIRE), jnp.int8(WEATHER_SUN))
        assert int(dmg) == 150
        atk_types = jnp.array([TYPE_FIRE, TYPE_NONE], dtype=jnp.int8)
        dmg = apply_stab_modifier(dmg, jnp.int32(TYPE_FIRE), atk_types,
                                  jnp.bool_(False))
        assert int(dmg) == 225

    def test_weather_stacks_with_type_effectiveness(self):
        """Rain + Water move vs Fire (SE): rain 1.5x, type 2x = 3x."""
        dmg = jnp.int32(100)
        dmg = apply_weather_modifier(dmg, jnp.int32(TYPE_WATER), jnp.int8(WEATHER_RAIN))
        assert int(dmg) == 150
        dmg = apply_type_modifier(dmg, jnp.float32(2.0))
        assert int(dmg) == 300

    def test_full_chain_physical_fire_in_sun(self):
        """
        Full chain: 100 base damage, no burn, no screen, sun, no crit,
        max roll, STAB fire, SE vs grass.
        Expected: 100 → sun 150 → STAB 225 → 2x type = 450
        """
        dmg = jnp.int32(100)
        dmg = apply_burn_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                  jnp.int8(STATUS_NONE), jnp.bool_(False))
        dmg = apply_screen_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                    jnp.bool_(False), jnp.bool_(False),
                                    jnp.bool_(False))
        dmg = apply_weather_modifier(dmg, jnp.int32(TYPE_FIRE), jnp.int8(WEATHER_SUN))
        dmg = apply_crit_modifier(dmg, jnp.bool_(False))
        dmg = apply_random_modifier(dmg, jnp.float32(1.0))
        atk_types = jnp.array([TYPE_FIRE, TYPE_NONE], dtype=jnp.int8)
        dmg = apply_stab_modifier(dmg, jnp.int32(TYPE_FIRE), atk_types,
                                  jnp.bool_(False))
        dmg = apply_type_modifier(dmg, jnp.float32(2.0))
        assert int(dmg) == 450

    def test_full_chain_burned_physical_vs_reflect(self):
        """
        Burn + Reflect: 100 → burn 50 → reflect 25
        """
        dmg = jnp.int32(100)
        dmg = apply_burn_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                  jnp.int8(STATUS_BRN), jnp.bool_(False))
        dmg = apply_screen_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                    jnp.bool_(True), jnp.bool_(False),
                                    jnp.bool_(False))
        assert int(dmg) == 25

    def test_crit_with_burn_physical(self):
        """
        Gen 4 crit: burn still applies (unlike Gen 6+).
        100 → burn 50 → crit 100
        """
        dmg = jnp.int32(100)
        dmg = apply_burn_modifier(dmg, jnp.int8(CATEGORY_PHYSICAL),
                                  jnp.int8(STATUS_BRN), jnp.bool_(False))
        assert int(dmg) == 50
        dmg = apply_crit_modifier(dmg, jnp.bool_(True))
        assert int(dmg) == 100


# ═══════════════════════════════════════════════════════════════════════════
# Gen 4 stat calculation with real Pokemon values
# ═══════════════════════════════════════════════════════════════════════════

class TestGen4StatCalcRealPokemon:
    """Stat calculations matching known Pokemon Showdown calculator values."""

    @pytest.mark.parametrize("base,level,ev,iv,nature,expected,desc", [
        # Garchomp: 108 base ATK, Lv100, 252 EV, 31 IV, Jolly (+Spe,-SpA)
        # ATK: floor((floor((216+31+63)*100/100)+5)*1.0) = floor(310+5) = 315
        (108, 100, 252, 31, 1.0, 315, "Garchomp ATK neutral"),
        # Scizor: 130 base ATK, Adamant (+ATK)
        # floor((floor((260+31+63)*100/100)+5)*1.1) = floor(359*1.1) = 394
        (130, 100, 252, 31, 1.1, 394, "Scizor ATK Adamant"),
        # Blissey: 10 base ATK, no EV, neutral
        # floor((floor((20+31+0)*100/100)+5)*1.0) = 56
        (10, 100, 0, 31, 1.0, 56, "Blissey ATK neutral"),
        # Deoxys-Speed: 150 base Spe, max invest, Timid
        # floor((floor((300+31+63)*100/100)+5)*1.1) = floor(399*1.1) = 438
        (150, 100, 252, 31, 1.1, 438, "Deoxys-S Spe Timid"),
        # Level 50 tournament: 100 base, 252 EV, 31 IV, neutral
        # floor((floor((200+31+63)*50/100)+5)*1.0) = floor(147+5) = 152
        (100, 50, 252, 31, 1.0, 152, "Lv50 100base maxEV neutral"),
    ])
    def test_stat_calc_real(self, base, level, ev, iv, nature, expected, desc):
        result = calc_stat(jnp.int32(base), jnp.int32(level),
                           jnp.int32(ev), jnp.int32(iv), jnp.float32(nature))
        assert int(result) == expected, f"{desc}: got {int(result)}, expected {expected}"

    @pytest.mark.parametrize("base_hp,level,ev,iv,expected,desc", [
        # Blissey: 255 base HP, max EV
        # floor((510+31+63)*100/100)+100+10 = 604+110 = 714
        (255, 100, 252, 31, 714, "Blissey HP maxEV"),
        # Shedinja: 1 base HP (special case)
        # floor((2+31+0)*100/100)+100+10 = 33+110 = 143
        (1, 100, 0, 31, 143, "Shedinja-like HP"),
        # Lv50 HP: 100 base, 252 EV
        # floor((200+31+63)*50/100)+50+10 = 147+60 = 207
        (100, 50, 252, 31, 207, "Lv50 HP maxEV"),
    ])
    def test_hp_calc_real(self, base_hp, level, ev, iv, expected, desc):
        result = calc_hp(jnp.int32(base_hp), jnp.int32(level),
                         jnp.int32(ev), jnp.int32(iv))
        assert int(result) == expected, f"{desc}: got {int(result)}, expected {expected}"


# ═══════════════════════════════════════════════════════════════════════════
# Boost multiplier parity
# ═══════════════════════════════════════════════════════════════════════════

class TestBoostMultiplierParity:
    """
    PS boost formula: max(2, 2+stage) / max(2, 2-stage).
    Must match exactly for all 13 stages (-6 to +6).
    """

    @pytest.mark.parametrize("stage,expected_num,expected_den", [
        (-6, 2, 8), (-5, 2, 7), (-4, 2, 6), (-3, 2, 5),
        (-2, 2, 4), (-1, 2, 3), (0, 2, 2),
        (1, 3, 2), (2, 4, 2), (3, 5, 2), (4, 6, 2), (5, 7, 2), (6, 8, 2),
    ])
    def test_boost_multiplier_exact(self, tables4, stage, expected_num, expected_den):
        mult = tables4.get_boost_multiplier(jnp.int32(stage))
        expected = expected_num / expected_den
        assert float(mult) == pytest.approx(expected, rel=1e-5), \
            f"Stage {stage}: got {float(mult)}, expected {expected}"


# ═══════════════════════════════════════════════════════════════════════════
# Full type chart exhaustive check
# ═══════════════════════════════════════════════════════════════════════════

class TestFullTypeChartGen4:
    """
    Exhaustive Gen 4 type chart (18x18, no Fairy).
    Each entry must match Pokemon Showdown's TypeChart.
    """

    # Gen 4 type chart: effectiveness[atk_type][def_type]
    # 0=immune, 0.5=NVE, 1=neutral, 2=SE
    EXPECTED_CHART = {
        # Normal attacking
        (TYPE_NORMAL, TYPE_ROCK): 0.5,
        (TYPE_NORMAL, TYPE_GHOST): 0.0,
        (TYPE_NORMAL, TYPE_STEEL): 0.5,
        # Fire attacking
        (TYPE_FIRE, TYPE_FIRE): 0.5,
        (TYPE_FIRE, TYPE_WATER): 0.5,
        (TYPE_FIRE, TYPE_GRASS): 2.0,
        (TYPE_FIRE, TYPE_ICE): 2.0,
        (TYPE_FIRE, TYPE_BUG): 2.0,
        (TYPE_FIRE, TYPE_ROCK): 0.5,
        (TYPE_FIRE, TYPE_DRAGON): 0.5,
        (TYPE_FIRE, TYPE_STEEL): 2.0,
        # Water attacking
        (TYPE_WATER, TYPE_FIRE): 2.0,
        (TYPE_WATER, TYPE_WATER): 0.5,
        (TYPE_WATER, TYPE_GRASS): 0.5,
        (TYPE_WATER, TYPE_GROUND): 2.0,
        (TYPE_WATER, TYPE_ROCK): 2.0,
        (TYPE_WATER, TYPE_DRAGON): 0.5,
        # Electric attacking
        (TYPE_ELECTRIC, TYPE_WATER): 2.0,
        (TYPE_ELECTRIC, TYPE_ELECTRIC): 0.5,
        (TYPE_ELECTRIC, TYPE_GRASS): 0.5,
        (TYPE_ELECTRIC, TYPE_GROUND): 0.0,
        (TYPE_ELECTRIC, TYPE_FLYING): 2.0,
        (TYPE_ELECTRIC, TYPE_DRAGON): 0.5,
        # Grass attacking
        (TYPE_GRASS, TYPE_FIRE): 0.5,
        (TYPE_GRASS, TYPE_WATER): 2.0,
        (TYPE_GRASS, TYPE_GRASS): 0.5,
        (TYPE_GRASS, TYPE_POISON): 0.5,
        (TYPE_GRASS, TYPE_GROUND): 2.0,
        (TYPE_GRASS, TYPE_FLYING): 0.5,
        (TYPE_GRASS, TYPE_BUG): 0.5,
        (TYPE_GRASS, TYPE_ROCK): 2.0,
        (TYPE_GRASS, TYPE_DRAGON): 0.5,
        (TYPE_GRASS, TYPE_STEEL): 0.5,
        # Ice attacking
        (TYPE_ICE, TYPE_FIRE): 0.5,
        (TYPE_ICE, TYPE_WATER): 0.5,
        (TYPE_ICE, TYPE_GRASS): 2.0,
        (TYPE_ICE, TYPE_ICE): 0.5,
        (TYPE_ICE, TYPE_GROUND): 2.0,
        (TYPE_ICE, TYPE_FLYING): 2.0,
        (TYPE_ICE, TYPE_DRAGON): 2.0,
        (TYPE_ICE, TYPE_STEEL): 0.5,
        # Fighting attacking
        (TYPE_FIGHTING, TYPE_NORMAL): 2.0,
        (TYPE_FIGHTING, TYPE_ICE): 2.0,
        (TYPE_FIGHTING, TYPE_POISON): 0.5,
        (TYPE_FIGHTING, TYPE_FLYING): 0.5,
        (TYPE_FIGHTING, TYPE_PSYCHIC): 0.5,
        (TYPE_FIGHTING, TYPE_BUG): 0.5,
        (TYPE_FIGHTING, TYPE_ROCK): 2.0,
        (TYPE_FIGHTING, TYPE_GHOST): 0.0,
        (TYPE_FIGHTING, TYPE_DARK): 2.0,
        (TYPE_FIGHTING, TYPE_STEEL): 2.0,
        # Poison attacking
        (TYPE_POISON, TYPE_GRASS): 2.0,
        (TYPE_POISON, TYPE_POISON): 0.5,
        (TYPE_POISON, TYPE_GROUND): 0.5,
        (TYPE_POISON, TYPE_ROCK): 0.5,
        (TYPE_POISON, TYPE_GHOST): 0.5,
        (TYPE_POISON, TYPE_STEEL): 0.0,
        # Ground attacking
        (TYPE_GROUND, TYPE_FIRE): 2.0,
        (TYPE_GROUND, TYPE_ELECTRIC): 2.0,
        (TYPE_GROUND, TYPE_GRASS): 0.5,
        (TYPE_GROUND, TYPE_POISON): 2.0,
        (TYPE_GROUND, TYPE_FLYING): 0.0,
        (TYPE_GROUND, TYPE_BUG): 0.5,
        (TYPE_GROUND, TYPE_ROCK): 2.0,
        (TYPE_GROUND, TYPE_STEEL): 2.0,
        # Flying attacking
        (TYPE_FLYING, TYPE_ELECTRIC): 0.5,
        (TYPE_FLYING, TYPE_GRASS): 2.0,
        (TYPE_FLYING, TYPE_FIGHTING): 2.0,
        (TYPE_FLYING, TYPE_BUG): 2.0,
        (TYPE_FLYING, TYPE_ROCK): 0.5,
        (TYPE_FLYING, TYPE_STEEL): 0.5,
        # Psychic attacking
        (TYPE_PSYCHIC, TYPE_FIGHTING): 2.0,
        (TYPE_PSYCHIC, TYPE_POISON): 2.0,
        (TYPE_PSYCHIC, TYPE_PSYCHIC): 0.5,
        (TYPE_PSYCHIC, TYPE_DARK): 0.0,
        (TYPE_PSYCHIC, TYPE_STEEL): 0.5,
        # Bug attacking
        (TYPE_BUG, TYPE_FIRE): 0.5,
        (TYPE_BUG, TYPE_GRASS): 2.0,
        (TYPE_BUG, TYPE_FIGHTING): 0.5,
        (TYPE_BUG, TYPE_POISON): 0.5,
        (TYPE_BUG, TYPE_FLYING): 0.5,
        (TYPE_BUG, TYPE_PSYCHIC): 2.0,
        (TYPE_BUG, TYPE_GHOST): 0.5,
        (TYPE_BUG, TYPE_DARK): 2.0,
        (TYPE_BUG, TYPE_STEEL): 0.5,
        # Rock attacking
        (TYPE_ROCK, TYPE_FIRE): 2.0,
        (TYPE_ROCK, TYPE_ICE): 2.0,
        (TYPE_ROCK, TYPE_FIGHTING): 0.5,
        (TYPE_ROCK, TYPE_GROUND): 0.5,
        (TYPE_ROCK, TYPE_FLYING): 2.0,
        (TYPE_ROCK, TYPE_BUG): 2.0,
        (TYPE_ROCK, TYPE_STEEL): 0.5,
        # Ghost attacking
        (TYPE_GHOST, TYPE_NORMAL): 0.0,
        (TYPE_GHOST, TYPE_PSYCHIC): 2.0,
        (TYPE_GHOST, TYPE_GHOST): 2.0,
        (TYPE_GHOST, TYPE_DARK): 0.5,
        (TYPE_GHOST, TYPE_STEEL): 0.5,
        # Dragon attacking
        (TYPE_DRAGON, TYPE_DRAGON): 2.0,
        (TYPE_DRAGON, TYPE_STEEL): 0.5,
        # Dark attacking
        (TYPE_DARK, TYPE_FIGHTING): 0.5,
        (TYPE_DARK, TYPE_PSYCHIC): 2.0,
        (TYPE_DARK, TYPE_GHOST): 2.0,
        (TYPE_DARK, TYPE_DARK): 0.5,
        (TYPE_DARK, TYPE_STEEL): 0.5,
        # Steel attacking
        (TYPE_STEEL, TYPE_FIRE): 0.5,
        (TYPE_STEEL, TYPE_WATER): 0.5,
        (TYPE_STEEL, TYPE_ELECTRIC): 0.5,
        (TYPE_STEEL, TYPE_ICE): 2.0,
        (TYPE_STEEL, TYPE_ROCK): 2.0,
        (TYPE_STEEL, TYPE_STEEL): 0.5,
    }

    @pytest.mark.parametrize("atk_type,def_type,expected", [
        (atk, dfn, eff) for (atk, dfn), eff in EXPECTED_CHART.items()
    ])
    def test_type_effectiveness_single(self, tables4, atk_type, def_type, expected):
        eff = tables4.get_type_effectiveness(
            jnp.int32(atk_type), jnp.int32(def_type), jnp.int32(TYPE_NONE)
        )
        assert float(eff) == pytest.approx(expected, abs=1e-4), \
            f"{atk_type} vs {def_type}: got {float(eff)}, expected {expected}"

    def test_all_neutral_when_same_type_not_in_chart(self, tables4):
        """Types not specifically listed in the chart should be neutral (1.0)."""
        for atk in range(1, 18):
            for dfn in range(1, 18):
                if (atk, dfn) not in self.EXPECTED_CHART:
                    eff = tables4.get_type_effectiveness(
                        jnp.int32(atk), jnp.int32(dfn), jnp.int32(TYPE_NONE)
                    )
                    assert float(eff) == pytest.approx(1.0, abs=1e-4), \
                        f"Expected neutral for {atk} vs {dfn}, got {float(eff)}"


# ═══════════════════════════════════════════════════════════════════════════
# Dual-type effectiveness
# ═══════════════════════════════════════════════════════════════════════════

class TestDualTypeEffectiveness:
    """Dual typing multiplies individual type matchups."""

    @pytest.mark.parametrize("atk,def1,def2,expected,desc", [
        # 4x super effective
        (TYPE_ICE, TYPE_GROUND, TYPE_FLYING, 4.0, "Ice vs Ground/Flying"),
        (TYPE_FIRE, TYPE_GRASS, TYPE_ICE, 4.0, "Fire vs Grass/Ice"),
        (TYPE_FIRE, TYPE_GRASS, TYPE_BUG, 4.0, "Fire vs Grass/Bug"),
        (TYPE_FIGHTING, TYPE_NORMAL, TYPE_ROCK, 4.0, "Fight vs Normal/Rock"),
        (TYPE_GROUND, TYPE_FIRE, TYPE_ELECTRIC, 4.0, "Ground vs Fire/Electric"),
        (TYPE_GROUND, TYPE_FIRE, TYPE_ROCK, 4.0, "Ground vs Fire/Rock"),
        # 0.25x double resist
        (TYPE_FIRE, TYPE_WATER, TYPE_ROCK, 0.25, "Fire vs Water/Rock"),
        (TYPE_GRASS, TYPE_FIRE, TYPE_FLYING, 0.25, "Grass vs Fire/Flying"),
        (TYPE_GRASS, TYPE_FIRE, TYPE_DRAGON, 0.25, "Grass vs Fire/Dragon"),
        # Immunity overrides SE
        (TYPE_ELECTRIC, TYPE_WATER, TYPE_GROUND, 0.0, "Elec vs Water/Ground"),
        (TYPE_GROUND, TYPE_ELECTRIC, TYPE_FLYING, 0.0, "Ground vs Elec/Flying"),
        (TYPE_FIGHTING, TYPE_NORMAL, TYPE_GHOST, 0.0, "Fight vs Normal/Ghost"),
        # Neutral combos
        (TYPE_FIRE, TYPE_GRASS, TYPE_WATER, 1.0, "Fire vs Grass/Water"),
        (TYPE_ICE, TYPE_FIRE, TYPE_WATER, 0.25, "Ice vs Fire/Water"),
        # Mixed: one SE, one neutral
        (TYPE_WATER, TYPE_FIRE, TYPE_NORMAL, 2.0, "Water vs Fire/Normal"),
        # Mixed: one SE, one NVE
        (TYPE_WATER, TYPE_FIRE, TYPE_GRASS, 1.0, "Water vs Fire/Grass"),
    ])
    def test_dual_type(self, tables4, atk, def1, def2, expected, desc):
        eff = tables4.get_type_effectiveness(
            jnp.int32(atk), jnp.int32(def1), jnp.int32(def2)
        )
        assert float(eff) == pytest.approx(expected, abs=1e-4), \
            f"{desc}: got {float(eff)}, expected {expected}"


# ═══════════════════════════════════════════════════════════════════════════
# Random roll range verification
# ═══════════════════════════════════════════════════════════════════════════

class TestRandomRollRange:
    """PS: Random roll is uniform 85-100 in integer steps, divided by 100."""

    def test_roll_range_85_to_100(self):
        """All possible random rolls should produce damage in [85%, 100%]."""
        for pct in range(85, 101):
            roll = pct / 100.0
            result = int(apply_random_modifier(jnp.int32(200), jnp.float32(roll)))
            expected = int(200 * roll)
            assert result == expected, f"Roll {pct}%: got {result}, expected {expected}"

    def test_floor_is_applied(self):
        """Floor division ensures damage is always an integer."""
        # 137 * 0.87 = 119.19 → floor = 119
        assert int(apply_random_modifier(jnp.int32(137), jnp.float32(0.87))) == 119
        # 253 * 0.91 = 230.23 → floor = 230
        assert int(apply_random_modifier(jnp.int32(253), jnp.float32(0.91))) == 230


# ═══════════════════════════════════════════════════════════════════════════
# Screen modifier parity
# ═══════════════════════════════════════════════════════════════════════════

class TestScreenModifierParity:
    """
    PS: Reflect halves physical damage, Light Screen halves special damage.
    Critical hits bypass screens. Both screens don't stack on same category.
    """

    def test_reflect_only_physical(self):
        """Reflect only affects physical moves."""
        # Physical
        assert int(apply_screen_modifier(
            jnp.int32(100), jnp.int8(CATEGORY_PHYSICAL),
            jnp.bool_(True), jnp.bool_(False), jnp.bool_(False)
        )) == 50
        # Special: unaffected by Reflect
        assert int(apply_screen_modifier(
            jnp.int32(100), jnp.int8(CATEGORY_SPECIAL),
            jnp.bool_(True), jnp.bool_(False), jnp.bool_(False)
        )) == 100
        # Status: unaffected
        assert int(apply_screen_modifier(
            jnp.int32(100), jnp.int8(CATEGORY_STATUS),
            jnp.bool_(True), jnp.bool_(False), jnp.bool_(False)
        )) == 100

    def test_light_screen_only_special(self):
        """Light Screen only affects special moves."""
        assert int(apply_screen_modifier(
            jnp.int32(100), jnp.int8(CATEGORY_SPECIAL),
            jnp.bool_(False), jnp.bool_(True), jnp.bool_(False)
        )) == 50
        assert int(apply_screen_modifier(
            jnp.int32(100), jnp.int8(CATEGORY_PHYSICAL),
            jnp.bool_(False), jnp.bool_(True), jnp.bool_(False)
        )) == 100

    def test_both_screens_physical_only_reflect(self):
        """If both screens up, physical is only halved once (by Reflect)."""
        assert int(apply_screen_modifier(
            jnp.int32(100), jnp.int8(CATEGORY_PHYSICAL),
            jnp.bool_(True), jnp.bool_(True), jnp.bool_(False)
        )) == 50

    def test_both_screens_special_only_light_screen(self):
        """If both screens up, special is only halved once (by LS)."""
        assert int(apply_screen_modifier(
            jnp.int32(100), jnp.int8(CATEGORY_SPECIAL),
            jnp.bool_(True), jnp.bool_(True), jnp.bool_(False)
        )) == 50
