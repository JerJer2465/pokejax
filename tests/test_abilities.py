"""
L2 interaction tests for Phase 5 ability and item implementations.

Tests:
  - populate_ability_tables installs handlers into events.py dispatch tables
  - ModifyAtk event: Huge Power ×2, Hustle ×1.5, Guts ×1.5 when statused
  - BasePower event: Technician ×1.5 if BP ≤ 60, Iron Fist ×1.2 for punch moves
  - TryHit event: Levitate immune to Ground, Water Absorb immune to Water + heals
  - SwitchIn state: Intimidate lowers foe ATK by 1, Drizzle sets rain
  - SwitchOut state: Natural Cure clears status
  - Residual state: Speed Boost +1 SPE, Leftovers heals 1/16 HP
  - Item: Choice Band ×1.5 ATK, Life Orb ×1.3 damage
  - Integration: compute full damage with Huge Power doubles output vs baseline
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE,
    STATUS_BRN, STATUS_PAR, STATUS_NONE,
    WEATHER_RAIN, WEATHER_SUN, WEATHER_NONE,
    TYPE_WATER, TYPE_ELECTRIC, TYPE_GROUND, TYPE_GRASS, TYPE_NORMAL,
)
from pokejax.data.tables import load_tables
from pokejax.config import GenConfig
from pokejax.core.state import make_battle_state
from pokejax.mechanics import events
from pokejax.mechanics.abilities import populate_ability_tables
from pokejax.mechanics.items import populate_item_tables
from pokejax.mechanics.events import (
    run_event_modify_atk, run_event_modify_spa,
    run_event_base_power, run_event_modify_damage,
    run_event_try_hit, run_event_try_hit_state,
    run_event_switch_in, run_event_switch_out,
    run_event_residual_state,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tables():
    return load_tables(gen=4)


@pytest.fixture(scope="module")
def cfg():
    return GenConfig.for_gen(4)


# Stable ability IDs used throughout tests (matches mock name_to_id below)
_ABILITY_IDS = {
    "Huge Power":    1,
    "Pure Power":    2,
    "Hustle":        3,
    "Guts":          4,
    "Technician":    5,
    "Iron Fist":     6,
    "Reckless":      7,
    "Blaze":         8,
    "Torrent":       9,
    "Overgrow":      10,
    "Swarm":         11,
    "Solar Power":   12,
    "Marvel Scale":  13,
    "Water Absorb":  14,
    "Volt Absorb":   15,
    "Flash Fire":    16,
    "Levitate":      17,
    "Motor Drive":   18,
    "Sap Sipper":    19,
    "Storm Drain":   20,
    "Lightning Rod": 21,
    "Intimidate":    22,
    "Drizzle":       23,
    "Drought":       24,
    "Sand Stream":   25,
    "Snow Warning":  26,
    "Download":      27,
    "Natural Cure":  28,
    "Speed Boost":   29,
    "Poison Heal":   30,
    "Adaptability":  31,
}

_ITEM_IDS = {
    "Choice Band":  1,
    "Choice Specs": 2,
    "Life Orb":     3,
    "Expert Belt":  4,
    "Leftovers":    5,
    "Black Sludge": 6,
    "Sitrus Berry": 7,
    "Lum Berry":    8,
}


@pytest.fixture(scope="module", autouse=True)
def populate_tables(tables):
    """Install ability/item handlers using test IDs before all tests."""
    populate_ability_tables(_ABILITY_IDS, tables)
    populate_item_tables(_ITEM_IDS, tables)


def _make_state(tables, cfg, *,
                p1_ability=0, p2_ability=0,
                p1_item=0, p2_item=0,
                p1_status=0, p2_status=0,
                p1_hp_frac=1.0, p2_hp_frac=1.0,
                p1_types=None, p2_types=None):
    """Build a minimal BattleState for ability tests."""
    n = 6
    base = np.array([[100, 100, 100, 100, 100, 100]] * n, dtype=np.int16)
    max_hp = 300

    abilities_p1 = np.zeros(n, dtype=np.int16)
    abilities_p1[0] = p1_ability
    abilities_p2 = np.zeros(n, dtype=np.int16)
    abilities_p2[0] = p2_ability

    items_p1 = np.zeros(n, dtype=np.int16)
    items_p1[0] = p1_item
    items_p2 = np.zeros(n, dtype=np.int16)
    items_p2[0] = p2_item

    hp_arr = np.full(n, max_hp, dtype=np.int16)
    hp_p1 = hp_arr.copy()
    hp_p1[0] = int(max_hp * p1_hp_frac)
    hp_p2 = hp_arr.copy()
    hp_p2[0] = int(max_hp * p2_hp_frac)

    status_arr = np.zeros(n, dtype=np.int8)
    status_p1 = status_arr.copy(); status_p1[0] = p1_status
    status_p2 = status_arr.copy(); status_p2[0] = p2_status

    if p1_types is None:
        types_p1 = np.zeros((n, 2), dtype=np.int8)
        types_p1[:, 0] = TYPE_NORMAL
    else:
        types_p1 = np.array([p1_types] * n, dtype=np.int8)

    if p2_types is None:
        types_p2 = np.zeros((n, 2), dtype=np.int8)
        types_p2[:, 0] = TYPE_NORMAL
    else:
        types_p2 = np.array([p2_types] * n, dtype=np.int8)

    move_ids = np.tile(np.arange(4, dtype=np.int16), (n, 1))
    move_pp = np.full((n, 4), 35, dtype=np.int8)
    levels = np.full(n, 100, dtype=np.int8)
    genders = np.zeros(n, dtype=np.int8)
    natures = np.zeros(n, dtype=np.int8)
    weights = np.full(n, 100, dtype=np.int16)

    state = make_battle_state(
        p1_species=np.zeros(n, dtype=np.int16),
        p2_species=np.zeros(n, dtype=np.int16),
        p1_abilities=abilities_p1, p2_abilities=abilities_p2,
        p1_items=items_p1, p2_items=items_p2,
        p1_types=types_p1, p2_types=types_p2,
        p1_base_stats=base, p2_base_stats=base,
        p1_max_hp=hp_arr, p2_max_hp=hp_arr,
        p1_move_ids=move_ids, p2_move_ids=move_ids,
        p1_move_pp=move_pp, p2_move_pp=move_pp,
        p1_move_max_pp=move_pp, p2_move_max_pp=move_pp,
        p1_levels=levels, p2_levels=levels,
        p1_genders=genders, p2_genders=genders,
        p1_natures=natures, p2_natures=natures,
        p1_weights_hg=weights, p2_weights_hg=weights,
        rng_key=jax.random.PRNGKey(0),
    )
    # Set HP separately for HP fraction tests
    hp_p1_full = np.full(n, max_hp, dtype=np.int16)
    hp_p1_full[0] = int(max_hp * p1_hp_frac)
    hp_p2_full = np.full(n, max_hp, dtype=np.int16)
    hp_p2_full[0] = int(max_hp * p2_hp_frac)
    new_hp = state.sides_team_hp.at[0].set(jnp.array(hp_p1_full)).at[1].set(jnp.array(hp_p2_full))
    # Set status
    new_status = state.sides_team_status.at[0, 0].set(jnp.int8(p1_status))
    return state._replace(sides_team_hp=new_hp, sides_team_status=new_status)


def _dummy_move_id(move_id: int = 0) -> jnp.ndarray:
    return jnp.int16(move_id)


# ---------------------------------------------------------------------------
# ModifyAtk relay tests
# ---------------------------------------------------------------------------

class TestModifyAtk:
    def test_huge_power_doubles_atk_relay(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Huge Power"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_atk(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(2.0)

    def test_pure_power_doubles_atk_relay(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Pure Power"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_atk(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(2.0)

    def test_hustle_one_point_five_atk(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Hustle"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_atk(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(1.5)

    def test_guts_no_boost_without_status(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Guts"], p1_status=0)
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_atk(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(1.0)

    def test_guts_boost_with_status(self, tables, cfg):
        state = _make_state(tables, cfg,
                             p1_ability=_ABILITY_IDS["Guts"],
                             p1_status=int(STATUS_BRN))
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_atk(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(1.5)

    def test_no_ability_no_boost(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=0)
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_atk(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BasePower relay tests
# ---------------------------------------------------------------------------

class TestBasePower:
    def test_technician_boosts_low_bp(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Technician"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        # relay = 40.0 (BP ≤ 60) → should become 60.0
        relay = run_event_base_power(
            jnp.float32(40.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(60.0)

    def test_technician_no_boost_high_bp(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Technician"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        # relay = 80.0 (BP > 60) → unchanged
        relay = run_event_base_power(
            jnp.float32(80.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(80.0)

    def test_technician_boundary_60bp(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Technician"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        # relay = 60.0 (BP == 60, should boost)
        relay = run_event_base_power(
            jnp.float32(60.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(90.0)

    def test_no_technician_no_change(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=0)
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_base_power(
            jnp.float32(40.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# TryHit (immunity) tests
# ---------------------------------------------------------------------------

class TestTryHit:
    # ------------------------------------------------------------------ helpers
    def _patch_tables(self, move_type_val: int):
        """
        Patch events._TABLES_REF so that move 0 has the given type.
        Returns the original _TABLES_REF for restoration.
        """
        from pokejax.mechanics import events as ev_mod
        from pokejax.core.damage import MF_TYPE
        from dataclasses import replace as dc_replace
        if ev_mod._TABLES_REF is None:
            return None
        move_data_np = np.array(ev_mod._TABLES_REF.moves)
        move_data_np[0, MF_TYPE] = move_type_val
        patched = dc_replace(ev_mod._TABLES_REF, moves=jnp.array(move_data_np))
        original = ev_mod._TABLES_REF
        ev_mod._TABLES_REF = patched
        return original

    def _restore_tables(self, original):
        from pokejax.mechanics import events as ev_mod
        ev_mod._TABLES_REF = original

    # ------------------------------------------------------------------ tests
    def test_levitate_immune_to_ground(self, tables, cfg):
        from pokejax.mechanics import events as ev_mod
        if ev_mod._TABLES_REF is None:
            pytest.skip("events._TABLES_REF not set; run after populate")

        state = _make_state(tables, cfg, p2_ability=_ABILITY_IDS["Levitate"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]

        orig = self._patch_tables(TYPE_GROUND)
        try:
            _, cancelled = run_event_try_hit(
                jnp.bool_(True), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
            )
        finally:
            self._restore_tables(orig)
        assert bool(cancelled) == True   # ground move should be blocked

    def test_levitate_allows_non_ground(self, tables, cfg):
        from pokejax.mechanics import events as ev_mod
        if ev_mod._TABLES_REF is None:
            pytest.skip("events._TABLES_REF not set")

        state = _make_state(tables, cfg, p2_ability=_ABILITY_IDS["Levitate"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]

        orig = self._patch_tables(TYPE_WATER)
        try:
            _, cancelled = run_event_try_hit(
                jnp.bool_(True), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
            )
        finally:
            self._restore_tables(orig)
        assert bool(cancelled) == False  # water move should NOT be blocked

    def test_water_absorb_cancels_water(self, tables, cfg):
        from pokejax.mechanics import events as ev_mod
        if ev_mod._TABLES_REF is None:
            pytest.skip("events._TABLES_REF not set")

        state = _make_state(tables, cfg, p2_ability=_ABILITY_IDS["Water Absorb"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]

        orig = self._patch_tables(TYPE_WATER)
        try:
            _, cancelled = run_event_try_hit(
                jnp.bool_(True), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
            )
        finally:
            self._restore_tables(orig)
        assert bool(cancelled) == True   # water absorbed → cancel


# ---------------------------------------------------------------------------
# TryHit state tests (absorption heals / boosts)
# ---------------------------------------------------------------------------

class TestTryHitState:
    def test_water_absorb_heals_on_water(self, tables, cfg):
        # Defender at 75% HP; Water move → heal 25% → back to full (or capped)
        max_hp = 300
        start_hp = 225  # 75%
        state = _make_state(tables, cfg, p2_ability=_ABILITY_IDS["Water Absorb"])
        # Set defender HP to 225
        new_hp = state.sides_team_hp.at[1, 0].set(jnp.int16(start_hp))
        state = state._replace(sides_team_hp=new_hp)

        state2 = run_event_try_hit_state(
            state, 0, jnp.int32(0), 1, jnp.int32(0),
            jnp.int16(0), jnp.int32(TYPE_WATER)
        )
        healed_hp = int(state2.sides_team_hp[1, 0])
        assert healed_hp == min(max_hp, start_hp + max_hp // 4)

    def test_water_absorb_no_heal_on_fire(self, tables, cfg):
        start_hp = 225
        state = _make_state(tables, cfg, p2_ability=_ABILITY_IDS["Water Absorb"])
        new_hp = state.sides_team_hp.at[1, 0].set(jnp.int16(start_hp))
        state = state._replace(sides_team_hp=new_hp)

        state2 = run_event_try_hit_state(
            state, 0, jnp.int32(0), 1, jnp.int32(0),
            jnp.int16(0), jnp.int32(2)  # TYPE_FIRE = 2
        )
        assert int(state2.sides_team_hp[1, 0]) == start_hp  # unchanged

    def test_motor_drive_boosts_spe_on_electric(self, tables, cfg):
        state = _make_state(tables, cfg, p2_ability=_ABILITY_IDS["Motor Drive"])
        state2 = run_event_try_hit_state(
            state, 0, jnp.int32(0), 1, jnp.int32(0),
            jnp.int16(0), jnp.int32(TYPE_ELECTRIC)
        )
        spe_boost = int(state2.sides_team_boosts[1, 0, BOOST_SPE])
        assert spe_boost == 1

    def test_motor_drive_no_boost_non_electric(self, tables, cfg):
        state = _make_state(tables, cfg, p2_ability=_ABILITY_IDS["Motor Drive"])
        state2 = run_event_try_hit_state(
            state, 0, jnp.int32(0), 1, jnp.int32(0),
            jnp.int16(0), jnp.int32(TYPE_WATER)
        )
        spe_boost = int(state2.sides_team_boosts[1, 0, BOOST_SPE])
        assert spe_boost == 0


# ---------------------------------------------------------------------------
# SwitchIn state tests
# ---------------------------------------------------------------------------

class TestSwitchInState:
    def test_intimidate_lowers_foe_atk(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Intimidate"])
        # Side 0 switches in → Intimidate fires → side 1's ATK drops by 1
        state2 = run_event_switch_in(state, 0, state.sides_active_idx[0])
        foe_atk_boost = int(state2.sides_team_boosts[1, 0, BOOST_ATK])
        assert foe_atk_boost == -1

    def test_intimidate_caps_at_minus_six(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Intimidate"])
        # Pre-set foe ATK to -6
        boosts = state.sides_team_boosts.at[1, 0, BOOST_ATK].set(jnp.int8(-6))
        state = state._replace(sides_team_boosts=boosts)
        state2 = run_event_switch_in(state, 0, state.sides_active_idx[0])
        foe_atk_boost = int(state2.sides_team_boosts[1, 0, BOOST_ATK])
        assert foe_atk_boost == -6  # still capped

    def test_drizzle_sets_rain(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Drizzle"])
        state2 = run_event_switch_in(state, 0, state.sides_active_idx[0])
        assert int(state2.field.weather) == WEATHER_RAIN
        assert int(state2.field.weather_turns) == 5

    def test_drought_sets_sun(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Drought"])
        state2 = run_event_switch_in(state, 0, state.sides_active_idx[0])
        assert int(state2.field.weather) == WEATHER_SUN

    def test_no_ability_no_switch_effect(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=0)
        weather_before = int(state.field.weather)
        state2 = run_event_switch_in(state, 0, state.sides_active_idx[0])
        assert int(state2.field.weather) == weather_before

    def test_download_boosts_atk_when_foe_spd_lower(self, tables, cfg):
        # Foe with SPD < DEF → Download boosts ATK
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Download"])
        # Default base stats: all 100. Manually lower foe SpD to 50 < 100.
        foe_stats = np.array([[100, 100, 100, 100, 50, 100]] * 6, dtype=np.int16)
        new_base = state.sides_team_base_stats.at[1].set(jnp.array(foe_stats))
        state = state._replace(sides_team_base_stats=new_base)
        state2 = run_event_switch_in(state, 0, state.sides_active_idx[0])
        atk_boost = int(state2.sides_team_boosts[0, 0, BOOST_ATK])
        spa_boost = int(state2.sides_team_boosts[0, 0, BOOST_SPA])
        assert atk_boost == 1 and spa_boost == 0

    def test_download_boosts_spa_when_foe_def_lower(self, tables, cfg):
        # Foe with DEF < SPD → Download boosts SpA
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Download"])
        foe_stats = np.array([[100, 100, 50, 100, 100, 100]] * 6, dtype=np.int16)
        new_base = state.sides_team_base_stats.at[1].set(jnp.array(foe_stats))
        state = state._replace(sides_team_base_stats=new_base)
        state2 = run_event_switch_in(state, 0, state.sides_active_idx[0])
        atk_boost = int(state2.sides_team_boosts[0, 0, BOOST_ATK])
        spa_boost = int(state2.sides_team_boosts[0, 0, BOOST_SPA])
        assert spa_boost == 1 and atk_boost == 0


# ---------------------------------------------------------------------------
# SwitchOut state tests
# ---------------------------------------------------------------------------

class TestSwitchOutState:
    def test_natural_cure_clears_status(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Natural Cure"],
                             p1_status=int(STATUS_PAR))
        assert int(state.sides_team_status[0, 0]) == STATUS_PAR
        idx = state.sides_active_idx[0]
        state2 = run_event_switch_out(state, 0, idx)
        assert int(state2.sides_team_status[0, 0]) == STATUS_NONE

    def test_natural_cure_no_status_no_change(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Natural Cure"],
                             p1_status=0)
        idx = state.sides_active_idx[0]
        state2 = run_event_switch_out(state, 0, idx)
        assert int(state2.sides_team_status[0, 0]) == STATUS_NONE

    def test_no_ability_no_clear(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=0, p1_status=int(STATUS_BRN))
        idx = state.sides_active_idx[0]
        state2 = run_event_switch_out(state, 0, idx)
        assert int(state2.sides_team_status[0, 0]) == STATUS_BRN  # unchanged


# ---------------------------------------------------------------------------
# Residual state tests
# ---------------------------------------------------------------------------

class TestResidualState:
    def test_speed_boost_adds_one_spe(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Speed Boost"])
        idx = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state2, _ = run_event_residual_state(state, key, 0, idx)
        spe_boost = int(state2.sides_team_boosts[0, 0, BOOST_SPE])
        assert spe_boost == 1

    def test_speed_boost_caps_at_plus_six(self, tables, cfg):
        state = _make_state(tables, cfg, p1_ability=_ABILITY_IDS["Speed Boost"])
        boosts = state.sides_team_boosts.at[0, 0, BOOST_SPE].set(jnp.int8(6))
        state = state._replace(sides_team_boosts=boosts)
        idx = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state2, _ = run_event_residual_state(state, key, 0, idx)
        spe_boost = int(state2.sides_team_boosts[0, 0, BOOST_SPE])
        assert spe_boost == 6  # capped

    def test_leftovers_heals_1_16(self, tables, cfg):
        max_hp = 300
        start_hp = 150
        state = _make_state(tables, cfg, p1_item=_ITEM_IDS["Leftovers"])
        new_hp = state.sides_team_hp.at[0, 0].set(jnp.int16(start_hp))
        state = state._replace(sides_team_hp=new_hp)
        idx = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state2, _ = run_event_residual_state(state, key, 0, idx)
        expected = min(max_hp, start_hp + max(1, max_hp // 16))
        assert int(state2.sides_team_hp[0, 0]) == expected

    def test_leftovers_caps_at_max_hp(self, tables, cfg):
        state = _make_state(tables, cfg, p1_item=_ITEM_IDS["Leftovers"])
        # HP already full
        idx = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state2, _ = run_event_residual_state(state, key, 0, idx)
        assert int(state2.sides_team_hp[0, 0]) == 300  # capped at max

    def test_black_sludge_heals_poison_type(self, tables, cfg):
        max_hp = 300
        start_hp = 200
        state = _make_state(tables, cfg, p1_item=_ITEM_IDS["Black Sludge"],
                             p1_types=[TYPE_GRASS, 8])  # Normal + Poison
        new_hp = state.sides_team_hp.at[0, 0].set(jnp.int16(start_hp))
        state = state._replace(sides_team_hp=new_hp)
        idx = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state2, _ = run_event_residual_state(state, key, 0, idx)
        assert int(state2.sides_team_hp[0, 0]) > start_hp  # healed

    def test_no_item_no_residual(self, tables, cfg):
        state = _make_state(tables, cfg, p1_item=0)
        hp_before = int(state.sides_team_hp[0, 0])
        idx = state.sides_active_idx[0]
        key = jax.random.PRNGKey(0)
        state2, _ = run_event_residual_state(state, key, 0, idx)
        assert int(state2.sides_team_hp[0, 0]) == hp_before


# ---------------------------------------------------------------------------
# ModifyDamage (items)
# ---------------------------------------------------------------------------

class TestModifyDamage:
    def test_life_orb_multiplies_damage(self, tables, cfg):
        state = _make_state(tables, cfg, p1_item=_ITEM_IDS["Life Orb"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_damage(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(1.3, abs=1e-5)

    def test_choice_band_multiplies_atk(self, tables, cfg):
        state = _make_state(tables, cfg, p1_item=_ITEM_IDS["Choice Band"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_atk(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(1.5)

    def test_choice_specs_multiplies_spa(self, tables, cfg):
        state = _make_state(tables, cfg, p1_item=_ITEM_IDS["Choice Specs"])
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_spa(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(1.5)

    def test_no_item_no_modifier(self, tables, cfg):
        state = _make_state(tables, cfg, p1_item=0)
        atk_idx = state.sides_active_idx[0]
        def_idx = state.sides_active_idx[1]
        relay = run_event_modify_damage(
            jnp.float32(1.0), state, 0, atk_idx, 1, def_idx, _dummy_move_id()
        )
        assert float(relay) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: compute_damage respects ability relays
# ---------------------------------------------------------------------------

class TestDamageIntegration:
    def test_huge_power_roughly_doubles_damage(self, tables, cfg):
        """With Huge Power, physical damage should roughly double vs no ability."""
        from pokejax.core.damage import compute_damage
        import jax

        # Build two states: one with Huge Power, one without
        state_hp = _make_state(tables, cfg,
                                p1_ability=_ABILITY_IDS["Huge Power"])
        state_no = _make_state(tables, cfg, p1_ability=0)

        # Use Tackle (move 0 in placeholder tables): BP=40, Physical, Normal
        move_id = jnp.int16(0)
        key = jax.random.PRNGKey(42)

        # With Huge Power: run_event_modify_atk → 2x → passed to compute_damage
        atk_idx = state_hp.sides_active_idx[0]
        def_idx = state_hp.sides_active_idx[1]
        atk_relay_hp = run_event_modify_atk(
            jnp.float32(1.0), state_hp, 0, atk_idx, 1, def_idx, move_id
        )
        _, dmg_hp, _, _ = compute_damage(
            state_hp, tables, 0, 1, move_id, key, atk_relay=atk_relay_hp
        )

        # Without ability
        _, dmg_no, _, _ = compute_damage(
            state_no, tables, 0, 1, move_id, key, atk_relay=jnp.float32(1.0)
        )

        # Huge Power should roughly double damage (within 10% due to rounding)
        ratio = float(dmg_hp) / max(1, float(dmg_no))
        assert 1.8 <= ratio <= 2.2, f"Expected ~2x damage ratio, got {ratio:.2f}"
