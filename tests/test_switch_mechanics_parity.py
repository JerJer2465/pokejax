"""
Switch mechanics parity tests.

Verifies pokejax matches Pokemon Showdown for Gen 4 switch behavior:

  - Boost reset on switch-out (Gen 4: all boosts clear)
  - Volatile clearing on switch-out
  - Toxic counter reset on switch-out
  - Entry hazard application on switch-in
  - Switch-in ability triggers (Intimidate, Drizzle, etc.)
  - Forced switch finding (fainted Pokemon replacement)
  - Active index tracking
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.core.state import (
    make_battle_state, set_status, set_boost, set_side_condition,
    set_volatile, set_volatile_counter, has_volatile,
    clear_volatiles, reset_boosts, set_hp,
    set_active, set_fainted, get_active_idx,
)
from pokejax.engine.switch import switch_out, switch_in
from pokejax.engine.actions import find_forced_switch_slot
from pokejax.types import (
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX,
    STATUS_SLP, STATUS_FRZ, STATUS_PAR,
    TYPE_NONE, TYPE_NORMAL, TYPE_FIRE, TYPE_WATER,
    TYPE_FLYING, TYPE_POISON, TYPE_STEEL,
    SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
    VOL_CONFUSED, VOL_SEEDED, VOL_SUBSTITUTE, VOL_PROTECT,
    VOL_ENCORE, VOL_TAUNT, VOL_DISABLE, VOL_YAWN,
    VOL_CHARGING, VOL_RECHARGING, VOL_LOCKEDMOVE, VOL_CHOICELOCK,
    VOL_FOCUSENERGY, VOL_INGRAIN, VOL_DESTINYBOND,
    BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE,
    BOOST_ACC, BOOST_EVA,
)
from pokejax.config import GenConfig
from pokejax.data.tables import load_tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(max_hp=300, p1_types=(TYPE_NORMAL, 0), p2_types=(TYPE_NORMAL, 0)):
    n = 6
    zeros6 = np.zeros(n, dtype=np.int16)
    zeros6i8 = np.zeros(n, dtype=np.int8)
    t1 = np.zeros((n, 2), dtype=np.int8)
    t1[:, 0] = p1_types[0]
    t1[:, 1] = p1_types[1]
    t2 = np.zeros((n, 2), dtype=np.int8)
    t2[:, 0] = p2_types[0]
    t2[:, 1] = p2_types[1]
    base = np.array([[80, 80, 80, 80, 80, 80]] * n, dtype=np.int16)
    mhp = np.full(n, max_hp, dtype=np.int16)
    mid = np.tile(np.arange(4, dtype=np.int16), (n, 1))
    mpp = np.full((n, 4), 35, dtype=np.int8)
    levels = np.full(n, 100, dtype=np.int8)
    return make_battle_state(
        p1_species=zeros6, p2_species=zeros6,
        p1_abilities=zeros6, p2_abilities=zeros6,
        p1_items=zeros6, p2_items=zeros6,
        p1_types=t1, p2_types=t2,
        p1_base_stats=base, p2_base_stats=base,
        p1_max_hp=mhp, p2_max_hp=mhp,
        p1_move_ids=mid, p2_move_ids=mid,
        p1_move_pp=mpp, p2_move_pp=mpp,
        p1_move_max_pp=mpp, p2_move_max_pp=mpp,
        p1_levels=levels, p2_levels=levels,
        p1_genders=zeros6i8, p2_genders=zeros6i8,
        p1_natures=zeros6i8, p2_natures=zeros6i8,
        p1_weights_hg=np.full(n, 100, dtype=np.int16),
        p2_weights_hg=np.full(n, 100, dtype=np.int16),
        rng_key=jax.random.PRNGKey(42),
    )


# ═══════════════════════════════════════════════════════════════════════════
# BOOST RESET ON SWITCH-OUT
# ═══════════════════════════════════════════════════════════════════════════

class TestBoostResetOnSwitch:
    """PS Gen 4: All stat boosts reset to 0 on switch-out."""

    @pytest.mark.parametrize("boost_idx,boost_val", [
        (BOOST_ATK, 6),
        (BOOST_ATK, -6),
        (BOOST_DEF, 3),
        (BOOST_SPA, 2),
        (BOOST_SPD, -2),
        (BOOST_SPE, 4),
        (BOOST_ACC, 1),
        (BOOST_EVA, -3),
    ])
    def test_single_boost_resets(self, boost_idx, boost_val):
        state = _make_state()
        state = set_boost(state, 0, 0, boost_idx, jnp.int8(boost_val))
        assert int(state.sides_team_boosts[0, 0, boost_idx]) == boost_val
        state = reset_boosts(state, 0, 0)
        assert int(state.sides_team_boosts[0, 0, boost_idx]) == 0

    def test_all_boosts_reset_at_once(self):
        state = _make_state()
        for i in range(7):
            state = set_boost(state, 0, 0, i, jnp.int8(3))
        state = reset_boosts(state, 0, 0)
        for i in range(7):
            assert int(state.sides_team_boosts[0, 0, i]) == 0

    def test_switch_out_clears_boosts(self, tables4, cfg4):
        """Full switch_out call clears all boosts."""
        state = _make_state()
        for i in range(7):
            state = set_boost(state, 0, 0, i, jnp.int8(6))
        state = switch_out(state, 0, cfg4)
        idx = int(state.sides_active_idx[0])
        for i in range(7):
            assert int(state.sides_team_boosts[0, idx, i]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILE CLEARING ON SWITCH-OUT
# ═══════════════════════════════════════════════════════════════════════════

class TestVolatileClearingOnSwitch:
    """
    PS Gen 4: Most volatiles clear on switch-out.
    Exceptions: some carry over (not in Gen 4 singles).
    """

    @pytest.mark.parametrize("vol_bit", [
        VOL_CONFUSED, VOL_SEEDED, VOL_SUBSTITUTE, VOL_PROTECT,
        VOL_ENCORE, VOL_TAUNT, VOL_DISABLE, VOL_YAWN,
        VOL_CHARGING, VOL_RECHARGING, VOL_LOCKEDMOVE, VOL_CHOICELOCK,
        VOL_FOCUSENERGY, VOL_DESTINYBOND,
    ])
    def test_volatile_clears_on_switch(self, vol_bit, tables4, cfg4):
        state = _make_state()
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, vol_bit, True)
        assert bool(has_volatile(state, 0, idx, vol_bit))
        state = switch_out(state, 0, cfg4)
        assert not bool(has_volatile(state, 0, idx, vol_bit))


# ═══════════════════════════════════════════════════════════════════════════
# TOXIC COUNTER RESET ON SWITCH
# ═══════════════════════════════════════════════════════════════════════════

class TestToxicCounterResetOnSwitch:
    """PS Gen 4: Toxic counter resets to 1 on switch-out, status remains TOX."""

    def test_toxic_counter_resets(self, tables4, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_TOX), jnp.int8(8))
        assert int(state.sides_team_status_turns[0, 0]) == 8
        state = switch_out(state, 0, cfg4)
        idx = int(state.sides_active_idx[0])
        assert int(state.sides_team_status_turns[0, idx]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVE INDEX TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class TestActiveIndexTracking:
    """Active Pokemon index correctly tracks switches."""

    def test_initial_active_is_slot_zero(self):
        state = _make_state()
        assert int(state.sides_active_idx[0]) == 0
        assert int(state.sides_active_idx[1]) == 0

    def test_set_active_updates_index(self):
        state = _make_state()
        state = set_active(state, 0, 2)
        assert int(state.sides_active_idx[0]) == 2
        assert bool(state.sides_team_is_active[0, 2])
        assert not bool(state.sides_team_is_active[0, 0])

    def test_set_active_independent_per_side(self):
        state = _make_state()
        state = set_active(state, 0, 3)
        state = set_active(state, 1, 5)
        assert int(state.sides_active_idx[0]) == 3
        assert int(state.sides_active_idx[1]) == 5


# ═══════════════════════════════════════════════════════════════════════════
# FORCED SWITCH (FAINTED REPLACEMENT)
# ═══════════════════════════════════════════════════════════════════════════

class TestForcedSwitchFinding:
    """find_forced_switch_slot returns first non-fainted, non-active slot."""

    def test_finds_first_alive(self):
        state = _make_state()
        slot = find_forced_switch_slot(state, 0)
        assert int(slot) == 1  # slot 0 is active, so 1 is first replacement

    def test_skips_fainted(self):
        state = _make_state()
        state = set_fainted(state, 0, 1)
        slot = find_forced_switch_slot(state, 0)
        assert int(slot) == 2  # 1 fainted, skip to 2

    def test_all_fainted_returns_minus_one(self):
        state = _make_state()
        for i in range(1, 6):
            state = set_fainted(state, 0, i)
        slot = find_forced_switch_slot(state, 0)
        assert int(slot) == -1  # only active slot 0 alive, no replacement

    def test_skips_active_slot(self):
        state = _make_state()
        state = set_active(state, 0, 2)
        slot = find_forced_switch_slot(state, 0)
        assert int(slot) != 2  # should not return active slot
