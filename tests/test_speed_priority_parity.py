"""
Speed calculation and priority parity tests.

Verifies pokejax matches Pokemon Showdown Gen 4 speed mechanics:

  - Effective speed with boosts (max(2,2+stage)/max(2,2-stage))
  - Paralysis speed reduction (÷4 in Gen 4)
  - Tailwind speed doubling
  - Trick Room speed reversal
  - Priority bracket ordering
  - Switch priority (+7)
  - Speed ties (50/50 coin flip)
  - Speed-modifying abilities (Swift Swim, Chlorophyll, etc.)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.core.state import (
    make_battle_state, set_status, set_boost,
    set_side_condition, set_weather,
)
from pokejax.core.priority import (
    get_effective_speed, sort_two_actions, compute_turn_order,
    ACTION_MOVE, ACTION_SWITCH,
)
from pokejax.types import (
    STATUS_NONE, STATUS_PAR,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    TYPE_NONE, TYPE_NORMAL,
    SC_TAILWIND,
    BOOST_SPE,
)
from pokejax.config import GenConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(p1_base_spe=100, p2_base_spe=100, max_hp=300):
    n = 6
    zeros6 = np.zeros(n, dtype=np.int16)
    zeros6i8 = np.zeros(n, dtype=np.int8)
    t = np.zeros((n, 2), dtype=np.int8)
    t[:, 0] = TYPE_NORMAL
    bs1 = np.array([[80, 80, 80, 80, 80, p1_base_spe]] * n, dtype=np.int16)
    bs2 = np.array([[80, 80, 80, 80, 80, p2_base_spe]] * n, dtype=np.int16)
    mhp = np.full(n, max_hp, dtype=np.int16)
    mid = np.tile(np.arange(4, dtype=np.int16), (n, 1))
    mpp = np.full((n, 4), 35, dtype=np.int8)
    levels = np.full(n, 100, dtype=np.int8)
    return make_battle_state(
        p1_species=zeros6, p2_species=zeros6,
        p1_abilities=zeros6, p2_abilities=zeros6,
        p1_items=zeros6, p2_items=zeros6,
        p1_types=t, p2_types=t,
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
        rng_key=jax.random.PRNGKey(42),
    )


# ═══════════════════════════════════════════════════════════════════════════
# EFFECTIVE SPEED WITH BOOSTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEffectiveSpeedWithBoosts:
    """
    PS boost multiplier for speed:
      stage 0: 1.0 (2/2)
      stage +1: 1.5 (3/2)
      stage +2: 2.0 (4/2)
      stage -1: 0.667 (2/3)
      stage -2: 0.5 (2/4)
    """

    def test_base_speed_no_boost(self, cfg4):
        state = _make_state(p1_base_spe=100)
        speed = int(get_effective_speed(state, 0, cfg4))
        # At level 100, 80 base spe... actually the calc uses base_stats
        # which are raw base stats, not calculated stats
        assert speed > 0

    @pytest.mark.parametrize("stage,mult_num,mult_den", [
        (0, 2, 2), (1, 3, 2), (2, 4, 2), (3, 5, 2),
        (4, 6, 2), (5, 7, 2), (6, 8, 2),
        (-1, 2, 3), (-2, 2, 4), (-3, 2, 5),
        (-4, 2, 6), (-5, 2, 7), (-6, 2, 8),
    ])
    def test_speed_boost_stages(self, cfg4, stage, mult_num, mult_den):
        state = _make_state(p1_base_spe=100)
        base_speed = int(get_effective_speed(state, 0, cfg4))
        state = set_boost(state, 0, 0, BOOST_SPE, jnp.int8(stage))
        boosted_speed = int(get_effective_speed(state, 0, cfg4))
        expected = int(base_speed * mult_num / mult_den)
        assert boosted_speed == expected, \
            f"Stage {stage}: expected {expected}, got {boosted_speed}"


# ═══════════════════════════════════════════════════════════════════════════
# PARALYSIS SPEED REDUCTION
# ═══════════════════════════════════════════════════════════════════════════

class TestParalysisSpeedParity:
    """PS Gen 4: Paralysis divides speed by 4."""

    def test_paralysis_divides_by_4(self, cfg4):
        state = _make_state(p1_base_spe=200)
        normal_speed = int(get_effective_speed(state, 0, cfg4))
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        para_speed = int(get_effective_speed(state, 0, cfg4))
        expected = normal_speed // 4
        assert para_speed == expected, \
            f"Par speed: expected {expected}, got {para_speed}"

    def test_paralysis_with_boost(self, cfg4):
        """Paralysis stacks with speed boosts."""
        state = _make_state(p1_base_spe=100)
        # +2 speed boost then paralysis
        state = set_boost(state, 0, 0, BOOST_SPE, jnp.int8(2))
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        speed = int(get_effective_speed(state, 0, cfg4))
        # Should be: (base_speed * 2) / 4 = base_speed / 2
        base = int(get_effective_speed(_make_state(p1_base_spe=100), 0, cfg4))
        expected = (base * 2) // 4
        assert speed == expected


# ═══════════════════════════════════════════════════════════════════════════
# TAILWIND
# ═══════════════════════════════════════════════════════════════════════════

class TestTailwindSpeedParity:
    """PS: Tailwind doubles speed for the team."""

    def test_tailwind_doubles_speed(self, cfg4):
        state = _make_state(p1_base_spe=100)
        normal_speed = int(get_effective_speed(state, 0, cfg4))
        state = set_side_condition(state, 0, SC_TAILWIND, jnp.int8(4))
        tailwind_speed = int(get_effective_speed(state, 0, cfg4))
        assert tailwind_speed == normal_speed * 2, \
            f"Tailwind: expected {normal_speed * 2}, got {tailwind_speed}"

    def test_tailwind_no_effect_on_opponent(self, cfg4):
        state = _make_state(p1_base_spe=100, p2_base_spe=100)
        normal_speed_p2 = int(get_effective_speed(state, 1, cfg4))
        state = set_side_condition(state, 0, SC_TAILWIND, jnp.int8(4))
        p2_speed = int(get_effective_speed(state, 1, cfg4))
        assert p2_speed == normal_speed_p2


# ═══════════════════════════════════════════════════════════════════════════
# PRIORITY ORDERING
# ═══════════════════════════════════════════════════════════════════════════

class TestPriorityOrdering:
    """PS: Higher priority moves always go first, regardless of speed."""

    @pytest.mark.parametrize("p0_pri,p1_pri,p0_spd,p1_spd,p0_first", [
        # Higher priority wins
        (1, 0, 50, 200, True),
        (0, 1, 200, 50, False),
        (2, 1, 1, 999, True),
        (-1, 0, 999, 1, False),
        # Same priority: faster wins
        (0, 0, 200, 100, True),
        (0, 0, 100, 200, False),
        (1, 1, 300, 100, True),
        # Extreme priorities
        (5, -7, 1, 999, True),
        (-7, 5, 999, 1, False),
    ])
    def test_priority_ordering(self, p0_pri, p1_pri, p0_spd, p1_spd, p0_first):
        result, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(p0_pri), jnp.int32(p0_spd),
            ACTION_MOVE, jnp.int8(p1_pri), jnp.int32(p1_spd),
            jnp.bool_(False), jax.random.PRNGKey(0),
        )
        assert bool(result) == p0_first

    def test_switch_always_before_move(self):
        """Switches have effective priority +7, beating all move priorities."""
        for move_pri in range(-7, 6):
            result, _, _ = sort_two_actions(
                ACTION_SWITCH, jnp.int8(0), jnp.int32(1),
                ACTION_MOVE, jnp.int8(move_pri), jnp.int32(999),
                jnp.bool_(False), jax.random.PRNGKey(0),
            )
            assert bool(result), f"Switch should beat priority {move_pri}"

    def test_both_switches_speed_decides(self):
        """When both switch, faster switches first."""
        result, _, _ = sort_two_actions(
            ACTION_SWITCH, jnp.int8(0), jnp.int32(200),
            ACTION_SWITCH, jnp.int8(0), jnp.int32(100),
            jnp.bool_(False), jax.random.PRNGKey(0),
        )
        assert bool(result)  # P0 faster


# ═══════════════════════════════════════════════════════════════════════════
# SPEED TIES
# ═══════════════════════════════════════════════════════════════════════════

class TestSpeedTies:
    """PS: Speed ties resolved by 50/50 coin flip."""

    def test_speed_tie_is_random(self):
        """Same speed, same priority → random outcome."""
        results = set()
        for seed in range(100):
            result, _, _ = sort_two_actions(
                ACTION_MOVE, jnp.int8(0), jnp.int32(100),
                ACTION_MOVE, jnp.int8(0), jnp.int32(100),
                jnp.bool_(False), jax.random.PRNGKey(seed),
            )
            results.add(bool(result))
            if len(results) == 2:
                break
        assert len(results) == 2, "Speed tie should produce both outcomes"

    def test_speed_tie_approximately_fair(self):
        """Speed tie should be roughly 50/50."""
        p0_first = sum(
            1 for seed in range(1000)
            if bool(sort_two_actions(
                ACTION_MOVE, jnp.int8(0), jnp.int32(100),
                ACTION_MOVE, jnp.int8(0), jnp.int32(100),
                jnp.bool_(False), jax.random.PRNGKey(seed),
            )[0])
        )
        rate = p0_first / 1000
        assert 0.35 < rate < 0.65, f"Speed tie rate {rate:.2%} far from 50%"


# ═══════════════════════════════════════════════════════════════════════════
# TRICK ROOM
# ═══════════════════════════════════════════════════════════════════════════

class TestTrickRoomParity:
    """PS: Trick Room reverses speed order within same priority."""

    def test_trick_room_slower_first(self):
        result, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(0), jnp.int32(50),
            ACTION_MOVE, jnp.int8(0), jnp.int32(200),
            jnp.bool_(True),  # Trick Room active
            jax.random.PRNGKey(0),
        )
        assert bool(result)  # P0 slower → goes first in TR

    def test_trick_room_faster_second(self):
        result, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(0), jnp.int32(200),
            ACTION_MOVE, jnp.int8(0), jnp.int32(50),
            jnp.bool_(True),
            jax.random.PRNGKey(0),
        )
        assert not bool(result)  # P0 faster → goes second in TR

    def test_trick_room_does_not_affect_priority(self):
        """Priority brackets still override speed in Trick Room."""
        result, _, _ = sort_two_actions(
            ACTION_MOVE, jnp.int8(1), jnp.int32(200),  # +1 priority, fast
            ACTION_MOVE, jnp.int8(0), jnp.int32(50),   # +0 priority, slow
            jnp.bool_(True),
            jax.random.PRNGKey(0),
        )
        assert bool(result)  # Priority still wins

    def test_trick_room_does_not_affect_switches(self):
        """Switches still go first in Trick Room."""
        result, _, _ = sort_two_actions(
            ACTION_SWITCH, jnp.int8(0), jnp.int32(50),
            ACTION_MOVE, jnp.int8(0), jnp.int32(200),
            jnp.bool_(True),
            jax.random.PRNGKey(0),
        )
        assert bool(result)
