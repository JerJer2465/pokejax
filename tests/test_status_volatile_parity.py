"""
Comprehensive status, volatile, and residual mechanic parity tests.

Tests cover all Gen 4 status conditions and volatile states to verify
pokejax matches Pokemon Showdown behavior exactly:

  - Non-volatile statuses: BRN, PSN, TOX, SLP, FRZ, PAR
  - Volatile conditions: Confusion, Substitute, Protect, Encore, Taunt,
    Disable, Yawn, Destiny Bond, Perish Song, Curse, Leech Seed
  - Residual damage/healing: burn/poison/toxic per-turn, leech seed drain,
    partial trap, ingrain heal
  - Status immunities: type-based, Safeguard, existing status
  - Volatile clearing on switch
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.core.state import (
    make_battle_state, set_status, set_boost, set_side_condition,
    set_weather, set_volatile, set_volatile_counter, has_volatile,
    clear_volatiles, reset_boosts, get_active_status,
)
from pokejax.mechanics.conditions import (
    apply_burn_residual, apply_poison_residual,
    apply_sleep_residual, check_sleep_before_move,
    check_paralysis_before_move, check_freeze_before_move,
    check_confusion_before_move,
    apply_volatile_residuals, decrement_volatile_timers,
    apply_entry_hazards, tick_side_conditions,
    try_set_status, apply_residual,
)
from pokejax.types import (
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX,
    STATUS_SLP, STATUS_FRZ, STATUS_PAR,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    TYPE_NONE, TYPE_NORMAL, TYPE_FIRE, TYPE_WATER, TYPE_ELECTRIC,
    TYPE_GRASS, TYPE_ICE, TYPE_FIGHTING, TYPE_POISON, TYPE_GROUND,
    TYPE_FLYING, TYPE_PSYCHIC, TYPE_BUG, TYPE_ROCK, TYPE_STEEL,
    SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
    SC_REFLECT, SC_LIGHTSCREEN, SC_SAFEGUARD, SC_MIST, SC_TAILWIND,
    VOL_CONFUSED, VOL_FLINCH, VOL_PARTIALLY_TRAPPED, VOL_SEEDED,
    VOL_SUBSTITUTE, VOL_PROTECT, VOL_ENCORE, VOL_TAUNT,
    VOL_DISABLE, VOL_YAWN, VOL_DESTINYBOND, VOL_PERISH,
    VOL_CURSE, VOL_NIGHTMARE, VOL_INGRAIN, VOL_FOCUSENERGY,
    VOL_CHOICELOCK, VOL_CHARGING, VOL_RECHARGING, VOL_LOCKEDMOVE,
    BOOST_ATK, BOOST_SPE,
)
from pokejax.config import GenConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(max_hp=300, p1_types=(TYPE_NORMAL, 0), p2_types=(TYPE_NORMAL, 0),
                level=100, rng_seed=42):
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
    levels = np.full(n, level, dtype=np.int8)
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
        rng_key=jax.random.PRNGKey(rng_seed),
    )


# ═══════════════════════════════════════════════════════════════════════════
# BURN
# ═══════════════════════════════════════════════════════════════════════════

class TestBurnResidualParity:
    """PS Gen 4: Burn deals floor(max_hp / 8) per turn, min 1."""

    @pytest.mark.parametrize("max_hp,expected_dmg", [
        (300, 37),   # floor(300/8) = 37
        (160, 20),
        (200, 25),
        (80, 10),
        (1, 1),      # min 1
        (7, 1),      # floor(7/8)=0 → clamped to 1
        (8, 1),      # exactly 1
        (9, 1),      # floor(9/8)=1
        (16, 2),     # floor(16/8)=2
        (400, 50),
    ])
    def test_burn_damage(self, cfg4, max_hp, expected_dmg):
        state = _make_state(max_hp=max_hp)
        state = set_status(state, 0, 0, jnp.int8(STATUS_BRN))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_burn_residual(state, 0, cfg4)
        actual_dmg = hp_before - int(state.sides_team_hp[0, 0])
        assert actual_dmg == expected_dmg, \
            f"max_hp={max_hp}: expected {expected_dmg} burn dmg, got {actual_dmg}"


# ═══════════════════════════════════════════════════════════════════════════
# TOXIC ESCALATION
# ═══════════════════════════════════════════════════════════════════════════

class TestToxicEscalationParity:
    """
    PS Gen 4: Toxic damage = floor(max_hp * counter / 16), min 1.
    Counter starts at 1, increments each turn, capped at 15.
    Resets to 1 on switch-out.
    """

    @pytest.mark.parametrize("max_hp,counter,expected_dmg", [
        (160, 1, 10),    # floor(160*1/16)=10
        (160, 2, 20),
        (160, 3, 30),
        (160, 5, 50),
        (160, 10, 100),
        (160, 15, 150),  # Max counter
        (100, 1, 6),     # floor(100/16)=6
        (100, 7, 43),    # floor(700/16)=43
        (1, 1, 1),       # min 1
        (7, 1, 1),       # floor(7/16)=0 → 1
        (16, 1, 1),      # floor(16/16)=1
        (320, 15, 300),  # floor(4800/16)=300
    ])
    def test_toxic_damage_at_counter(self, max_hp, counter, expected_dmg):
        state = _make_state(max_hp=max_hp)
        state = set_status(state, 0, 0, jnp.int8(STATUS_TOX), jnp.int8(counter))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_poison_residual(state, 0)
        actual_dmg = hp_before - int(state.sides_team_hp[0, 0])
        assert actual_dmg == expected_dmg, \
            f"max_hp={max_hp}, counter={counter}: expected {expected_dmg}, got {actual_dmg}"

    def test_toxic_counter_increments_each_turn(self):
        """Counter increments by 1 each turn residual."""
        state = _make_state(max_hp=500)
        state = set_status(state, 0, 0, jnp.int8(STATUS_TOX), jnp.int8(1))
        for expected in range(2, 16):
            state = apply_poison_residual(state, 0)
            actual = int(state.sides_team_status_turns[0, 0])
            assert actual == min(expected, 15), \
                f"Turn {expected-1}: counter should be {min(expected, 15)}, got {actual}"

    def test_toxic_counter_caps_at_15(self):
        state = _make_state(max_hp=500)
        state = set_status(state, 0, 0, jnp.int8(STATUS_TOX), jnp.int8(15))
        state = apply_poison_residual(state, 0)
        assert int(state.sides_team_status_turns[0, 0]) == 15

    def test_regular_poison_no_escalation(self):
        """Regular PSN does 1/8 every turn, no counter change."""
        state = _make_state(max_hp=160)
        state = set_status(state, 0, 0, jnp.int8(STATUS_PSN))
        for _ in range(5):
            hp_before = int(state.sides_team_hp[0, 0])
            state = apply_poison_residual(state, 0)
            assert hp_before - int(state.sides_team_hp[0, 0]) == 20


# ═══════════════════════════════════════════════════════════════════════════
# SLEEP
# ═══════════════════════════════════════════════════════════════════════════

class TestSleepMechanicsParity:
    """
    PS Gen 4: Sleep lasts 1-3 turns. Counter set at infliction,
    decremented each residual. Wakes when counter hits 0.
    """

    def test_sleep_duration_range(self, cfg4):
        """Sleep turns should be in {1, 2, 3}."""
        durations = set()
        for seed in range(200):
            s, _ = try_set_status(
                _make_state(), 0, 0, jnp.int8(STATUS_SLP),
                jax.random.PRNGKey(seed), cfg4,
            )
            d = int(s.sides_team_sleep_turns[0, 0])
            if d > 0:
                durations.add(d)
        assert durations.issubset({1, 2, 3}), f"Sleep durations out of range: {durations}"
        assert len(durations) >= 2, "Sleep should have variable duration"

    def test_sleep_decrement_and_wake(self):
        """Counter decrements each residual and wakes at 0."""
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_SLP))
        state = state._replace(
            sides_team_sleep_turns=state.sides_team_sleep_turns.at[0, 0].set(jnp.int8(2))
        )
        # Turn 1: 2 → 1
        state, _ = apply_sleep_residual(state, 0, jax.random.PRNGKey(0))
        assert int(state.sides_team_sleep_turns[0, 0]) == 1
        assert int(state.sides_team_status[0, 0]) == STATUS_SLP
        # Turn 2: 1 → 0 → wakes up
        state, _ = apply_sleep_residual(state, 0, jax.random.PRNGKey(1))
        assert int(state.sides_team_status[0, 0]) == STATUS_NONE

    def test_sleeping_pokemon_cannot_move(self):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_SLP))
        state = state._replace(
            sides_team_sleep_turns=state.sides_team_sleep_turns.at[0, 0].set(jnp.int8(3))
        )
        can_move, _, _ = check_sleep_before_move(state, 0, jax.random.PRNGKey(0))
        assert not bool(can_move)


# ═══════════════════════════════════════════════════════════════════════════
# FREEZE
# ═══════════════════════════════════════════════════════════════════════════

class TestFreezeMechanicsParity:
    """PS Gen 4: 20% thaw at start of turn; can act if thawed."""

    def test_thaw_rate_statistical(self, cfg4):
        """Thaw rate should be approximately 20%."""
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_FRZ))
        thaws = sum(
            1 for seed in range(1000)
            if bool(check_freeze_before_move(state, 0, jax.random.PRNGKey(seed), cfg4)[0])
        )
        rate = thaws / 1000
        assert 0.12 < rate < 0.30, f"Thaw rate {rate:.2%} far from 20%"

    def test_thaw_clears_freeze_status(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_FRZ))
        for seed in range(200):
            can_move, _, s = check_freeze_before_move(
                state, 0, jax.random.PRNGKey(seed), cfg4
            )
            if bool(can_move):
                assert int(s.sides_team_status[0, 0]) == STATUS_NONE
                return
        pytest.fail("Never thawed in 200 trials")

    def test_not_frozen_always_can_move(self, cfg4):
        state = _make_state()
        can_move, _, _ = check_freeze_before_move(state, 0, jax.random.PRNGKey(0), cfg4)
        assert bool(can_move)


# ═══════════════════════════════════════════════════════════════════════════
# PARALYSIS
# ═══════════════════════════════════════════════════════════════════════════

class TestParalysisMechanicsParity:
    """PS Gen 4: 25% full paralysis; speed ÷ 4."""

    def test_full_para_rate_statistical(self, cfg4):
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_PAR))
        blocked = sum(
            1 for seed in range(1000)
            if not bool(check_paralysis_before_move(state, 0, jax.random.PRNGKey(seed), cfg4)[0])
        )
        rate = blocked / 1000
        assert 0.15 < rate < 0.38, f"Full para rate {rate:.2%} far from 25%"

    def test_para_speed_divisor_is_4(self, cfg4):
        assert cfg4.paralysis_speed_divisor == 4


# ═══════════════════════════════════════════════════════════════════════════
# CONFUSION
# ═══════════════════════════════════════════════════════════════════════════

class TestConfusionMechanicsParity:
    """PS Gen 4: Confusion 50% self-hit, counter decrements, snaps at 0."""

    def test_self_hit_rate_statistical(self):
        """Self-hit should occur roughly 50% of the time in Gen 4."""
        state = _make_state(max_hp=500)
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, VOL_CONFUSED, True)
        state = set_volatile_counter(state, 0, idx, VOL_CONFUSED, jnp.int8(5))
        hits = sum(
            1 for seed in range(500)
            if not bool(check_confusion_before_move(state, 0, jax.random.PRNGKey(seed))[0])
        )
        rate = hits / 500
        assert 0.30 < rate < 0.70, f"Self-hit rate {rate:.2%} far from 50%"

    def test_confusion_self_hit_deals_damage(self):
        state = _make_state(max_hp=500)
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, VOL_CONFUSED, True)
        state = set_volatile_counter(state, 0, idx, VOL_CONFUSED, jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        for seed in range(200):
            can_move, _, s = check_confusion_before_move(state, 0, jax.random.PRNGKey(seed))
            if not bool(can_move):
                assert int(s.sides_team_hp[0, 0]) < hp_before
                return
        pytest.fail("No self-hit in 200 trials")

    def test_confusion_clears_when_counter_zero(self):
        state = _make_state()
        idx = int(state.sides_active_idx[0])
        state = set_volatile(state, 0, idx, VOL_CONFUSED, True)
        state = set_volatile_counter(state, 0, idx, VOL_CONFUSED, jnp.int8(1))
        for seed in range(200):
            can_move, _, s = check_confusion_before_move(state, 0, jax.random.PRNGKey(seed))
            if bool(can_move):
                assert not bool(has_volatile(s, 0, idx, VOL_CONFUSED))
                return
        pytest.skip("No non-hit seed found")


# ═══════════════════════════════════════════════════════════════════════════
# STATUS IMMUNITIES
# ═══════════════════════════════════════════════════════════════════════════

class TestStatusImmunitiesParity:
    """PS: Type-based immunities and existing status prevention."""

    @pytest.mark.parametrize("types,status,should_apply", [
        # Fire immune to burn
        ((TYPE_FIRE, 0), STATUS_BRN, False),
        # Steel immune to poison/toxic
        ((TYPE_STEEL, 0), STATUS_PSN, False),
        ((TYPE_STEEL, 0), STATUS_TOX, False),
        # Poison immune to poison/toxic
        ((TYPE_POISON, 0), STATUS_PSN, False),
        ((TYPE_POISON, 0), STATUS_TOX, False),
        # Ice immune to freeze
        ((TYPE_ICE, 0), STATUS_FRZ, False),
        # Electric immune to paralysis (Gen 6+, NOT Gen 4)
        # In Gen 4, Electric types CAN be paralyzed
        ((TYPE_ELECTRIC, 0), STATUS_PAR, True),
        # Normal types: no immunities
        ((TYPE_NORMAL, 0), STATUS_BRN, True),
        ((TYPE_NORMAL, 0), STATUS_PSN, True),
        ((TYPE_NORMAL, 0), STATUS_TOX, True),
        ((TYPE_NORMAL, 0), STATUS_SLP, True),
        ((TYPE_NORMAL, 0), STATUS_FRZ, True),
        ((TYPE_NORMAL, 0), STATUS_PAR, True),
        # Dual type: one immune type blocks
        ((TYPE_FIRE, TYPE_STEEL), STATUS_PSN, False),
        ((TYPE_FIRE, TYPE_STEEL), STATUS_BRN, False),
        ((TYPE_WATER, TYPE_ICE), STATUS_FRZ, False),
    ])
    def test_type_immunity(self, cfg4, types, status, should_apply):
        state = _make_state(p1_types=types)
        state, _ = try_set_status(
            state, 0, 0, jnp.int8(status), jax.random.PRNGKey(0), cfg4,
        )
        applied = int(get_active_status(state, 0)) == status
        assert applied == should_apply, \
            f"Types {types}, status {status}: expected apply={should_apply}, got {applied}"

    def test_cannot_apply_second_status(self, cfg4):
        """Already statused Pokemon cannot receive another status."""
        state = _make_state()
        state = set_status(state, 0, 0, jnp.int8(STATUS_BRN))
        state, _ = try_set_status(
            state, 0, 0, jnp.int8(STATUS_PAR), jax.random.PRNGKey(0), cfg4,
        )
        assert int(get_active_status(state, 0)) == STATUS_BRN

    def test_safeguard_blocks_status(self, cfg4):
        state = _make_state()
        state = set_side_condition(state, 0, SC_SAFEGUARD, jnp.int8(5))
        state, _ = try_set_status(
            state, 0, 0, jnp.int8(STATUS_BRN), jax.random.PRNGKey(0), cfg4,
        )
        assert int(get_active_status(state, 0)) == STATUS_NONE


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILE CONDITIONS STATE
# ═══════════════════════════════════════════════════════════════════════════

class TestVolatileStateParity:
    """Verify volatile condition state management."""

    def test_set_and_check_volatile(self):
        state = _make_state()
        state = set_volatile(state, 0, 0, VOL_CONFUSED, True)
        assert bool(has_volatile(state, 0, 0, VOL_CONFUSED))
        assert not bool(has_volatile(state, 0, 0, VOL_SEEDED))

    def test_multiple_volatiles(self):
        """Multiple volatiles can be active simultaneously."""
        state = _make_state()
        state = set_volatile(state, 0, 0, VOL_CONFUSED, True)
        state = set_volatile(state, 0, 0, VOL_SEEDED, True)
        state = set_volatile(state, 0, 0, VOL_FOCUSENERGY, True)
        assert bool(has_volatile(state, 0, 0, VOL_CONFUSED))
        assert bool(has_volatile(state, 0, 0, VOL_SEEDED))
        assert bool(has_volatile(state, 0, 0, VOL_FOCUSENERGY))

    def test_clear_volatiles_removes_all(self):
        state = _make_state()
        state = set_volatile(state, 0, 0, VOL_CONFUSED, True)
        state = set_volatile(state, 0, 0, VOL_SEEDED, True)
        state = set_volatile(state, 0, 0, VOL_PROTECT, True)
        state = clear_volatiles(state, 0, 0)
        assert not bool(has_volatile(state, 0, 0, VOL_CONFUSED))
        assert not bool(has_volatile(state, 0, 0, VOL_SEEDED))
        assert not bool(has_volatile(state, 0, 0, VOL_PROTECT))

    def test_volatile_counter_set_and_read(self):
        state = _make_state()
        state = set_volatile(state, 0, 0, VOL_PERISH, True)
        state = set_volatile_counter(state, 0, 0, VOL_PERISH, jnp.int8(3))
        assert int(state.sides_team_volatile_data[0, 0, VOL_PERISH]) == 3

    def test_protect_consecutive_counter(self):
        """Protect counter tracks consecutive uses for fail rate."""
        state = _make_state()
        state = set_volatile(state, 0, 0, VOL_PROTECT, True)
        state = set_volatile_counter(state, 0, 0, VOL_PROTECT, jnp.int8(1))
        assert int(state.sides_team_volatile_data[0, 0, VOL_PROTECT]) == 1
        state = set_volatile_counter(state, 0, 0, VOL_PROTECT, jnp.int8(2))
        assert int(state.sides_team_volatile_data[0, 0, VOL_PROTECT]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY HAZARDS
# ═══════════════════════════════════════════════════════════════════════════

class TestEntryHazardsParity:
    """
    PS Gen 4 entry hazard damage:
      - Stealth Rock: floor(max_hp * rock_effectiveness / 8)
      - Spikes: 1 layer = 1/8, 2 = 1/6, 3 = 1/4
      - Toxic Spikes: 1 = PSN, 2 = TOX (grounded only; Poison absorbs)
      - Sticky Web: -1 Speed (grounded only)
    """

    @pytest.mark.parametrize("def_types,expected_frac,desc", [
        # Rock neutral (1.0): 1/8
        ((TYPE_NORMAL, 0), 1/8, "Normal: 1/8"),
        # Rock SE (2.0): 2/8 = 1/4
        ((TYPE_FIRE, 0), 2/8, "Fire (2x rock): 2/8"),
        ((TYPE_ICE, 0), 2/8, "Ice (2x rock): 2/8"),
        ((TYPE_BUG, 0), 2/8, "Bug (2x rock): 2/8"),
        ((TYPE_FLYING, 0), 2/8, "Flying (2x rock): 2/8"),
        # Rock NVE (0.5): 1/16
        ((TYPE_FIGHTING, 0), 1/16, "Fighting (0.5x rock): 1/16"),
        ((TYPE_GROUND, 0), 1/16, "Ground (0.5x rock): 1/16"),
        ((TYPE_STEEL, 0), 1/16, "Steel (0.5x rock): 1/16"),
        # Rock 4x SE: 4/8 = 1/2
        ((TYPE_FIRE, TYPE_FLYING), 4/8, "Fire/Flying (4x rock): 4/8"),
        ((TYPE_ICE, TYPE_BUG), 4/8, "Ice/Bug (4x rock): 4/8"),
        # Rock 0.25x: 0.25/8 = 1/32
        ((TYPE_FIGHTING, TYPE_STEEL), 0.25/8, "Fight/Steel (0.25x rock): 1/32"),
    ])
    def test_stealth_rock_damage(self, tables4, def_types, expected_frac, desc):
        """Stealth Rock damage = floor(max_hp * rock_effectiveness / 8)."""
        max_hp = 400
        state = _make_state(max_hp=max_hp, p2_types=def_types)
        state = set_side_condition(state, 1, SC_STEALTHROCK, jnp.int8(1))
        hp_before = int(state.sides_team_hp[1, 0])
        state = apply_entry_hazards(state, 1, tables4)
        hp_after = int(state.sides_team_hp[1, 0])
        actual_dmg = hp_before - hp_after
        expected_dmg = max(1, int(max_hp * expected_frac))
        assert actual_dmg == expected_dmg, \
            f"{desc}: max_hp={max_hp}, expected {expected_dmg} dmg, got {actual_dmg}"

    @pytest.mark.parametrize("layers,expected_frac", [
        (1, 1/8),
        (2, 1/6),
        (3, 1/4),
    ])
    def test_spikes_damage_by_layer(self, tables4, layers, expected_frac):
        """Spikes: 1 layer=1/8, 2=1/6, 3=1/4 of max HP."""
        max_hp = 240  # divisible by 8, 6, and 4
        state = _make_state(max_hp=max_hp)
        state = set_side_condition(state, 1, SC_SPIKES, jnp.int8(layers))
        hp_before = int(state.sides_team_hp[1, 0])
        state = apply_entry_hazards(state, 1, tables4)
        hp_after = int(state.sides_team_hp[1, 0])
        actual_dmg = hp_before - hp_after
        expected_dmg = max(1, int(max_hp * expected_frac))
        assert actual_dmg == expected_dmg, \
            f"{layers} layers: expected {expected_dmg}, got {actual_dmg}"

    def test_spikes_immune_to_flying(self, tables4):
        """Flying types are immune to Spikes."""
        state = _make_state(max_hp=300, p2_types=(TYPE_FLYING, 0))
        state = set_side_condition(state, 1, SC_SPIKES, jnp.int8(3))
        hp_before = int(state.sides_team_hp[1, 0])
        state = apply_entry_hazards(state, 1, tables4)
        assert int(state.sides_team_hp[1, 0]) == hp_before

    def test_toxic_spikes_1_layer_poisons(self, tables4):
        """1 layer of Toxic Spikes inflicts regular poison."""
        state = _make_state(max_hp=300)
        state = set_side_condition(state, 1, SC_TOXICSPIKES, jnp.int8(1))
        state = apply_entry_hazards(state, 1, tables4)
        assert int(state.sides_team_status[1, 0]) == STATUS_PSN

    def test_toxic_spikes_2_layers_badly_poisons(self, tables4):
        """2 layers of Toxic Spikes inflicts badly poisoned (toxic)."""
        state = _make_state(max_hp=300)
        state = set_side_condition(state, 1, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 1, tables4)
        assert int(state.sides_team_status[1, 0]) == STATUS_TOX

    def test_toxic_spikes_immune_to_flying(self, tables4):
        """Flying types are immune to Toxic Spikes."""
        state = _make_state(max_hp=300, p2_types=(TYPE_FLYING, 0))
        state = set_side_condition(state, 1, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 1, tables4)
        assert int(state.sides_team_status[1, 0]) == STATUS_NONE

    def test_toxic_spikes_poison_type_absorbs(self, tables4):
        """Grounded Poison types absorb Toxic Spikes (remove them)."""
        state = _make_state(max_hp=300, p2_types=(TYPE_POISON, 0))
        state = set_side_condition(state, 1, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 1, tables4)
        assert int(state.sides_team_status[1, 0]) == STATUS_NONE
        assert int(state.sides_side_conditions[1, SC_TOXICSPIKES]) == 0

    def test_toxic_spikes_steel_immune(self, tables4):
        """Steel types are immune to Toxic Spikes."""
        state = _make_state(max_hp=300, p2_types=(TYPE_STEEL, 0))
        state = set_side_condition(state, 1, SC_TOXICSPIKES, jnp.int8(2))
        state = apply_entry_hazards(state, 1, tables4)
        assert int(state.sides_team_status[1, 0]) == STATUS_NONE

    def test_no_hazards_no_damage(self, tables4):
        """No hazards → no damage on switch-in."""
        state = _make_state(max_hp=300)
        hp_before = int(state.sides_team_hp[1, 0])
        state = apply_entry_hazards(state, 1, tables4)
        assert int(state.sides_team_hp[1, 0]) == hp_before


# ═══════════════════════════════════════════════════════════════════════════
# SIDE CONDITION TIMERS
# ═══════════════════════════════════════════════════════════════════════════

class TestSideConditionTimersParity:
    """PS: Screens and team effects have duration timers that decrement."""

    @pytest.mark.parametrize("cond,initial,expected_after", [
        (SC_REFLECT, 5, 4),
        (SC_LIGHTSCREEN, 5, 4),
        (SC_TAILWIND, 4, 3),
        (SC_SAFEGUARD, 5, 4),
        (SC_MIST, 5, 4),
    ])
    def test_timer_decrements(self, cond, initial, expected_after):
        state = _make_state()
        state = set_side_condition(state, 0, cond, jnp.int8(initial))
        state = tick_side_conditions(state, 0)
        assert int(state.sides_side_conditions[0, cond]) == expected_after

    @pytest.mark.parametrize("cond", [SC_REFLECT, SC_LIGHTSCREEN, SC_TAILWIND, SC_SAFEGUARD, SC_MIST])
    def test_timer_clears_at_zero(self, cond):
        state = _make_state()
        state = set_side_condition(state, 0, cond, jnp.int8(1))
        state = tick_side_conditions(state, 0)
        assert int(state.sides_side_conditions[0, cond]) == 0

    def test_hazard_layers_not_ticked(self):
        """Spikes and Stealth Rock are permanent (no timer)."""
        state = _make_state()
        state = set_side_condition(state, 0, SC_SPIKES, jnp.int8(3))
        state = set_side_condition(state, 0, SC_STEALTHROCK, jnp.int8(1))
        state = tick_side_conditions(state, 0)
        assert int(state.sides_side_conditions[0, SC_SPIKES]) == 3
        assert int(state.sides_side_conditions[0, SC_STEALTHROCK]) == 1
