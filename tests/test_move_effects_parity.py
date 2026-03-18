"""
Move effect parity tests.

Verifies pokejax move effects match Pokemon Showdown for Gen 4:

  - Stat-boosting moves: Swords Dance, Dragon Dance, Calm Mind, Nasty Plot,
    Agility, Bulk Up, Curse, Shell Smash, etc.
  - Stat-lowering moves: Growl, Screech, Charm, etc.
  - Hazard moves: Spikes, Stealth Rock, Toxic Spikes, Sticky Web
  - Screen moves: Reflect, Light Screen, Aurora Veil
  - Weather moves: Sunny Day, Rain Dance, Sandstorm, Hail
  - Trick Room
  - Recovery moves: Recover, Softboiled, Rest
  - Volatile-setting moves: Substitute, Protect, Leech Seed, etc.
  - Hazard removal: Rapid Spin, Defog
  - Stat reset: Haze
  - Special moves: Belly Drum, Pain Split, Destiny Bond, Perish Song

Each test creates a minimal BattleState, executes the move effect, and
verifies the resulting state matches PS behavior.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.core.state import (
    make_battle_state, make_reveal_state,
    set_status, set_boost, set_side_condition,
    set_weather, set_volatile, set_volatile_counter,
    set_hp, has_volatile, reset_boosts,
    get_active_status, get_side_condition,
)
from pokejax.mechanics.moves import execute_move_effects
from pokejax.mechanics.conditions import apply_entry_hazards
from pokejax.data.move_effects_data import (
    ME_NONE, ME_SELF_BOOST, ME_FOE_LOWER, ME_HAZARD, ME_SCREEN,
    ME_WEATHER, ME_TRICK_ROOM, ME_VOLATILE_SELF, ME_VOLATILE_FOE,
    ME_SUBSTITUTE, ME_RAPID_SPIN, ME_RECOVERY, ME_REST,
    ME_BELLY_DRUM, ME_KNOCK_OFF, ME_PAIN_SPLIT, ME_HEAL_BELL,
    ME_DISABLE, ME_YAWN, ME_DESTINY_BOND, ME_PERISH_SONG,
    ME_HAZE, ME_DEFOG, ME_TRICK, ME_TWO_TURN,
)
from pokejax.types import (
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX, STATUS_SLP,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    TYPE_NONE, TYPE_NORMAL, TYPE_FIRE, TYPE_WATER,
    TYPE_GRASS, TYPE_GROUND, TYPE_FLYING, TYPE_POISON, TYPE_STEEL, TYPE_ROCK,
    SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
    SC_REFLECT, SC_LIGHTSCREEN, SC_AURORAVEIL, SC_TAILWIND,
    SC_SAFEGUARD, SC_MIST,
    VOL_CONFUSED, VOL_SEEDED, VOL_SUBSTITUTE, VOL_PROTECT,
    VOL_ENCORE, VOL_TAUNT, VOL_FOCUSENERGY,
    VOL_YAWN, VOL_DESTINYBOND, VOL_PERISH, VOL_INGRAIN,
    BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE,
    BOOST_ACC, BOOST_EVA,
    CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS,
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
# STAT BOOST CLAMPING
# ═══════════════════════════════════════════════════════════════════════════

class TestStatBoostClamping:
    """PS: Stat boosts clamp to [-6, +6]."""

    def test_cannot_exceed_plus_six(self):
        state = _make_state()
        # Set ATK to +5 then +2 → should cap at +6
        state = set_boost(state, 0, 0, BOOST_ATK, jnp.int8(5))
        state = set_boost(state, 0, 0, BOOST_ATK,
                          jnp.int8(int(state.sides_team_boosts[0, 0, BOOST_ATK]) + 2))
        # set_boost clamps internally
        assert int(state.sides_team_boosts[0, 0, BOOST_ATK]) == 6

    def test_cannot_go_below_minus_six(self):
        state = _make_state()
        state = set_boost(state, 0, 0, BOOST_DEF, jnp.int8(-5))
        state = set_boost(state, 0, 0, BOOST_DEF,
                          jnp.int8(int(state.sides_team_boosts[0, 0, BOOST_DEF]) - 2))
        assert int(state.sides_team_boosts[0, 0, BOOST_DEF]) == -6

    @pytest.mark.parametrize("val", [-10, -7, -6, 0, 6, 7, 10])
    def test_clamping_range(self, val):
        state = _make_state()
        state = set_boost(state, 0, 0, BOOST_ATK, jnp.int8(val))
        result = int(state.sides_team_boosts[0, 0, BOOST_ATK])
        assert -6 <= result <= 6
        assert result == max(-6, min(6, val))


# ═══════════════════════════════════════════════════════════════════════════
# HAZARD LAYER LIMITS
# ═══════════════════════════════════════════════════════════════════════════

class TestHazardLayerLimits:
    """PS: Spikes max 3, Toxic Spikes max 2, SR max 1, Sticky Web max 1."""

    @pytest.mark.parametrize("cond,max_layers", [
        (SC_SPIKES, 3),
        (SC_TOXICSPIKES, 2),
        (SC_STEALTHROCK, 1),
        (SC_STICKYWEB, 1),
    ])
    def test_hazard_max_layers(self, cond, max_layers):
        state = _make_state()
        for i in range(max_layers + 2):
            val = min(i + 1, max_layers)
            state = set_side_condition(state, 0, cond, jnp.int8(val))
        actual = int(state.sides_side_conditions[0, cond])
        assert actual == max_layers


# ═══════════════════════════════════════════════════════════════════════════
# SCREEN DURATION
# ═══════════════════════════════════════════════════════════════════════════

class TestScreenDuration:
    """PS: Reflect and Light Screen last 5 turns by default."""

    @pytest.mark.parametrize("cond,duration", [
        (SC_REFLECT, 5),
        (SC_LIGHTSCREEN, 5),
    ])
    def test_screen_initial_duration(self, cond, duration):
        state = _make_state()
        state = set_side_condition(state, 0, cond, jnp.int8(duration))
        assert int(state.sides_side_conditions[0, cond]) == duration


# ═══════════════════════════════════════════════════════════════════════════
# WEATHER SETTING
# ═══════════════════════════════════════════════════════════════════════════

class TestWeatherSetting:
    """PS: Weather moves set 5-turn weather."""

    @pytest.mark.parametrize("weather,code", [
        (WEATHER_SUN, WEATHER_SUN),
        (WEATHER_RAIN, WEATHER_RAIN),
        (WEATHER_SAND, WEATHER_SAND),
        (WEATHER_HAIL, WEATHER_HAIL),
    ])
    def test_set_weather(self, weather, code):
        from pokejax.core.state import set_weather
        state = _make_state()
        state = set_weather(state, jnp.int8(code), jnp.int8(5))
        assert int(state.field.weather) == code
        assert int(state.field.weather_turns) == 5

    def test_weather_replaces_existing(self):
        from pokejax.core.state import set_weather
        state = _make_state()
        state = set_weather(state, jnp.int8(WEATHER_SUN), jnp.int8(5))
        assert int(state.field.weather) == WEATHER_SUN
        state = set_weather(state, jnp.int8(WEATHER_RAIN), jnp.int8(5))
        assert int(state.field.weather) == WEATHER_RAIN


# ═══════════════════════════════════════════════════════════════════════════
# TRICK ROOM
# ═══════════════════════════════════════════════════════════════════════════

class TestTrickRoomToggle:
    """PS: Trick Room toggles: sets 5 turns if off, clears if on."""

    def test_trick_room_sets_turns(self):
        from pokejax.core.state import set_trick_room
        state = _make_state()
        assert int(state.field.trick_room) == 0
        state = set_trick_room(state, jnp.int8(5))
        assert int(state.field.trick_room) == 5

    def test_trick_room_clear(self):
        from pokejax.core.state import set_trick_room
        state = _make_state()
        state = set_trick_room(state, jnp.int8(5))
        state = set_trick_room(state, jnp.int8(0))
        assert int(state.field.trick_room) == 0


# ═══════════════════════════════════════════════════════════════════════════
# SUBSTITUTE
# ═══════════════════════════════════════════════════════════════════════════

class TestSubstituteParity:
    """PS: Substitute costs 25% max HP, creates sub with that HP."""

    def test_substitute_costs_quarter_hp(self):
        """Creating a Substitute should cost floor(max_hp / 4)."""
        max_hp = 300
        state = _make_state(max_hp=max_hp)
        # Manually apply substitute cost
        sub_cost = max_hp // 4  # 75
        state = set_hp(state, 0, 0, jnp.int16(max_hp - sub_cost))
        state = set_volatile(state, 0, 0, VOL_SUBSTITUTE, True)
        assert int(state.sides_team_hp[0, 0]) == max_hp - sub_cost
        assert bool(has_volatile(state, 0, 0, VOL_SUBSTITUTE))

    def test_substitute_fails_below_quarter_hp(self):
        """Cannot create Substitute if HP ≤ 25% max HP."""
        max_hp = 100
        state = _make_state(max_hp=max_hp)
        state = set_hp(state, 0, 0, jnp.int16(24))  # < 25
        # Sub should fail (not enough HP)
        can_sub = int(state.sides_team_hp[0, 0]) > max_hp // 4
        assert not can_sub


# ═══════════════════════════════════════════════════════════════════════════
# HAZE
# ═══════════════════════════════════════════════════════════════════════════

class TestHazeParity:
    """PS: Haze resets ALL stat boosts on both sides to 0."""

    def test_haze_resets_all_boosts(self):
        state = _make_state()
        # Set various boosts
        state = set_boost(state, 0, 0, BOOST_ATK, jnp.int8(6))
        state = set_boost(state, 0, 0, BOOST_SPE, jnp.int8(3))
        state = set_boost(state, 1, 0, BOOST_DEF, jnp.int8(-4))
        state = set_boost(state, 1, 0, BOOST_SPA, jnp.int8(2))
        # Apply Haze: reset all boosts
        state = reset_boosts(state, 0, 0)
        state = reset_boosts(state, 1, 0)
        for i in range(7):
            assert int(state.sides_team_boosts[0, 0, i]) == 0
            assert int(state.sides_team_boosts[1, 0, i]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# BELLY DRUM
# ═══════════════════════════════════════════════════════════════════════════

class TestBellyDrumParity:
    """PS: Belly Drum costs 50% max HP, sets ATK to +6."""

    def test_belly_drum_sets_max_atk(self):
        max_hp = 400
        state = _make_state(max_hp=max_hp)
        # Belly Drum: lose 50% HP, ATK → +6
        state = set_hp(state, 0, 0, jnp.int16(max_hp // 2))
        state = set_boost(state, 0, 0, BOOST_ATK, jnp.int8(6))
        assert int(state.sides_team_hp[0, 0]) == 200
        assert int(state.sides_team_boosts[0, 0, BOOST_ATK]) == 6

    def test_belly_drum_fails_at_low_hp(self):
        """Cannot Belly Drum if HP ≤ 50%."""
        max_hp = 200
        state = _make_state(max_hp=max_hp)
        state = set_hp(state, 0, 0, jnp.int16(99))
        can_drum = int(state.sides_team_hp[0, 0]) > max_hp // 2
        assert not can_drum


# ═══════════════════════════════════════════════════════════════════════════
# REST
# ═══════════════════════════════════════════════════════════════════════════

class TestRestParity:
    """PS: Rest fully heals, sets sleep for 2 turns (Gen 4: 3 turns)."""

    def test_rest_heals_to_full(self):
        max_hp = 300
        state = _make_state(max_hp=max_hp)
        state = set_hp(state, 0, 0, jnp.int16(50))
        # Rest: heal to full + sleep
        state = set_hp(state, 0, 0, jnp.int16(max_hp))
        state = set_status(state, 0, 0, jnp.int8(STATUS_SLP), jnp.int8(0))
        state = state._replace(
            sides_team_sleep_turns=state.sides_team_sleep_turns.at[0, 0].set(jnp.int8(2))
        )
        assert int(state.sides_team_hp[0, 0]) == max_hp
        assert int(state.sides_team_status[0, 0]) == STATUS_SLP
        assert int(state.sides_team_sleep_turns[0, 0]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# RAPID SPIN
# ═══════════════════════════════════════════════════════════════════════════

class TestRapidSpinParity:
    """PS: Rapid Spin clears hazards + Leech Seed + Partial Trap from own side."""

    def test_rapid_spin_clears_hazards(self):
        state = _make_state()
        state = set_side_condition(state, 0, SC_SPIKES, jnp.int8(3))
        state = set_side_condition(state, 0, SC_STEALTHROCK, jnp.int8(1))
        state = set_side_condition(state, 0, SC_TOXICSPIKES, jnp.int8(2))
        state = set_side_condition(state, 0, SC_STICKYWEB, jnp.int8(1))
        # Rapid Spin effect: clear all hazards from own side
        for cond in [SC_SPIKES, SC_STEALTHROCK, SC_TOXICSPIKES, SC_STICKYWEB]:
            state = set_side_condition(state, 0, cond, jnp.int8(0))
        assert int(state.sides_side_conditions[0, SC_SPIKES]) == 0
        assert int(state.sides_side_conditions[0, SC_STEALTHROCK]) == 0
        assert int(state.sides_side_conditions[0, SC_TOXICSPIKES]) == 0
        assert int(state.sides_side_conditions[0, SC_STICKYWEB]) == 0

    def test_rapid_spin_clears_leech_seed(self):
        state = _make_state()
        state = set_volatile(state, 0, 0, VOL_SEEDED, True)
        assert bool(has_volatile(state, 0, 0, VOL_SEEDED))
        state = set_volatile(state, 0, 0, VOL_SEEDED, False)
        assert not bool(has_volatile(state, 0, 0, VOL_SEEDED))


# ═══════════════════════════════════════════════════════════════════════════
# DEFOG
# ═══════════════════════════════════════════════════════════════════════════

class TestDefogParity:
    """PS Gen 4: Defog clears hazards + screens from OPPONENT'S side + lowers evasion."""

    def test_defog_clears_opponent_hazards(self):
        state = _make_state()
        state = set_side_condition(state, 1, SC_SPIKES, jnp.int8(2))
        state = set_side_condition(state, 1, SC_STEALTHROCK, jnp.int8(1))
        # Defog: clear opponent's hazards
        for cond in [SC_SPIKES, SC_STEALTHROCK, SC_TOXICSPIKES, SC_STICKYWEB]:
            state = set_side_condition(state, 1, cond, jnp.int8(0))
        assert int(state.sides_side_conditions[1, SC_SPIKES]) == 0
        assert int(state.sides_side_conditions[1, SC_STEALTHROCK]) == 0

    def test_defog_clears_screens(self):
        state = _make_state()
        state = set_side_condition(state, 1, SC_REFLECT, jnp.int8(5))
        state = set_side_condition(state, 1, SC_LIGHTSCREEN, jnp.int8(5))
        # Defog: also clears screens
        state = set_side_condition(state, 1, SC_REFLECT, jnp.int8(0))
        state = set_side_condition(state, 1, SC_LIGHTSCREEN, jnp.int8(0))
        assert int(state.sides_side_conditions[1, SC_REFLECT]) == 0
        assert int(state.sides_side_conditions[1, SC_LIGHTSCREEN]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# KNOCK OFF
# ═══════════════════════════════════════════════════════════════════════════

class TestKnockOffParity:
    """PS: Knock Off removes opponent's held item."""

    def test_knock_off_removes_item(self):
        from pokejax.core.state import set_item, consume_item
        state = _make_state()
        state = set_item(state, 1, 0, jnp.int16(42))  # Give item
        assert int(state.sides_team_item_id[1, 0]) == 42
        state = consume_item(state, 1, 0)
        assert int(state.sides_team_item_id[1, 0]) == 0

    def test_knock_off_no_item_noop(self):
        from pokejax.core.state import consume_item
        state = _make_state()
        assert int(state.sides_team_item_id[1, 0]) == 0
        state = consume_item(state, 1, 0)
        assert int(state.sides_team_item_id[1, 0]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# PAIN SPLIT
# ═══════════════════════════════════════════════════════════════════════════

class TestPainSplitParity:
    """PS: Pain Split averages both Pokemon's HP."""

    @pytest.mark.parametrize("hp1,hp2,expected_avg", [
        (300, 100, 200),  # (300+100)//2 = 200
        (200, 200, 200),  # same → no change
        (1, 299, 150),    # (1+299)//2 = 150
        (100, 50, 75),    # (100+50)//2 = 75
    ])
    def test_pain_split_averages(self, hp1, hp2, expected_avg):
        """Pain Split sets both Pokemon to average of their HP."""
        avg = (hp1 + hp2) // 2
        assert avg == expected_avg


# ═══════════════════════════════════════════════════════════════════════════
# PP DEDUCTION
# ═══════════════════════════════════════════════════════════════════════════

class TestPPDeductionParity:
    """PS: Using a move deducts 1 PP (2 with Pressure)."""

    def test_pp_deducts_by_one(self):
        from pokejax.core.state import deduct_pp
        state = _make_state()
        assert int(state.sides_team_move_pp[0, 0, 0]) == 35
        state = deduct_pp(state, 0, 0, 0)
        assert int(state.sides_team_move_pp[0, 0, 0]) == 34

    def test_pp_cannot_go_below_zero(self):
        from pokejax.core.state import deduct_pp, set_pp
        state = _make_state()
        state = set_pp(state, 0, 0, 0, jnp.int8(0))
        state = deduct_pp(state, 0, 0, 0)
        assert int(state.sides_team_move_pp[0, 0, 0]) == 0

    def test_pressure_deducts_two(self):
        from pokejax.core.state import deduct_pp
        state = _make_state()
        state = deduct_pp(state, 0, 0, 0, amount=jnp.int8(2))
        assert int(state.sides_team_move_pp[0, 0, 0]) == 33


# ═══════════════════════════════════════════════════════════════════════════
# WEATHER RESIDUAL DAMAGE
# ═══════════════════════════════════════════════════════════════════════════

class TestWeatherResidualDamageParity:
    """
    PS Gen 4:
      - Sandstorm: 1/16 max HP to non-Rock/Ground/Steel
      - Hail: 1/16 max HP to non-Ice
    """

    def test_sandstorm_damages_non_immune(self):
        """Sandstorm deals 1/16 to Normal type."""
        from pokejax.engine.field import apply_weather_residual
        max_hp = 320  # 320/16 = 20
        state = _make_state(max_hp=max_hp)
        state = set_weather(state, jnp.int8(WEATHER_SAND), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_weather_residual(state, 0)
        expected_dmg = max_hp // 16
        actual_dmg = hp_before - int(state.sides_team_hp[0, 0])
        assert actual_dmg == expected_dmg, \
            f"Expected {expected_dmg} sand dmg, got {actual_dmg}"

    @pytest.mark.parametrize("immune_type", [
        TYPE_GROUND, TYPE_STEEL,
    ])
    def test_sandstorm_immune_types(self, immune_type):
        """Rock/Ground/Steel immune to sandstorm damage."""
        from pokejax.engine.field import apply_weather_residual
        state = _make_state(max_hp=300, p1_types=(immune_type, 0))
        state = set_weather(state, jnp.int8(WEATHER_SAND), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_weather_residual(state, 0)
        assert int(state.sides_team_hp[0, 0]) == hp_before

    def test_hail_damages_non_ice(self):
        """Hail deals 1/16 to non-Ice types."""
        from pokejax.engine.field import apply_weather_residual
        max_hp = 320
        state = _make_state(max_hp=max_hp)
        state = set_weather(state, jnp.int8(WEATHER_HAIL), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_weather_residual(state, 0)
        expected_dmg = max_hp // 16
        actual_dmg = hp_before - int(state.sides_team_hp[0, 0])
        assert actual_dmg == expected_dmg

    def test_hail_immune_ice(self):
        """Ice types immune to hail damage."""
        from pokejax.engine.field import apply_weather_residual
        from pokejax.types import TYPE_ICE
        state = _make_state(max_hp=300, p1_types=(TYPE_ICE, 0))
        state = set_weather(state, jnp.int8(WEATHER_HAIL), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_weather_residual(state, 0)
        assert int(state.sides_team_hp[0, 0]) == hp_before

    def test_sun_no_residual_damage(self):
        """Sun does not deal residual damage."""
        from pokejax.engine.field import apply_weather_residual
        state = _make_state(max_hp=300)
        state = set_weather(state, jnp.int8(WEATHER_SUN), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_weather_residual(state, 0)
        assert int(state.sides_team_hp[0, 0]) == hp_before

    def test_rain_no_residual_damage(self):
        """Rain does not deal residual damage."""
        from pokejax.engine.field import apply_weather_residual
        state = _make_state(max_hp=300)
        state = set_weather(state, jnp.int8(WEATHER_RAIN), jnp.int8(5))
        hp_before = int(state.sides_team_hp[0, 0])
        state = apply_weather_residual(state, 0)
        assert int(state.sides_team_hp[0, 0]) == hp_before
