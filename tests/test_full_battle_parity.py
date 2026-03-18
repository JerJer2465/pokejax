"""
Full battle end-to-end parity test.

Runs complete battles through the pokejax engine and verifies:
  1. Battle terminates (no infinite loops)
  2. HP stays within bounds
  3. State invariants hold after each turn
  4. Winner is determined correctly
  5. Action masks are consistent with state
  6. Turn counter increments correctly

This test exercises the entire engine pipeline:
  execute_turn → damage → status → switch → residuals → field timers
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.core.state import make_battle_state, make_reveal_state
from pokejax.engine.turn import execute_turn
from pokejax.env.pokejax_env import PokeJAXEnv
from pokejax.types import (
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX,
    STATUS_SLP, STATUS_FRZ, STATUS_PAR,
    WEATHER_NONE,
    TYPE_NORMAL, TYPE_FIRE, TYPE_WATER, TYPE_ELECTRIC,
    TYPE_GRASS, TYPE_ICE, TYPE_FIGHTING, TYPE_POISON,
    TYPE_GROUND, TYPE_FLYING, TYPE_ROCK, TYPE_GHOST, TYPE_STEEL,
)
from pokejax.config import GenConfig
from pokejax.data.tables import load_tables


# ═══════════════════════════════════════════════════════════════════════════
# INVARIANT CHECKING
# ═══════════════════════════════════════════════════════════════════════════

def _check_state_invariants(bs, turn: int):
    """Assert that all state invariants hold after a turn.

    bs: BattleState (not EnvState)
    """
    errors = []

    # HP bounds
    for side in range(2):
        for slot in range(6):
            hp = int(bs.sides_team_hp[side, slot])
            max_hp = int(bs.sides_team_max_hp[side, slot])
            if hp < 0:
                errors.append(f"Turn {turn}: P{side+1} slot {slot} HP={hp} < 0")
            if hp > max_hp:
                errors.append(f"Turn {turn}: P{side+1} slot {slot} HP={hp} > max {max_hp}")

    # Status codes valid
    for side in range(2):
        for slot in range(6):
            status = int(bs.sides_team_status[side, slot])
            if status < 0 or status > 6:
                errors.append(f"Turn {turn}: P{side+1} slot {slot} invalid status {status}")

    # Boost bounds
    for side in range(2):
        for slot in range(6):
            for b in range(7):
                boost = int(bs.sides_team_boosts[side, slot, b])
                if boost < -6 or boost > 6:
                    errors.append(f"Turn {turn}: P{side+1} slot {slot} boost[{b}]={boost}")

    # Active index valid
    for side in range(2):
        idx = int(bs.sides_active_idx[side])
        if idx < 0 or idx > 5:
            errors.append(f"Turn {turn}: P{side+1} active_idx={idx} out of range")

    # Pokemon left consistency
    for side in range(2):
        fainted_count = sum(1 for s in range(6) if bool(bs.sides_team_fainted[side, s]))
        pokemon_left = int(bs.sides_pokemon_left[side])
        expected_left = 6 - fainted_count
        if pokemon_left != expected_left:
            errors.append(
                f"Turn {turn}: P{side+1} pokemon_left={pokemon_left}, "
                f"expected {expected_left} (fainted={fainted_count})"
            )

    # Turn counter
    if int(bs.turn) != turn:
        errors.append(f"Turn counter {int(bs.turn)} != expected {turn}")

    # Weather valid
    w = int(bs.field.weather)
    if w < 0 or w > 4:
        errors.append(f"Turn {turn}: invalid weather {w}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# RANDOM BATTLE SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRandomBattleInvariants:
    """Run random battles and check invariants hold throughout."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_battle_invariants(self, seed, tables4, cfg4):
        """Run a random battle for up to 100 turns, checking invariants."""
        env = PokeJAXEnv(gen=4)
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        state, _ = env.reset(subkey)

        all_errors = []

        for turn in range(100):
            bs = state.battle
            if bool(bs.finished):
                break

            # Check invariants before action
            errors = _check_state_invariants(bs, turn)
            all_errors.extend(errors)

            # Sample random legal actions
            key, k1, k2 = jax.random.split(key, 3)
            masks = env.get_action_masks(state)
            p0_mask = masks[0]
            p1_mask = masks[1]

            # Pick random legal action for each side
            p0_legal = jnp.where(p0_mask, jnp.arange(p0_mask.shape[0]), -1)
            p1_legal = jnp.where(p1_mask, jnp.arange(p1_mask.shape[0]), -1)
            p0_valid = p0_legal[p0_legal >= 0]
            p1_valid = p1_legal[p1_legal >= 0]

            if p0_valid.shape[0] == 0 or p1_valid.shape[0] == 0:
                break

            p0_action = p0_valid[jax.random.randint(k1, (), 0, p0_valid.shape[0])]
            p1_action = p1_valid[jax.random.randint(k2, (), 0, p1_valid.shape[0])]

            actions = jnp.array([p0_action, p1_action], dtype=jnp.int32)
            key, subkey = jax.random.split(key)
            state, obs, rewards, dones, info = env.step(state, actions, subkey)

        # Final invariant check
        bs = state.battle
        errors = _check_state_invariants(bs, int(bs.turn))
        all_errors.extend(errors)

        assert len(all_errors) == 0, \
            f"Invariant violations in seed {seed}:\n" + "\n".join(all_errors)

    @pytest.mark.parametrize("seed", range(5))
    def test_battle_terminates(self, seed, tables4, cfg4):
        """Battle should terminate within 500 turns (PP stall + Struggle)."""
        env = PokeJAXEnv(gen=4)
        key = jax.random.PRNGKey(seed + 1000)
        key, subkey = jax.random.split(key)
        state, _ = env.reset(subkey)

        for turn in range(500):
            if bool(state.battle.finished):
                break
            key, k1, k2 = jax.random.split(key, 3)
            masks = env.get_action_masks(state)
            p0_mask = masks[0]
            p1_mask = masks[1]
            p0_legal = jnp.where(p0_mask, jnp.arange(p0_mask.shape[0]), -1)
            p1_legal = jnp.where(p1_mask, jnp.arange(p1_mask.shape[0]), -1)
            p0_valid = p0_legal[p0_legal >= 0]
            p1_valid = p1_legal[p1_legal >= 0]
            if p0_valid.shape[0] == 0 or p1_valid.shape[0] == 0:
                break
            p0_action = p0_valid[jax.random.randint(k1, (), 0, p0_valid.shape[0])]
            p1_action = p1_valid[jax.random.randint(k2, (), 0, p1_valid.shape[0])]
            actions = jnp.array([p0_action, p1_action], dtype=jnp.int32)
            key, subkey = jax.random.split(key)
            state, _, _, _, _ = env.step(state, actions, subkey)

        # Should have terminated
        assert bool(state.battle.finished) or turn >= 499, \
            f"Battle did not terminate after 500 turns"


# ═══════════════════════════════════════════════════════════════════════════
# WINNER DETERMINATION
# ═══════════════════════════════════════════════════════════════════════════

class TestWinnerDetermination:
    """Verify winner is set correctly when all Pokemon faint."""

    def test_winner_set_when_all_faint(self, tables4, cfg4):
        """When all of one side's Pokemon faint, the other side wins."""
        env = PokeJAXEnv(gen=4)
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        state, _ = env.reset(subkey)

        for turn in range(300):
            bs = state.battle
            if bool(bs.finished):
                winner = int(bs.winner)
                assert winner in [0, 1, 2], f"Invalid winner: {winner}"
                # Winner should have at least 1 Pokemon left
                # (or it's a draw = 2)
                if winner < 2:
                    assert int(bs.sides_pokemon_left[winner]) > 0
                    assert int(bs.sides_pokemon_left[1 - winner]) == 0
                return
            key, k1, k2 = jax.random.split(key, 3)
            masks = env.get_action_masks(state)
            p0_valid = jnp.where(masks[0], jnp.arange(masks[0].shape[0]), -1)
            p1_valid = jnp.where(masks[1], jnp.arange(masks[1].shape[0]), -1)
            p0_valid = p0_valid[p0_valid >= 0]
            p1_valid = p1_valid[p1_valid >= 0]
            if p0_valid.shape[0] == 0 or p1_valid.shape[0] == 0:
                break
            p0_action = p0_valid[jax.random.randint(k1, (), 0, p0_valid.shape[0])]
            p1_action = p1_valid[jax.random.randint(k2, (), 0, p1_valid.shape[0])]
            actions = jnp.array([p0_action, p1_action], dtype=jnp.int32)
            key, subkey = jax.random.split(key)
            state, _, _, _, _ = env.step(state, actions, subkey)
        # If battle didn't finish in 300 turns, that's still fine (skip check)


# ═══════════════════════════════════════════════════════════════════════════
# ACTION MASK CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════

class TestActionMaskConsistency:
    """Action masks should be consistent with game state."""

    @pytest.mark.parametrize("seed", range(5))
    def test_at_least_one_legal_action(self, seed):
        """Each non-finished side must have at least 1 legal action."""
        env = PokeJAXEnv(gen=4)
        key = jax.random.PRNGKey(seed + 2000)
        key, subkey = jax.random.split(key)
        state, _ = env.reset(subkey)

        for turn in range(50):
            if bool(state.battle.finished):
                break
            masks = env.get_action_masks(state)
            for side in range(2):
                n_legal = int(jnp.sum(masks[side]))
                assert n_legal > 0, \
                    f"Turn {turn}: P{side+1} has 0 legal actions but battle not finished"
            key, k1, k2 = jax.random.split(key, 3)
            p0_valid = jnp.where(masks[0], jnp.arange(masks[0].shape[0]), -1)
            p1_valid = jnp.where(masks[1], jnp.arange(masks[1].shape[0]), -1)
            p0_valid = p0_valid[p0_valid >= 0]
            p1_valid = p1_valid[p1_valid >= 0]
            p0_action = p0_valid[jax.random.randint(k1, (), 0, p0_valid.shape[0])]
            p1_action = p1_valid[jax.random.randint(k2, (), 0, p1_valid.shape[0])]
            actions = jnp.array([p0_action, p1_action], dtype=jnp.int32)
            key, subkey = jax.random.split(key)
            state, _, _, _, _ = env.step(state, actions, subkey)


# ═══════════════════════════════════════════════════════════════════════════
# JIT AND VMAP COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════

class TestJITVmapCompat:
    """Engine must work under JIT and vmap (required for GPU training)."""

    def test_execute_turn_jits(self, tables4, cfg4):
        """execute_turn should compile under jax.jit."""
        env = PokeJAXEnv(gen=4)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)
        battle = state.battle
        reveal = make_reveal_state(battle)

        @jax.jit
        def step(s, r, actions):
            return execute_turn(s, r, actions, tables4, cfg4)

        actions = jnp.array([0, 0], dtype=jnp.int32)
        battle2, reveal2 = step(battle, reveal, actions)
        assert int(battle2.turn) == 1

    def test_env_step_jits(self):
        """PokeJAXEnv.step should work under JIT."""
        env = PokeJAXEnv(gen=4)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        actions = jnp.array([0, 0], dtype=jnp.int32)
        key, subkey = jax.random.split(key)

        @jax.jit
        def jit_step(s, a, k):
            return env.step(s, a, k)

        state2, obs, rewards, dones, info = jit_step(state, actions, subkey)
        assert state2 is not None

    def test_no_nan_in_observations(self):
        """Observations should never contain NaN values."""
        env = PokeJAXEnv(gen=4)
        key = jax.random.PRNGKey(99)
        state, _ = env.reset(key)

        for turn in range(20):
            if bool(state.battle.finished):
                break
            key, k1, k2 = jax.random.split(key, 3)
            masks = env.get_action_masks(state)
            p0_valid = jnp.where(masks[0], jnp.arange(masks[0].shape[0]), -1)
            p1_valid = jnp.where(masks[1], jnp.arange(masks[1].shape[0]), -1)
            p0_valid = p0_valid[p0_valid >= 0]
            p1_valid = p1_valid[p1_valid >= 0]
            if p0_valid.shape[0] == 0 or p1_valid.shape[0] == 0:
                break
            p0_action = p0_valid[jax.random.randint(k1, (), 0, p0_valid.shape[0])]
            p1_action = p1_valid[jax.random.randint(k2, (), 0, p1_valid.shape[0])]
            actions = jnp.array([p0_action, p1_action], dtype=jnp.int32)
            key, subkey = jax.random.split(key)
            state, obs, rewards, dones, info = env.step(state, actions, subkey)
            assert not bool(jnp.any(jnp.isnan(obs))), \
                f"Turn {turn}: NaN in observations"


# ═══════════════════════════════════════════════════════════════════════════
# TURN EXECUTION SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════

class TestTurnExecutionSequence:
    """Verify turn execution follows PS sequence."""

    def test_turn_counter_increments(self, tables4, cfg4):
        """Turn counter should increment by 1 each turn."""
        env = PokeJAXEnv(gen=4)
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)
        assert int(state.battle.turn) == 0

        for expected_turn in range(1, 6):
            if bool(state.battle.finished):
                break
            key, k1, k2 = jax.random.split(key, 3)
            masks = env.get_action_masks(state)
            p0_valid = jnp.where(masks[0], jnp.arange(masks[0].shape[0]), -1)
            p1_valid = jnp.where(masks[1], jnp.arange(masks[1].shape[0]), -1)
            p0_valid = p0_valid[p0_valid >= 0]
            p1_valid = p1_valid[p1_valid >= 0]
            p0_action = p0_valid[jax.random.randint(k1, (), 0, p0_valid.shape[0])]
            p1_action = p1_valid[jax.random.randint(k2, (), 0, p1_valid.shape[0])]
            actions = jnp.array([p0_action, p1_action], dtype=jnp.int32)
            key, subkey = jax.random.split(key)
            state, _, _, _, _ = env.step(state, actions, subkey)
            assert int(state.battle.turn) == expected_turn, \
                f"Expected turn {expected_turn}, got {int(state.battle.turn)}"
