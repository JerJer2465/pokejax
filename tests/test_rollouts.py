"""
L3 rollout tests for Phase 6.

Tests:
  - execute_turn runs without error and produces valid state
  - execute_turn is JIT-compilable (jax.jit)
  - execute_turn is vmap-able (jax.vmap over batched states)
  - Full rollout (100 turns) terminates or hits max turns
  - Batched rollout (N envs via vmap) produces consistent output shapes
  - lax.scan rollout gives same result as unrolled loop
  - PokeJAXEnv.step_jit and step_vmap are callable
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.data.tables import load_tables
from pokejax.config import GenConfig
from pokejax.core.state import make_battle_state
from pokejax.engine.turn import execute_turn
from pokejax.env.pokejax_env import PokeJAXEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_team():
    """Minimal team spec (6 identical Pokemon)."""
    n = 6
    base = np.array([[80, 80, 80, 80, 80, 80]] * n, dtype=np.int16)
    max_hp = np.full(n, 250, dtype=np.int16)
    types = np.zeros((n, 2), dtype=np.int8)
    types[:, 0] = 1  # Normal
    move_ids = np.zeros((n, 4), dtype=np.int16)   # all move 0 (Tackle-ish)
    move_pp = np.full((n, 4), 35, dtype=np.int8)
    levels = np.full(n, 100, dtype=np.int8)
    return dict(
        species=np.zeros(n, dtype=np.int16),
        abilities=np.zeros(n, dtype=np.int16),
        items=np.zeros(n, dtype=np.int16),
        types=types,
        base_stats=base,
        max_hp=max_hp,
        move_ids=move_ids,
        move_pp=move_pp,
        move_max_pp=move_pp,
        levels=levels,
        genders=np.zeros(n, dtype=np.int8),
        natures=np.zeros(n, dtype=np.int8),
        weights_hg=np.full(n, 900, dtype=np.int16),
    )


def _make_state(key, tables, cfg):
    t0 = _make_team()
    t1 = _make_team()
    return make_battle_state(
        p1_species=t0['species'],    p2_species=t1['species'],
        p1_abilities=t0['abilities'],p2_abilities=t1['abilities'],
        p1_items=t0['items'],        p2_items=t1['items'],
        p1_types=t0['types'],        p2_types=t1['types'],
        p1_base_stats=t0['base_stats'], p2_base_stats=t1['base_stats'],
        p1_max_hp=t0['max_hp'],      p2_max_hp=t1['max_hp'],
        p1_move_ids=t0['move_ids'],  p2_move_ids=t1['move_ids'],
        p1_move_pp=t0['move_pp'],    p2_move_pp=t1['move_pp'],
        p1_move_max_pp=t0['move_max_pp'], p2_move_max_pp=t1['move_max_pp'],
        p1_levels=t0['levels'],      p2_levels=t1['levels'],
        p1_genders=t0['genders'],    p2_genders=t1['genders'],
        p1_natures=t0['natures'],    p2_natures=t1['natures'],
        p1_weights_hg=t0['weights_hg'], p2_weights_hg=t1['weights_hg'],
        rng_key=key,
    )


@pytest.fixture(scope="module")
def tables():
    return load_tables(gen=4)


@pytest.fixture(scope="module")
def cfg():
    return GenConfig.for_gen(4)


@pytest.fixture(scope="module")
def init_state(tables, cfg):
    key = jax.random.PRNGKey(42)
    return _make_state(key, tables, cfg)


# ---------------------------------------------------------------------------
# Basic execute_turn tests
# ---------------------------------------------------------------------------

class TestExecuteTurn:
    def test_turn_runs_without_error(self, tables, cfg, init_state):
        actions = jnp.array([0, 0], dtype=jnp.int32)  # both use move 0
        state2 = execute_turn(init_state, actions, tables, cfg)
        assert state2 is not None
        assert hasattr(state2, 'turn')

    def test_turn_counter_increments(self, tables, cfg, init_state):
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2 = execute_turn(init_state, actions, tables, cfg)
        assert int(state2.turn) == int(init_state.turn) + 1

    def test_hp_decreases_after_move(self, tables, cfg, init_state):
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2 = execute_turn(init_state, actions, tables, cfg)
        total_hp_before = int(init_state.sides_team_hp.sum())
        total_hp_after  = int(state2.sides_team_hp.sum())
        assert total_hp_after <= total_hp_before

    def test_switch_action(self, tables, cfg, init_state):
        # Action 4 = switch to slot 0 (which is active → slot 1 effectively)
        # Action encoding: 4 = switch to team slot 0, 5 = slot 1, etc.
        actions = jnp.array([5, 0], dtype=jnp.int32)  # P0 switches to slot 1
        state2 = execute_turn(init_state, actions, tables, cfg)
        assert int(state2.turn) == int(init_state.turn) + 1
        # P0 active index should have changed
        assert int(state2.sides_active_idx[0]) == 1

    def test_finished_flag_consistent(self, tables, cfg, init_state):
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2 = execute_turn(init_state, actions, tables, cfg)
        # A brand-new battle with full HP shouldn't be finished after 1 turn
        assert not bool(state2.finished)


# ---------------------------------------------------------------------------
# JIT compilation tests
# ---------------------------------------------------------------------------

class TestJIT:
    def test_execute_turn_jit_compiles(self, tables, cfg, init_state):
        """execute_turn should be JIT-compilable without errors."""
        jit_turn = jax.jit(lambda s, a: execute_turn(s, a, tables, cfg))
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2 = jit_turn(init_state, actions)
        assert int(state2.turn) == int(init_state.turn) + 1

    def test_execute_turn_jit_consistent_with_eager(self, tables, cfg, init_state):
        """JIT and eager modes should produce identical results."""
        jit_turn = jax.jit(lambda s, a: execute_turn(s, a, tables, cfg))
        actions = jnp.array([0, 0], dtype=jnp.int32)

        state_eager = execute_turn(init_state, actions, tables, cfg)
        state_jit   = jit_turn(init_state, actions)

        assert int(state_eager.turn) == int(state_jit.turn)
        np.testing.assert_array_equal(
            np.array(state_eager.sides_team_hp),
            np.array(state_jit.sides_team_hp),
        )


# ---------------------------------------------------------------------------
# vmap tests
# ---------------------------------------------------------------------------

class TestVmap:
    def test_execute_turn_vmap(self, tables, cfg):
        """vmap over a batch of states should work."""
        N = 4
        keys = jax.random.split(jax.random.PRNGKey(0), N)
        states = jax.vmap(lambda k: _make_state(k, tables, cfg))(keys)

        actions = jnp.zeros((N, 2), dtype=jnp.int32)
        vmap_turn = jax.jit(jax.vmap(lambda s, a: execute_turn(s, a, tables, cfg)))
        states2 = vmap_turn(states, actions)

        # Turn counter should be 1 for all envs
        assert states2.turn.shape == (N,)
        np.testing.assert_array_equal(np.array(states2.turn), np.ones(N, dtype=np.int16))

    def test_vmap_hp_shapes(self, tables, cfg):
        N = 8
        keys = jax.random.split(jax.random.PRNGKey(7), N)
        states = jax.vmap(lambda k: _make_state(k, tables, cfg))(keys)
        actions = jnp.zeros((N, 2), dtype=jnp.int32)
        vmap_turn = jax.jit(jax.vmap(lambda s, a: execute_turn(s, a, tables, cfg)))
        states2 = vmap_turn(states, actions)
        assert states2.sides_team_hp.shape == (N, 2, 6)


# ---------------------------------------------------------------------------
# Full rollout tests
# ---------------------------------------------------------------------------

class TestRollout:
    def test_rollout_terminates(self, tables, cfg, init_state):
        """A rollout should eventually end (finished=True) or reach max_turns."""
        max_turns = 200
        jit_turn = jax.jit(lambda s, a: execute_turn(s, a, tables, cfg))
        actions = jnp.array([0, 0], dtype=jnp.int32)

        state = init_state
        for _ in range(max_turns):
            if bool(state.finished):
                break
            state = jit_turn(state, actions)

        # Either finished (one side ran out of Pokemon) or hit max_turns
        total_hp = int(state.sides_team_hp.sum())
        assert bool(state.finished) or int(state.turn) >= max_turns or total_hp < int(init_state.sides_team_hp.sum())

    def test_lax_scan_rollout(self, tables, cfg, init_state):
        """lax.scan rollout should match an unrolled loop."""
        N_STEPS = 10
        jit_turn = jax.jit(lambda s, a: execute_turn(s, a, tables, cfg))

        # Unrolled
        state_unrolled = init_state
        for _ in range(N_STEPS):
            state_unrolled = jit_turn(state_unrolled, jnp.array([0, 0], dtype=jnp.int32))

        # lax.scan
        def scan_step(s, _):
            s2 = execute_turn(s, jnp.array([0, 0], dtype=jnp.int32), tables, cfg)
            return s2, s2.turn

        state_scan, turns = jax.lax.scan(scan_step, init_state, None, length=N_STEPS)

        np.testing.assert_array_equal(
            np.array(state_unrolled.sides_team_hp),
            np.array(state_scan.sides_team_hp),
        )
        assert int(state_scan.turn) == N_STEPS

    def test_scan_turn_sequence(self, tables, cfg, init_state):
        """Turns emitted by scan should be 1, 2, ..., N."""
        N = 5
        def scan_step(s, _):
            s2 = execute_turn(s, jnp.array([0, 0], dtype=jnp.int32), tables, cfg)
            return s2, s2.turn

        _, turns = jax.lax.scan(scan_step, init_state, None, length=N)
        np.testing.assert_array_equal(np.array(turns), np.arange(1, N + 1, dtype=np.int16))


# ---------------------------------------------------------------------------
# PokeJAXEnv tests
# ---------------------------------------------------------------------------

class TestPokeJAXEnv:
    @pytest.fixture(scope="class")
    def env(self):
        return PokeJAXEnv(gen=4)

    def test_reset(self, env):
        key = jax.random.PRNGKey(0)
        state, obs = env.reset(key)
        assert obs.shape == (env.obs_dim,)
        assert not bool(state.finished)

    def test_step(self, env):
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, obs, rewards, dones, _ = env.step(state, actions, key)
        assert obs.shape == (2, env.obs_dim)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert int(state2.turn) == 1

    def test_step_jit(self, env):
        key = jax.random.PRNGKey(1)
        state, _ = env.reset(key)
        actions = jnp.array([0, 0], dtype=jnp.int32)
        state2, obs, rewards, dones, _ = env.step_jit(state, actions, key)
        assert obs.shape == (2, env.obs_dim)
        assert int(state2.turn) == 1

    def test_step_vmap(self, env):
        N = 4
        keys = jax.random.split(jax.random.PRNGKey(2), N)
        states = jax.vmap(env.reset)(keys)
        # states is (BattleState[N], obs[N, OBS_DIM]) — unpack
        states_batch, _ = states
        actions = jnp.zeros((N, 2), dtype=jnp.int32)
        step_vmap = jax.jit(jax.vmap(lambda s, a, k: env.step(s, a, k)))
        result = step_vmap(states_batch, actions, keys)
        states2, obs2, rewards2, dones2, _ = result
        assert obs2.shape == (N, 2, env.obs_dim)
        assert rewards2.shape == (N, 2)

    def test_get_action_masks(self, env):
        key = jax.random.PRNGKey(3)
        state, _ = env.reset(key)
        masks = env.get_action_masks(state)
        assert masks.shape == (2, env.n_actions)
        # At least one action should be legal per side
        assert bool(masks[0].any())
        assert bool(masks[1].any())
