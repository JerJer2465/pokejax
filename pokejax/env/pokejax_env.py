"""
PokeJAX RL Environment — Gymnax-compatible interface.

Usage:
    pool = load_team_pool("data/team_pool.npz")
    env = PokeJAXEnv(gen=4, team_pool=pool)
    tables = env.tables
    cfg    = env.cfg

    key = jax.random.PRNGKey(0)
    state, obs = env.reset(key)
    state, obs, rewards, dones, info = env.step(state, actions, key)

Vectorization:
    v_reset = jax.vmap(env.reset)
    v_step  = jax.vmap(env.step)

    keys   = jax.random.split(key, N_ENVS)
    states, obss = v_reset(keys)
    # ... each training step:
    states, obss, rewards, dones, _ = v_step(states, actions, keys)

Rollout fusion with lax.scan:
    def rollout_step(carry, action):
        state, key = carry
        new_key, step_key = jax.random.split(key)
        new_state, obs, reward, done, _ = env.step(state, action, step_key)
        return (new_state, new_key), (obs, reward, done)

    (final_state, _), (obss, rewards, dones) = jax.lax.scan(
        rollout_step, (state, key), actions_seq
    )
"""

from dataclasses import dataclass
from typing import Tuple

import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import functools

from pokejax.config import GenConfig
from pokejax.data.tables import Tables, load_tables
from typing import NamedTuple
from pokejax.types import BattleState, RevealState
from pokejax.core.state import make_battle_state, make_reveal_state
from pokejax.engine.turn import execute_turn
from pokejax.engine.switch import switch_in, get_valid_switch_mask
from pokejax.env.obs import build_observation, OBS_DIM
from pokejax.env.action_mask import get_action_mask, N_ACTIONS
from pokejax.env.team_gen import load_team_pool, sample_teams, team_to_state_arrays


class EnvState(NamedTuple):
    """Combined battle + information-masking state.
    A single JAX pytree passed through jit/vmap/scan.
    """
    battle: BattleState
    reveal: RevealState


@dataclass
class EnvParams:
    gen: int = 4
    max_turns: int = 300


class PokeJAXEnv:
    """
    Pokemon battle environment following gymnax conventions.

    Tables, cfg, and team_pool are NOT JAX state — they are Python-level
    constants captured in the closure of jit-compiled step/reset functions.
    """

    def __init__(self, gen: int = 4, showdown_path: str | None = None,
                 team_pool: jnp.ndarray | None = None,
                 team_pool_path: str | None = None):
        self.cfg    = GenConfig.for_gen(gen)
        self.tables = load_tables(gen, showdown_path=showdown_path)
        self.obs_dim     = OBS_DIM
        self.n_actions   = N_ACTIONS
        self.params      = EnvParams(gen=gen)

        # Load team pool (auto-detect if not specified)
        if team_pool is not None:
            self.team_pool = team_pool
        elif team_pool_path is not None:
            self.team_pool = load_team_pool(team_pool_path)
        else:
            # Auto-detect: look for default team pool
            default_path = str(Path(__file__).resolve().parent.parent.parent / 'data' / 'team_pool.npz')
            if os.path.exists(default_path):
                self.team_pool = load_team_pool(default_path)
            else:
                self.team_pool = None

        # Pre-compile JIT/vmap versions.
        def _step_pure(env_state, actions, key):
            return self.step(env_state, actions, key)

        self.step_jit  = jax.jit(_step_pure)
        self.step_vmap = jax.jit(jax.vmap(_step_pure))

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, key: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        """
        Initialize a new battle with random teams from the pool.
        Returns (EnvState, obs_for_p0).
        """
        key, team_key, battle_key = jax.random.split(key, 3)

        if self.team_pool is not None:
            return self._reset_from_pool(team_key, battle_key)
        else:
            return self._reset_placeholder(team_key, battle_key)

    def _reset_from_pool(self, team_key, battle_key):
        """Reset using pre-generated team pool (JIT/vmap compatible)."""
        raw0, raw1 = sample_teams(self.team_pool, team_key)
        team0 = team_to_state_arrays(raw0)
        team1 = team_to_state_arrays(raw1)

        state = make_battle_state(
            p1_species=team0['species'],   p2_species=team1['species'],
            p1_abilities=team0['abilities'],p2_abilities=team1['abilities'],
            p1_items=team0['items'],        p2_items=team1['items'],
            p1_types=team0['types'],        p2_types=team1['types'],
            p1_base_stats=team0['base_stats'], p2_base_stats=team1['base_stats'],
            p1_max_hp=team0['max_hp'],      p2_max_hp=team1['max_hp'],
            p1_move_ids=team0['move_ids'],  p2_move_ids=team1['move_ids'],
            p1_move_pp=team0['move_pp'],    p2_move_pp=team1['move_pp'],
            p1_move_max_pp=team0['move_max_pp'], p2_move_max_pp=team1['move_max_pp'],
            p1_levels=team0['levels'],      p2_levels=team1['levels'],
            p1_genders=team0['genders'],    p2_genders=team1['genders'],
            p1_natures=team0['natures'],    p2_natures=team1['natures'],
            p1_weights_hg=team0['weights_hg'], p2_weights_hg=team1['weights_hg'],
            rng_key=battle_key,
        )

        reveal = make_reveal_state(state)
        obs = build_observation(state, player=0, tables=self.tables)
        return EnvState(battle=state, reveal=reveal), obs

    def _reset_placeholder(self, team_key, battle_key):
        """Fallback reset with hardcoded placeholder teams (for testing)."""
        n = 6
        species = np.zeros(n, dtype=np.int16)
        abilities = np.zeros(n, dtype=np.int16)
        items = np.zeros(n, dtype=np.int16)
        types = np.zeros((n, 2), dtype=np.int8)
        types[:, 0] = 1  # Normal type

        base_stats = np.array([
            [255, 10, 10, 75, 135, 55],
            [80,  130, 80, 65, 60, 105],
            [79,  100, 123, 79, 100, 51],
            [80,  80, 80, 105, 80, 110],
            [91,  90, 90, 90, 90, 91],
            [45,  49, 49, 65, 65, 45],
        ], dtype=np.int16)

        hp_vals = np.array([
            (2 * base_stats[i, 0] + 31 + 21) + 110
            for i in range(n)
        ], dtype=np.int16)

        levels = np.full(n, 100, dtype=np.int8)
        genders = np.zeros(n, dtype=np.int8)
        natures = np.zeros(n, dtype=np.int8)
        weights_hg = np.array([469, 404, 4000, 486, 900, 69], dtype=np.int16)

        move_ids = np.zeros((n, 4), dtype=np.int16)
        move_ids[:, :] = np.arange(4, dtype=np.int16)
        move_pp = np.full((n, 4), 35, dtype=np.int8)
        move_max_pp = np.full((n, 4), 35, dtype=np.int8)

        team = {
            'species': species, 'abilities': abilities, 'items': items,
            'types': types, 'base_stats': base_stats, 'max_hp': hp_vals,
            'move_ids': move_ids, 'move_pp': move_pp, 'move_max_pp': move_max_pp,
            'levels': levels, 'genders': genders, 'natures': natures,
            'weights_hg': weights_hg,
        }

        state = make_battle_state(
            p1_species=team['species'],   p2_species=team['species'],
            p1_abilities=team['abilities'],p2_abilities=team['abilities'],
            p1_items=team['items'],        p2_items=team['items'],
            p1_types=team['types'],        p2_types=team['types'],
            p1_base_stats=team['base_stats'], p2_base_stats=team['base_stats'],
            p1_max_hp=team['max_hp'],      p2_max_hp=team['max_hp'],
            p1_move_ids=team['move_ids'],  p2_move_ids=team['move_ids'],
            p1_move_pp=team['move_pp'],    p2_move_pp=team['move_pp'],
            p1_move_max_pp=team['move_max_pp'], p2_move_max_pp=team['move_max_pp'],
            p1_levels=team['levels'],      p2_levels=team['levels'],
            p1_genders=team['genders'],    p2_genders=team['genders'],
            p1_natures=team['natures'],    p2_natures=team['natures'],
            p1_weights_hg=team['weights_hg'], p2_weights_hg=team['weights_hg'],
            rng_key=battle_key,
        )

        reveal = make_reveal_state(state)
        obs = build_observation(state, player=0, tables=self.tables)
        return EnvState(battle=state, reveal=reveal), obs

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        env_state: EnvState,
        actions: jnp.ndarray,   # int32[2]
        key: jnp.ndarray,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """
        Execute one turn with forced switch handling.

        Actions 0-3 = moves, 4-9 = switch to team slot 0-5.

        Returns: (new_env_state, obs[2, OBS_DIM], rewards[2], dones[2], info)
        """
        state, reveal = env_state.battle, env_state.reveal
        new_state, new_reveal = execute_turn(state, reveal, actions, self.tables, self.cfg)

        # --- Forced switch handling ---
        # If a Pokemon fainted but the side still has Pokemon left,
        # auto-switch to the first alive non-active slot.
        new_state = self._handle_forced_switch(new_state, 0)
        new_state = self._handle_forced_switch(new_state, 1)

        new_env_state = EnvState(battle=new_state, reveal=new_reveal)

        # Observations for both players
        obs0 = build_observation(new_state, player=0, tables=self.tables)
        obs1 = build_observation(new_state, player=1, tables=self.tables)
        obs = jnp.stack([obs0, obs1], axis=0)  # shape (2, OBS_DIM)

        # Reward: HP advantage
        def hp_fraction(s: BattleState, side: int) -> jnp.ndarray:
            total_hp = s.sides_team_hp[side].sum().astype(jnp.float32)
            total_max = s.sides_team_max_hp[side].sum().astype(jnp.float32)
            return jnp.where(total_max > 0, total_hp / total_max, jnp.float32(0.0))

        p0_hp = hp_fraction(new_state, 0)
        p1_hp = hp_fraction(new_state, 1)

        # Terminal reward based on winner
        win_reward = jnp.where(
            new_state.finished,
            jnp.where(new_state.winner == jnp.int8(0), jnp.float32(1.0),
            jnp.where(new_state.winner == jnp.int8(1), jnp.float32(-1.0),
                       jnp.float32(0.0))),
            jnp.float32(0.0)
        )

        # Shaping: HP advantage delta + faint bonus
        old_p0_hp = hp_fraction(state, 0)
        old_p1_hp = hp_fraction(state, 1)
        delta_p0 = (p0_hp - old_p0_hp) - (p1_hp - old_p1_hp)

        # Faint-based shaping: clear discrete signal for KOs
        old_p0_fainted = state.sides_team_fainted[0].sum().astype(jnp.float32)
        old_p1_fainted = state.sides_team_fainted[1].sum().astype(jnp.float32)
        new_p0_fainted = new_state.sides_team_fainted[0].sum().astype(jnp.float32)
        new_p1_fainted = new_state.sides_team_fainted[1].sum().astype(jnp.float32)
        faint_delta = (new_p1_fainted - old_p1_fainted) - (new_p0_fainted - old_p0_fainted)

        r0 = win_reward + delta_p0 * 0.05 + faint_delta * 0.15
        r1 = -win_reward - delta_p0 * 0.05 - faint_delta * 0.15

        rewards = jnp.array([r0, r1])
        dones   = jnp.broadcast_to(new_state.finished, (2,))

        return new_env_state, obs, rewards, dones, {}

    def step_lean(
        self,
        env_state: EnvState,
        actions: jnp.ndarray,   # int32[2]
        key: jnp.ndarray,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray]:
        """
        Lean env step: execute turn + compute rewards, but skip obs building.

        Use this in rollout loops where the caller builds obs separately.
        Saves 2× build_observation calls per step (~50% of env step cost).

        Returns: (new_env_state, rewards[2], dones[2])
        """
        state, reveal = env_state.battle, env_state.reveal
        new_state, new_reveal = execute_turn(state, reveal, actions, self.tables, self.cfg)

        new_state = self._handle_forced_switch(new_state, 0)
        new_state = self._handle_forced_switch(new_state, 1)

        new_env_state = EnvState(battle=new_state, reveal=new_reveal)

        # Reward computation (same as step())
        def hp_fraction(s: BattleState, side: int) -> jnp.ndarray:
            total_hp = s.sides_team_hp[side].sum().astype(jnp.float32)
            total_max = s.sides_team_max_hp[side].sum().astype(jnp.float32)
            return jnp.where(total_max > 0, total_hp / total_max, jnp.float32(0.0))

        p0_hp = hp_fraction(new_state, 0)
        p1_hp = hp_fraction(new_state, 1)

        win_reward = jnp.where(
            new_state.finished,
            jnp.where(new_state.winner == jnp.int8(0), jnp.float32(1.0),
            jnp.where(new_state.winner == jnp.int8(1), jnp.float32(-1.0),
                       jnp.float32(0.0))),
            jnp.float32(0.0)
        )

        old_p0_hp = hp_fraction(state, 0)
        old_p1_hp = hp_fraction(state, 1)
        delta_p0 = (p0_hp - old_p0_hp) - (p1_hp - old_p1_hp)

        # Faint-based shaping: clear discrete signal for KOs
        old_p0_fainted = state.sides_team_fainted[0].sum().astype(jnp.float32)
        old_p1_fainted = state.sides_team_fainted[1].sum().astype(jnp.float32)
        new_p0_fainted = new_state.sides_team_fainted[0].sum().astype(jnp.float32)
        new_p1_fainted = new_state.sides_team_fainted[1].sum().astype(jnp.float32)
        faint_delta = (new_p1_fainted - old_p1_fainted) - (new_p0_fainted - old_p0_fainted)

        r0 = win_reward + delta_p0 * 0.05 + faint_delta * 0.15
        r1 = -win_reward - delta_p0 * 0.05 - faint_delta * 0.15

        rewards = jnp.array([r0, r1])
        dones   = jnp.broadcast_to(new_state.finished, (2,))

        return new_env_state, rewards, dones

    def _handle_forced_switch(self, state: BattleState, side: int) -> BattleState:
        """
        If the active Pokemon on `side` has fainted and there are alive
        replacements, switch in the first available one.
        """
        active_idx = state.sides_active_idx[side]
        active_fainted = state.sides_team_fainted[side, active_idx]
        has_alive = state.sides_pokemon_left[side] > jnp.int8(0)
        needs_switch = active_fainted & has_alive & ~state.finished

        # Find first alive non-active slot
        switch_mask = get_valid_switch_mask(state, side)  # bool[6]
        # Weighted argmax trick: multiply by descending weights so slot 0 wins ties
        slot_priorities = jnp.array([6, 5, 4, 3, 2, 1], dtype=jnp.int32)
        weighted = switch_mask.astype(jnp.int32) * slot_priorities
        target_slot = jnp.argmax(weighted).astype(jnp.int8)
        # If no valid slot, keep current (shouldn't happen if has_alive)
        any_valid = switch_mask.any()

        do_switch = needs_switch & any_valid
        new_state = jax.lax.cond(
            do_switch,
            lambda s: switch_in(s, side, target_slot, self.tables, self.cfg),
            lambda s: s,
            state,
        )
        return new_state

    # ------------------------------------------------------------------
    # Legal action mask
    # ------------------------------------------------------------------

    def get_action_masks(self, env_state: EnvState) -> jnp.ndarray:
        """Return bool[2, 10] legal action masks for both players."""
        state = env_state.battle
        mask0 = get_action_mask(state, 0)
        mask1 = get_action_mask(state, 1)
        return jnp.stack([mask0, mask1], axis=0)

    # ------------------------------------------------------------------
    # Auto-reset wrapper (for training loops)
    # ------------------------------------------------------------------

    def step_autoreset(
        self,
        env_state: EnvState,
        actions: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """
        Step with automatic reset on episode end.
        Returns the NEW episode's first obs when done=True.
        """
        new_env_state, obs, rewards, dones, info = self.step(env_state, actions, key)

        # Auto-reset: if done, reset to new episode
        key, reset_key = jax.random.split(key)
        reset_env_state, reset_obs = self.reset(reset_key)

        finished = new_env_state.battle.finished

        # Use reset state/obs if episode is done
        final_env_state = jax.tree.map(
            lambda r, n: jnp.where(finished, r, n),
            reset_env_state, new_env_state,
        )
        final_obs = jnp.where(
            finished[None, None],  # broadcast
            jnp.stack([reset_obs, reset_obs], axis=0),
            obs,
        )

        return final_env_state, final_obs, rewards, dones, info
