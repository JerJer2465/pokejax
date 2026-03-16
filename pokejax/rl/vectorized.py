"""
Vectorized game runner — batched stepping with vmap.

Runs N environments in parallel on GPU. Each "step" processes all N envs
with a single JIT kernel (vmap over action selection + env.step).

Post-JIT throughput: ~100K steps/s on RTX 3080 with N=1024.

Usage:
    runner = BatchedRunner(env, n_envs=1024)
    # Eval
    result = runner.eval_heuristic_vs_random(seed=0)
    # BC collection
    data = runner.collect_bc(n_transitions=500_000)
"""

from __future__ import annotations

import time as _time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.env.pokejax_env import PokeJAXEnv, EnvState
from pokejax.rl.obs_builder import build_obs
from pokejax.rl.heuristic import heuristic_action, random_action, _build_move_categories


MAX_TURNS = 200


def _where_broadcast(mask, old, new):
    """jnp.where with auto broadcasting: (N,) mask vs (N, ...) arrays."""
    m = mask
    for _ in range(old.ndim - 1):
        m = m[..., None]
    return jnp.where(m, old, new)


class BatchedRunner:
    """
    Persistent batched game runner. JIT-compiles once, reuses across calls.

    Args:
        env: PokeJAXEnv instance
        n_envs: number of parallel environments (default 1024)
    """

    def __init__(self, env: PokeJAXEnv, n_envs: int = 1024):
        self.env = env
        self.N = n_envs
        self.tables = env.tables

        # Pre-build heuristic cache
        _build_move_categories(self.tables)

        # Build persistent JIT functions
        self._build_eval_step()
        self._build_bc_step()
        self._jit_reset = jax.jit(jax.vmap(env.reset))
        self._compiled = {"eval": False, "bc": False}

    def _build_eval_step(self):
        N = self.N
        env = self.env
        tables = self.tables

        def heuristic_policy(battle, reveal, key):
            return heuristic_action(battle, 0, tables, key)

        def random_policy(battle, reveal, key):
            return random_action(battle, 1, key)

        @jax.jit
        def eval_step(states, done, turns, key):
            key, k0, k1, sk = jax.random.split(key, 4)
            keys0 = jax.random.split(k0, N)
            keys1 = jax.random.split(k1, N)
            step_keys = jax.random.split(sk, N)

            acts0 = jax.vmap(heuristic_policy)(states.battle, states.reveal, keys0)
            acts1 = jax.vmap(random_policy)(states.battle, states.reveal, keys1)
            actions = jnp.stack([acts0, acts1], axis=1)

            new_states, _, _, _, _ = jax.vmap(env.step)(states, actions, step_keys)

            final_states = jax.tree.map(
                lambda o, n: _where_broadcast(done, o, n), states, new_states,
            )
            new_done = done | new_states.battle.finished
            new_turns = turns + (~done).astype(jnp.int32)
            return final_states, new_done, new_turns, key

        self._eval_step = eval_step

    def _build_bc_step(self):
        N = self.N
        env = self.env
        tables = self.tables

        @jax.jit
        def bc_step(states, key):
            key, hk, rk, sk = jax.random.split(key, 4)
            h_keys = jax.random.split(hk, N)
            r_keys = jax.random.split(rk, N)
            s_keys = jax.random.split(sk, N)

            alive = ~states.battle.finished

            obs = jax.vmap(
                lambda b, r: build_obs(b, r, 0, tables)
            )(states.battle, states.reveal)

            teacher_acts = jax.vmap(
                lambda b, k: heuristic_action(b, 0, tables, k)
            )(states.battle, h_keys)

            opp_acts = jax.vmap(
                lambda b, k: random_action(b, 1, k)
            )(states.battle, r_keys)

            actions = jnp.stack([teacher_acts, opp_acts], axis=1)
            new_states, _, _, _, _ = jax.vmap(env.step_autoreset)(states, actions, s_keys)

            return new_states, obs, teacher_acts, alive, key

        self._bc_step = bc_step

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval_heuristic_vs_random(
        self, seed: int = 9999, max_turns: int = MAX_TURNS, verbose: bool = True,
    ) -> dict:
        """Run N games (heuristic vs random), return win rate + avg turns."""
        N = self.N

        keys = jax.random.split(jax.random.PRNGKey(seed), N)
        states, _ = self._jit_reset(keys)

        done = jnp.zeros(N, dtype=jnp.bool_)
        turns = jnp.zeros(N, dtype=jnp.int32)
        key = jax.random.PRNGKey(seed + 1)

        if verbose:
            print(f"  Batched eval: {N} games, max_turns={max_turns}")
            if not self._compiled["eval"]:
                print(f"  JIT compiling (one-time)...")

        t0 = _time.time()

        for turn in range(max_turns):
            states, done, turns, key = self._eval_step(states, done, turns, key)

            if turn % 20 == 19:
                if bool(done.all()):
                    if verbose:
                        print(f"  All games finished at turn {turn + 1}")
                    break

        if not self._compiled["eval"]:
            jax.block_until_ready(jax.tree.leaves(states))
            self._compiled["eval"] = True

        elapsed = _time.time() - t0
        if verbose:
            print(f"  Completed in {elapsed:.1f}s")

        winners = np.array(states.battle.winner)
        turns_np = np.array(turns)
        done_np = np.array(done).astype(bool)
        finished = done_np
        wins = ((winners == 0) & finished).sum()
        total = finished.sum()

        return {
            "win_rate": wins / max(total, 1),
            "avg_turns": turns_np[finished].mean() if finished.any() else 0.0,
            "finished": int(total),
            "total": N,
        }

    # ------------------------------------------------------------------
    # BC collection
    # ------------------------------------------------------------------

    def collect_bc(
        self,
        n_transitions: int,
        seed: int = 0,
        verbose: bool = True,
    ) -> dict:
        """
        Collect BC training data at GPU speed using autoreset.

        Pre-allocates output arrays to avoid OOM from list accumulation.
        Returns dict with int_ids, float_feats, legal_mask, actions.
        """
        N = self.N

        keys = jax.random.split(jax.random.PRNGKey(seed), N)
        states, _ = self._jit_reset(keys)
        key = jax.random.PRNGKey(seed + 1)

        # Pre-allocate output arrays
        out_int_ids = np.zeros((n_transitions, 15, 8), dtype=np.int32)
        out_float_feats = np.zeros((n_transitions, 15, 394), dtype=np.float32)
        out_legal_mask = np.zeros((n_transitions, 10), dtype=np.float32)
        out_actions = np.zeros(n_transitions, dtype=np.int32)
        write_idx = 0
        step_count = 0

        if verbose:
            print(f"  Batched BC collection: n_envs={N}, target={n_transitions}")
            if not self._compiled["bc"]:
                print(f"  JIT compiling (one-time)...")

        t0 = _time.time()

        while write_idx < n_transitions:
            states, obs, acts, alive, key = self._bc_step(states, key)

            if not self._compiled["bc"] and step_count == 0:
                jax.block_until_ready(jax.tree.leaves(states))
                self._compiled["bc"] = True
                jit_time = _time.time() - t0
                if verbose:
                    print(f"  JIT compiled in {jit_time:.1f}s")
                t0 = _time.time()

            # Pull to CPU and write valid transitions directly
            alive_np = np.array(alive).astype(bool)
            n_valid = int(alive_np.sum())

            if n_valid > 0:
                # How many can we still write?
                n_write = min(n_valid, n_transitions - write_idx)
                valid_idx = np.where(alive_np)[0][:n_write]

                end = write_idx + n_write
                out_int_ids[write_idx:end] = np.array(obs["int_ids"])[valid_idx]
                out_float_feats[write_idx:end] = np.array(obs["float_feats"])[valid_idx]
                out_legal_mask[write_idx:end] = np.array(obs["legal_mask"])[valid_idx]
                out_actions[write_idx:end] = np.array(acts)[valid_idx]
                write_idx = end

            step_count += 1

            if verbose and step_count % 100 == 0:
                elapsed = _time.time() - t0
                rate = write_idx / max(elapsed, 0.001)
                eta = (n_transitions - write_idx) / max(rate, 0.001)
                print(f"  Step {step_count}: {write_idx}/{n_transitions} "
                      f"({rate:.0f} trans/s, ETA {eta:.0f}s)")

        elapsed = _time.time() - t0
        if verbose:
            print(f"  Done: {write_idx} transitions in {elapsed:.1f}s "
                  f"({write_idx/max(elapsed, 0.001):.0f} trans/s)")

        return {
            "int_ids": out_int_ids,
            "float_feats": out_float_feats,
            "legal_mask": out_legal_mask,
            "actions": out_actions,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (create runner, run once)
# ---------------------------------------------------------------------------

def eval_heuristic_vs_random_batched(
    env, n_games=1024, seed=9999, max_turns=MAX_TURNS, verbose=True,
):
    runner = BatchedRunner(env, n_envs=n_games)
    return runner.eval_heuristic_vs_random(seed=seed, max_turns=max_turns, verbose=verbose)


def collect_bc_batched(
    env, n_transitions, seed=0, n_envs=1024, verbose=True,
):
    runner = BatchedRunner(env, n_envs=n_envs)
    return runner.collect_bc(n_transitions=n_transitions, seed=seed, verbose=verbose)
