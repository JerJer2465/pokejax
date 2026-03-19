"""
GPU-Parallel Expectimax Search for Pokemon battles.

Uses the pokejax engine + neural network to look ahead and select the best
action by simulating all (our_action × opponent_action) pairs in parallel.

PERFORMANCE: Split into 3 smaller JIT'd kernels (root_eval, simulate, aggregate)
to avoid OOM during XLA compilation. Each compiles in ~10-30s and is cached.
The 3 kernel launches add negligible overhead vs. a single fused kernel.

Algorithm (depth-1):
  1. Get legal actions + policy priors for both players.
  2. Enumerate all 10×10 = 100 action pairs.
  3. Simulate each pair N_SAMPLES times (different RNG keys → stochasticity).
  4. Evaluate resulting states with value network.
  5. V(a) = Σ_b P(b|s) · mean_k[V(s_{a,b,k})]
  6. Return argmax_a V(a) over legal actions.
"""

from __future__ import annotations
from pathlib import Path
from typing import NamedTuple
import time as _time

import jax
import jax.numpy as jnp
import numpy as np

# Persistent XLA compilation cache — compile once, reuse across runs.
_CACHE_DIR = str(Path(__file__).resolve().parent.parent.parent / ".jax_cache")
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from pokejax.env.pokejax_env import PokeJAXEnv, EnvState
from pokejax.env.action_mask import N_ACTIONS
from pokejax.rl.obs_builder import build_obs as build_obs_jax
from pokejax.rl.model import PokeTransformer  # used for type hints


class SearchResult(NamedTuple):
    """Result of a search from the root state."""
    best_action: int           # argmax action index (0-9)
    action_values: np.ndarray  # float[10] expected value per action
    root_value: float          # value network estimate of root state
    search_policy: np.ndarray  # float[10] softmax of action values (for logging)


class ExpectiMaxSearch:
    """GPU-parallel expectimax search using pokejax engine.

    Split into 3 JIT'd kernels to keep compilation fast and within memory:
      1. root_eval: obs → policy priors + value for both players
      2. simulate: replicate state → vmap step_lean → vmap build_obs → model
      3. aggregate: leaf values → weighted expectimax action values

    Parameters
    ----------
    env : PokeJAXEnv
        The pokejax environment.
    model : PokeTransformer
        Actor-critic model.
    params : dict
        Flax model parameters.
    n_samples : int
        RNG samples per action pair (default 16).
    opp_temperature : float
        Temperature for opponent policy (0 = greedy, 1 = raw policy).
    """

    def __init__(
        self,
        env: PokeJAXEnv,
        model: PokeTransformer,
        params,
        n_samples: int = 16,
        opp_temperature: float = 0.5,
        warmup: bool = True,
    ):
        self.env = env
        self.model = model
        self.params = params
        self.n_samples = n_samples
        self.opp_temperature = opp_temperature

        # Total sims: 10 our × 10 opp × n_samples
        self.n_total = N_ACTIONS * N_ACTIONS * n_samples

        # Pre-build the fixed action pair grid: (n_total, 2)
        our_grid = np.repeat(np.arange(N_ACTIONS), N_ACTIONS * n_samples)
        opp_grid = np.tile(np.repeat(np.arange(N_ACTIONS), n_samples), N_ACTIONS)
        self._action_pairs = jnp.array(
            np.stack([our_grid, opp_grid], axis=-1), dtype=jnp.int32
        )  # (n_total, 2)

        tables = env.tables
        _n_samples = n_samples
        _opp_temp = opp_temperature
        _action_pairs = self._action_pairs

        # ── Kernel 1: Root evaluation ──
        # Gets policy priors + value for both players from current state.
        @jax.jit
        def _root_eval(params, env_state):
            battle = env_state.battle
            reveal = env_state.reveal

            obs_p0 = build_obs_jax(battle, reveal, player=0, tables=tables)
            obs_p1 = build_obs_jax(battle, reveal, player=1, tables=tables)

            our_log_probs, _, our_value = model.apply(
                params,
                obs_p0['int_ids'][None],
                obs_p0['float_feats'][None],
                obs_p0['legal_mask'][None],
            )
            opp_log_probs, _, _ = model.apply(
                params,
                obs_p1['int_ids'][None],
                obs_p1['float_feats'][None],
                obs_p1['legal_mask'][None],
            )

            return (our_log_probs[0], our_value[0],
                    obs_p0['legal_mask'], opp_log_probs[0], obs_p1['legal_mask'])

        self._root_eval = _root_eval

        # ── Kernel 2: Simulate all action pairs ──
        # Hybrid approach: vmap within chunks, lax.map across chunks.
        # This gives GPU parallelism (vmap) without exploding compile times.
        # Chunk size 100 = 10 opp_actions × n_samples (one "our action" per chunk).
        _chunk_size = N_ACTIONS * n_samples  # 10 × 16 = 160

        # vmap over a single chunk
        def _sim_chunk_body(env_state, action_pairs_chunk, keys_chunk):
            """Simulate one chunk of action pairs in parallel."""
            batched = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (_chunk_size,) + x.shape),
                env_state,
            )
            next_states, rewards_all, dones_all = jax.vmap(env.step_lean)(
                batched, action_pairs_chunk, keys_chunk,
            )
            rewards_p0 = rewards_all[:, 0]
            dones_p0 = dones_all[:, 0]

            all_obs = jax.vmap(
                lambda b, r: build_obs_jax(b, r, player=0, tables=tables)
            )(next_states.battle, next_states.reveal)

            _, _, leaf_vals = model.apply(
                params,
                all_obs['int_ids'],
                all_obs['float_feats'],
                all_obs['legal_mask'],
            )
            return jnp.where(dones_p0, rewards_p0, leaf_vals)

        @jax.jit
        def _simulate(params, env_state, keys):
            # Reshape into chunks: (n_chunks, chunk_size, ...)
            n_chunks = N_ACTIONS  # one chunk per our_action
            ap_chunks = _action_pairs.reshape(n_chunks, _chunk_size, 2)
            key_chunks = keys.reshape(n_chunks, _chunk_size, -1)

            def scan_body(carry, chunk_data):
                ap_chunk, key_chunk = chunk_data
                leaf_vals = _sim_chunk_body(env_state, ap_chunk, key_chunk)
                return carry, leaf_vals

            _, all_leaf_values = jax.lax.scan(
                scan_body, None, (ap_chunks, key_chunks),
            )
            # all_leaf_values: (n_chunks, chunk_size)
            return all_leaf_values.reshape(-1)

        self._simulate = _simulate

        # ── Kernel 3: Aggregate ──
        # Combines leaf values with opponent priors into action values.
        @jax.jit
        def _aggregate(leaf_values, opp_lp, opp_mask, our_mask):
            # Temperature-scaled opponent probs
            scaled = jnp.where(
                _opp_temp > 0,
                opp_lp / jnp.maximum(_opp_temp, 1e-4),
                opp_lp,
            )
            scaled = jnp.where(opp_mask > 0, scaled, -1e9)
            opp_probs = jax.nn.softmax(scaled)
            opp_probs = opp_probs * opp_mask
            opp_probs = opp_probs / jnp.maximum(opp_probs.sum(), 1e-8)

            # Reshape (n_total,) → (10, 10, n_samples), average, weight
            lv = leaf_values.reshape(N_ACTIONS, N_ACTIONS, _n_samples)
            mean_values = lv.mean(axis=2)  # (10, 10)
            our_values = (mean_values * opp_probs[None, :]).sum(axis=1)  # (10,)
            our_values = jnp.where(our_mask > 0, our_values, -1e9)
            return our_values

        self._aggregate = _aggregate

        # ── Warm up: force XLA compilation with dummy data ──
        if warmup:
            t0 = _time.time()
            print("[Search] Compiling 3 search kernels (first run only, cached after)...", flush=True)

            dummy_key = jax.random.PRNGKey(0)
            dummy_state, _ = env.reset(dummy_key)
            dummy_keys = jax.random.split(dummy_key, self.n_total)

            # Kernel 1
            print("[Search]   1/3 root_eval...", flush=True)
            t1 = _time.time()
            out = _root_eval(params, dummy_state)
            out[0].block_until_ready()
            print(f"[Search]   1/3 done ({_time.time() - t1:.1f}s)", flush=True)

            # Kernel 2 (the big one)
            print("[Search]   2/3 simulate...", flush=True)
            t2 = _time.time()
            lv = _simulate(params, dummy_state, dummy_keys)
            lv.block_until_ready()
            print(f"[Search]   2/3 done ({_time.time() - t2:.1f}s)", flush=True)

            # Kernel 3
            print("[Search]   3/3 aggregate...", flush=True)
            t3 = _time.time()
            av = _aggregate(lv, out[3], out[4], out[2])
            av.block_until_ready()
            print(f"[Search]   3/3 done ({_time.time() - t3:.1f}s)", flush=True)

            print(f"[Search] All kernels ready! Total: {_time.time() - t0:.1f}s", flush=True)

    def search(
        self,
        env_state: EnvState,
        key: jnp.ndarray,
    ) -> SearchResult:
        """Run expectimax search. Fast after init (3 small kernel launches)."""
        # Generate RNG keys for all sims
        sim_keys = jax.random.split(key, self.n_total)

        # Kernel 1: root evaluation
        our_lp, root_val, our_mask, opp_lp, opp_mask = self._root_eval(
            self.params, env_state,
        )

        # Kernel 2: simulate all action pairs
        leaf_values = self._simulate(self.params, env_state, sim_keys)

        # Kernel 3: aggregate
        action_values_jax = self._aggregate(leaf_values, opp_lp, opp_mask, our_mask)

        # Convert to numpy for action selection
        action_values = np.array(action_values_jax)
        root_value = float(root_val)

        best_action = int(np.argmax(action_values))

        # Softmax of action values for logging
        valid = action_values > -1e8
        policy = np.zeros(N_ACTIONS, dtype=np.float32)
        if valid.any():
            v = action_values[valid]
            v = v - v.max()
            e = np.exp(v)
            policy[valid] = e / e.sum()

        return SearchResult(
            best_action=best_action,
            action_values=action_values,
            root_value=root_value,
            search_policy=policy,
        )


def run_search(
    env: PokeJAXEnv,
    model: PokeTransformer,
    params,
    env_state: EnvState,
    key: jnp.ndarray,
    n_samples: int = 16,
    opp_temperature: float = 0.5,
) -> SearchResult:
    """Convenience: create searcher and run one search."""
    searcher = ExpectiMaxSearch(
        env=env, model=model, params=params,
        n_samples=n_samples, opp_temperature=opp_temperature,
    )
    return searcher.search(env_state, key)
