"""
GPU-Accelerated Monte Carlo Tree Search for Pokemon battles.

Uses the pokejax engine + neural network for multi-turn lookahead.

At each node:
  - Our action is selected via PUCT (policy prior + UCB exploration)
  - Opponent action is sampled from neural network policy
  - Environment is stepped using the JAX engine (GPU)
  - Leaf nodes are evaluated by the value network

Algorithm:
  1. Evaluate root state → policy priors + value for both players.
  2. For each simulation:
     a. SELECT: Walk tree using PUCT until reaching an unexpanded child.
     b. EXPAND: Step env (sample opponent action + RNG), evaluate with model.
     c. BACKUP: Propagate value up the path.
  3. Return action with most visits at root.

Key features:
  - Multi-turn search (depth determined by simulation budget)
  - Neural network guided (AlphaZero-style PUCT)
  - GPU-accelerated (JIT-compiled env stepping + model inference)
  - Handles stochasticity via RNG sampling
  - Opponent modeled by policy network with temperature
  - Dirichlet noise at root for exploration

PERFORMANCE: Three JIT kernels (eval_root, expand_leaf, expand_batch)
compiled once and cached via persistent XLA cache.
"""

from __future__ import annotations
from pathlib import Path
from typing import NamedTuple, Optional
import time as _time

import jax
import jax.numpy as jnp
import numpy as np

# Persistent XLA compilation cache
_CACHE_DIR = str(Path(__file__).resolve().parent.parent.parent / ".jax_cache")
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from pokejax.env.pokejax_env import PokeJAXEnv, EnvState
from pokejax.env.action_mask import N_ACTIONS
from pokejax.rl.obs_builder import build_obs as build_obs_jax
from pokejax.rl.model import PokeTransformer


class SearchResult(NamedTuple):
    """Result of a search from the root state."""
    best_action: int           # argmax visit-count action (0-9)
    action_values: np.ndarray  # float[10] mean Q-value per action
    root_value: float          # value network estimate of root state
    search_policy: np.ndarray  # float[10] normalized visit counts (for logging)


class _MCTSNode:
    """Single node in the MCTS tree.

    Stored on CPU (numpy). Env states stored as JAX pytrees on GPU.
    """
    __slots__ = (
        'env_state', 'our_prior', 'our_legal_mask',
        'opp_prior', 'opp_legal_mask',
        'children', 'visit_count', 'value_sum', 'total_visits',
        'is_terminal', 'terminal_value', 'depth',
    )

    def __init__(
        self,
        env_state: EnvState,
        our_prior: np.ndarray,     # float[10] policy logits
        our_legal_mask: np.ndarray, # float[10]
        opp_prior: np.ndarray,     # float[10] opponent policy logits
        opp_legal_mask: np.ndarray, # float[10]
        is_terminal: bool = False,
        terminal_value: float = 0.0,
        depth: int = 0,
    ):
        self.env_state = env_state
        self.our_prior = our_prior
        self.our_legal_mask = our_legal_mask
        self.opp_prior = opp_prior
        self.opp_legal_mask = opp_legal_mask
        self.is_terminal = is_terminal
        self.terminal_value = terminal_value
        self.depth = depth

        # Children: action_idx → _MCTSNode or None
        self.children: list[Optional[_MCTSNode]] = [None] * N_ACTIONS
        # Per-action stats
        self.visit_count = np.zeros(N_ACTIONS, dtype=np.int32)
        self.value_sum = np.zeros(N_ACTIONS, dtype=np.float32)
        self.total_visits = 0

    @property
    def is_expanded(self) -> bool:
        """True if at least one legal child has been created."""
        return any(
            self.children[a] is not None
            for a in range(N_ACTIONS)
            if self.our_legal_mask[a] > 0
        )

    @property
    def is_fully_expanded(self) -> bool:
        """True if all legal children have been created."""
        return all(
            self.children[a] is not None
            for a in range(N_ACTIONS)
            if self.our_legal_mask[a] > 0
        )


class MCTSSearch:
    """GPU-parallel MCTS search using pokejax engine.

    JIT-compiles the expensive operations (env stepping + model inference)
    and runs tree operations on CPU.

    Parameters
    ----------
    env : PokeJAXEnv
        The pokejax environment.
    model : PokeTransformer
        Actor-critic model.
    params : dict
        Flax model parameters.
    n_simulations : int
        Number of MCTS simulations per search (default 128).
    c_puct : float
        PUCT exploration constant (default 2.5).
    opp_temperature : float
        Temperature for opponent policy sampling (default 0.5).
    max_depth : int
        Maximum search depth in turns (default 10).
    dirichlet_alpha : float
        Dirichlet noise alpha for root exploration (default 0.3).
    dirichlet_frac : float
        Fraction of Dirichlet noise mixed into root prior (default 0.25).
    batch_size : int
        Number of leaves to expand per GPU batch (default 8).
    warmup : bool
        If True, compile JIT kernels at init (default True).
    """

    def __init__(
        self,
        env: PokeJAXEnv,
        model: PokeTransformer,
        params,
        n_simulations: int = 128,
        c_puct: float = 2.5,
        opp_temperature: float = 0.5,
        max_depth: int = 10,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        batch_size: int = 8,
        warmup: bool = True,
    ):
        self.env = env
        self.model = model
        self.params = params
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.opp_temperature = opp_temperature
        self.max_depth = max_depth
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.batch_size = batch_size

        tables = env.tables

        # ── Kernel 1: Evaluate a state for both players ──
        @jax.jit
        def _eval_state(params, env_state):
            battle = env_state.battle
            reveal = env_state.reveal
            obs_p0 = build_obs_jax(battle, reveal, player=0, tables=tables)
            obs_p1 = build_obs_jax(battle, reveal, player=1, tables=tables)

            # Batch both players in a single forward pass
            int_ids = jnp.stack([obs_p0['int_ids'], obs_p1['int_ids']])
            float_feats = jnp.stack([obs_p0['float_feats'], obs_p1['float_feats']])
            legal_masks = jnp.stack([obs_p0['legal_mask'], obs_p1['legal_mask']])

            log_probs, _, values = model.apply(params, int_ids, float_feats, legal_masks)

            return (
                log_probs[0], values[0], obs_p0['legal_mask'],  # our
                log_probs[1], obs_p1['legal_mask'],             # opponent
            )

        self._eval_state = _eval_state

        # ── Kernel 2: Expand a single leaf (step env + evaluate) ──
        @jax.jit
        def _expand_leaf(params, env_state, our_action, opp_action, key):
            actions = jnp.array([our_action, opp_action], dtype=jnp.int32)
            new_env_state, rewards, dones = env.step_lean(env_state, actions, key)

            # Terminal check
            done = dones[0]
            reward = rewards[0]

            # Evaluate new state for both players
            battle = new_env_state.battle
            reveal = new_env_state.reveal
            obs_p0 = build_obs_jax(battle, reveal, player=0, tables=tables)
            obs_p1 = build_obs_jax(battle, reveal, player=1, tables=tables)

            int_ids = jnp.stack([obs_p0['int_ids'], obs_p1['int_ids']])
            float_feats = jnp.stack([obs_p0['float_feats'], obs_p1['float_feats']])
            legal_masks = jnp.stack([obs_p0['legal_mask'], obs_p1['legal_mask']])

            log_probs, _, values = model.apply(params, int_ids, float_feats, legal_masks)

            return (
                new_env_state,
                log_probs[0], values[0], obs_p0['legal_mask'],  # our
                log_probs[1], obs_p1['legal_mask'],             # opponent
                done, reward,
            )

        self._expand_leaf = _expand_leaf

        # ── Kernel 3: Batched expansion (step N envs + evaluate) ──
        @jax.jit
        def _expand_batch(params, env_states, our_actions, opp_actions, keys):
            """Expand B leaves in parallel."""
            actions = jnp.stack([our_actions, opp_actions], axis=-1)  # (B, 2)

            new_env_states, rewards_all, dones_all = jax.vmap(env.step_lean)(
                env_states, actions, keys,
            )
            dones = dones_all[:, 0]    # (B,)
            rewards = rewards_all[:, 0]  # (B,)

            # Build obs for both players for all B states
            def build_both(battle, reveal):
                obs_p0 = build_obs_jax(battle, reveal, player=0, tables=tables)
                obs_p1 = build_obs_jax(battle, reveal, player=1, tables=tables)
                return obs_p0, obs_p1

            obs_p0, obs_p1 = jax.vmap(build_both)(
                new_env_states.battle, new_env_states.reveal,
            )

            # Stack all B×2 observations and run model
            B = our_actions.shape[0]
            int_ids = jnp.concatenate([obs_p0['int_ids'], obs_p1['int_ids']])      # (2B, 15, 8)
            float_feats = jnp.concatenate([obs_p0['float_feats'], obs_p1['float_feats']])
            legal_masks = jnp.concatenate([obs_p0['legal_mask'], obs_p1['legal_mask']])

            log_probs, _, values = model.apply(params, int_ids, float_feats, legal_masks)

            our_lp = log_probs[:B]       # (B, 10)
            our_val = values[:B]         # (B,)
            our_legal = legal_masks[:B]  # (B, 10)
            opp_lp = log_probs[B:]       # (B, 10)
            opp_legal = legal_masks[B:]  # (B, 10)

            return (
                new_env_states,
                our_lp, our_val, our_legal,
                opp_lp, opp_legal,
                dones, rewards,
            )

        self._expand_batch = _expand_batch

        # ── Warm up JIT compilation ──
        if warmup:
            t0 = _time.time()
            print("[MCTS] Compiling search kernels (first run only, cached after)...",
                  flush=True)

            dummy_key = jax.random.PRNGKey(0)
            dummy_state, _ = env.reset(dummy_key)

            # Kernel 1: eval_state
            print("[MCTS]   1/2 eval_state...", flush=True)
            t1 = _time.time()
            out = _eval_state(params, dummy_state)
            out[0].block_until_ready()
            print(f"[MCTS]   1/2 done ({_time.time() - t1:.1f}s)", flush=True)

            # Kernel 2: expand_leaf
            print("[MCTS]   2/2 expand_leaf...", flush=True)
            t2 = _time.time()
            out2 = _expand_leaf(
                params, dummy_state, jnp.int32(0), jnp.int32(0), dummy_key,
            )
            out2[0].battle.turn.block_until_ready()
            print(f"[MCTS]   2/2 done ({_time.time() - t2:.1f}s)", flush=True)

            print(f"[MCTS] All kernels ready! Total: {_time.time() - t0:.1f}s",
                  flush=True)

    # ── PUCT action selection ──

    def _select_action(self, node: _MCTSNode) -> int:
        """Select action at node using PUCT formula."""
        N = node.visit_count.astype(np.float32)
        Q = np.where(N > 0, node.value_sum / np.maximum(N, 1), 0.0)

        # Softmax the prior logits (masked)
        prior_logits = node.our_prior.copy()
        prior_logits[node.our_legal_mask <= 0] = -1e9
        prior_logits -= prior_logits.max()
        exp_p = np.exp(prior_logits)
        P = exp_p / (exp_p.sum() + 1e-8)

        total_N = float(node.total_visits)
        exploration = self.c_puct * P * np.sqrt(total_N + 1) / (1 + N)
        score = Q + exploration

        # Mask illegal
        score[node.our_legal_mask <= 0] = -np.inf

        return int(np.argmax(score))

    def _sample_opponent_action(
        self, node: _MCTSNode, rng: np.random.Generator,
    ) -> int:
        """Sample opponent action from policy prior with temperature."""
        logits = node.opp_prior.copy()
        logits[node.opp_legal_mask <= 0] = -1e9

        if self.opp_temperature > 0:
            logits = logits / max(self.opp_temperature, 1e-4)

        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / (exp_l.sum() + 1e-8)
        probs = probs * (node.opp_legal_mask > 0).astype(np.float32)
        total = probs.sum()
        if total < 1e-8:
            # Fallback: uniform over legal
            legal = (node.opp_legal_mask > 0).astype(np.float32)
            probs = legal / max(legal.sum(), 1)
        else:
            probs = probs / total

        return int(rng.choice(N_ACTIONS, p=probs))

    def _add_dirichlet_noise(self, node: _MCTSNode, rng: np.random.Generator):
        """Add Dirichlet noise to root prior for exploration."""
        legal = node.our_legal_mask > 0
        n_legal = int(legal.sum())
        if n_legal <= 1:
            return

        noise = np.zeros(N_ACTIONS, dtype=np.float32)
        noise[legal] = rng.dirichlet([self.dirichlet_alpha] * n_legal)

        # Mix noise into prior logits (convert prior to probs, mix, convert back)
        prior_logits = node.our_prior.copy()
        prior_logits[~legal] = -1e9
        prior_logits -= prior_logits.max()
        exp_p = np.exp(prior_logits)
        probs = exp_p / (exp_p.sum() + 1e-8)

        mixed = (1 - self.dirichlet_frac) * probs + self.dirichlet_frac * noise
        mixed = np.maximum(mixed, 1e-8)
        node.our_prior = np.log(mixed)

    # ── Core search ──

    def search(
        self,
        env_state: EnvState,
        key: jnp.ndarray,
    ) -> SearchResult:
        """Run MCTS search from the given state.

        Parameters
        ----------
        env_state : EnvState
            Current game state (from BattleBridge or JAX env).
        key : jnp.ndarray
            JAX PRNG key for stochastic simulations.

        Returns
        -------
        SearchResult
            best_action, action_values, root_value, search_policy
        """
        rng = np.random.default_rng(int(key[0]))

        # Evaluate root state
        our_lp, root_val, our_mask, opp_lp, opp_mask = self._eval_state(
            self.params, env_state,
        )
        our_lp = np.array(our_lp)
        root_value = float(root_val)
        our_mask = np.array(our_mask)
        opp_lp = np.array(opp_lp)
        opp_mask = np.array(opp_mask)

        # Create root node
        root = _MCTSNode(
            env_state=env_state,
            our_prior=our_lp,
            our_legal_mask=our_mask,
            opp_prior=opp_lp,
            opp_legal_mask=opp_mask,
            depth=0,
        )

        # Add Dirichlet noise to root
        self._add_dirichlet_noise(root, rng)

        # ── Run simulations ──
        key, sim_key = jax.random.split(key)
        sim_keys = jax.random.split(sim_key, self.n_simulations)

        for sim_idx in range(self.n_simulations):
            sim_rng_key = sim_keys[sim_idx]

            # SELECT: walk tree via PUCT
            node = root
            path = []  # list of (node, action)

            while not node.is_terminal and node.depth < self.max_depth:
                action = self._select_action(node)
                path.append((node, action))

                child = node.children[action]
                if child is None:
                    # Unexpanded → expand this child
                    break
                node = child

            if node.is_terminal:
                # Terminal node: use actual game outcome
                value = node.terminal_value
            elif node.depth >= self.max_depth:
                # Max depth: use value estimate (Q or network value)
                value = root_value  # fallback
            elif path:
                # EXPAND the leaf
                parent_node, parent_action = path[-1]

                # Sample opponent action
                opp_action = self._sample_opponent_action(parent_node, rng)

                # Step env + evaluate (GPU)
                (
                    new_env_state,
                    child_our_lp, child_val, child_our_mask,
                    child_opp_lp, child_opp_mask,
                    done, reward,
                ) = self._expand_leaf(
                    self.params,
                    parent_node.env_state,
                    jnp.int32(parent_action),
                    jnp.int32(opp_action),
                    sim_rng_key,
                )

                is_terminal = bool(done)
                terminal_value = float(reward) if is_terminal else 0.0

                # Value for backup: terminal reward if done, else network value
                if is_terminal:
                    value = terminal_value
                else:
                    value = float(child_val)

                # Create child node
                child = _MCTSNode(
                    env_state=new_env_state,
                    our_prior=np.array(child_our_lp),
                    our_legal_mask=np.array(child_our_mask),
                    opp_prior=np.array(child_opp_lp),
                    opp_legal_mask=np.array(child_opp_mask),
                    is_terminal=is_terminal,
                    terminal_value=terminal_value,
                    depth=parent_node.depth + 1,
                )
                parent_node.children[parent_action] = child
            else:
                # Shouldn't happen, but safety
                value = root_value

            # BACKUP: propagate value up the path
            for bp_node, bp_action in reversed(path):
                bp_node.visit_count[bp_action] += 1
                bp_node.value_sum[bp_action] += value
                bp_node.total_visits += 1

        # ── Extract results from root ──
        visit_counts = root.visit_count.astype(np.float32)
        total = visit_counts.sum()

        # Action values (mean Q)
        action_values = np.where(
            root.visit_count > 0,
            root.value_sum / np.maximum(root.visit_count.astype(np.float32), 1),
            -1e9,
        )
        action_values[our_mask <= 0] = -1e9

        # Best action: most visited
        best_action = int(np.argmax(visit_counts))

        # Search policy: normalized visit counts
        search_policy = np.zeros(N_ACTIONS, dtype=np.float32)
        if total > 0:
            search_policy = visit_counts / total

        return SearchResult(
            best_action=best_action,
            action_values=action_values,
            root_value=root_value,
            search_policy=search_policy,
        )

    def search_batched(
        self,
        env_state: EnvState,
        key: jnp.ndarray,
    ) -> SearchResult:
        """Run MCTS with virtual loss for path diversity.

        Uses virtual loss to select multiple paths before processing them,
        but expands leaves individually (avoiding expensive pytree stacking).
        This gives better path diversity than pure sequential while avoiding
        the overhead of batched pytree operations.

        Parameters
        ----------
        env_state : EnvState
            Current game state.
        key : jnp.ndarray
            JAX PRNG key.

        Returns
        -------
        SearchResult
        """
        rng = np.random.default_rng(int(key[0]))
        B = self.batch_size

        # Evaluate root state
        our_lp, root_val, our_mask, opp_lp, opp_mask = self._eval_state(
            self.params, env_state,
        )
        our_lp = np.array(our_lp)
        root_value = float(root_val)
        our_mask = np.array(our_mask)
        opp_lp = np.array(opp_lp)
        opp_mask = np.array(opp_mask)

        # Create root node
        root = _MCTSNode(
            env_state=env_state,
            our_prior=our_lp,
            our_legal_mask=our_mask,
            opp_prior=opp_lp,
            opp_legal_mask=opp_mask,
            depth=0,
        )
        self._add_dirichlet_noise(root, rng)

        VIRTUAL_LOSS = 3.0

        key, sim_key = jax.random.split(key)
        sim_keys = jax.random.split(sim_key, self.n_simulations)
        sim_idx = 0

        while sim_idx < self.n_simulations:
            # Collect paths with virtual loss for diversity
            pending = []  # (path, parent_node, our_action, opp_action, key)

            for _ in range(min(B, self.n_simulations - sim_idx)):
                node = root
                path = []

                while not node.is_terminal and node.depth < self.max_depth:
                    action = self._select_action(node)
                    path.append((node, action))
                    child = node.children[action]
                    if child is None:
                        break
                    node = child

                if not path:
                    sim_idx += 1
                    continue

                parent_node, parent_action = path[-1]

                if node.is_terminal:
                    value = node.terminal_value
                    for bp_node, bp_action in reversed(path):
                        bp_node.visit_count[bp_action] += 1
                        bp_node.value_sum[bp_action] += value
                        bp_node.total_visits += 1
                    sim_idx += 1
                    continue

                if parent_node.children[parent_action] is not None:
                    value = root_value
                    for bp_node, bp_action in reversed(path):
                        bp_node.visit_count[bp_action] += 1
                        bp_node.value_sum[bp_action] += value
                        bp_node.total_visits += 1
                    sim_idx += 1
                    continue

                # Apply virtual loss
                for bp_node, bp_action in path:
                    bp_node.visit_count[bp_action] += 1
                    bp_node.value_sum[bp_action] -= VIRTUAL_LOSS
                    bp_node.total_visits += 1

                opp_action = self._sample_opponent_action(parent_node, rng)
                pending.append((
                    path, parent_node, parent_action,
                    opp_action, sim_keys[sim_idx],
                ))
                sim_idx += 1

            # Expand all pending leaves individually (avoids pytree stacking)
            for path, parent_node, parent_action, opp_action, rng_key in pending:
                (
                    new_env_state,
                    child_our_lp, child_val, child_our_mask,
                    child_opp_lp, child_opp_mask,
                    done, reward,
                ) = self._expand_leaf(
                    self.params,
                    parent_node.env_state,
                    jnp.int32(parent_action),
                    jnp.int32(opp_action),
                    rng_key,
                )

                is_terminal = bool(done)
                terminal_value = float(reward) if is_terminal else 0.0
                value = terminal_value if is_terminal else float(child_val)

                child = _MCTSNode(
                    env_state=new_env_state,
                    our_prior=np.array(child_our_lp),
                    our_legal_mask=np.array(child_our_mask),
                    opp_prior=np.array(child_opp_lp),
                    opp_legal_mask=np.array(child_opp_mask),
                    is_terminal=is_terminal,
                    terminal_value=terminal_value,
                    depth=parent_node.depth + 1,
                )
                parent_node.children[parent_action] = child

                # Remove virtual loss and apply real value
                for bp_node, bp_action in reversed(path):
                    bp_node.value_sum[bp_action] += VIRTUAL_LOSS + value

        # ── Extract results ──
        visit_counts = root.visit_count.astype(np.float32)
        total = visit_counts.sum()

        action_values = np.where(
            root.visit_count > 0,
            root.value_sum / np.maximum(root.visit_count.astype(np.float32), 1),
            -1e9,
        )
        action_values[our_mask <= 0] = -1e9

        best_action = int(np.argmax(visit_counts))

        search_policy = np.zeros(N_ACTIONS, dtype=np.float32)
        if total > 0:
            search_policy = visit_counts / total

        return SearchResult(
            best_action=best_action,
            action_values=action_values,
            root_value=root_value,
            search_policy=search_policy,
        )
