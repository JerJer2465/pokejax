"""
Behavioral Cloning (BC) for PokeJAX.

Collects (obs, expert_action) pairs from heuristic vs random battles,
then trains the PokeTransformer via supervised cross-entropy loss.

This module runs on CPU/GPU but does NOT use jax.jit for the data collection
loop (the heuristic is Python-level). The training loop IS jit-compiled.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from pokejax.rl.obs_builder import build_obs
from pokejax.rl.model import PokeTransformer
from pokejax.env.pokejax_env import PokeJAXEnv, EnvState
from pokejax.env.heuristic import smart_heuristic_action, random_action, _state_to_numpy
from pokejax.data.tables import Tables


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BCConfig:
    lr: float = 3e-4
    lr_end: float = 1e-5
    batch_size: int = 1024
    max_grad_norm: float = 0.5
    total_steps: int = 500_000     # total transitions to train on
    eval_interval: int = 50        # eval every N updates
    eval_games: int = 50           # games per eval
    collect_batch: int = 2048      # transitions per collection batch
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10         # log every N updates


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

class BCBatch(NamedTuple):
    """A batch of (obs, action) pairs for supervised learning."""
    int_ids: np.ndarray      # (B, 15, 8)
    float_feats: np.ndarray  # (B, 15, 394)
    legal_mask: np.ndarray   # (B, 10)
    actions: np.ndarray      # (B,) int32


def collect_bc_data(
    env: PokeJAXEnv,
    n_transitions: int,
    seed: int = 0,
    teacher_side: int = 0,
    verbose: bool = True,
) -> BCBatch:
    """
    Collect BC training data by running heuristic (teacher) vs random (opponent).

    teacher_side: which side the heuristic plays (0 or 1).
    The opponent plays random legal moves.

    Returns BCBatch with n_transitions samples.
    """
    tables = env.tables
    rng = np.random.RandomState(seed)
    key = jax.random.PRNGKey(seed)

    all_int_ids = []
    all_float_feats = []
    all_legal_mask = []
    all_actions = []

    collected = 0
    games = 0
    opp_side = 1 - teacher_side

    # Pre-JIT the step and obs functions for speed
    @jax.jit
    def jit_step(env_state, actions, step_key):
        return env.step(env_state, actions, step_key)

    @jax.jit
    def jit_obs(battle, reveal):
        return build_obs(battle, reveal, teacher_side, tables)

    # Warm up JIT with a dummy run
    if verbose:
        print("  JIT compiling step + obs (first time only)...")
    key, warmup_key = jax.random.split(key)
    warmup_state, _ = env.reset(warmup_key)
    warmup_obs = jit_obs(warmup_state.battle, warmup_state.reveal)
    warmup_actions = jnp.array([0, 0], dtype=jnp.int32)
    key, warmup_step_key = jax.random.split(key)
    _ = jit_step(warmup_state, warmup_actions, warmup_step_key)
    # Force compilation to complete
    jax.block_until_ready(warmup_obs["int_ids"])
    if verbose:
        print("  JIT compiled. Collecting data...")

    import time as _time
    t0 = _time.time()

    while collected < n_transitions:
        # Reset environment
        key, reset_key = jax.random.split(key)
        env_state, _ = env.reset(reset_key)
        state = env_state.battle
        reveal = env_state.reveal

        turn = 0
        while not bool(state.finished) and turn < 300:
            # Build observation for teacher (JIT-compiled)
            obs = jit_obs(state, reveal)

            # Bulk convert state to numpy once per turn
            np_state = _state_to_numpy(state)

            # Teacher picks action (CPU heuristic, uses cached numpy)
            teacher_action = smart_heuristic_action(state, teacher_side, tables, _np_cache=np_state)

            # Random opponent picks action
            opp_action = random_action(state, opp_side)

            # Record (obs, action)
            all_int_ids.append(np.array(obs["int_ids"]))
            all_float_feats.append(np.array(obs["float_feats"]))
            all_legal_mask.append(np.array(obs["legal_mask"]))
            all_actions.append(teacher_action)
            collected += 1

            # Step environment (JIT-compiled)
            actions = jnp.array([0, 0], dtype=jnp.int32)
            actions = actions.at[teacher_side].set(teacher_action)
            actions = actions.at[opp_side].set(opp_action)

            key, step_key = jax.random.split(key)
            env_state, _, _, _, _ = jit_step(env_state, actions, step_key)
            state = env_state.battle
            reveal = env_state.reveal
            turn += 1

            if collected >= n_transitions:
                break

        games += 1
        if verbose and games % 50 == 0:
            elapsed = _time.time() - t0
            rate = collected / max(elapsed, 0.001)
            eta = (n_transitions - collected) / max(rate, 0.001)
            print(f"  {collected}/{n_transitions} from {games} games "
                  f"({rate:.0f} trans/s, ETA {eta:.0f}s)")

    if verbose:
        print(f"  Done: {collected} transitions from {games} games")

    return BCBatch(
        int_ids=np.stack(all_int_ids[:n_transitions]),
        float_feats=np.stack(all_float_feats[:n_transitions]),
        legal_mask=np.stack(all_legal_mask[:n_transitions]),
        actions=np.array(all_actions[:n_transitions], dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# BC loss function (JIT-compatible)
# ---------------------------------------------------------------------------

def bc_loss(params, model, int_ids, float_feats, legal_mask, actions):
    """Cross-entropy loss: -log p(expert_action | obs).

    Handles edge case where expert action may be masked (illegal) by
    temporarily making it legal in the mask.
    """
    # Ensure expert action is always legal in the mask
    # This handles edge cases where mask/heuristic disagree
    B = actions.shape[0]
    action_onehot = jax.nn.one_hot(actions, 10)
    safe_mask = jnp.maximum(legal_mask, action_onehot)

    log_probs, _, _ = model.apply(params, int_ids, float_feats, safe_mask)
    # log_probs: (B, 10)
    # actions: (B,)
    action_log_probs = jnp.take_along_axis(
        log_probs, actions[:, None], axis=1
    ).squeeze(1)  # (B,)

    # Clip to avoid extreme values from numerical issues
    action_log_probs = jnp.clip(action_log_probs, -20.0, 0.0)
    loss = -action_log_probs.mean()

    # Metrics
    preds = jnp.argmax(log_probs, axis=-1)
    accuracy = (preds == actions).mean()
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()

    return loss, {"loss": loss, "accuracy": accuracy, "entropy": entropy}


# ---------------------------------------------------------------------------
# BC training state & step
# ---------------------------------------------------------------------------

class BCTrainState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    step: jnp.ndarray


def create_bc_train_state(
    model: PokeTransformer,
    cfg: BCConfig,
    key: jnp.ndarray,
    init_params: Optional[dict] = None,
) -> tuple:
    """Initialize model, optimizer, and train state for BC."""
    if init_params is None:
        B = 1
        dummy_int = jnp.zeros((B, 15, 8), dtype=jnp.int32)
        dummy_float = jnp.zeros((B, 15, 394), dtype=jnp.float32)
        dummy_mask = jnp.ones((B, 10), dtype=jnp.float32)
        init_params = model.init(key, dummy_int, dummy_float, dummy_mask)

    n_updates = cfg.total_steps // cfg.batch_size
    # alpha controls the minimum: lr * alpha at the end
    alpha = cfg.lr_end / max(cfg.lr, 1e-10)
    lr_schedule = optax.cosine_decay_schedule(
        cfg.lr, decay_steps=max(n_updates, 1), alpha=alpha,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(init_params)

    train_state = BCTrainState(
        params=init_params,
        opt_state=opt_state,
        step=jnp.int32(0),
    )
    return model, optimizer, train_state


def make_bc_step(model, optimizer):
    """Return a JIT-compiled BC gradient step function."""

    @jax.jit
    def _step(train_state: BCTrainState, int_ids, float_feats, legal_mask, actions):
        loss_fn = lambda p: bc_loss(p, model, int_ids, float_feats, legal_mask, actions)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params
        )
        updates, new_opt_state = optimizer.update(
            grads, train_state.opt_state, train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)
        new_state = BCTrainState(
            params=new_params,
            opt_state=new_opt_state,
            step=train_state.step + 1,
        )
        return new_state, metrics

    return _step


# ---------------------------------------------------------------------------
# Evaluation: win rate vs random
# ---------------------------------------------------------------------------

def eval_vs_random(
    model,
    params: dict,
    env: PokeJAXEnv,
    n_games: int = 50,
    seed: int = 9999,
) -> dict:
    """
    Evaluate model win rate vs random opponent.
    Model plays as side 0, random plays as side 1.
    Returns dict with win_rate, avg_turns.
    """
    tables = env.tables
    key = jax.random.PRNGKey(seed)

    wins = 0
    total_turns = 0

    # JIT the model forward pass, env step, and obs builder
    @jax.jit
    def get_action(params, int_ids, float_feats, legal_mask):
        log_probs, _, _ = model.apply(
            params,
            int_ids[None],
            float_feats[None],
            legal_mask[None],
        )
        return jnp.argmax(log_probs[0])

    @jax.jit
    def jit_step(env_state, actions, step_key):
        return env.step(env_state, actions, step_key)

    @jax.jit
    def jit_obs(battle, reveal):
        return build_obs(battle, reveal, 0, tables)

    for g in range(n_games):
        key, reset_key = jax.random.split(key)
        env_state, _ = env.reset(reset_key)
        state = env_state.battle
        reveal = env_state.reveal

        turn = 0
        while not bool(state.finished) and turn < 300:
            # Model picks action for side 0
            obs = jit_obs(state, reveal)
            model_action = int(get_action(params, obs["int_ids"], obs["float_feats"], obs["legal_mask"]))

            # Random picks for side 1
            opp_action = random_action(state, 1)

            actions = jnp.array([model_action, opp_action], dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            env_state, _, _, _, _ = jit_step(env_state, actions, step_key)
            state = env_state.battle
            reveal = env_state.reveal
            turn += 1

        if bool(state.finished) and int(state.winner) == 0:
            wins += 1
        total_turns += turn

    return {
        "win_rate": wins / max(n_games, 1),
        "avg_turns": total_turns / max(n_games, 1),
    }


def eval_vs_heuristic(
    model,
    params: dict,
    env: PokeJAXEnv,
    n_games: int = 50,
    seed: int = 7777,
) -> dict:
    """
    Evaluate model win rate vs heuristic opponent.
    Model plays as side 0, heuristic plays as side 1.
    """
    tables = env.tables
    key = jax.random.PRNGKey(seed)

    wins = 0
    total_turns = 0

    @jax.jit
    def get_action(params, int_ids, float_feats, legal_mask):
        log_probs, _, _ = model.apply(
            params,
            int_ids[None],
            float_feats[None],
            legal_mask[None],
        )
        return jnp.argmax(log_probs[0])

    @jax.jit
    def jit_step(env_state, actions, step_key):
        return env.step(env_state, actions, step_key)

    @jax.jit
    def jit_obs(battle, reveal):
        return build_obs(battle, reveal, 0, tables)

    for g in range(n_games):
        key, reset_key = jax.random.split(key)
        env_state, _ = env.reset(reset_key)
        state = env_state.battle
        reveal = env_state.reveal

        turn = 0
        while not bool(state.finished) and turn < 300:
            # Model picks action for side 0
            obs = jit_obs(state, reveal)
            model_action = int(get_action(params, obs["int_ids"], obs["float_feats"], obs["legal_mask"]))

            # Heuristic picks for side 1
            opp_action = smart_heuristic_action(state, 1, tables)

            actions = jnp.array([model_action, opp_action], dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            env_state, _, _, _, _ = jit_step(env_state, actions, step_key)
            state = env_state.battle
            reveal = env_state.reveal
            turn += 1

        if bool(state.finished) and int(state.winner) == 0:
            wins += 1
        total_turns += turn

    return {
        "win_rate": wins / max(n_games, 1),
        "avg_turns": total_turns / max(n_games, 1),
    }
