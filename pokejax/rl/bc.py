"""
Behavioral Cloning (BC) for PokeJAX.

Collects (obs, expert_action) pairs from heuristic vs random/heuristic battles,
then trains the PokeTransformer via supervised cross-entropy loss.

Data collection is fully vectorized (vmap + lax.scan) for GPU speed.
The training loop is also JIT-compiled.
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
from pokejax.rl.heuristic import heuristic_action, random_action
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


# ---------------------------------------------------------------------------
# Vectorized BC data collection (vmap + lax.scan, fully on GPU)
# ---------------------------------------------------------------------------

class _BCStepCarry(NamedTuple):
    env_state: EnvState
    key: jnp.ndarray

class _BCStepOutput(NamedTuple):
    int_ids: jnp.ndarray      # (15, 8)
    float_feats: jnp.ndarray  # (15, 394)
    legal_mask: jnp.ndarray   # (10,)
    action: jnp.ndarray       # scalar int32
    valid: jnp.ndarray        # scalar bool (True if game was active)


def collect_bc_data_vectorized(
    env: PokeJAXEnv,
    n_envs: int = 512,
    n_steps: int = 256,
    seed: int = 0,
    teacher_side: int = 0,
    opp_mode: str = "random",
    verbose: bool = True,
) -> BCBatch:
    """
    Collect BC training data fully on GPU using vmap + lax.scan.

    Runs n_envs games in parallel for n_steps each, collecting
    (obs, heuristic_action) pairs at every step. Auto-resets finished
    games so all steps produce useful data.

    Args:
        env: PokeJAXEnv instance
        n_envs: number of parallel environments (vmap width)
        n_steps: steps per environment (lax.scan length)
        seed: random seed
        teacher_side: which side the heuristic plays (0 or 1)
        opp_mode: "random" or "heuristic" — opponent policy
        verbose: print progress

    Returns:
        BCBatch with n_envs * n_steps transitions
    """
    tables = env.tables
    opp_side = 1 - teacher_side

    # Pre-build heuristic move categories BEFORE JIT to avoid tracer leaks.
    # The heuristic caches these by id(tables), but during JIT tracing
    # the cached values become stale tracers. Force-build them as concrete
    # JAX arrays here so they are captured as constants in the JIT closure.
    from pokejax.rl.heuristic import _build_move_categories
    _build_move_categories(tables)

    def _bc_step(carry: _BCStepCarry, _) -> tuple[_BCStepCarry, _BCStepOutput]:
        env_state, key = carry
        key, heur_key, opp_key, step_key, reset_key = jax.random.split(key, 5)

        state = env_state.battle
        reveal = env_state.reveal

        # Build observation for teacher
        obs = build_obs(state, reveal, teacher_side, tables)

        # Teacher picks action via heuristic
        teacher_action = heuristic_action(state, teacher_side, tables, heur_key)

        # Opponent picks action
        if opp_mode == "heuristic":
            opp_action = heuristic_action(state, opp_side, tables, opp_key)
        else:
            opp_action = random_action(state, opp_side, opp_key)

        # Assemble actions array
        actions = jnp.zeros(2, dtype=jnp.int32)
        actions = actions.at[teacher_side].set(teacher_action)
        actions = actions.at[opp_side].set(opp_action)

        # Track whether game is active (valid data point)
        valid = ~state.finished

        # Step environment
        new_env_state, _, _, _, _ = env.step(env_state, actions, step_key)

        # Auto-reset if game finished
        reset_state, _ = env.reset(reset_key)
        done = new_env_state.battle.finished
        final_env_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n),
            reset_state, new_env_state,
        )

        output = _BCStepOutput(
            int_ids=obs["int_ids"],
            float_feats=obs["float_feats"],
            legal_mask=obs["legal_mask"],
            action=teacher_action.astype(jnp.int32),
            valid=valid,
        )

        return _BCStepCarry(env_state=final_env_state, key=key), output

    # Build the full vectorized collection function.
    # Use a fixed chunk size (32 envs × 64 steps) that compiles quickly,
    # then run multiple chunks to reach the desired total.
    CHUNK_ENVS = min(n_envs, 32)
    CHUNK_STEPS = min(n_steps, 64)
    n_rounds = max(1, (n_envs * n_steps) // (CHUNK_ENVS * CHUNK_STEPS))

    def _collect_one_env(init_carry: _BCStepCarry) -> _BCStepOutput:
        _, outputs = jax.lax.scan(_bc_step, init_carry, None, length=CHUNK_STEPS)
        return outputs

    if verbose:
        chunk_total = CHUNK_ENVS * CHUNK_STEPS
        print(f"  Vectorized BC: {n_rounds} rounds × "
              f"({CHUNK_ENVS} envs × {CHUNK_STEPS} steps = {chunk_total}), "
              f"opp={opp_mode}")

    import time as _time

    # JIT the chunk-sized vmap+scan pipeline (compiles once, reused for all rounds)
    @jax.jit
    def _collect_chunk(key):
        env_keys = jax.random.split(key, CHUNK_ENVS + 1)
        master_key, env_keys = env_keys[0], env_keys[1:]

        init_states, _ = jax.vmap(env.reset)(env_keys)

        scan_keys = jax.random.split(master_key, CHUNK_ENVS)
        init_carries = jax.vmap(lambda s, k: _BCStepCarry(env_state=s, key=k))(
            init_states, scan_keys,
        )

        outputs = jax.vmap(_collect_one_env)(init_carries)
        return outputs  # each field: (CHUNK_ENVS, CHUNK_STEPS, ...)

    # Warmup / compile
    if verbose:
        print(f"  JIT compiling chunk ({CHUNK_ENVS} × {CHUNK_STEPS})...")
    t0 = _time.time()
    key = jax.random.PRNGKey(seed)
    key, warmup_key = jax.random.split(key)
    warmup_out = _collect_chunk(warmup_key)
    jax.block_until_ready(warmup_out.action)
    if verbose:
        print(f"  Compiled in {_time.time() - t0:.1f}s. Collecting {n_rounds} rounds...")

    # Run all rounds, accumulating results on CPU
    all_int_ids = []
    all_float_feats = []
    all_legal_mask = []
    all_actions = []
    all_valid = []

    t0 = _time.time()
    for r in range(n_rounds):
        key, round_key = jax.random.split(key)
        outputs = _collect_chunk(round_key)

        chunk_sz = CHUNK_ENVS * CHUNK_STEPS
        all_int_ids.append(np.array(outputs.int_ids.reshape(chunk_sz, 15, 8)))
        all_float_feats.append(np.array(outputs.float_feats.reshape(chunk_sz, 15, 394)))
        all_legal_mask.append(np.array(outputs.legal_mask.reshape(chunk_sz, 10)))
        all_actions.append(np.array(outputs.action.reshape(chunk_sz)))
        all_valid.append(np.array(outputs.valid.reshape(chunk_sz)))

        if verbose and (r + 1) % max(1, n_rounds // 10) == 0:
            elapsed = _time.time() - t0
            collected = (r + 1) * chunk_sz
            rate = collected / max(elapsed, 0.001)
            print(f"  Round {r+1}/{n_rounds}: {collected:,} transitions "
                  f"({rate:.0f} trans/s)")

    elapsed = _time.time() - t0
    total_collected = n_rounds * CHUNK_ENVS * CHUNK_STEPS
    if verbose:
        print(f"  Collection done in {elapsed:.1f}s ({total_collected:,} total, "
              f"{total_collected/max(elapsed,0.001):.0f} trans/s)")

    # Concatenate and filter valid
    int_ids_flat = np.concatenate(all_int_ids)
    float_feats_flat = np.concatenate(all_float_feats)
    legal_mask_flat = np.concatenate(all_legal_mask)
    actions_flat = np.concatenate(all_actions)
    valid_flat = np.concatenate(all_valid)

    valid_idx = np.where(valid_flat)[0]
    n_valid = len(valid_idx)
    if verbose:
        print(f"  Valid transitions: {n_valid}/{total_collected} "
              f"({100*n_valid/total_collected:.1f}%)")

    return BCBatch(
        int_ids=int_ids_flat[valid_idx],
        float_feats=float_feats_flat[valid_idx],
        legal_mask=legal_mask_flat[valid_idx],
        actions=actions_flat[valid_idx].astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Legacy sequential collection (kept for compatibility)
# ---------------------------------------------------------------------------

def collect_bc_data(
    env: PokeJAXEnv,
    n_transitions: int,
    seed: int = 0,
    teacher_side: int = 0,
    verbose: bool = True,
) -> BCBatch:
    """
    Collect BC training data by running heuristic (teacher) vs random (opponent).

    Uses JIT-compiled heuristic and random opponent for GPU acceleration.
    teacher_side: which side the heuristic plays (0 or 1).

    Returns BCBatch with n_transitions samples.

    NOTE: Prefer collect_bc_data_vectorized() for much faster GPU collection.
    """
    tables = env.tables
    key = jax.random.PRNGKey(seed)

    all_int_ids = []
    all_float_feats = []
    all_legal_mask = []
    all_actions = []

    collected = 0
    games = 0
    opp_side = 1 - teacher_side

    # Pre-JIT all functions for GPU speed
    @jax.jit
    def jit_step(env_state, actions, step_key):
        return env.step(env_state, actions, step_key)

    @jax.jit
    def jit_obs(battle, reveal):
        return build_obs(battle, reveal, teacher_side, tables)

    @jax.jit
    def jit_heuristic(battle, rng_key):
        return heuristic_action(battle, teacher_side, tables, rng_key)

    @jax.jit
    def jit_random(battle, rng_key):
        return random_action(battle, opp_side, rng_key)

    # Warm up JIT with a dummy run
    if verbose:
        print("  JIT compiling step + obs + heuristic (first time only)...")
    key, warmup_key = jax.random.split(key)
    warmup_state, _ = env.reset(warmup_key)
    warmup_obs = jit_obs(warmup_state.battle, warmup_state.reveal)
    key, hk = jax.random.split(key)
    warmup_teacher = jit_heuristic(warmup_state.battle, hk)
    key, rk = jax.random.split(key)
    warmup_random = jit_random(warmup_state.battle, rk)
    warmup_actions = jnp.array([0, 0], dtype=jnp.int32)
    key, warmup_step_key = jax.random.split(key)
    _ = jit_step(warmup_state, warmup_actions, warmup_step_key)
    # Force compilation to complete
    jax.block_until_ready(warmup_obs["int_ids"])
    jax.block_until_ready(warmup_teacher)
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

            # Teacher picks action (JIT heuristic on GPU)
            key, heur_key = jax.random.split(key)
            teacher_action = jit_heuristic(state, heur_key)

            # Random opponent picks action (JIT on GPU)
            key, rand_key = jax.random.split(key)
            opp_action = jit_random(state, rand_key)

            # Record (obs, action)
            all_int_ids.append(np.array(obs["int_ids"]))
            all_float_feats.append(np.array(obs["float_feats"]))
            all_legal_mask.append(np.array(obs["legal_mask"]))
            all_actions.append(int(teacher_action))
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

    @jax.jit
    def jit_random(battle, rng_key):
        return random_action(battle, 1, rng_key)

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

            # Random picks for side 1 (JIT)
            key, rand_key = jax.random.split(key)
            opp_action = int(jit_random(state, rand_key))

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

    @jax.jit
    def jit_heuristic(battle, rng_key):
        return heuristic_action(battle, 1, tables, rng_key)

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

            # Heuristic picks for side 1 (JIT)
            key, heur_key = jax.random.split(key)
            opp_action = int(jit_heuristic(state, heur_key))

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
