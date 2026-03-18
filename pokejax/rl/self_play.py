"""
Self-play trainer: wraps PPO + rollout into a training loop.

Features:
  - TensorBoard logging (losses, SPS, win rates, LR, diagnostics)
  - Historical opponent pool (asymmetric self-play)
  - Periodic eval vs heuristic + random
  - Episode tracking (win rate, episode length)
  - Checkpoint saving with configurable interval
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os
import pickle
import subprocess
import time
import random as py_random

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pokejax.rl.model import PokeTransformer, MODEL_CONFIG
from pokejax.rl.ppo import PPOConfig, TrainState, create_train_state, ppo_step, make_jit_ppo_epochs
from pokejax.rl.rollout import (
    RolloutConfig, collect_rollout, make_jit_rollout, RolloutBatch,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    ppo:     PPOConfig     = field(default_factory=PPOConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    total_timesteps: int   = 250_000_000
    log_interval:    int   = 1       # PPO updates between TB logs (log every update)
    print_interval:  int   = 10      # PPO updates between stdout prints
    checkpoint_interval: int = 100   # PPO updates between checkpoints
    checkpoint_dir:  str   = "checkpoints"
    tensorboard_dir: str   = "runs"

    # Eval config
    eval_interval:   int   = 50      # PPO updates between eval rounds
    eval_games:      int   = 64      # number of eval games (vs heuristic + vs random)

    # PS server eval (requires local Showdown server running)
    ps_eval:          bool  = True    # enable eval on local PS server
    ps_eval_interval: int   = 100    # PPO updates between PS eval rounds
    ps_eval_games:    int   = 20     # games per PS eval round

    # Opponent pool config
    pool_size:         int   = 20     # max historical checkpoints in pool
    pool_save_interval: int  = 50     # save to pool every N updates
    pool_latest_ratio:  float = 0.75  # probability of playing against latest vs pool

    # LR warmup
    lr_warmup_steps: int = 1000      # linear warmup steps before cosine decay
    lr_min:          float = 1e-5    # minimum LR floor (cosine decay alpha)


# ---------------------------------------------------------------------------
# Opponent pool
# ---------------------------------------------------------------------------

class OpponentPool:
    """Maintains a ring buffer of historical parameter checkpoints."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.pool: list[dict] = []  # list of param dicts
        self.pool_steps: list[int] = []  # update step when added

    def add(self, params: dict, step: int):
        """Add a checkpoint to the pool. Evicts oldest if full."""
        # Copy params to CPU numpy to save GPU memory
        params_np = jax.tree.map(lambda x: np.array(x), params)
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)
            self.pool_steps.pop(0)
        self.pool.append(params_np)
        self.pool_steps.append(step)

    def sample(self) -> Optional[dict]:
        """Sample a random checkpoint from the pool. Returns JAX arrays."""
        if not self.pool:
            return None
        idx = py_random.randint(0, len(self.pool) - 1)
        return jax.tree.map(lambda x: jnp.array(x), self.pool[idx])

    def __len__(self):
        return len(self.pool)


# ---------------------------------------------------------------------------
# Episode stats extraction
# ---------------------------------------------------------------------------

def _pull_scalars(metrics: dict, info: dict) -> dict:
    """Pull all scalars from GPU to CPU in one batch.

    Consolidates all float() calls into a single sync point to minimize
    CPU↔GPU transfer overhead. Called only at print_interval, not every step.
    """
    # Gather all JAX scalars into a single dict
    all_vals = {}
    for k, v in metrics.items():
        all_vals[f"m/{k}"] = v
    for k, v in info.items():
        all_vals[f"i/{k}"] = v

    # Single block_until_ready on all values, then convert
    leaves = list(all_vals.values())
    jax.block_until_ready(leaves)
    return {k: float(v) for k, v in all_vals.items()}


# ---------------------------------------------------------------------------
# Eval: JIT-batched eval vs heuristic and random
# ---------------------------------------------------------------------------

# Module-level cache for eval JIT functions (avoid recompilation)
_EVAL_JIT_CACHE = {}


def _get_eval_step_fn(model, env, tables, n_games, opp_type):
    """Build or retrieve a JIT-compiled eval step function.

    opp_type: "heuristic" or "random"
    """
    cache_key = (opp_type, n_games)
    if cache_key in _EVAL_JIT_CACHE:
        return _EVAL_JIT_CACHE[cache_key]

    from pokejax.rl.heuristic import heuristic_action, random_action
    from pokejax.rl.obs_builder import build_obs

    N = n_games

    def _where_bcast(mask, old, new):
        m = mask
        for _ in range(old.ndim - 1):
            m = m[..., None]
        return jnp.where(m, old, new)

    @jax.jit
    def eval_step(params, states, done, key):
        key, mk, ok, sk = jax.random.split(key, 4)
        model_keys = jax.random.split(mk, N)
        opp_keys = jax.random.split(ok, N)
        step_keys = jax.random.split(sk, N)

        # Model acts as player 0 (greedy)
        def model_action(battle, reveal, k):
            obs = build_obs(battle, reveal, 0, tables)
            log_probs, _, _ = model.apply(
                params,
                obs["int_ids"][None],
                obs["float_feats"][None],
                obs["legal_mask"][None],
            )
            return jnp.argmax(log_probs[0]).astype(jnp.int32)

        acts0 = jax.vmap(model_action)(states.battle, states.reveal, model_keys)

        # Opponent acts as player 1
        if opp_type == "heuristic":
            acts1 = jax.vmap(
                lambda b, k: heuristic_action(b, 1, tables, k)
            )(states.battle, opp_keys)
        else:
            acts1 = jax.vmap(
                lambda b, k: random_action(b, 1, k)
            )(states.battle, opp_keys)

        actions = jnp.stack([acts0, acts1], axis=1)
        new_states, _, rewards, _, _ = jax.vmap(env.step)(states, actions, step_keys)

        # Freeze finished games
        final_states = jax.tree.map(
            lambda o, n: _where_bcast(done, o, n), states, new_states,
        )
        new_done = done | new_states.battle.finished

        # Track reward at terminal state (player 0's perspective)
        terminal_reward = jnp.where(
            new_states.battle.finished & ~done,
            rewards[:, 0],
            0.0,
        )

        return final_states, new_done, terminal_reward, key

    _EVAL_JIT_CACHE[cache_key] = eval_step
    return eval_step


def eval_vs_opponents(model, params, env, tables, n_games: int = 64) -> dict:
    """Run fast JIT-batched eval: model vs heuristic and model vs random.

    All n_games run in parallel on GPU via vmap. Each eval step is a single
    JIT kernel — much faster than sequential game loops.
    """
    key = jax.random.PRNGKey(int(time.time()) % 2**31)
    v_reset = jax.jit(jax.vmap(env.reset))
    results = {}

    for opp_type in ["heuristic", "random"]:
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, n_games)
        states, _ = v_reset(reset_keys)

        done = jnp.zeros(n_games, dtype=jnp.bool_)
        total_rewards = jnp.zeros(n_games, dtype=jnp.float32)

        eval_step = _get_eval_step_fn(model, env, tables, n_games, opp_type)

        for turn in range(200):
            states, done, term_rewards, key = eval_step(params, states, done, key)
            total_rewards = total_rewards + term_rewards
            if turn % 20 == 19 and bool(done.all()):
                break

        # Compute win rate
        done_np = np.array(done).astype(bool)
        rewards_np = np.array(total_rewards)
        wins = (rewards_np[done_np] > 0).sum() if done_np.any() else 0
        total = done_np.sum()
        wr = wins / max(total, 1)
        results[f"eval/win_rate_vs_{opp_type}"] = wr

    return results


# ---------------------------------------------------------------------------
# PS server eval (runs play_local_heuristic.py as subprocess)
# ---------------------------------------------------------------------------

def _run_ps_eval(checkpoint_path: str, n_games: int, writer, global_step: int):
    """Run eval on local Pokemon Showdown server via subprocess.

    Requires a local PS server running (node pokemon-showdown start --no-security).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(repo_root, "scripts", "play_local_heuristic.py")
    if not os.path.exists(script):
        print("  PS eval script not found, skipping")
        return None

    results = {}
    for vs in ["heuristic", "random"]:
        try:
            proc = subprocess.run(
                [
                    "python3", script,
                    "--checkpoint", checkpoint_path,
                    "--games", str(n_games),
                    "--vs", vs,
                    "--output-dir", "/tmp/ps_eval",
                ],
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                print(f"  PS eval vs {vs} failed: {proc.stderr[:200]}")
                continue

            import json
            summary_path = "/tmp/ps_eval/local_summary.json"
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                wr = summary.get("win_rate", 0.0)
                results[f"ps_eval/win_rate_vs_{vs}"] = wr
                writer.add_scalar(f"ps_eval/win_rate_vs_{vs}", wr, global_step)
                print(f"  PS eval vs {vs}: {wr:.0%} "
                      f"({summary['wins']}W/{summary['losses']}L/"
                      f"{summary.get('ties', 0)}T)")
        except subprocess.TimeoutExpired:
            print(f"  PS eval vs {vs} timed out")
        except Exception as e:
            print(f"  PS eval vs {vs} error: {e}")

    return results if results else None


# ---------------------------------------------------------------------------
# Model + optimizer creation
# ---------------------------------------------------------------------------

def create_model_and_state(
    cfg: TrainConfig,
    key: jnp.ndarray,
    init_params: Optional[dict] = None,
) -> tuple:
    """Initialize model parameters and optimizer state.

    Uses linear warmup + cosine decay LR schedule.
    If init_params is provided (e.g. from BC), use those instead of random init.
    """
    model = PokeTransformer()

    if init_params is None:
        B = 1
        dummy_int_ids    = jnp.zeros((B, 15, 8), dtype=jnp.int32)
        dummy_float_feats = jnp.zeros((B, 15, 394), dtype=jnp.float32)
        dummy_legal      = jnp.ones((B, 10), dtype=jnp.float32)
        params = model.init(key, dummy_int_ids, dummy_float_feats, dummy_legal)
    else:
        params = init_params

    # Linear warmup then cosine decay
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=cfg.ppo.lr,
        transition_steps=cfg.lr_warmup_steps,
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=cfg.ppo.lr,
        decay_steps=max(cfg.total_timesteps - cfg.lr_warmup_steps, 1),
        alpha=cfg.lr_min / max(cfg.ppo.lr, 1e-10),
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[cfg.lr_warmup_steps],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )

    train_state = TrainState(
        params=params,
        opt_state=optimizer.init(params),
        step=jnp.int32(0),
    )
    return model, optimizer, train_state, lr_schedule


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    env,
    tables,
    cfg: TrainConfig,
    key: jnp.ndarray,
    init_params: Optional[dict] = None,
    resume_checkpoint: Optional[dict] = None,
) -> TrainState:
    """Main PPO self-play training loop with full logging.

    env:              PokeJAXEnv instance
    tables:           Tables object
    cfg:              TrainConfig
    key:              JAX PRNGKey
    init_params:      Optional pre-trained params (from BC) to warm-start
    resume_checkpoint: Optional full checkpoint dict for resuming training
    """
    # TensorBoard
    from tensorboardX import SummaryWriter
    os.makedirs(cfg.tensorboard_dir, exist_ok=True)
    run_name = f"ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(os.path.join(cfg.tensorboard_dir, run_name))

    # Log hyperparameters
    hparams = {
        "lr": cfg.ppo.lr,
        "clip_eps": cfg.ppo.clip_eps,
        "gamma": cfg.ppo.gamma,
        "gae_lambda": cfg.ppo.gae_lambda,
        "vf_coef": cfg.ppo.vf_coef,
        "ent_coef": cfg.ppo.ent_coef,
        "n_epochs": cfg.ppo.n_epochs,
        "minibatch_size": cfg.ppo.minibatch_size,
        "n_envs": cfg.rollout.n_envs,
        "n_steps": cfg.rollout.n_steps,
        "total_timesteps": cfg.total_timesteps,
        "pool_size": cfg.pool_size,
        "pool_latest_ratio": cfg.pool_latest_ratio,
        "lr_warmup_steps": cfg.lr_warmup_steps,
    }
    writer.add_text("hyperparameters", str(hparams), 0)

    # Initialize model + optimizer
    key, init_key = jax.random.split(key)
    resume_params = resume_checkpoint["params"] if resume_checkpoint else init_params
    model, optimizer, train_state, lr_schedule = create_model_and_state(
        cfg, init_key, resume_params
    )

    # Restore full training state on resume
    if resume_checkpoint is not None:
        train_state = TrainState(
            params=resume_checkpoint["params"],
            opt_state=resume_checkpoint["opt_state"],
            step=resume_checkpoint["train_step"],
        )

    # JIT compile rollout + PPO
    # Symmetric self-play only — asymmetric doubles XLA memory and OOMs on 10GB.
    # Opponent diversity comes from periodic eval checkpoints instead.
    jit_rollout = make_jit_rollout(model, env, tables, cfg.rollout)
    jit_epochs = make_jit_ppo_epochs(model, optimizer, cfg.ppo)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    timesteps_collected = resume_checkpoint["timesteps"] if resume_checkpoint else 0
    update_idx = resume_checkpoint["update_idx"] if resume_checkpoint else 0
    metrics = {}
    t_start = time.time()
    best_win_rate = resume_checkpoint.get("best_win_rate", 0.0) if resume_checkpoint else 0.0

    transitions_per_rollout = cfg.rollout.n_envs * cfg.rollout.n_steps

    print(f"PPO Self-Play Training:")
    print(f"  Envs: {cfg.rollout.n_envs}, Steps/rollout: {cfg.rollout.n_steps}")
    print(f"  Transitions/update: {transitions_per_rollout:,}")
    print(f"  Total timesteps: {cfg.total_timesteps:,}")
    print(f"  LR: {cfg.ppo.lr}, Gamma: {cfg.ppo.gamma}, GAE λ: {cfg.ppo.gae_lambda}")
    print(f"  Entropy coeff: {cfg.ppo.ent_coef}, Clip ε: {cfg.ppo.clip_eps}")
    print(f"  Epochs: {cfg.ppo.n_epochs}, Minibatch: {cfg.ppo.minibatch_size}")
    print(f"  Self-play: symmetric (latest vs latest)")
    print(f"  TensorBoard: {cfg.tensorboard_dir}/{run_name}")
    if resume_checkpoint is not None:
        print(f"  Resumed from checkpoint: update_idx={update_idx}, "
              f"timesteps={timesteps_collected:,}, best_wr={best_win_rate:.2%}, "
              f"opt_step={int(train_state.step)}")
    elif init_params is not None:
        print(f"  Warm-started from pre-trained params")
    print()

    while timesteps_collected < cfg.total_timesteps:
        key, rollout_key, ppo_key = jax.random.split(key, 3)

        # Symmetric self-play: rollout + PPO, fully async on GPU
        _, batch, info = jit_rollout(train_state.params, rollout_key)
        train_state, metrics, _ = jit_epochs(train_state, batch, ppo_key)

        timesteps_collected += transitions_per_rollout
        update_idx += 1

        # --- Logging (only at print_interval to minimize GPU sync) ---
        # All float() calls consolidated into one sync point
        if update_idx % cfg.print_interval == 0:
            # Single batched GPU→CPU transfer for all scalars
            vals = _pull_scalars(metrics, info)
            elapsed = time.time() - t_start
            sps = timesteps_collected / max(elapsed, 1)
            global_step = timesteps_collected

            # TensorBoard
            writer.add_scalar("loss/total",   vals["m/loss/total"],   global_step)
            writer.add_scalar("loss/policy",  vals["m/loss/policy"],  global_step)
            writer.add_scalar("loss/value",   vals["m/loss/value"],   global_step)
            writer.add_scalar("loss/entropy", vals["m/loss/entropy"], global_step)
            writer.add_scalar("diagnostics/approx_kl",    vals["m/diagnostics/approx_kl"],    global_step)
            writer.add_scalar("diagnostics/clip_fraction", vals["m/diagnostics/clip_fraction"], global_step)
            writer.add_scalar("diagnostics/ratio_mean",    vals["m/ratio/mean"],    global_step)
            writer.add_scalar("diagnostics/ratio_max",     vals["m/ratio/max"],     global_step)
            lr_val = float(lr_schedule(train_state.step))
            writer.add_scalar("charts/learning_rate", lr_val, global_step)
            writer.add_scalar("charts/ent_coef", vals["m/diagnostics/ent_coef"], global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/selfplay_win_rate",    vals["i/win_rate"],    global_step)
            writer.add_scalar("charts/episodes_per_rollout", vals["i/n_episodes"],  global_step)
            writer.add_scalar("charts/mean_episode_reward",  vals["i/mean_reward"], global_step)

            # Stdout
            print(f"[{timesteps_collected:>12,}] "
                  f"loss={vals['m/loss/total']:.4f}  "
                  f"pol={vals['m/loss/policy']:.4f}  "
                  f"val={vals['m/loss/value']:.4f}  "
                  f"ent={vals['m/loss/entropy']:.4f}  "
                  f"kl={vals['m/diagnostics/approx_kl']:.4f}  "
                  f"clip={vals['m/diagnostics/clip_fraction']:.3f}  "
                  f"wr={vals['i/win_rate']:.2f}  "
                  f"ep={vals['i/n_episodes']:.0f}  "
                  f"SPS={sps:,.0f}")

        # --- Eval vs heuristic + random ---
        if update_idx % cfg.eval_interval == 0:
            print(f"  Eval ({cfg.eval_games} games each)...", end=" ", flush=True)
            eval_results = eval_vs_opponents(
                model, train_state.params, env, tables, n_games=cfg.eval_games
            )
            for k, v in eval_results.items():
                writer.add_scalar(k, v, timesteps_collected)

            wr_h = eval_results["eval/win_rate_vs_heuristic"]
            wr_r = eval_results["eval/win_rate_vs_random"]
            print(f"vs_heur={wr_h:.0%}  vs_rand={wr_r:.0%}")

            # Save best checkpoint (by heuristic win rate)
            if wr_h > best_win_rate:
                best_win_rate = wr_h
                ckpt = {
                    "params": train_state.params,
                    "opt_state": train_state.opt_state,
                    "train_step": train_state.step,
                    "update_idx": update_idx,
                    "timesteps": timesteps_collected,
                    "best_win_rate": best_win_rate,
                    "eval_win_rate_heuristic": wr_h,
                }
                path = os.path.join(cfg.checkpoint_dir, "ppo_best.pkl")
                with open(path, "wb") as f:
                    pickle.dump(ckpt, f)
                print(f"  New best! win_rate_vs_heuristic={wr_h:.0%} saved to {path}")

        # --- Checkpoint ---
        if update_idx % cfg.checkpoint_interval == 0:
            ckpt = {
                "params": train_state.params,
                "opt_state": train_state.opt_state,
                "train_step": train_state.step,
                "update_idx": update_idx,
                "timesteps": timesteps_collected,
                "best_win_rate": best_win_rate,
            }
            path = os.path.join(cfg.checkpoint_dir, f"ppo_{update_idx:06d}.pkl")
            with open(path, "wb") as f:
                pickle.dump(ckpt, f)
            latest_path = os.path.join(cfg.checkpoint_dir, "ppo_latest.pkl")
            with open(latest_path, "wb") as f:
                pickle.dump(ckpt, f)
            print(f"  Checkpoint saved: {path}")

        # --- PS server eval ---
        if cfg.ps_eval and update_idx % cfg.ps_eval_interval == 0:
            latest_ckpt = os.path.join(cfg.checkpoint_dir, "ppo_latest.pkl")
            if os.path.exists(latest_ckpt):
                print(f"  PS server eval ({cfg.ps_eval_games} games each)...")
                _run_ps_eval(latest_ckpt, cfg.ps_eval_games, writer, timesteps_collected)

    # Final checkpoint
    ckpt = {
        "params": train_state.params,
        "opt_state": train_state.opt_state,
        "train_step": train_state.step,
        "update_idx": update_idx,
        "timesteps": timesteps_collected,
        "best_win_rate": best_win_rate,
    }
    path = os.path.join(cfg.checkpoint_dir, "ppo_final.pkl")
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"Final checkpoint: {path}")

    writer.close()
    return train_state
