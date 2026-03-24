#!/usr/bin/env python3
"""Optuna hyperparameter tuning for PokeMLP PPO training.

Searches over training hyperparameters (lr, gamma, entropy, etc.) with fixed
default MLP architecture. Each trial runs for a short training budget
(default 25M steps) and is evaluated by win rate vs heuristic.

Usage:
    python scripts/tune_mlp.py --n-trials 15 --trial-steps 25000000

Results are saved to tuning/optuna.db (SQLite) and tuning/best_params.json.
"""

import argparse
import json
import os
import pickle
import sys
import traceback

# Unbuffered stdout for real-time logging
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

import jax

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def objective(trial, args):
    """Single Optuna trial: train MLP for trial_steps, return heuristic win rate."""
    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.rl.self_play import TrainConfig, PPOConfig, RolloutConfig, train

    # --- Suggest training hyperparameters ---
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.99, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.005, 0.05, log=True)
    ent_coef_end = trial.suggest_float("ent_coef_end", 0.001, 0.01, log=True)
    n_epochs = trial.suggest_int("n_epochs", 2, 4)
    minibatch_size = trial.suggest_categorical("minibatch_size", [4096, 8192, 16384])
    vf_coef = trial.suggest_float("vf_coef", 0.5, 2.0)
    clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)

    # Entropy decay: cover ~25% of trial training
    n_envs, n_steps = 512, 128
    transitions_per_update = n_envs * n_steps
    total_updates = args.trial_steps // transitions_per_update
    ent_decay_steps = max(int(total_updates * n_epochs * 0.25), 100)

    trial_dir = os.path.join("tuning", f"trial_{trial.number:03d}")
    os.makedirs(os.path.join(trial_dir, "checkpoints"), exist_ok=True)

    cfg = TrainConfig(
        ppo=PPOConfig(
            lr=lr,
            clip_eps=clip_eps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            ent_coef_end=ent_coef_end,
            ent_coef_decay_steps=ent_decay_steps,
            n_epochs=n_epochs,
            minibatch_size=minibatch_size,
        ),
        rollout=RolloutConfig(
            n_envs=n_envs,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
        ),
        total_timesteps=args.trial_steps,
        checkpoint_dir=os.path.join(trial_dir, "checkpoints"),
        checkpoint_interval=max(total_updates // 5, 1),
        tensorboard_dir=os.path.join(trial_dir, "runs"),
        eval_interval=max(total_updates // 10, 1),
        eval_games=64,
        ps_eval=False,
        arch="mlp",
        lr_warmup_steps=500,
        lr_min=1e-5,
    )

    # Pruning callback: report win rate to Optuna after each eval
    report_idx = [0]

    def prune_callback(win_rate, timesteps):
        trial.report(win_rate, report_idx[0])
        report_idx[0] += 1
        if trial.should_prune():
            return True
        return False

    print(f"\n{'=' * 70}")
    print(f"Trial {trial.number}: lr={lr:.2e} gamma={gamma:.4f} "
          f"ent={ent_coef:.3f}\u2192{ent_coef_end:.4f} epochs={n_epochs} "
          f"mb={minibatch_size} vf={vf_coef:.2f} clip={clip_eps:.3f}")
    print(f"{'=' * 70}")

    env = PokeJAXEnv(gen=4)
    key = jax.random.PRNGKey(trial.number + 42)

    try:
        train(
            env, env.tables, cfg, key,
            eval_callback=prune_callback,
        )
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        return 0.0

    # Read best win rate from checkpoint (avoids JIT tracer leak on re-eval)
    best_path = os.path.join(trial_dir, "checkpoints", "ppo_best.pkl")
    if os.path.exists(best_path):
        with open(best_path, "rb") as f:
            best_ckpt = pickle.load(f)
        win_rate = best_ckpt.get("eval_win_rate_heuristic", 0.0)
    else:
        win_rate = 0.0

    print(f"Trial {trial.number} finished: vs_heuristic={win_rate:.1%}")
    return win_rate


def main():
    parser = argparse.ArgumentParser(
        description="Optuna HP tuning for PokeMLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trial-steps", type=int, default=25_000_000,
                        help="Env steps per trial")
    parser.add_argument("--n-trials", type=int, default=15,
                        help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, default="pokejax_mlp_tune",
                        help="Optuna study name")
    parser.add_argument("--db-path", type=str, default="tuning/optuna.db",
                        help="SQLite DB for Optuna persistence")
    args = parser.parse_args()

    os.makedirs("tuning", exist_ok=True)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Trial budget: {args.trial_steps:,} steps per trial")
    print(f"Total trials: {args.n_trials}")

    storage = f"sqlite:///{args.db_path}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=3,
        ),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
    )

    # Print results
    print(f"\n{'=' * 70}")
    print("TUNING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best win rate vs heuristic: {study.best_value:.1%}")
    print(f"Best params:")
    best = study.best_params
    for k, v in sorted(best.items()):
        print(f"  {k}: {v}")

    summary = {
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": best,
        "n_trials_completed": len(study.trials),
    }

    out_path = "tuning/best_params.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest params saved to {out_path}")

    # Print ready-to-use train_ppo.py command
    print(f"\n--- Ready-to-use command for full training ---")
    print(f"python3 scripts/train_ppo.py \\")
    print(f"  --arch mlp \\")
    print(f"  --total-steps 1000000000 \\")
    print(f"  --lr {best['lr']:.6f} \\")
    print(f"  --gamma {best['gamma']:.6f} \\")
    print(f"  --gae-lambda {best['gae_lambda']:.6f} \\")
    print(f"  --ent-coef {best['ent_coef']:.6f} \\")
    print(f"  --ent-coef-end {best['ent_coef_end']:.6f} \\")
    print(f"  --n-epochs {best['n_epochs']} \\")
    print(f"  --minibatch-size {best['minibatch_size']} \\")
    print(f"  --vf-coef {best['vf_coef']:.6f} \\")
    print(f"  --clip-eps {best['clip_eps']:.6f} \\")
    print(f"  --checkpoint-dir checkpoints/tuned_mlp_1B")


if __name__ == "__main__":
    main()
