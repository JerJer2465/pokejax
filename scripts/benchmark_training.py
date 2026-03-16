#!/usr/bin/env python3
"""
Benchmark pokejax training throughput at different n_envs and minibatch sizes.
Designed for RTX 3080 16GB VRAM.
"""

import os
import time
import argparse

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compile_cache")


def benchmark_config(env, tables, n_envs, n_steps, mb_size, use_scan_ppo=True, n_updates=5, seed=42):
    """Run a few PPO updates and return SPS."""
    from pokejax.rl.self_play import TrainConfig, RolloutConfig, PPOConfig, create_model_and_state
    from pokejax.rl.rollout import make_jit_rollout
    from pokejax.rl.ppo import make_jit_ppo_epochs, ppo_step

    cfg = TrainConfig(
        ppo=PPOConfig(minibatch_size=mb_size),
        rollout=RolloutConfig(n_envs=n_envs, n_steps=n_steps),
    )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    model, optimizer, train_state, _lr = create_model_and_state(cfg, init_key)

    jit_rollout = make_jit_rollout(model, env, tables, cfg.rollout)

    transitions_per = n_envs * n_steps

    if use_scan_ppo:
        jit_epochs = make_jit_ppo_epochs(model, optimizer, cfg.ppo)

        # Warmup
        label = f"scan n_envs={n_envs} mb={mb_size}"
        print(f"  Compiling ({label})...")
        key, rk, pk = jax.random.split(key, 3)
        _, batch, _info = jit_rollout(train_state.params, rk)
        train_state, metrics, _ = jit_epochs(train_state, batch, pk)
        jax.block_until_ready(train_state.params)

        # Timed
        t0 = time.perf_counter()
        for i in range(n_updates):
            key, rk, pk = jax.random.split(key, 3)
            _, batch, _info = jit_rollout(train_state.params, rk)
            train_state, metrics, _ = jit_epochs(train_state, batch, pk)
        jax.block_until_ready(train_state.params)
        elapsed = time.perf_counter() - t0
    else:
        # Python-loop PPO (no lax.scan fusion)
        jit_step = jax.jit(lambda ts, mb: ppo_step(ts, model, mb, cfg.ppo, optimizer))
        B = transitions_per
        n_mb = B // mb_size
        n_epochs = cfg.ppo.n_epochs

        label = f"loop n_envs={n_envs} mb={mb_size}"
        print(f"  Compiling ({label})...")
        key, rk = jax.random.split(key)
        _, batch, _info = jit_rollout(train_state.params, rk)
        # Warmup one step
        mb = jax.tree.map(lambda x: x[:mb_size], batch)
        train_state, _ = jit_step(train_state, mb)
        jax.block_until_ready(train_state.params)

        # Timed
        t0 = time.perf_counter()
        for i in range(n_updates):
            key, rk = jax.random.split(key)
            _, batch, _info = jit_rollout(train_state.params, rk)
            for epoch in range(n_epochs):
                key, perm_key = jax.random.split(key)
                perm = jax.random.permutation(perm_key, B)
                batch_s = jax.tree.map(lambda x: x[perm], batch)
                for mb_i in range(n_mb):
                    start = mb_i * mb_size
                    mb = jax.tree.map(lambda x: x[start:start+mb_size], batch_s)
                    train_state, metrics = jit_step(train_state, mb)
        jax.block_until_ready(train_state.params)
        elapsed = time.perf_counter() - t0

    total_steps = transitions_per * n_updates
    sps = total_steps / elapsed
    hrs_250m = 250_000_000 / sps / 3600

    return sps, elapsed, hrs_250m, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--n-updates", type=int, default=5)
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    from pokejax.env.pokejax_env import PokeJAXEnv
    env = PokeJAXEnv(gen=args.gen)
    tables = env.tables

    configs = [
        # (n_envs, n_steps, mb_size, use_scan_ppo)
        (256,  128, 4096, False),   # Python loop, small
        (512,  128, 4096, False),   # Python loop, medium
        (256,  128, 4096, True),    # lax.scan, small
        (512,  128, 4096, True),    # lax.scan, medium
        (512,  128, 8192, False),   # Python loop, large mb
        (1024, 128, 4096, False),   # Python loop, large envs
    ]

    print("\n" + "=" * 80)
    print(f"{'label':>35} {'SPS':>10} {'250M hrs':>10}")
    print("=" * 80)

    for n_envs, n_steps, mb_size, use_scan in configs:
        try:
            sps, elapsed, hrs, label = benchmark_config(
                env, tables, n_envs, n_steps, mb_size,
                use_scan_ppo=use_scan,
                n_updates=args.n_updates,
            )
            print(f"{label:>35} {sps:>10,.0f} {hrs:>10.1f}")
        except Exception as e:
            tag = f"{'scan' if use_scan else 'loop'} n_envs={n_envs} mb={mb_size}"
            print(f"{tag:>35} {'OOM/ERR':>10} {'---':>10}  ({type(e).__name__})")

    print("=" * 80)


if __name__ == "__main__":
    main()
