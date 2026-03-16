#!/usr/bin/env python3
"""Benchmark SPS: measure rollout + PPO update time breakdown."""

import sys
import time
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compile_cache")

print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

from pokejax.env.pokejax_env import PokeJAXEnv
from pokejax.rl.model import PokeTransformer
from pokejax.rl.rollout import RolloutConfig, make_jit_rollout
from pokejax.rl.ppo import PPOConfig, make_jit_ppo_epochs
from pokejax.rl.self_play import create_model_and_state, TrainConfig

env = PokeJAXEnv(gen=4)

import sys as _sys
_n_envs = int(_sys.argv[1]) if len(_sys.argv) > 1 else 256
_n_steps = int(_sys.argv[2]) if len(_sys.argv) > 2 else 128
print(f"Config: n_envs={_n_envs}, n_steps={_n_steps}")

cfg = TrainConfig(
    ppo=PPOConfig(minibatch_size=min(4096, _n_envs * _n_steps)),
    rollout=RolloutConfig(n_envs=_n_envs, n_steps=_n_steps),
)

key = jax.random.PRNGKey(42)
key, init_key = jax.random.split(key)
model, optimizer, train_state, lr_schedule = create_model_and_state(cfg, init_key)

# Count params
param_count = sum(x.size for x in jax.tree.leaves(train_state.params))
print(f"\nModel params: {param_count:,}")

transitions_per_update = cfg.rollout.n_envs * cfg.rollout.n_steps
print(f"Transitions per update: {transitions_per_update:,}")

# Build JIT functions
print("\nBuilding JIT functions...")
jit_rollout = make_jit_rollout(model, env, env.tables, cfg.rollout)
jit_epochs = make_jit_ppo_epochs(model, optimizer, cfg.ppo)

# Warmup / compile
print("Compiling (first run)... this may take a while")
t0 = time.time()
key, rk, pk = jax.random.split(key, 3)
_, batch, info = jit_rollout(train_state.params, rk)
jax.block_until_ready(batch)
t_compile_rollout = time.time() - t0
print(f"  Rollout compile: {t_compile_rollout:.1f}s")

t0 = time.time()
train_state, metrics, _ = jit_epochs(train_state, batch, pk)
jax.block_until_ready(metrics)
t_compile_ppo = time.time() - t0
print(f"  PPO compile: {t_compile_ppo:.1f}s")

# Benchmark: time rollout and PPO separately
N_WARMUP = 3
N_BENCH = 20

print(f"\nWarming up {N_WARMUP} iterations...")
for i in range(N_WARMUP):
    key, rk, pk = jax.random.split(key, 3)
    _, batch, info = jit_rollout(train_state.params, rk)
    train_state, metrics, _ = jit_epochs(train_state, batch, pk)
    jax.block_until_ready(metrics)

print(f"Benchmarking {N_BENCH} iterations...")

# Benchmark rollout only
rollout_times = []
for i in range(N_BENCH):
    key, rk = jax.random.split(key)
    t0 = time.time()
    _, batch, info = jit_rollout(train_state.params, rk)
    jax.block_until_ready(batch)
    rollout_times.append(time.time() - t0)

# Benchmark PPO only
ppo_times = []
for i in range(N_BENCH):
    key, pk = jax.random.split(key)
    t0 = time.time()
    train_state, metrics, _ = jit_epochs(train_state, batch, pk)
    jax.block_until_ready(metrics)
    ppo_times.append(time.time() - t0)

# Benchmark combined (rollout + PPO)
combined_times = []
for i in range(N_BENCH):
    key, rk, pk = jax.random.split(key, 3)
    t0 = time.time()
    _, batch, info = jit_rollout(train_state.params, rk)
    train_state, metrics, _ = jit_epochs(train_state, batch, pk)
    jax.block_until_ready(metrics)
    combined_times.append(time.time() - t0)

# Results
avg_rollout = sum(rollout_times) / len(rollout_times)
avg_ppo = sum(ppo_times) / len(ppo_times)
avg_combined = sum(combined_times) / len(combined_times)
sps = transitions_per_update / avg_combined

print(f"\n{'='*60}")
print(f"BENCHMARK RESULTS ({N_BENCH} iterations)")
print(f"{'='*60}")
print(f"  Rollout (env+inference):  {avg_rollout:.3f}s  (min={min(rollout_times):.3f}, max={max(rollout_times):.3f})")
print(f"  PPO epochs (3 epochs):    {avg_ppo:.3f}s  (min={min(ppo_times):.3f}, max={max(ppo_times):.3f})")
print(f"  Combined (full update):   {avg_combined:.3f}s  (min={min(combined_times):.3f}, max={max(combined_times):.3f})")
print(f"")
print(f"  Transitions/update:  {transitions_per_update:,}")
print(f"  SPS (steps/second):  {sps:,.0f}")
print(f"")
print(f"  Time split: rollout={avg_rollout/avg_combined*100:.1f}%  PPO={avg_ppo/avg_combined*100:.1f}%")
print(f"")
total_steps = 250_000_000
total_updates = total_steps / transitions_per_update
total_time_s = total_updates * avg_combined
total_time_h = total_time_s / 3600
total_time_d = total_time_h / 24
print(f"  Estimated 250M steps:")
print(f"    {total_updates:,.0f} updates × {avg_combined:.3f}s = {total_time_s:,.0f}s")
print(f"    = {total_time_h:.1f} hours = {total_time_d:.1f} days")
print(f"{'='*60}")

# VRAM check
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode == 0:
        used, total = result.stdout.strip().split(', ')
        print(f"\nGPU VRAM: {used}MB / {total}MB ({int(used)/int(total)*100:.1f}% used)")
except Exception:
    pass
