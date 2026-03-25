"""
JAX PRNG helpers for PokeJAX.

JAX uses an explicit functional PRNG (Threefry-based).  Every random
operation consumes a key and must split it to produce independent sub-keys.

All helpers here are pure functions compatible with jit/vmap/lax.scan.
"""

import jax
import jax.numpy as jnp


def split(key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split key into (new_key, subkey). Mirrors jax.random.split signature."""
    k1, k2 = jax.random.split(key)
    return k1, k2


def split_n(key: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split key into (new_key, subkeys[n])."""
    keys = jax.random.split(key, n + 1)
    return keys[0], keys[1:]


def rand_int(key: jnp.ndarray, low: int, high: int) -> jnp.ndarray:
    """Uniform integer in [low, high).  Returns int32 scalar."""
    return jax.random.randint(key, shape=(), minval=low, maxval=high, dtype=jnp.int32)


def rand_float(key: jnp.ndarray) -> jnp.ndarray:
    """Uniform float in [0, 1).  Returns float32 scalar."""
    return jax.random.uniform(key, shape=(), dtype=jnp.float32)


def rand_bool(key: jnp.ndarray, prob: float | jnp.ndarray) -> jnp.ndarray:
    """Returns True with probability `prob`."""
    return jax.random.uniform(key, shape=(), dtype=jnp.float32) < prob


def rand_bool_pct(key: jnp.ndarray, pct: int | jnp.ndarray) -> jnp.ndarray:
    """Returns True with probability pct/100.

    In Pokemon, many effects use integer percentages (e.g., 30% flinch).
    This mirrors Showdown's PRNG: roll < pct where roll is in [0, 100).
    """
    roll = jax.random.randint(key, shape=(), minval=0, maxval=100, dtype=jnp.int32)
    return roll < jnp.int32(pct)


def damage_roll(key: jnp.ndarray) -> jnp.ndarray:
    """
    Pokemon damage roll: one of 16 discrete values from 85/100 to 100/100.

    Showdown PRNG: roll = random(0, 16); multiplier = (85 + roll) / 100
    We return the multiplier as a float32.
    """
    roll = jax.random.randint(key, shape=(), minval=0, maxval=16, dtype=jnp.int32)
    return (jnp.int32(85) + roll).astype(jnp.float32) / jnp.float32(100.0)


def accuracy_roll(key: jnp.ndarray, accuracy: jnp.ndarray) -> jnp.ndarray:
    """
    Returns True if the move hits.

    accuracy: int in [1, 100] (or a special 'always hits' sentinel handled
    by the caller).  Showdown: roll < accuracy where roll in [0, 100).
    """
    roll = jax.random.randint(key, shape=(), minval=0, maxval=100, dtype=jnp.int32)
    return roll < jnp.int32(accuracy)


def critical_hit_roll(key: jnp.ndarray, ratio: jnp.ndarray) -> jnp.ndarray:
    """
    Gen 4-5 crit roll.

    ratio matches Pokemon Showdown's critRatio (1-based):
      ratio 1 → 1/16   (default, no modifiers)
      ratio 2 → 1/8    (high-crit move, or +1 item with no other boosts)
      ratio 3 → 1/4    (e.g. normal + Focus Energy, or high-crit + Scope Lens)
      ratio 4 → 1/3    (e.g. high-crit + Focus Energy)
      ratio 5 → 1/2    (max; e.g. high-crit + FE + Scope Lens)
      ratio 6+ → always crit

    Uses roll out of 48 to represent all stages as integers (LCM of 16, 3, 2).
    critMult = [0, 16, 8, 4, 3, 2] in PS → thresholds out of 48: [0, 3, 6, 12, 16, 24, 48, 48]

    Returns True if crit.
    """
    # Thresholds out of 48: index = critRatio, value = how many rolls trigger crit
    # [0]: impossible (ratio 0 never used), [1]=3/48=1/16, [2]=6/48=1/8,
    # [3]=12/48=1/4, [4]=16/48=1/3, [5]=24/48=1/2, [6+]=always
    thresholds = jnp.array([0, 3, 6, 12, 16, 24, 48, 48], dtype=jnp.int32)
    threshold = thresholds[jnp.clip(ratio, 0, 7)]
    roll = jax.random.randint(key, shape=(), minval=0, maxval=48, dtype=jnp.int32)
    return roll < threshold


def sleep_roll(key: jnp.ndarray) -> jnp.ndarray:
    """Roll sleep duration: 1-3 turns (uniform, Gen 4 PS: random(1, 4))."""
    return jax.random.randint(key, shape=(), minval=1, maxval=4, dtype=jnp.int8)


def confusion_roll(key: jnp.ndarray) -> jnp.ndarray:
    """Roll confusion duration: 2-5 turns."""
    return jax.random.randint(key, shape=(), minval=2, maxval=6, dtype=jnp.int8)


def multi_hit_roll(key: jnp.ndarray, min_hits: int, max_hits: int) -> jnp.ndarray:
    """Roll number of hits for multi-hit moves.

    Standard 2-5 hit distribution (Gen 5+): 2=37.5%, 3=37.5%, 4=12.5%, 5=12.5%
    For fixed-range moves (e.g., 2-hit Bonemerang), min==max or a simple range.
    """
    if min_hits == max_hits:
        return jnp.int8(min_hits)
    if min_hits == 2 and max_hits == 5:
        # Showdown's discrete distribution: [2,2,2,3,3,3,4,5]
        rolls = jnp.array([2, 2, 2, 3, 3, 3, 4, 5], dtype=jnp.int8)
        idx = jax.random.randint(key, shape=(), minval=0, maxval=8, dtype=jnp.int32)
        return rolls[idx]
    return jax.random.randint(
        key, shape=(), minval=min_hits, maxval=max_hits + 1, dtype=jnp.int8
    )


def freeze_thaw_roll(key: jnp.ndarray) -> jnp.ndarray:
    """20% chance to thaw out of freeze."""
    return rand_bool_pct(key, 20)


def flinch_roll(key: jnp.ndarray, chance_pct: int | jnp.ndarray) -> jnp.ndarray:
    """Roll for flinch (typically 10% or 30%)."""
    return rand_bool_pct(key, chance_pct)


def paralysis_full_roll(key: jnp.ndarray) -> jnp.ndarray:
    """25% chance of full paralysis (cannot move)."""
    return rand_bool_pct(key, 25)


def secondary_effect_roll(key: jnp.ndarray, chance_pct: int | jnp.ndarray) -> jnp.ndarray:
    """Roll for a move's secondary effect."""
    return rand_bool_pct(key, chance_pct)


def speed_tie_roll(key: jnp.ndarray) -> jnp.ndarray:
    """Resolve a speed tie: returns 0 or 1 (which side goes first)."""
    return jax.random.randint(key, shape=(), minval=0, maxval=2, dtype=jnp.int32)


def metronome_roll(key: jnp.ndarray, n_moves: int) -> jnp.ndarray:
    """Pick a random move index for Metronome."""
    return jax.random.randint(key, shape=(), minval=0, maxval=n_moves, dtype=jnp.int32)
