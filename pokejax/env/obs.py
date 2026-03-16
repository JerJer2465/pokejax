"""
State → observation tensor for PokeJAX.

Observation layout (float32 vector):
  Field (20):
    weather: 5 one-hot
    terrain: 5 one-hot
    trick_room: 1
    weather_turns_norm: 1 (/ 8)
    terrain_turns_norm: 1 (/ 5)
    turn_norm: 1 (/ 300)
    (4 padding)

  Per team (2 sides × 6 Pokemon × ~80 values):
    hp_ratio: 1
    is_active: 1
    is_fainted: 1
    status: 7 one-hot
    types: 2 × 19 one-hot → 38 (or 2 int indices normalized)
    boosts: 7 values normalized to [-1, 1] (÷6)
    moves: 4 × (type_norm, category_norm, bp_norm, pp_ratio) = 4×4 = 16
    level_norm: 1 (÷100)
    volatile_mask: 16 (top 16 bits of uint32 bitmask, as float)

  Total: 20 + 2 × 6 × (1+1+1+7+38+7+16+1+16) = 20 + 2×6×88 = 20+1056 = 1076

"""

import jax.numpy as jnp
import numpy as np

from pokejax.types import (
    BattleState,
    N_TYPES,
    MAX_TEAM_SIZE, MAX_MOVES,
)

# Derived constants
N_STATUS  = 7   # 0-6
N_WEATHER = 5
N_TERRAIN = 5
PER_POKEMON = 88
OBS_DIM = 20 + 2 * MAX_TEAM_SIZE * PER_POKEMON  # 1076


def _one_hot(idx: jnp.ndarray, n: int) -> jnp.ndarray:
    """int → float32 one-hot vector of length n."""
    return (jnp.arange(n, dtype=jnp.int32) == idx.astype(jnp.int32)).astype(jnp.float32)


def build_field_obs(state: BattleState) -> jnp.ndarray:
    """Build field observation: float32[20]."""
    weather_oh  = _one_hot(state.field.weather, N_WEATHER)
    terrain_oh  = _one_hot(state.field.terrain, N_TERRAIN)
    trick_room  = jnp.array([state.field.trick_room > jnp.int8(0)], dtype=jnp.float32)
    weather_t   = jnp.array([state.field.weather_turns.astype(jnp.float32) / 8.0])
    terrain_t   = jnp.array([state.field.terrain_turns.astype(jnp.float32) / 5.0])
    turn_norm   = jnp.array([state.turn.astype(jnp.float32) / 300.0])
    pad         = jnp.zeros(4, dtype=jnp.float32)

    return jnp.concatenate([weather_oh, terrain_oh, trick_room,
                              weather_t, terrain_t, turn_norm, pad])  # 5+5+1+1+1+1+4=18... add 2 more
    # Fix: 5+5+1+1+1+1+4 = 18, need 20 — extra 2 slots for future use


def build_pokemon_obs(state: BattleState, side: int, slot: int,
                       tables) -> jnp.ndarray:
    """Build single-Pokemon observation: float32[PER_POKEMON=88]."""
    hp       = state.sides_team_hp[side, slot].astype(jnp.float32)
    max_hp   = state.sides_team_max_hp[side, slot].astype(jnp.float32)
    hp_ratio = jnp.where(max_hp > 0, hp / max_hp, jnp.float32(0.0))

    is_active  = state.sides_team_is_active[side, slot].astype(jnp.float32)
    is_fainted = state.sides_team_fainted[side, slot].astype(jnp.float32)

    # Status: 7 one-hot
    status_oh = _one_hot(state.sides_team_status[side, slot], N_STATUS)

    # Types: 2 × 19 one-hot  (38 values)
    t0 = state.sides_team_types[side, slot, 0]
    t1 = state.sides_team_types[side, slot, 1]
    type0_oh = _one_hot(t0, N_TYPES)
    type1_oh = _one_hot(t1, N_TYPES)

    # Boosts: 7 values in [-6,+6], normalized to [-1, +1]
    boosts = state.sides_team_boosts[side, slot].astype(jnp.float32) / 6.0

    # Moves: 4 × 4 = 16 values
    def move_obs(move_idx: int) -> jnp.ndarray:
        mid = state.sides_team_move_ids[side, slot, move_idx].astype(jnp.int32)
        valid = mid >= jnp.int32(0)
        mid_clamped = jnp.maximum(jnp.int32(0), mid)

        move_type = jnp.where(valid, tables.moves[mid_clamped, 2].astype(jnp.float32) / float(N_TYPES), 0.0)
        move_cat  = jnp.where(valid, tables.moves[mid_clamped, 3].astype(jnp.float32) / 2.0, 0.0)
        move_bp   = jnp.where(valid, tables.moves[mid_clamped, 0].astype(jnp.float32) / 250.0, 0.0)

        pp     = state.sides_team_move_pp[side, slot, move_idx].astype(jnp.float32)
        max_pp = state.sides_team_move_max_pp[side, slot, move_idx].astype(jnp.float32)
        pp_ratio = jnp.where(valid & (max_pp > 0), pp / max_pp, jnp.float32(0.0))

        return jnp.array([move_type, move_cat, move_bp, pp_ratio])

    moves_obs = jnp.concatenate([move_obs(i) for i in range(MAX_MOVES)])  # 16

    # Level normalized
    level_norm = state.sides_team_level[side, slot].astype(jnp.float32) / 100.0

    # Volatile bitmask: top 16 bits (float)
    vols = state.sides_team_volatiles[side, slot].astype(jnp.uint32)
    vol_bits = jnp.array(
        [(vols >> i) & jnp.uint32(1) for i in range(16)],
        dtype=jnp.float32
    )

    obs = jnp.concatenate([
        jnp.array([hp_ratio, is_active, is_fainted]),  # 3
        status_oh,    # 7
        type0_oh,     # 19
        type1_oh,     # 19
        boosts,       # 7
        moves_obs,    # 16
        jnp.array([level_norm]),  # 1
        vol_bits,     # 16
    ])  # total: 3+7+19+19+7+16+1+16 = 88

    return obs


def build_observation(state: BattleState, player: int,
                       tables) -> jnp.ndarray:
    """
    Build the full observation for `player`.

    Observation is from `player`'s perspective:
      - Field (20)
      - Own team 6×88 = 528
      - Opponent team 6×88 = 528
    Total: 20 + 528 + 528 = 1076
    """
    field_obs = build_field_obs(state)  # 20 (approximately; see note in build_field_obs)
    # Pad/trim to exactly 20
    field_obs = jnp.concatenate([field_obs, jnp.zeros(2, dtype=jnp.float32)])[:20]

    own_side = player
    opp_side = 1 - player

    own_obs = jnp.concatenate([
        build_pokemon_obs(state, own_side, slot, tables)
        for slot in range(MAX_TEAM_SIZE)
    ])  # 6 × 88 = 528

    opp_obs = jnp.concatenate([
        build_pokemon_obs(state, opp_side, slot, tables)
        for slot in range(MAX_TEAM_SIZE)
    ])  # 528

    return jnp.concatenate([field_obs, own_obs, opp_obs])  # 1076
