"""
Action priority and speed sorting for PokeJAX.

Reproduces Showdown's BattleQueue.sort() logic, branchlessly.

In Pokemon battles, actions are sorted by:
  1. Priority bracket (higher first). e.g., Quick Attack +1, Protect +4, Pursuit -1
  2. Speed (higher first, or reversed under Trick Room)
  3. Speed ties broken by random coin flip

For simplicity, actions are represented as fixed-size arrays.
In singles, there are exactly 2 actions per turn (one per player).
We compute which action goes first and return an ordering index.

Action encoding (int8[ACTION_FIELDS]):
  [0] action_type:  0=move, 1=switch, 2=pass, 3=mega, 4=dynamax
  [1] player:       0 or 1
  [2] move_slot:    0-3 (move actions) or team_slot (switch actions)
  [3] priority:     int8 — base priority of move (from move table)
  [4] sub_priority: int8 — fractional priority (e.g., Prankster, Quick Claw)
"""

import jax
import jax.numpy as jnp

from pokejax.types import BattleState
from pokejax.core import rng as rng_utils
from pokejax.mechanics.events import run_event_speed


# Action type codes
ACTION_MOVE   = jnp.int8(0)
ACTION_SWITCH = jnp.int8(1)
ACTION_PASS   = jnp.int8(2)

# Action field indices
AF_TYPE      = 0
AF_PLAYER    = 1
AF_SLOT      = 2
AF_PRIORITY  = 3
AF_SUBPRI    = 4


def get_effective_speed(state: BattleState, side: int, cfg) -> jnp.ndarray:
    """
    Compute effective Speed stat for the active Pokemon.

    Applies:
      - Stat boosts (from boosts array)
      - Paralysis: divide by cfg.paralysis_speed_divisor (4 in Gen 4-6, 2 in Gen 7+)
      - Tailwind: 2x
      - Trick Room: speed is inverted (handled in sort_two_actions, not here)

    Returns int32 effective speed.
    """
    idx = state.sides_active_idx[side]
    base_spe = state.sides_team_base_stats[side, idx, 5].astype(jnp.float32)  # STAT_SPE=5
    boost    = state.sides_team_boosts[side, idx, 4].astype(jnp.int32)        # BOOST_SPE=4

    # Boost multiplier: max(2,2+stage)/max(2,2-stage)
    num = jnp.maximum(2, 2 + boost).astype(jnp.float32)
    den = jnp.maximum(2, 2 - boost).astype(jnp.float32)
    boosted_spe = jnp.floor(base_spe * num / den).astype(jnp.int32)

    # Paralysis: speed halved/quartered based on generation
    is_par = state.sides_team_status[side, idx] == jnp.int8(6)  # STATUS_PAR
    para_speed = boosted_spe // cfg.paralysis_speed_divisor
    boosted_spe = jnp.where(is_par, para_speed, boosted_spe)

    # Tailwind: 2x speed for this side's active Pokemon
    tailwind = state.sides_side_conditions[side, 7] > jnp.int8(0)  # SC_TAILWIND
    boosted_spe = jnp.where(tailwind, boosted_spe * 2, boosted_spe)

    # Ability/item speed modifier (Swift Swim, Chlorophyll, Choice Scarf, etc.)
    speed_mult = run_event_speed(state, side, idx)
    boosted_spe = jnp.floor(boosted_spe.astype(jnp.float32) * speed_mult).astype(jnp.int32)

    return jnp.maximum(jnp.int32(1), boosted_spe)


def get_move_priority(state: BattleState, side: int, move_slot: int,
                       tables) -> jnp.ndarray:
    """Look up a move's base priority from the move table."""
    idx = state.sides_active_idx[side]
    move_id = state.sides_team_move_ids[side, idx, move_slot].astype(jnp.int32)
    # Clamp to valid range (handle -1 / invalid move IDs)
    move_id = jnp.maximum(jnp.int32(0), move_id)
    return tables.moves[move_id, 4].astype(jnp.int8)  # MF_PRIORITY = 4


def sort_two_actions(
    action0_type: jnp.ndarray,   # int8: ACTION_MOVE / ACTION_SWITCH / ACTION_PASS
    action0_priority: jnp.ndarray, # int8
    action0_speed: jnp.ndarray,    # int32
    action1_type: jnp.ndarray,
    action1_priority: jnp.ndarray,
    action1_speed: jnp.ndarray,
    trick_room: jnp.ndarray,       # bool: Trick Room active
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Determine which of two actions goes first.

    Returns: (p0_first: bool, new_key, tie: bool)
      p0_first = True  → action 0 (player 0's action) goes first
      tie       = True → speed tie (coin flip was used)
    """
    # Switches always have priority +7 (higher than any move)
    eff_pri0 = jnp.where(action0_type == ACTION_SWITCH, jnp.int8(7), action0_priority)
    eff_pri1 = jnp.where(action1_type == ACTION_SWITCH, jnp.int8(7), action1_priority)

    # Compare priority brackets
    priority_diff = eff_pri0.astype(jnp.int32) - eff_pri1.astype(jnp.int32)
    priority_p0_wins = priority_diff > 0
    priority_tie     = priority_diff == 0

    # Speed comparison (trick room reverses)
    faster_p0 = jnp.where(
        trick_room,
        action0_speed < action1_speed,   # Trick Room: slower goes first
        action0_speed > action1_speed,   # Normal: faster goes first
    )
    speed_tie = action0_speed == action1_speed

    # Speed tie: coin flip
    key, subkey = rng_utils.split(key)
    coin = rng_utils.speed_tie_roll(subkey)  # 0 or 1
    coin_p0_wins = coin == jnp.int32(0)

    # Final ordering:
    # 1. Higher priority wins
    # 2. If priority tied → speed determines
    # 3. If speed tied → coin flip
    p0_first = jnp.where(
        priority_tie,
        jnp.where(speed_tie, coin_p0_wins, faster_p0),
        priority_p0_wins,
    )

    return p0_first, key, (priority_tie & speed_tie)


def compute_turn_order(
    state: BattleState,
    p0_move_slot: jnp.ndarray,   # int8: move slot chosen by P0 (-1 if switching)
    p1_move_slot: jnp.ndarray,   # int8: move slot chosen by P1
    p0_is_switch: jnp.ndarray,   # bool
    p1_is_switch: jnp.ndarray,   # bool
    tables,
    cfg,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute which player goes first for the turn.

    Returns:
        first_side: int32 — 0 if P0 acts first, 1 if P1 acts first
        new_key
        p0_speed: int32 (for diagnostics)
    """
    p0_speed = get_effective_speed(state, 0, cfg)
    p1_speed = get_effective_speed(state, 1, cfg)

    # Get move priorities (switches use +7)
    p0_idx = state.sides_active_idx[0]
    p1_idx = state.sides_active_idx[1]

    p0_move_id = state.sides_team_move_ids[0, p0_idx, jnp.maximum(jnp.int8(0), p0_move_slot).astype(jnp.int32)]
    p1_move_id = state.sides_team_move_ids[1, p1_idx, jnp.maximum(jnp.int8(0), p1_move_slot).astype(jnp.int32)]

    p0_move_id_clamped = jnp.maximum(jnp.int32(0), p0_move_id.astype(jnp.int32))
    p1_move_id_clamped = jnp.maximum(jnp.int32(0), p1_move_id.astype(jnp.int32))

    p0_priority = jnp.where(
        p0_is_switch, jnp.int8(7),
        tables.moves[p0_move_id_clamped, 4].astype(jnp.int8)
    )
    p1_priority = jnp.where(
        p1_is_switch, jnp.int8(7),
        tables.moves[p1_move_id_clamped, 4].astype(jnp.int8)
    )

    trick_room = state.field.trick_room > jnp.int8(0)

    p0_first, key, _ = sort_two_actions(
        jnp.where(p0_is_switch, ACTION_SWITCH, ACTION_MOVE),
        p0_priority, p0_speed,
        jnp.where(p1_is_switch, ACTION_SWITCH, ACTION_MOVE),
        p1_priority, p1_speed,
        trick_room, key,
    )

    first_side = jnp.where(p0_first, jnp.int32(0), jnp.int32(1))
    return first_side, key, p0_speed
