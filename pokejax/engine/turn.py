"""
Turn loop orchestrator — JIT-compatible.

execute_turn(state, reveal, actions, tables, cfg) -> tuple[BattleState, RevealState]

Implements the full Gen 4 turn sequence:
  1. Sort actions by priority/speed
  2. Execute first player's action (lax.cond: switch vs move)
  3. Check faint / forced switch
  4. Execute second player's action (lax.cond: skip if finished)
  5. Check faint / forced switch
  6. Field residual (weather damage, terrain heal)
  7. Status residual (burn, poison, sleep, freeze)
  8. Volatile / side condition ticking
  9. Weather / terrain timer tick
 10. Increment turn counter, clear single-turn flags

All Python-level branches on JAX-traced values have been replaced with
jax.lax.cond so this function is fully jit-compatible.
"""

import jax
import jax.numpy as jnp

from pokejax.types import BattleState, RevealState
from pokejax.core import rng as rng_utils
from pokejax.core.priority import compute_turn_order
from pokejax.engine.actions import (
    execute_move_action, execute_switch_action,
    find_forced_switch_slot, check_fainted, check_win,
)
from pokejax.engine.field import apply_field_residual, tick_all_field_timers, apply_terrain_residual
from pokejax.mechanics.conditions import apply_residual
from pokejax.engine.switch import switch_in


# Action encoding for external callers:
#   actions[side] = encoded_action (int32)
#   encoding: 0-3  → use move slot 0-3
#             4-9  → switch to team slot 0-5
#   10+: reserved (Dynamax, etc.)

def decode_action(action: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Decode an action integer.
    Returns: (is_switch: bool, move_slot: int8, switch_slot: int8)
    """
    is_switch   = action >= jnp.int32(4)
    move_slot   = jnp.clip(action, 0, 3).astype(jnp.int8)
    switch_slot = (action - jnp.int32(4)).astype(jnp.int8)
    return is_switch, move_slot, switch_slot


def _select_state(cond: jnp.ndarray, true_state: BattleState,
                   false_state: BattleState) -> BattleState:
    """
    Branchlessly select between two BattleStates field-by-field.
    Required for JIT compatibility.
    """
    return jax.tree_util.tree_map(
        lambda x, y: jnp.where(cond, x, y),
        true_state, false_state
    )


def handle_forced_replacement_jit(state: BattleState, side: int,
                                   tables, cfg,
                                   key: jnp.ndarray) -> tuple[BattleState, jnp.ndarray]:
    """JIT-compatible forced replacement using branchless select."""
    idx = state.sides_active_idx[side]
    fainted = state.sides_team_fainted[side, idx]

    replacement = find_forced_switch_slot(state, side)
    has_replacement = replacement >= jnp.int32(0)
    should_switch = fainted & has_replacement & ~state.finished

    # Compute switched state (both branches traced in JIT)
    safe_slot = jnp.maximum(jnp.int32(0), replacement)
    switched_state = switch_in(state, side, safe_slot, tables, cfg)
    switched_state = check_fainted(switched_state, side)

    # Select branchlessly
    result = _select_state(should_switch, switched_state, state)
    return result, key


def _exec_action(state: BattleState, key: jnp.ndarray, side: int,
                  is_switch: jnp.ndarray,
                  move_slot: jnp.ndarray, switch_slot: jnp.ndarray,
                  tables, cfg) -> tuple[BattleState, jnp.ndarray]:
    """
    Execute move or switch for a given side.
    `side` must be a Python int (compile-time constant).
    `is_switch`, `move_slot`, `switch_slot` are JAX arrays.
    """
    def do_switch(sk):
        s, k = sk
        return execute_switch_action(tables, s, side, switch_slot, k, cfg)

    def do_move(sk):
        s, k = sk
        return execute_move_action(tables, s, side, move_slot, k, cfg)

    return jax.lax.cond(is_switch, do_switch, do_move, (state, key))


def _update_reveal(
    reveal: RevealState,
    state_after: BattleState,
    pre_active: jnp.ndarray,       # int8[2] active indices before action phase
    p0_is_switch: jnp.ndarray,
    p0_move_slot: jnp.ndarray,
    p0_switch_slot: jnp.ndarray,
    p1_is_switch: jnp.ndarray,
    p1_move_slot: jnp.ndarray,
    p1_switch_slot: jnp.ndarray,
) -> RevealState:
    """Update RevealState based on what happened this turn.

    All operations are branchless jnp.where / masking so this is fully
    JIT- and vmap-compatible.

    Move revelation: if player p actually moved (sides_team_move_this_turn),
      the opponent sees the move slot used on the pre-action active slot.
    Switch revelation: when any switch happens (voluntary or forced after faint),
      detected via change in sides_active_idx, the new lead is revealed.
    """
    # ----- Move revelations -----
    pre_active_0 = pre_active[0]
    pre_active_1 = pre_active[1]

    p0_did_move = state_after.sides_team_move_this_turn[0, pre_active_0]
    p1_did_move = state_after.sides_team_move_this_turn[1, pre_active_1]

    # P0 used move_slot → P1 gains knowledge: revealed_moves[1, pre_active_0, move_slot]
    p0_slot_mask = (jnp.arange(6, dtype=jnp.int32) == pre_active_0.astype(jnp.int32))  # [6]
    p0_move_mask = (jnp.arange(4, dtype=jnp.int32) == p0_move_slot.astype(jnp.int32))  # [4]
    p0_update = (p0_slot_mask[:, None] & p0_move_mask[None, :]) & (p0_did_move & ~p0_is_switch)
    new_moves = reveal.revealed_moves.at[1].set(reveal.revealed_moves[1] | p0_update)

    # P1 used move_slot → P0 gains knowledge: revealed_moves[0, pre_active_1, move_slot]
    p1_slot_mask = (jnp.arange(6, dtype=jnp.int32) == pre_active_1.astype(jnp.int32))  # [6]
    p1_move_mask = (jnp.arange(4, dtype=jnp.int32) == p1_move_slot.astype(jnp.int32))  # [4]
    p1_update = (p1_slot_mask[:, None] & p1_move_mask[None, :]) & (p1_did_move & ~p1_is_switch)
    new_moves = new_moves.at[0].set(new_moves[0] | p1_update)

    # ----- Pokemon revelations via active_idx change -----
    # Covers both voluntary switches AND forced switches after a faint.
    new_active_0 = state_after.sides_active_idx[0]
    new_active_1 = state_after.sides_active_idx[1]

    # P0's active changed → P1 sees it (revealed_pokemon[1, new_active_0])
    p0_switched = new_active_0 != pre_active_0
    p0_slot_new = (jnp.arange(6, dtype=jnp.int32) == new_active_0.astype(jnp.int32))
    new_pokemon = reveal.revealed_pokemon.at[1].set(
        reveal.revealed_pokemon[1] | (p0_slot_new & p0_switched)
    )

    # P1's active changed → P0 sees it (revealed_pokemon[0, new_active_1])
    p1_switched = new_active_1 != pre_active_1
    p1_slot_new = (jnp.arange(6, dtype=jnp.int32) == new_active_1.astype(jnp.int32))
    new_pokemon = new_pokemon.at[0].set(
        new_pokemon[0] | (p1_slot_new & p1_switched)
    )

    return reveal._replace(revealed_moves=new_moves, revealed_pokemon=new_pokemon)


def execute_turn(
    state: BattleState,
    reveal: RevealState,
    actions: jnp.ndarray,   # int32[2]: action for each player
    tables,
    cfg,
) -> tuple[BattleState, RevealState]:
    """
    Execute one full turn of battle. Fully JIT-compatible.

    This is the main entry point for the RL environment's step() function.
    actions[0] = P0's action, actions[1] = P1's action.
    Returns the updated (BattleState, RevealState) pair.
    """
    p0_action = actions[0]
    p1_action = actions[1]

    p0_is_switch, p0_move_slot, p0_switch_slot = decode_action(p0_action)
    p1_is_switch, p1_move_slot, p1_switch_slot = decode_action(p1_action)

    # Capture active indices before any actions for move/switch revelation
    pre_active = state.sides_active_idx  # int8[2]

    # --- Clear single-turn flags from last turn ---
    new_moved = jnp.zeros((2, 6), dtype=jnp.bool_)
    new_times_atk = jnp.zeros((2, 6), dtype=jnp.int8)
    # Clear Protect volatile (lasts 1 turn) and Flinch
    from pokejax.types import VOL_PROTECT, VOL_FLINCH
    p0_idx = state.sides_active_idx[0]
    p1_idx = state.sides_active_idx[1]
    vols0 = state.sides_team_volatiles[0, p0_idx] & ~jnp.uint32(1 << VOL_PROTECT) & ~jnp.uint32(1 << VOL_FLINCH)
    vols1 = state.sides_team_volatiles[1, p1_idx] & ~jnp.uint32(1 << VOL_PROTECT) & ~jnp.uint32(1 << VOL_FLINCH)
    new_vols = state.sides_team_volatiles.at[0, p0_idx].set(vols0).at[1, p1_idx].set(vols1)

    state = state._replace(
        sides_team_move_this_turn=new_moved,
        sides_team_times_attacked=new_times_atk,
        sides_team_volatiles=new_vols,
    )

    # --- PRNG: split key for this turn ---
    key = state.rng_key
    key, turn_key = rng_utils.split(key)
    state = state._replace(rng_key=key)

    # --- Sort actions: determine who goes first ---
    first_side, turn_key, _ = compute_turn_order(
        state,
        p0_move_slot, p1_move_slot,
        p0_is_switch, p1_is_switch,
        tables, cfg, turn_key,
    )

    # --- Execute actions in turn order via lax.cond ---
    # Two concrete branches (P0-first / P1-first) so that `side` is always
    # a compile-time Python int inside execute_move/switch_action.

    def exec_p0_first(sk: tuple) -> tuple:
        s, k = sk
        s, k = _exec_action(s, k, 0, p0_is_switch, p0_move_slot, p0_switch_slot, tables, cfg)
        s, k = handle_forced_replacement_jit(s, 1, tables, cfg, k)
        def p1_acts(sk2):
            s2, k2 = sk2
            return _exec_action(s2, k2, 1, p1_is_switch, p1_move_slot, p1_switch_slot, tables, cfg)
        s, k = jax.lax.cond(s.finished, lambda sk2: sk2, p1_acts, (s, k))
        s, k = handle_forced_replacement_jit(s, 0, tables, cfg, k)
        return s, k

    def exec_p1_first(sk: tuple) -> tuple:
        s, k = sk
        s, k = _exec_action(s, k, 1, p1_is_switch, p1_move_slot, p1_switch_slot, tables, cfg)
        s, k = handle_forced_replacement_jit(s, 0, tables, cfg, k)
        def p0_acts(sk2):
            s2, k2 = sk2
            return _exec_action(s2, k2, 0, p0_is_switch, p0_move_slot, p0_switch_slot, tables, cfg)
        s, k = jax.lax.cond(s.finished, lambda sk2: sk2, p0_acts, (s, k))
        s, k = handle_forced_replacement_jit(s, 1, tables, cfg, k)
        return s, k

    state, turn_key = jax.lax.cond(
        first_side == jnp.int32(0),
        exec_p0_first,
        exec_p1_first,
        (state, turn_key),
    )

    # --- Residual phase (skipped if battle is over) ---
    def do_residual(sk: tuple) -> tuple:
        s, k = sk

        # 1. Field weather damage
        s = apply_field_residual(s)

        # 2. Grassy terrain heal
        s = apply_terrain_residual(s, 0)
        s = apply_terrain_residual(s, 1)

        # 3. Status/volatile residual for each side
        s, k = apply_residual(s, 0, k, cfg)
        s, k = apply_residual(s, 1, k, cfg)

        # 3b. Ability + item residual (Speed Boost, Leftovers, Black Sludge, Shed Skin, etc.)
        from pokejax.mechanics.events import run_event_residual_state
        k, k_res0 = jax.random.split(k)
        s, k_res0 = run_event_residual_state(s, k_res0, 0, s.sides_active_idx[0])
        k, k_res1 = jax.random.split(k)
        s, k_res1 = run_event_residual_state(s, k_res1, 1, s.sides_active_idx[1])

        # 4. Check for residual faints
        s = check_fainted(s, 0)
        s = check_fainted(s, 1)
        s = check_win(s)

        # 5. Field timer ticking
        s = tick_all_field_timers(s)

        return s, k

    state, turn_key = jax.lax.cond(
        state.finished,
        lambda sk: sk,
        do_residual,
        (state, turn_key),
    )

    # --- Update RevealState based on actions taken ---
    reveal = _update_reveal(
        reveal, state,
        pre_active,
        p0_is_switch, p0_move_slot, p0_switch_slot,
        p1_is_switch, p1_move_slot, p1_switch_slot,
    )

    # --- Increment turn counter ---
    new_turn = state.turn + jnp.int16(1)
    state = state._replace(turn=new_turn, rng_key=turn_key)

    return state, reveal
