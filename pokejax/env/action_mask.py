"""
Legal action mask generation.

For each player, compute which of the 10 actions (4 moves + 6 switches)
are currently legal.

Legal move conditions:
  - Move has PP > 0
  - Move is not disabled
  - Not locked into a different move (Choice item, Encore)
  - If Encore: only the encored move is legal
  - Has at least one legal move; else Struggle is forced (action=0 with special handling)

Legal switch conditions:
  - Target slot is not active
  - Target Pokemon is not fainted
  - Not trapped (partially trapped, Mean Look, Block, Arena Trap, Shadow Tag)
"""

import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    VOL_PARTIALLY_TRAPPED, VOL_INGRAIN, VOL_ENCORE, VOL_CHOICELOCK,
    MAX_TEAM_SIZE, MAX_MOVES,
)


N_ACTIONS = 10  # 4 moves + 6 switches


def get_move_mask(state: BattleState, side: int) -> jnp.ndarray:
    """
    Returns bool[4] — which move slots are legal to choose.
    """
    idx = state.sides_active_idx[side]

    # PP available and not disabled
    pp = state.sides_team_move_pp[side, idx]           # int8[4]
    disabled = state.sides_team_move_disabled[side, idx]  # bool[4]
    move_ids = state.sides_team_move_ids[side, idx]       # int16[4]

    has_pp    = pp > jnp.int8(0)
    has_move  = move_ids >= jnp.int16(0)
    base_mask = has_pp & has_move & ~disabled

    # Encore: only the encored move slot is legal
    vols = state.sides_team_volatiles[side, idx]
    encore_active = (vols & jnp.uint32(1 << VOL_ENCORE)) != jnp.uint32(0)
    # Encored move slot stored in volatile_data[VOL_ENCORE] (reuse counter as slot index)
    encored_slot = state.sides_team_volatile_data[side, idx, VOL_ENCORE].astype(jnp.int32)
    slots = jnp.arange(4, dtype=jnp.int32)
    encore_mask = (slots == encored_slot)
    base_mask = jnp.where(encore_active, encore_mask & base_mask, base_mask)

    # Choice Lock: only the locked move slot is legal
    choicelock = (vols & jnp.uint32(1 << VOL_CHOICELOCK)) != jnp.uint32(0)
    locked_slot = state.sides_team_volatile_data[side, idx, VOL_CHOICELOCK].astype(jnp.int32)
    choice_mask = (slots == locked_slot)
    base_mask = jnp.where(choicelock, choice_mask & base_mask, base_mask)

    # If no legal move exists → Struggle (force all moves legal so the policy can pick)
    any_legal = base_mask.any()
    return jnp.where(any_legal, base_mask, jnp.ones(4, dtype=jnp.bool_))


def is_trapped(state: BattleState, side: int) -> jnp.ndarray:
    """
    Returns True if the active Pokemon cannot switch out.
    Checks: partial trap, ingrain, future trapping abilities (placeholder).
    """
    idx = state.sides_active_idx[side]
    vols = state.sides_team_volatiles[side, idx]

    partially_trapped = (vols & jnp.uint32(1 << VOL_PARTIALLY_TRAPPED)) != jnp.uint32(0)
    ingrained         = (vols & jnp.uint32(1 << VOL_INGRAIN))            != jnp.uint32(0)

    return partially_trapped | ingrained


def get_switch_mask(state: BattleState, side: int) -> jnp.ndarray:
    """
    Returns bool[6] — which team slots can be switched to.
    """
    active = state.sides_active_idx[side]
    slots  = jnp.arange(6, dtype=jnp.int32)

    not_active  = slots != active.astype(jnp.int32)
    not_fainted = ~state.sides_team_fainted[side]

    base_switch = not_active & not_fainted

    # Trapped: cannot switch
    trapped = is_trapped(state, side)
    return jnp.where(trapped, jnp.zeros(6, dtype=jnp.bool_), base_switch)


def get_action_mask(state: BattleState, side: int) -> jnp.ndarray:
    """
    Full action mask: bool[10].
    [0:4] = move mask, [4:10] = switch mask.
    """
    move_mask   = get_move_mask(state, side)    # bool[4]
    switch_mask = get_switch_mask(state, side)  # bool[6]
    return jnp.concatenate([move_mask, switch_mask])
