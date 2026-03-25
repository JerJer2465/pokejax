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
    VOL_RECHARGING, VOL_TAUNT,
    TYPE_FLYING, TYPE_GHOST, TYPE_STEEL,
    CATEGORY_STATUS,
    MAX_TEAM_SIZE, MAX_MOVES,
)


N_ACTIONS = 10  # 4 moves + 6 switches


def get_move_mask(state: BattleState, side: int, tables=None) -> jnp.ndarray:
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

    # Recharging: no moves are legal (engine auto-skips the turn)
    recharging = (vols & jnp.uint32(1 << VOL_RECHARGING)) != jnp.uint32(0)
    base_mask = jnp.where(recharging, jnp.zeros(4, dtype=jnp.bool_), base_mask)

    # Taunt: cannot use status (category=2) moves while taunted
    if tables is not None:
        taunted = (vols & jnp.uint32(1 << VOL_TAUNT)) != jnp.uint32(0)
        move_ids_i32 = move_ids.astype(jnp.int32)
        # Category field index 3 in move table
        categories = tables.moves[move_ids_i32, 3]  # MF_CATEGORY = 3
        is_status = categories == jnp.int8(CATEGORY_STATUS)
        base_mask = jnp.where(taunted, base_mask & ~is_status, base_mask)

    # If no legal move exists → Struggle (force all moves legal so the policy can pick)
    any_legal = base_mask.any()
    return jnp.where(any_legal, base_mask, jnp.ones(4, dtype=jnp.bool_))


def is_trapped(state: BattleState, side: int) -> jnp.ndarray:
    """
    Returns True if the active Pokemon cannot switch out.
    Checks: partial trap, ingrain, Arena Trap, Shadow Tag, Magnet Pull.
    """
    from pokejax.mechanics.abilities import (
        ARENA_TRAP_ID, SHADOW_TAG_ID, MAGNET_PULL_ID, LEVITATE_ID,
    )

    idx = state.sides_active_idx[side]
    vols = state.sides_team_volatiles[side, idx]

    partially_trapped = (vols & jnp.uint32(1 << VOL_PARTIALLY_TRAPPED)) != jnp.uint32(0)
    ingrained         = (vols & jnp.uint32(1 << VOL_INGRAIN))            != jnp.uint32(0)

    # Opponent's ability-based trapping
    opp = 1 - side
    opp_idx = state.sides_active_idx[opp]
    opp_ability = state.sides_team_ability_id[opp, opp_idx].astype(jnp.int32)

    own_types = state.sides_team_types[side, idx]
    own_t0 = own_types[0].astype(jnp.int32)
    own_t1 = own_types[1].astype(jnp.int32)
    own_ability = state.sides_team_ability_id[side, idx].astype(jnp.int32)

    # Arena Trap: traps grounded opponents (not Flying, not Levitate)
    is_flying = (own_t0 == jnp.int32(TYPE_FLYING)) | (own_t1 == jnp.int32(TYPE_FLYING))
    has_levitate = (LEVITATE_ID >= 0) & (own_ability == jnp.int32(LEVITATE_ID))
    is_grounded = ~is_flying & ~has_levitate
    arena_trap = (ARENA_TRAP_ID >= 0) & (opp_ability == jnp.int32(ARENA_TRAP_ID)) & is_grounded

    # Shadow Tag: traps all (except Ghost-types in Gen 4)
    is_ghost = (own_t0 == jnp.int32(TYPE_GHOST)) | (own_t1 == jnp.int32(TYPE_GHOST))
    shadow_tag = (SHADOW_TAG_ID >= 0) & (opp_ability == jnp.int32(SHADOW_TAG_ID)) & ~is_ghost

    # Magnet Pull: traps Steel-types
    is_steel = (own_t0 == jnp.int32(TYPE_STEEL)) | (own_t1 == jnp.int32(TYPE_STEEL))
    magnet_pull = (MAGNET_PULL_ID >= 0) & (opp_ability == jnp.int32(MAGNET_PULL_ID)) & is_steel

    return partially_trapped | ingrained | arena_trap | shadow_tag | magnet_pull


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

    # Recharging: cannot switch
    vols = state.sides_team_volatiles[side, active]
    recharging = (vols & jnp.uint32(1 << VOL_RECHARGING)) != jnp.uint32(0)

    cant_switch = trapped | recharging
    return jnp.where(cant_switch, jnp.zeros(6, dtype=jnp.bool_), base_switch)


def get_action_mask(state: BattleState, side: int, tables=None) -> jnp.ndarray:
    """
    Full action mask: bool[10].
    [0:4] = move mask, [4:10] = switch mask.
    Pass tables to enable Taunt enforcement in move mask.
    """
    move_mask   = get_move_mask(state, side, tables)  # bool[4]
    switch_mask = get_switch_mask(state, side)         # bool[6]
    return jnp.concatenate([move_mask, switch_mask])
