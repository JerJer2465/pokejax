"""
Switch-in and switch-out logic.

Handles:
  - Switch-out: clear turn-limited volatiles, reset toxic counter (Gen 5+),
    not clearing boosts (Gen 4: boosts cleared on switch, Gen 5+: not)
  - Switch-in: set active Pokemon, apply entry hazards, trigger SwitchIn abilities
"""

import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    STATUS_TOX, STATUS_PSN,
    VOL_CONFUSED, VOL_FLINCH, VOL_PROTECT, VOL_ENCORE, VOL_TAUNT,
    VOL_HEALBLOCK, VOL_EMBARGO, VOL_CHARGING, VOL_RECHARGING,
    VOL_LOCKEDMOVE, VOL_CHOICELOCK, VOL_DISABLE, VOL_YAWN,
    VOL_PARTIALLY_TRAPPED, VOL_SEEDED, VOL_SUBSTITUTE,
    VOL_INGRAIN, VOL_MAGICCOAT, VOL_SNATCH,
    MAX_VOLATILES,
)
from pokejax.core.state import (
    set_active, set_volatile, clear_volatiles, reset_boosts,
)
from pokejax.mechanics.conditions import apply_entry_hazards
from pokejax.mechanics.events import run_event_switch_in, run_event_switch_out


# Volatile bits that are cleared on switch-out
# (single-turn, trapping, and other per-active-period effects)
_CLEAR_ON_SWITCH_OUT = [
    VOL_CONFUSED, VOL_FLINCH, VOL_PROTECT, VOL_ENCORE, VOL_TAUNT,
    VOL_HEALBLOCK, VOL_EMBARGO, VOL_CHARGING, VOL_RECHARGING,
    VOL_LOCKEDMOVE, VOL_DISABLE, VOL_YAWN,
    VOL_MAGICCOAT, VOL_SNATCH,
]
# Leech Seed (VOL_SEEDED), Substitute (VOL_SUBSTITUTE), Ingrain (VOL_INGRAIN),
# Choice Lock (VOL_CHOICELOCK), Partial Trap (VOL_PARTIALLY_TRAPPED)
# persist or are handled elsewhere.

# Volatiles that are always cleared on any switch
_ALWAYS_CLEAR = [
    VOL_CONFUSED, VOL_FLINCH, VOL_PROTECT, VOL_CHARGING, VOL_RECHARGING,
    VOL_MAGICCOAT, VOL_SNATCH, VOL_DISABLE, VOL_YAWN,
    VOL_PARTIALLY_TRAPPED, VOL_SEEDED,  # Gen 4: trap/seed cleared on switch
]


def switch_out(state: BattleState, side: int, cfg) -> BattleState:
    """
    Handle switch-out for the active Pokemon on `side`.

    Gen 4 effects:
      - Clear all volatile statuses
      - Reset stat boosts
      - Reset toxic counter (but keep status as PSN for Gen 4; Gen 5+ resets to PSN)
    """
    idx = state.sides_active_idx[side]

    # Clear volatile statuses (all in Gen 4)
    state = clear_volatiles(state, side, idx)

    # Gen 4: reset stat boosts on switch
    # Gen 5+: boosts persist (Baton Pass excluded — handled separately)
    state = reset_boosts(state, side, idx)

    # Toxic counter: in Gen 4, toxic counter resets to 0 (stays as status TOX)
    is_tox = state.sides_team_status[side, idx] == jnp.int8(STATUS_TOX)
    new_turns = jnp.where(is_tox, jnp.int8(1), state.sides_team_status_turns[side, idx])
    new_turns_arr = state.sides_team_status_turns.at[side, idx].set(new_turns)
    state = state._replace(sides_team_status_turns=new_turns_arr)

    # Choice Lock: clear (choice item allows new move selection on switch)
    state = set_volatile(state, side, idx, VOL_CHOICELOCK, False)

    # Clear move_this_turn flag
    new_moved = state.sides_team_move_this_turn.at[side, idx].set(False)
    state = state._replace(sides_team_move_this_turn=new_moved)

    # SwitchOut ability effects (e.g., Natural Cure clears status)
    state = run_event_switch_out(state, side, idx)

    return state


def switch_in(state: BattleState, side: int, new_slot: int,
              tables, cfg) -> BattleState:
    """
    Handle switch-in of the Pokemon at team slot `new_slot` on `side`.

    Steps:
      1. Set new_slot as active
      2. Apply entry hazards (Stealth Rock, Spikes, etc.)
      3. Trigger SwitchIn ability effects (delegated to events system)
    """
    # Set the new Pokemon as active
    state = set_active(state, side, new_slot)

    # Apply entry hazards (tables needed for Stealth Rock type-effectiveness)
    state = apply_entry_hazards(state, side, tables)

    # SwitchIn ability effects (Intimidate, Drizzle, Download, etc.)
    new_idx = state.sides_active_idx[side]
    state = run_event_switch_in(state, side, new_idx)

    return state


def force_switch(state: BattleState, side: int, new_slot: jnp.ndarray,
                 tables, cfg) -> BattleState:
    """
    Forced switch (e.g., from Roar, Whirlwind, Dragon Tail, phazing).
    Skips switch-out effects (no choice on switching out).
    """
    state = clear_volatiles(state, side, state.sides_active_idx[side])
    state = reset_boosts(state, side, state.sides_active_idx[side])
    state = set_active(state, side, new_slot)
    state = apply_entry_hazards(state, side, tables)
    return state


def is_valid_switch_target(state: BattleState, side: int,
                            slot: int) -> jnp.ndarray:
    """
    Returns True if `slot` is a valid switch target:
      - Not the currently active Pokemon
      - Not fainted
    """
    active = state.sides_active_idx[side]
    is_active = jnp.int8(slot) == active
    is_fainted = state.sides_team_fainted[side, slot]
    return ~is_active & ~is_fainted


def get_valid_switch_mask(state: BattleState, side: int) -> jnp.ndarray:
    """
    Returns bool[6] mask of valid switch targets for a side.
    """
    active = state.sides_active_idx[side]
    slots = jnp.arange(6, dtype=jnp.int8)
    not_active  = slots != active
    not_fainted = ~state.sides_team_fainted[side]
    return not_active & not_fainted
