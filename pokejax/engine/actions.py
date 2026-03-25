"""
Move and switch action resolution.

Implements:
  - execute_move_action: full move resolution for one attacker
  - execute_switch_action: resolve a voluntary switch
  - check_fainted: check if a Pokemon fainted and mark it
  - check_win: check if the battle is over

This module orchestrates hit_pipeline, switch, and conditions.
"""

import jax
import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    VOL_PROTECT, VOL_CHOICELOCK, VOL_RECHARGING, VOL_CHARGING, VOL_DESTINYBOND,
    VOL_TAUNT, VOL_FLINCH, VOL_LOCKEDMOVE,
    STATUS_NONE,
    CATEGORY_STATUS,
    MAX_TEAM_SIZE,
)
from pokejax.core.damage import (
    MF_CATEGORY, MF_BASE_POWER, MF_FLAGS_LO, FLAG_RECHARGE,
    apply_damage, apply_heal, fraction_of_max_hp,
)
from pokejax.core.state import (
    get_active_idx, set_fainted, set_active, set_last_move,
    set_volatile, clear_volatiles, deduct_pp,
)
from pokejax.core import rng as rng_utils
from pokejax.engine.hit_pipeline import execute_move_hit
from pokejax.engine.switch import switch_out, switch_in, force_switch
from pokejax.mechanics.moves import execute_move_effects


def check_fainted(state: BattleState, side: int) -> BattleState:
    """
    Check if the active Pokemon has fainted (HP == 0) and update state.
    Does not handle forced switch-in — that's done in the turn loop.
    """
    idx = state.sides_active_idx[side]
    hp = state.sides_team_hp[side, idx]
    just_fainted = (hp <= jnp.int16(0)) & ~state.sides_team_fainted[side, idx]

    new_fainted = state.sides_team_fainted.at[side, idx].set(
        state.sides_team_fainted[side, idx] | just_fainted
    )
    new_left = state.sides_pokemon_left.at[side].set(
        jnp.where(just_fainted,
                   state.sides_pokemon_left[side] - jnp.int8(1),
                   state.sides_pokemon_left[side])
    )
    return state._replace(sides_team_fainted=new_fainted, sides_pokemon_left=new_left)


def check_win(state: BattleState) -> BattleState:
    """Check if one side has no Pokemon left and set finished/winner."""
    p0_left = state.sides_pokemon_left[0]
    p1_left = state.sides_pokemon_left[1]

    p0_lost = p0_left <= jnp.int8(0)
    p1_lost = p1_left <= jnp.int8(0)

    both_lost = p0_lost & p1_lost
    finished  = p0_lost | p1_lost

    winner = jnp.where(both_lost, jnp.int8(2),    # draw
              jnp.where(p0_lost,  jnp.int8(1),    # P1 wins
              jnp.where(p1_lost,  jnp.int8(0),    # P0 wins
                                  jnp.int8(-1))))  # ongoing

    return state._replace(
        finished=finished | state.finished,
        winner=jnp.where(state.finished, state.winner, winner),
    )


def get_move_id_for_action(state: BattleState, side: int,
                            move_slot: jnp.ndarray) -> jnp.ndarray:
    """Get the move ID for a given move slot of the active Pokemon."""
    idx = state.sides_active_idx[side]
    move_slot_clamped = jnp.clip(move_slot.astype(jnp.int32), 0, 3)
    return state.sides_team_move_ids[side, idx, move_slot_clamped]


def execute_move_action(
    tables,
    state: BattleState,
    atk_side: int,
    move_slot: jnp.ndarray,   # int8: which move slot (0-3)
    key: jnp.ndarray,
    cfg,
) -> tuple[BattleState, jnp.ndarray]:
    """
    Execute one move for `atk_side`.

    Steps:
      1. Check if attacker can move (status, faint)
      2. Deduct PP
      3. Run BeforeMove checks (paralysis, sleep, confusion, freeze)
      4. Run the 8-step hit pipeline against defender
      5. Check for faints
      6. Update last_move tracking

    Returns: (new_state, new_key)
    """
    def_side = 1 - atk_side
    atk_idx  = state.sides_active_idx[atk_side]
    def_idx  = state.sides_active_idx[def_side]

    # Get move ID
    move_slot_i = jnp.clip(move_slot.astype(jnp.int32), 0, 3)

    # Locked move (Outrage/Thrash/Petal Dance): force use of last-used move slot
    atk_locked = (state.sides_team_volatiles[atk_side, atk_idx]
                  & jnp.uint32(1 << VOL_LOCKEDMOVE)) != jnp.uint32(0)
    locked_move_id = state.sides_team_last_move_id[atk_side, atk_idx]
    # Find the slot (0-3) that matches the locked move
    locked_slot = jnp.int32(0)
    for _s in range(4):
        sm = state.sides_team_move_ids[atk_side, atk_idx, _s]
        locked_slot = jnp.where(
            (sm == locked_move_id) & (locked_move_id >= jnp.int16(0)),
            jnp.int32(_s), locked_slot,
        )
    move_slot_i = jnp.where(atk_locked, locked_slot, move_slot_i)

    move_id = state.sides_team_move_ids[atk_side, atk_idx, move_slot_i]
    move_id = jnp.maximum(jnp.int16(0), move_id)  # clamp invalid slots

    # Check if attacker is already fainted (skip)
    atk_fainted = state.sides_team_fainted[atk_side, atk_idx]

    # Check if must recharge (Hyper Beam etc.)
    recharging = (state.sides_team_volatiles[atk_side, atk_idx]
                  & jnp.uint32(1 << VOL_RECHARGING)) != jnp.uint32(0)

    # Clear recharge volatile and Destiny Bond (expires when user takes any action)
    new_vols = (state.sides_team_volatiles[atk_side, atk_idx]
                & ~jnp.where(recharging, jnp.uint32(1 << VOL_RECHARGING), jnp.uint32(0))
                & ~jnp.uint32(1 << VOL_DESTINYBOND))
    new_vol_arr = state.sides_team_volatiles.at[atk_side, atk_idx].set(new_vols)
    state = state._replace(sides_team_volatiles=new_vol_arr)

    # BeforeMove checks (paralysis, sleep, freeze, confusion)
    from pokejax.mechanics.conditions import (
        check_paralysis_before_move, check_sleep_before_move,
        check_freeze_before_move, check_confusion_before_move,
    )

    can_move = ~atk_fainted & ~recharging

    # Paralysis (25% full paralysis in Gen 4)
    par_can_move, key, state = check_paralysis_before_move(state, atk_side, key, cfg)
    can_move = can_move & par_can_move

    # Sleep (cannot act while asleep)
    slp_can_move, key, state = check_sleep_before_move(state, atk_side, key)
    can_move = can_move & slp_can_move

    # Freeze (20% thaw chance; if thawed, Pokemon CAN act this turn — Gen 4+)
    frz_can_move, key, state = check_freeze_before_move(state, atk_side, key, cfg)
    can_move = can_move & frz_can_move

    # Confusion (deals self-hit and sets can_move=False if hit)
    conf_can_move, key, state = check_confusion_before_move(state, atk_side, key, tables)
    can_move = can_move & conf_can_move

    # Flinch: flinched Pokemon cannot act this turn
    atk_vols_flinch = state.sides_team_volatiles[atk_side, atk_idx]
    is_flinched = (atk_vols_flinch & jnp.uint32(1 << VOL_FLINCH)) != jnp.uint32(0)
    can_move = can_move & ~is_flinched

    # Taunt: cannot use status moves (CATEGORY_STATUS) while taunted
    atk_vols = state.sides_team_volatiles[atk_side, atk_idx]
    atk_taunted = (atk_vols & jnp.uint32(1 << VOL_TAUNT)) != jnp.uint32(0)
    move_category = tables.moves[move_id.astype(jnp.int32), MF_CATEGORY].astype(jnp.int32)
    is_status_move = (move_category == jnp.int32(CATEGORY_STATUS))
    can_move = can_move & ~(atk_taunted & is_status_move)

    # Attract/Infatuation: 50% chance not to act (Gen 4: randomChance(1, 2))
    from pokejax.types import VOL_ATTRACT as _VOL_ATTRACT
    atk_infatuated = (atk_vols & jnp.uint32(1 << _VOL_ATTRACT)) != jnp.uint32(0)
    key, attract_key = jax.random.split(key)
    attract_stops = atk_infatuated & (jax.random.randint(attract_key, shape=(), minval=0, maxval=2, dtype=jnp.int32) == jnp.int32(0))
    can_move = can_move & ~attract_stops

    # Torment: cannot use same move as last turn
    from pokejax.types import VOL_TORMENT as _VOL_TORMENT
    atk_tormented = (atk_vols & jnp.uint32(1 << _VOL_TORMENT)) != jnp.uint32(0)
    last_mid = state.sides_team_last_move_id[atk_side, atk_idx]
    same_as_last = (move_id.astype(jnp.int16) == last_mid) & (last_mid >= jnp.int16(0))
    can_move = can_move & ~(atk_tormented & same_as_last)

    # Deduct PP (always, even if frozen/paralyzed that turn)
    # Pressure: if the defender has Pressure, deduct 2 PP instead of 1
    from pokejax.mechanics.abilities import PRESSURE_ID
    def_ability = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)
    has_pressure = (PRESSURE_ID >= 0) & (def_ability == jnp.int32(PRESSURE_ID))
    pp_cost = jnp.where(has_pressure, jnp.int8(2), jnp.int8(1))
    state = deduct_pp(state, atk_side, atk_idx, move_slot_i, pp_cost)

    # ------------------------------------------------------------------
    # Two-turn move handling (Fly, Dig, Dive, Bounce, etc.)
    # Turn 1 (charge): set VOL_CHARGING, skip damage.
    # Turn 2 (release): clear VOL_CHARGING, deal damage normally.
    # ------------------------------------------------------------------
    from pokejax.data.move_effects_data import ME_TWO_TURN
    mid_i32 = move_id.astype(jnp.int32)
    effect_pre = tables.move_effects[mid_i32, 0].astype(jnp.int32)
    is_two_turn = (effect_pre == jnp.int32(ME_TWO_TURN)) & can_move

    currently_charging = ((state.sides_team_volatiles[atk_side, atk_idx]
                           & jnp.uint32(1 << VOL_CHARGING)) != jnp.uint32(0))
    will_charge  = is_two_turn & ~currently_charging  # first use → charge turn
    will_release = is_two_turn & currently_charging   # second use → release turn

    # Transition VOL_CHARGING state before the hit pipeline
    charge_vols   = state.sides_team_volatiles[atk_side, atk_idx] | jnp.uint32(1 << VOL_CHARGING)
    release_vols  = state.sides_team_volatiles[atk_side, atk_idx] & ~jnp.uint32(1 << VOL_CHARGING)
    new_tt_vols = jnp.where(will_charge, charge_vols,
                  jnp.where(will_release, release_vols,
                             state.sides_team_volatiles[atk_side, atk_idx]))
    state = state._replace(
        sides_team_volatiles=state.sides_team_volatiles.at[atk_side, atk_idx].set(new_tt_vols)
    )
    # Save state before hit pipeline so we can revert on charge turn
    state_pre_hit = state

    # Execute hit pipeline — returns cancelled so effects can be gated on it
    state, total_dmg, is_crit, key, move_cancelled = execute_move_hit(
        tables, state, atk_side, def_side, move_id, key, cfg
    )
    # Gate on can_move: if the attacker couldn't act, treat as cancelled
    move_cancelled = move_cancelled | ~can_move

    # On charge turn: revert hit-pipeline effects (no damage), mark cancelled
    state = jax.tree.map(
        lambda pre, post: jnp.where(will_charge, pre, post),
        state_pre_hit, state,
    )
    move_cancelled = move_cancelled | will_charge

    # Apply move-specific effects (stat boosts, hazards, screens, weather, etc.)
    state, key = execute_move_effects(
        tables, state, atk_side, def_side, move_id, move_cancelled, key, cfg
    )

    # Track last move used
    state = set_last_move(state, atk_side, atk_idx, move_id)

    # Mark that this Pokemon moved this turn
    new_moved = state.sides_team_move_this_turn.at[atk_side, atk_idx].set(True)
    state = state._replace(sides_team_move_this_turn=new_moved)

    # ------------------------------------------------------------------
    # Choice item lock: if attacker has Choice Band/Specs/Scarf and the
    # move was not cancelled, lock to this move slot for future turns.
    # Clears on switch-out (handled in switch.py).
    # ------------------------------------------------------------------
    from pokejax.mechanics.items import CHOICE_BAND_ID, CHOICE_SPECS_ID, CHOICE_SCARF_ID
    atk_item = state.sides_team_item_id[atk_side, atk_idx].astype(jnp.int32)
    has_choice = (
        ((CHOICE_BAND_ID >= 0) & (atk_item == jnp.int32(CHOICE_BAND_ID))) |
        ((CHOICE_SPECS_ID >= 0) & (atk_item == jnp.int32(CHOICE_SPECS_ID))) |
        ((CHOICE_SCARF_ID >= 0) & (atk_item == jnp.int32(CHOICE_SCARF_ID)))
    )
    should_lock = has_choice & ~move_cancelled
    atk_vols_choice = state.sides_team_volatiles[atk_side, atk_idx]
    new_vols_choice = jnp.where(
        should_lock,
        atk_vols_choice | jnp.uint32(1 << VOL_CHOICELOCK),
        atk_vols_choice,
    )
    # Store the locked move slot in volatile_data[VOL_CHOICELOCK]
    old_locked_slot = state.sides_team_volatile_data[atk_side, atk_idx, VOL_CHOICELOCK]
    new_locked_slot = jnp.where(should_lock, move_slot_i.astype(jnp.int8), old_locked_slot)
    new_vol_data_choice = state.sides_team_volatile_data.at[atk_side, atk_idx, VOL_CHOICELOCK].set(new_locked_slot)
    state = state._replace(
        sides_team_volatiles=state.sides_team_volatiles.at[atk_side, atk_idx].set(new_vols_choice),
        sides_team_volatile_data=new_vol_data_choice,
    )

    # Check for faints
    def_was_alive = ~state.sides_team_fainted[def_side, def_idx]
    state = check_fainted(state, def_side)

    # ------------------------------------------------------------------
    # Destiny Bond: if defender faints AND had VOL_DESTINYBOND, attacker
    # also faints (direct KO trigger — branchless).
    # ------------------------------------------------------------------
    def_just_fainted = def_was_alive & state.sides_team_fainted[def_side, def_idx]

    # ------------------------------------------------------------------
    # Recharge mechanic (Hyper Beam, Giga Impact, Blast Burn, etc.)
    # PS Gen 4: "if (!target.fainted)" → set mustrecharge volatile.
    # So recharge is required UNLESS the target just fainted this turn.
    # ------------------------------------------------------------------
    move_flags_lo = tables.moves[move_id.astype(jnp.int32), MF_FLAGS_LO].astype(jnp.int32)
    has_recharge_flag = (move_flags_lo & jnp.int32(FLAG_RECHARGE)) != jnp.int32(0)
    should_recharge = has_recharge_flag & ~move_cancelled & ~def_just_fainted
    old_atk_vols = state.sides_team_volatiles[atk_side, atk_idx]
    new_atk_vols_rc = jnp.where(
        should_recharge,
        old_atk_vols | jnp.uint32(1 << VOL_RECHARGING),
        old_atk_vols,
    )
    state = state._replace(
        sides_team_volatiles=state.sides_team_volatiles.at[atk_side, atk_idx].set(new_atk_vols_rc)
    )

    def_had_dbond = ((state.sides_team_volatiles[def_side, def_idx]
                      & jnp.uint32(1 << VOL_DESTINYBOND)) != jnp.uint32(0))
    dbond_trigger = def_just_fainted & def_had_dbond & ~move_cancelled
    atk_hp_after_dbond = jnp.where(dbond_trigger,
                                    jnp.int16(0),
                                    state.sides_team_hp[atk_side, atk_idx])
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[atk_side, atk_idx].set(atk_hp_after_dbond)
    )

    state = check_fainted(state, atk_side)  # recoil / crash damage / Destiny Bond

    # Check win condition
    state = check_win(state)

    # ------------------------------------------------------------------
    # U-turn / Baton Pass: user switches out after move executes
    # Auto-pivots to first available slot (branchless, JAX-compatible).
    # ------------------------------------------------------------------
    from pokejax.data.move_effects_data import ME_U_TURN, ME_BATON_PASS
    mid = move_id.astype(jnp.int32)
    effect_type = tables.move_effects[mid, 0].astype(jnp.int32)

    is_uturn     = (~move_cancelled) & (effect_type == jnp.int32(ME_U_TURN))
    is_baton_pass = (~move_cancelled) & (effect_type == jnp.int32(ME_BATON_PASS))
    needs_pivot   = is_uturn | is_baton_pass

    # Only pivot if the attacker is still alive and the battle isn't over
    atk_alive   = ~state.sides_team_fainted[atk_side, atk_idx]
    replacement = find_forced_switch_slot(state, atk_side)
    has_replacement = replacement >= jnp.int32(0)
    will_pivot  = needs_pivot & atk_alive & has_replacement & ~state.finished

    # Save boosts before switch_out clears them (for Baton Pass transfer)
    saved_boosts = state.sides_team_boosts[atk_side, atk_idx]

    # Execute the pivot switch (both branches traced; branchlessly selected)
    safe_slot = jnp.maximum(jnp.int32(0), replacement)
    pivoted = switch_out(state, atk_side, cfg)
    pivoted = switch_in(pivoted, atk_side, safe_slot, tables, cfg)

    # Baton Pass: restore saved boosts to the newly switched-in mon
    new_idx = pivoted.sides_active_idx[atk_side].astype(jnp.int32)
    restored_boosts = pivoted.sides_team_boosts.at[atk_side, new_idx].set(
        jnp.where(is_baton_pass, saved_boosts,
                  pivoted.sides_team_boosts[atk_side, new_idx])
    )
    pivoted = pivoted._replace(sides_team_boosts=restored_boosts)
    pivoted = check_fainted(pivoted, atk_side)
    pivoted = check_win(pivoted)

    # Branchless select between pivoted and non-pivoted state
    state = jax.tree.map(
        lambda x, y: jnp.where(will_pivot, x, y),
        pivoted, state,
    )

    # ------------------------------------------------------------------
    # ME_ROAR: force defender to switch to a random alive non-active mon.
    # Roar, Whirlwind, Dragon Tail, Circle Throw.
    # Fails if: cancelled, no valid replacement, defender is the only one left.
    # Entry hazards apply on the forced switch-in.
    # ------------------------------------------------------------------
    from pokejax.data.move_effects_data import ME_ROAR as _ME_ROAR
    effect_type_roar = tables.move_effects[move_id.astype(jnp.int32), 0].astype(jnp.int32)
    is_roar = (~move_cancelled) & (effect_type_roar == jnp.int32(_ME_ROAR))

    # Build validity mask for defender's team (alive, not currently active)
    def_active_now = state.sides_active_idx[def_side].astype(jnp.int32)
    slots_i32 = jnp.arange(6, dtype=jnp.int32)
    roar_valid = (
        ~state.sides_team_fainted[def_side] &
        (slots_i32 != def_active_now)
    )  # bool[6]
    has_roar_target = roar_valid.any()

    # Pick random slot from valid targets (branchless):
    # Generate per-slot random values, zero out invalid, pick argmax.
    key, roar_key = rng_utils.split(key)
    rand_vals = jax.random.uniform(roar_key, shape=(6,), dtype=jnp.float32)
    masked_rand = jnp.where(roar_valid, rand_vals, jnp.float32(-1.0))
    roar_slot = jnp.argmax(masked_rand).astype(jnp.int32)

    will_roar = is_roar & has_roar_target & ~state.finished

    # Compute forced-switch state (always traced, branchlessly selected)
    safe_roar_slot = jnp.maximum(jnp.int32(0), roar_slot)
    roared_state = force_switch(state, def_side, safe_roar_slot, tables, cfg)
    roared_state = check_fainted(roared_state, def_side)
    roared_state = check_win(roared_state)

    state = jax.tree.map(
        lambda x, y: jnp.where(will_roar, x, y),
        roared_state, state,
    )

    return state, key


def execute_switch_action(
    tables,
    state: BattleState,
    side: int,
    target_slot: jnp.ndarray,   # int8: which team slot to switch to
    key: jnp.ndarray,
    cfg,
) -> tuple[BattleState, jnp.ndarray]:
    """
    Execute a voluntary switch for `side`.

    Steps:
      1. Switch-out current active Pokemon
      2. Switch-in the new Pokemon (entry hazards, SwitchIn abilities)
    """
    state = switch_out(state, side, cfg)
    state = switch_in(state, side, target_slot, tables, cfg)

    # Check for faints from entry hazard damage
    state = check_fainted(state, side)
    state = check_win(state)

    return state, key


def find_forced_switch_slot(state: BattleState, side: int) -> jnp.ndarray:
    """
    Find the first valid (alive, not active) Pokemon for a forced replacement.
    Returns -1 if no valid replacement exists.
    """
    active = state.sides_active_idx[side]
    alive  = ~state.sides_team_fainted[side]           # bool[6]
    slots  = jnp.arange(6, dtype=jnp.int32)
    valid  = alive & (slots != active.astype(jnp.int32))

    # Find first valid slot (argmax on bool array returns first True)
    first_valid = jnp.argmax(valid)
    any_valid   = valid.any()
    return jnp.where(any_valid, first_valid, jnp.int32(-1))
