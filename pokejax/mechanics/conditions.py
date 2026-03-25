"""
Status condition, volatile status, and weather/terrain residual effects.

All functions are pure and branchless (jnp.where throughout).
They take a BattleState and return a new BattleState.

Corresponds to Showdown's data/conditions.ts handlers:
  - onResidual for each status/volatile/weather
  - onStart/onEnd for setup/teardown
  - onModify* for in-battle modifications

Conventions:
  - side: 0 or 1 (P1 or P2)
  - slot: 0-5 (team index)
  - Active Pokemon is always at sides_active_idx[side]
"""

import jax
import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX,
    STATUS_SLP, STATUS_FRZ, STATUS_PAR,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    SC_REFLECT, SC_LIGHTSCREEN, SC_AURORAVEIL, SC_TAILWIND, SC_SAFEGUARD, SC_MIST,
    SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
    VOL_CONFUSED, VOL_PARTIALLY_TRAPPED, VOL_SEEDED, VOL_SUBSTITUTE,
    VOL_PROTECT, VOL_ENCORE, VOL_TAUNT, VOL_HEALBLOCK, VOL_EMBARGO,
    VOL_INGRAIN, VOL_YAWN, VOL_RECHARGING, VOL_LOCKEDMOVE, VOL_CHOICELOCK,
    VOL_FOCUSENERGY, VOL_DISABLE, VOL_PERISH, VOL_NIGHTMARE, VOL_CURSE,
    TYPE_ROCK, TYPE_FLYING, TYPE_GROUND, TYPE_STEEL, TYPE_ICE,
    TYPE_FIRE, TYPE_WATER, TYPE_POISON,
)
from pokejax.core.state import (
    get_active_idx, get_active_hp, get_active_status, get_active_ability,
    get_active_volatiles, get_active_volatile_data,
    set_status, set_volatile, set_volatile_counter, clear_volatiles,
    set_fainted, set_hp,
    has_active_volatile, has_volatile, set_side_condition,
)
from pokejax.core.damage import apply_damage, apply_heal, fraction_of_max_hp
from pokejax.core import rng as rng_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_slot(state: BattleState, side: int) -> jnp.ndarray:
    return state.sides_active_idx[side]


def _active_type_is(state: BattleState, side: int, type_id: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    types = state.sides_team_types[side, idx]
    return (types[0] == jnp.int8(type_id)) | (types[1] == jnp.int8(type_id))


def _get_ability(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_ability_id[side, idx]


def _get_item(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_item_id[side, idx]


# ---------------------------------------------------------------------------
# Non-volatile status: residual damage
# ---------------------------------------------------------------------------

def apply_burn_residual(state: BattleState, side: int, cfg) -> BattleState:
    """
    Burn: deals 1/burn_damage_denom of max HP per turn.
    Gen 4: 1/8. Gen 7+: 1/16.
    """
    idx = _active_slot(state, side)
    is_burned = state.sides_team_status[side, idx] == jnp.int8(STATUS_BRN)
    dmg = fraction_of_max_hp(state, side, idx, 1, cfg.burn_damage_denom)
    # Apply damage only if burned, else 0
    dmg = jnp.where(is_burned, dmg, jnp.int32(0))
    new_hp = jnp.maximum(
        jnp.int16(0),
        state.sides_team_hp[side, idx] - dmg.astype(jnp.int16)
    )
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr)


def apply_poison_residual(state: BattleState, side: int) -> BattleState:
    """
    Poison: 1/8 max HP per turn.
    Toxic (badly poisoned): escalating 1/16 * (counter) per turn, capped at 15/16.
    status_turns tracks the toxic counter (0 for normal poison, 1-15 for toxic).
    """
    idx = _active_slot(state, side)
    status = state.sides_team_status[side, idx]

    is_psn = status == jnp.int8(STATUS_PSN)
    is_tox = status == jnp.int8(STATUS_TOX)
    is_either = is_psn | is_tox

    # Normal poison: 1/8 max HP
    psn_dmg = fraction_of_max_hp(state, side, idx, 1, 8)

    # Toxic: 1/16 * counter; counter increments each turn
    tox_counter = state.sides_team_status_turns[side, idx].astype(jnp.int32)
    tox_counter = jnp.clip(tox_counter, 1, 15)
    max_hp = state.sides_team_max_hp[side, idx].astype(jnp.int32)
    tox_dmg = jnp.maximum(jnp.int32(1), max_hp * tox_counter // 16)

    dmg = jnp.where(is_tox, tox_dmg, psn_dmg)
    dmg = jnp.where(is_either, dmg, jnp.int32(0))

    # Increment toxic counter for next turn
    new_tox_counter = jnp.where(
        is_tox,
        jnp.minimum(jnp.int8(15), state.sides_team_status_turns[side, idx] + jnp.int8(1)),
        state.sides_team_status_turns[side, idx]
    )
    new_turns = state.sides_team_status_turns.at[side, idx].set(new_tox_counter)

    new_hp = jnp.maximum(
        jnp.int16(0),
        state.sides_team_hp[side, idx] - dmg.astype(jnp.int16)
    )
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr, sides_team_status_turns=new_turns)


def apply_sleep_residual(state: BattleState, side: int,
                          key: jnp.ndarray) -> tuple[BattleState, jnp.ndarray]:
    """
    Sleep residual: no-op. Sleep counter is decremented in check_sleep_before_move
    (matching PS Gen 4 behaviour where the counter ticks at the START of the turn
    and the Pokemon wakes + CAN act when the counter hits 0).
    Kept for API compatibility.
    """
    return state, key


def apply_freeze_residual(state: BattleState, side: int,
                           key: jnp.ndarray) -> tuple[BattleState, jnp.ndarray]:
    """
    No-op: freeze thaw is handled at the start of the Pokemon's turn in
    check_freeze_before_move (Gen 4+ behaviour matches Showdown's onBeforeMove
    handler for the FRZ condition).  Kept for API compatibility.
    """
    return state, key


# ---------------------------------------------------------------------------
# Non-volatile status: BeforeMove checks
# ---------------------------------------------------------------------------

def check_paralysis_before_move(state: BattleState, side: int,
                                  key: jnp.ndarray,
                                  cfg) -> tuple[bool, jnp.ndarray, jnp.ndarray]:
    """
    Paralysis: paralysis_full_para_chance% cannot move.
    Returns (can_move, new_key, new_state).
    """
    idx = _active_slot(state, side)
    is_paralyzed = state.sides_team_status[side, idx] == jnp.int8(STATUS_PAR)
    key, subkey = rng_utils.split(key)
    full_para = rng_utils.rand_bool_pct(subkey, cfg.paralysis_full_para_chance)
    cannot_move = is_paralyzed & full_para
    return ~cannot_move, key, state


def check_sleep_before_move(state: BattleState, side: int,
                              key: jnp.ndarray) -> tuple[bool, jnp.ndarray, jnp.ndarray]:
    """
    Sleep: decrement the sleep counter at the START of the Pokemon's turn.
    If counter reaches 0, the Pokemon wakes up and CAN act this turn (PS Gen 4 behaviour).
    Early Bird: decrement by 2 instead of 1.
    """
    from pokejax.mechanics.abilities import EARLY_BIRD_ID

    idx = _active_slot(state, side)
    is_asleep = state.sides_team_status[side, idx] == jnp.int8(STATUS_SLP)

    # Early Bird: halve sleep duration
    ability_id = state.sides_team_ability_id[side, idx].astype(jnp.int32)
    has_early_bird = (EARLY_BIRD_ID >= 0) & (ability_id == jnp.int32(EARLY_BIRD_ID))
    decrement = jnp.where(has_early_bird, jnp.int8(2), jnp.int8(1))

    # Decrement sleep counter
    sleep_counter = state.sides_team_sleep_turns[side, idx]
    new_counter = jnp.maximum(jnp.int8(0), sleep_counter - decrement)
    woke_up = is_asleep & (new_counter == jnp.int8(0))

    # Clear status on wake
    new_status = jnp.where(woke_up, jnp.int8(STATUS_NONE),
                            state.sides_team_status[side, idx])
    new_sleep_arr = state.sides_team_sleep_turns.at[side, idx].set(
        jnp.where(is_asleep, new_counter, sleep_counter)
    )
    new_status_arr = state.sides_team_status.at[side, idx].set(new_status)
    state = state._replace(sides_team_status=new_status_arr,
                            sides_team_sleep_turns=new_sleep_arr)

    # Can move if: not asleep at all, OR woke up this turn
    can_move = ~is_asleep | woke_up
    return can_move, key, state


def check_freeze_before_move(state: BattleState, side: int,
                               key: jnp.ndarray,
                               cfg) -> tuple[jnp.ndarray, jnp.ndarray, BattleState]:
    """
    Freeze: 20% chance to thaw at the start of the Pokemon's turn (Gen 4+).
    If thawed this turn, the Pokemon CAN act.  If still frozen, it cannot move.
    Returns (can_move: bool, new_key, new_state).
    """
    idx = _active_slot(state, side)
    is_frozen = state.sides_team_status[side, idx] == jnp.int8(STATUS_FRZ)

    key, subkey = rng_utils.split(key)
    thawed = is_frozen & rng_utils.freeze_thaw_roll(subkey)

    new_status = jnp.where(thawed, jnp.int8(STATUS_NONE),
                            state.sides_team_status[side, idx])
    new_status_arr = state.sides_team_status.at[side, idx].set(new_status)
    state = state._replace(sides_team_status=new_status_arr)

    # Pokemon can move if it was never frozen, or if it just thawed this turn
    still_frozen = is_frozen & ~thawed
    return ~still_frozen, key, state


def check_confusion_before_move(state: BattleState, side: int,
                                  key: jnp.ndarray,
                                  tables=None) -> tuple[bool, jnp.ndarray, BattleState]:
    """
    Confusion: decrement counter. If it reaches 0, snap out.
    If still confused: Gen 4 = 50% chance to hurt self (typeless, physical, 40 bp).
    Damage uses boosted Attack/Defense (same as PS getDamage(pokemon, pokemon, 40)).
    Returns (can_move, new_key, new_state).
    """
    idx = _active_slot(state, side)
    is_confused = has_volatile(state, side, idx, VOL_CONFUSED)
    conf_turns  = state.sides_team_volatile_data[side, idx, VOL_CONFUSED]

    # Decrement counter
    new_turns = jnp.maximum(jnp.int8(0), conf_turns - jnp.int8(1))
    snapped_out = is_confused & (new_turns == jnp.int8(0))

    # Update counter and clear volatile if snapped out
    new_data = state.sides_team_volatile_data.at[side, idx, VOL_CONFUSED].set(
        jnp.where(is_confused, new_turns, conf_turns)
    )
    state = state._replace(sides_team_volatile_data=new_data)
    state = set_volatile(state, side, idx, VOL_CONFUSED,
                         is_confused & ~snapped_out)

    # Gen 4: 50% chance to hurt self (PS: randomChance(1, 2))
    key, subkey = rng_utils.split(key)
    self_hit = is_confused & ~snapped_out & rng_utils.speed_tie_roll(subkey).astype(jnp.bool_)

    # Confusion self-hit: 40 bp, typeless Physical, uses boosted Attack vs boosted Defense
    # PS: getDamage(pokemon, pokemon, 40) — includes stat boosts, no crit, no STAB/type
    atk_base  = state.sides_team_base_stats[side, idx, 1].astype(jnp.float32)  # ATK
    def_base  = state.sides_team_base_stats[side, idx, 2].astype(jnp.float32)  # DEF
    atk_boost = state.sides_team_boosts[side, idx, 0]  # BOOST_ATK = 0
    def_boost = state.sides_team_boosts[side, idx, 1]  # BOOST_DEF = 1
    level     = state.sides_team_level[side, idx].astype(jnp.int32)
    # Boost multiplier: max(2, 2+stage) / max(2, 2-stage)
    if tables is not None:
        atk_mult = tables.get_boost_multiplier(atk_boost)
        def_mult = tables.get_boost_multiplier(def_boost)
    else:
        # Fallback: compute inline without tables
        def _boost_mult(s):
            s = s.astype(jnp.float32)
            return jnp.maximum(2.0, 2.0 + s) / jnp.maximum(2.0, 2.0 - s)
        atk_mult = _boost_mult(atk_boost)
        def_mult = _boost_mult(def_boost)
    atk_eff  = jnp.maximum(jnp.int32(1), jnp.floor(atk_base * atk_mult).astype(jnp.int32))
    def_eff  = jnp.maximum(jnp.int32(1), jnp.floor(def_base * def_mult).astype(jnp.int32))
    conf_dmg = ((2 * level // 5 + 2) * 40 * atk_eff // def_eff) // 50 + 2
    conf_dmg = jnp.maximum(jnp.int32(1), conf_dmg)

    new_hp = jnp.where(
        self_hit,
        jnp.maximum(jnp.int16(0),
                    state.sides_team_hp[side, idx] - conf_dmg.astype(jnp.int16)),
        state.sides_team_hp[side, idx]
    )
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    state = state._replace(sides_team_hp=new_hp_arr)

    # Cannot move if confused and self-hit this turn
    can_move = ~self_hit
    return can_move, key, state


# ---------------------------------------------------------------------------
# Status: application (TrySetStatus → SetStatus)
# ---------------------------------------------------------------------------

def try_set_status(state: BattleState, side: int, slot: int,
                   new_status: jnp.ndarray,
                   key: jnp.ndarray,
                   cfg) -> tuple[BattleState, jnp.ndarray]:
    """
    Attempt to apply a status condition to a Pokemon.
    Fails if:
      - Already has a status
      - Is a type immune to the status (e.g., Fire can't be burned)
      - Safeguard is active for their side
      - Gen 4: sleep clause (already sleeping on side)
    """
    idx = _active_slot(state, side)
    current_status = state.sides_team_status[side, idx]
    already_statused = current_status != jnp.int8(STATUS_NONE)

    types = state.sides_team_types[side, idx]
    type0 = types[0].astype(jnp.int32)
    type1 = types[1].astype(jnp.int32)

    # Substitute blocks status application
    has_sub = (state.sides_team_volatiles[side, idx]
               & jnp.uint32(1 << VOL_SUBSTITUTE)) != jnp.uint32(0)

    # Type immunities
    fire_type  = (type0 == TYPE_FIRE)  | (type1 == TYPE_FIRE)
    steel_type = (type0 == TYPE_STEEL) | (type1 == TYPE_STEEL)
    ice_type   = (type0 == TYPE_ICE)   | (type1 == TYPE_ICE)

    burn_blocked  = (new_status == jnp.int8(STATUS_BRN)) & fire_type
    poison_blocked= ((new_status == jnp.int8(STATUS_PSN)) |
                     (new_status == jnp.int8(STATUS_TOX))) & (steel_type | poison_immune(type0, type1))
    freeze_blocked= (new_status == jnp.int8(STATUS_FRZ)) & ice_type

    type_blocked = burn_blocked | poison_blocked | freeze_blocked

    # Safeguard
    safeguard = state.sides_side_conditions[side, SC_SAFEGUARD] > jnp.int8(0)

    # Sleep Clause: only one opponent Pokemon asleep at a time
    # (Rest is exempt — it sets sleep directly, not through try_set_status)
    any_sleeping = jnp.any(
        (state.sides_team_status[side] == jnp.int8(STATUS_SLP)) &
        ~state.sides_team_fainted[side]
    )
    sleep_clause_blocks = (new_status == jnp.int8(STATUS_SLP)) & any_sleeping

    blocked = already_statused | type_blocked | safeguard | sleep_clause_blocks | has_sub

    # Set status
    init_turns = jnp.where(new_status == jnp.int8(STATUS_TOX), jnp.int8(1), jnp.int8(0))

    # Sleep: roll duration
    key, sleep_key = rng_utils.split(key)
    sleep_dur = rng_utils.sleep_roll(sleep_key)
    sleep_turns = jnp.where(new_status == jnp.int8(STATUS_SLP), sleep_dur, jnp.int8(0))

    final_status = jnp.where(blocked, current_status, new_status)
    final_turns  = jnp.where(blocked, state.sides_team_status_turns[side, idx], init_turns)
    final_sleep  = jnp.where(blocked, state.sides_team_sleep_turns[side, idx], sleep_turns)

    new_status_arr = state.sides_team_status.at[side, idx].set(final_status)
    new_turns_arr  = state.sides_team_status_turns.at[side, idx].set(final_turns)
    new_sleep_arr  = state.sides_team_sleep_turns.at[side, idx].set(final_sleep)
    state = state._replace(
        sides_team_status=new_status_arr,
        sides_team_status_turns=new_turns_arr,
        sides_team_sleep_turns=new_sleep_arr,
    )
    return state, key


def poison_immune(type0: jnp.ndarray, type1: jnp.ndarray) -> jnp.ndarray:
    """Poison-type is immune to being poisoned."""
    is_poison = (type0 == jnp.int32(TYPE_POISON)) | (type1 == jnp.int32(TYPE_POISON))
    return is_poison


# ---------------------------------------------------------------------------
# Volatile statuses
# ---------------------------------------------------------------------------

def apply_volatile_residuals(state: BattleState, side: int) -> BattleState:
    """
    Apply end-of-turn effects for all active volatile statuses.
    Handles: leech seed, partial trap, ingrain.
    (Confusion and other per-turn checks happen during BeforeMove.)
    """
    idx = _active_slot(state, side)
    opp_side = 1 - side

    # Leech Seed: drain 1/8 max HP, heal opponent
    seeded = has_volatile(state, side, idx, VOL_SEEDED)
    seed_dmg = fraction_of_max_hp(state, side, idx, 1, 8)
    seed_dmg = jnp.where(seeded, seed_dmg, jnp.int32(0))

    new_hp = jnp.maximum(
        jnp.int16(0),
        state.sides_team_hp[side, idx] - seed_dmg.astype(jnp.int16)
    )
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    state = state._replace(sides_team_hp=new_hp_arr)

    # Heal opponent with leech seed drain (up to max HP)
    opp_idx = state.sides_active_idx[opp_side]
    opp_max = state.sides_team_max_hp[opp_side, opp_idx]
    opp_hp  = state.sides_team_hp[opp_side, opp_idx]
    new_opp_hp = jnp.where(
        seeded,
        jnp.minimum(opp_max, opp_hp + seed_dmg.astype(jnp.int16)),
        opp_hp
    )
    new_opp_hp_arr = state.sides_team_hp.at[opp_side, opp_idx].set(new_opp_hp)
    state = state._replace(sides_team_hp=new_opp_hp_arr)

    # Partial Trap: 1/8 max HP per turn (Gen 5+); Gen 4: 1/16
    trapped = has_volatile(state, side, idx, VOL_PARTIALLY_TRAPPED)
    trap_turns = state.sides_team_volatile_data[side, idx, VOL_PARTIALLY_TRAPPED]
    trap_dmg = fraction_of_max_hp(state, side, idx, 1, 16)  # Gen 4: 1/16
    trap_dmg = jnp.where(trapped & (trap_turns > jnp.int8(0)), trap_dmg, jnp.int32(0))

    # Decrement trap counter, remove if expired
    new_trap_turns = jnp.maximum(jnp.int8(0), trap_turns - jnp.int8(1))
    trap_expired = trapped & (new_trap_turns == jnp.int8(0))
    new_data = state.sides_team_volatile_data.at[side, idx, VOL_PARTIALLY_TRAPPED].set(
        jnp.where(trapped, new_trap_turns, trap_turns)
    )
    state = state._replace(sides_team_volatile_data=new_data)
    state = set_volatile(state, side, idx, VOL_PARTIALLY_TRAPPED, trapped & ~trap_expired)

    new_hp = jnp.maximum(
        jnp.int16(0),
        state.sides_team_hp[side, idx] - trap_dmg.astype(jnp.int16)
    )
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    state = state._replace(sides_team_hp=new_hp_arr)

    # Ingrain: heal 1/16 max HP per turn
    ingrained = has_volatile(state, side, idx, VOL_INGRAIN)
    ingrain_heal = fraction_of_max_hp(state, side, idx, 1, 16)
    ingrain_heal = jnp.where(ingrained, ingrain_heal, jnp.int32(0))

    cur_hp  = state.sides_team_hp[side, idx]
    max_hp  = state.sides_team_max_hp[side, idx]
    new_hp  = jnp.minimum(max_hp, cur_hp + ingrain_heal.astype(jnp.int16))
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    state = state._replace(sides_team_hp=new_hp_arr)

    # Nightmare: lose 1/4 max HP per turn while asleep
    nightmared = has_volatile(state, side, idx, VOL_NIGHTMARE)
    is_asleep  = state.sides_team_status[side, idx] == jnp.int8(STATUS_SLP)
    nightmare_dmg = fraction_of_max_hp(state, side, idx, 1, 4)
    nightmare_dmg = jnp.where(nightmared & is_asleep, nightmare_dmg, jnp.int32(0))
    new_hp = jnp.maximum(jnp.int16(0),
                          state.sides_team_hp[side, idx] - nightmare_dmg.astype(jnp.int16))
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[side, idx].set(new_hp)
    )
    # Clear Nightmare if the Pokemon woke up (status no longer SLP)
    state = set_volatile(state, side, idx, VOL_NIGHTMARE, nightmared & is_asleep)

    # Ghost Curse: lose 1/4 max HP per turn
    cursed     = has_volatile(state, side, idx, VOL_CURSE)
    curse_dmg  = fraction_of_max_hp(state, side, idx, 1, 4)
    curse_dmg  = jnp.where(cursed, curse_dmg, jnp.int32(0))
    new_hp = jnp.maximum(jnp.int16(0),
                          state.sides_team_hp[side, idx] - curse_dmg.astype(jnp.int16))
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[side, idx].set(new_hp)
    )

    return state


def decrement_volatile_timers(state: BattleState, side: int,
                               key: jnp.ndarray) -> tuple[BattleState, jnp.ndarray]:
    """
    Decrement turn counters for time-limited volatiles and remove expired ones.
    Handles: Encore, Taunt, Heal Block, Embargo, Yawn.
    """
    idx = _active_slot(state, side)
    # Encore: packed encoding — bits[1:0] = slot, bits[7:2] = timer.
    # Decrement only the timer portion; keep slot bits intact.
    encore_active = has_volatile(state, side, idx, VOL_ENCORE)
    enc_data = state.sides_team_volatile_data[side, idx, VOL_ENCORE].astype(jnp.int32)
    enc_timer = enc_data >> 2
    enc_slot  = enc_data & jnp.int32(3)
    new_enc_timer = jnp.maximum(jnp.int32(0), enc_timer - jnp.int32(1))
    encore_expired = encore_active & (new_enc_timer == jnp.int32(0))
    new_enc_data = jnp.where(
        encore_active,
        ((new_enc_timer << 2) | enc_slot).astype(jnp.int8),
        state.sides_team_volatile_data[side, idx, VOL_ENCORE],
    )
    state = state._replace(
        sides_team_volatile_data=state.sides_team_volatile_data.at[side, idx, VOL_ENCORE].set(new_enc_data)
    )
    state = set_volatile(state, side, idx, VOL_ENCORE, encore_active & ~encore_expired)

    timed_vols = [
        (VOL_TAUNT, 3),     # Taunt lasts 3 turns
        (VOL_HEALBLOCK, 5), # Heal Block lasts 5 turns
        (VOL_EMBARGO, 5),   # Embargo lasts 5 turns
    ]

    for vol_bit, _ in timed_vols:
        active = has_volatile(state, side, idx, vol_bit)
        count  = state.sides_team_volatile_data[side, idx, vol_bit]
        new_count = jnp.maximum(jnp.int8(0), count - jnp.int8(1))
        expired = active & (new_count == jnp.int8(0))

        new_data = state.sides_team_volatile_data.at[side, idx, vol_bit].set(
            jnp.where(active, new_count, count)
        )
        state = state._replace(sides_team_volatile_data=new_data)
        state = set_volatile(state, side, idx, vol_bit, active & ~expired)

    # Locked move (Outrage/Thrash/Petal Dance): decrement counter, apply confusion on expiry
    locked_active = has_volatile(state, side, idx, VOL_LOCKEDMOVE)
    locked_count  = state.sides_team_volatile_data[side, idx, VOL_LOCKEDMOVE]
    new_locked_count = jnp.maximum(jnp.int8(0), locked_count - jnp.int8(1))
    locked_expired = locked_active & (new_locked_count == jnp.int8(0))

    new_locked_data = state.sides_team_volatile_data.at[side, idx, VOL_LOCKEDMOVE].set(
        jnp.where(locked_active, new_locked_count, locked_count)
    )
    state = state._replace(sides_team_volatile_data=new_locked_data)
    state = set_volatile(state, side, idx, VOL_LOCKEDMOVE, locked_active & ~locked_expired)

    # Apply confusion when lock expires (PS: always confused at end of lock)
    confusion_from_lock = locked_expired
    cur_confused = has_volatile(state, side, idx, VOL_CONFUSED)
    state = set_volatile(state, side, idx, VOL_CONFUSED, cur_confused | confusion_from_lock)
    key, conf_dur_key = rng_utils.split(key)
    conf_dur = rng_utils.confusion_roll(conf_dur_key)
    cur_conf_data = state.sides_team_volatile_data[side, idx, VOL_CONFUSED]
    new_conf_data = jnp.where(confusion_from_lock, conf_dur,
                               jnp.where(cur_confused, cur_conf_data, conf_dur))
    state = state._replace(
        sides_team_volatile_data=state.sides_team_volatile_data.at[side, idx, VOL_CONFUSED].set(
            jnp.where(confusion_from_lock, new_conf_data, cur_conf_data)
        )
    )

    # Disable: decrement timer; when expired, clear VOL_DISABLE bit and un-disable the move.
    dis_active = has_volatile(state, side, idx, VOL_DISABLE)
    dis_count = state.sides_team_volatile_data[side, idx, VOL_DISABLE]
    new_dis_count = jnp.maximum(jnp.int8(0), dis_count - jnp.int8(1))
    dis_expired = dis_active & (new_dis_count == jnp.int8(0))
    new_dis_data = state.sides_team_volatile_data.at[side, idx, VOL_DISABLE].set(
        jnp.where(dis_active, new_dis_count, dis_count)
    )
    state = state._replace(sides_team_volatile_data=new_dis_data)
    state = set_volatile(state, side, idx, VOL_DISABLE, dis_active & ~dis_expired)
    # Clear all move_disabled entries for this Pokemon when Disable expires
    for _ms in range(4):
        cur_d = state.sides_team_move_disabled[side, idx, _ms]
        state = state._replace(
            sides_team_move_disabled=state.sides_team_move_disabled.at[side, idx, _ms].set(
                jnp.where(dis_expired, jnp.bool_(False), cur_d)
            )
        )

    # Yawn: if counter reaches 0, apply sleep
    yawn_active = has_volatile(state, side, idx, VOL_YAWN)
    yawn_count  = state.sides_team_volatile_data[side, idx, VOL_YAWN]
    new_yawn_count = jnp.maximum(jnp.int8(0), yawn_count - jnp.int8(1))
    yawn_expired = yawn_active & (new_yawn_count == jnp.int8(0))

    new_data = state.sides_team_volatile_data.at[side, idx, VOL_YAWN].set(
        jnp.where(yawn_active, new_yawn_count, yawn_count)
    )
    state = state._replace(sides_team_volatile_data=new_data)
    state = set_volatile(state, side, idx, VOL_YAWN, yawn_active & ~yawn_expired)

    # Apply sleep when yawn expires (if not already statused)
    no_status = state.sides_team_status[side, idx] == jnp.int8(STATUS_NONE)
    apply_sleep = yawn_expired & no_status
    new_status = jnp.where(apply_sleep, jnp.int8(STATUS_SLP),
                            state.sides_team_status[side, idx])
    # Roll sleep duration (2-5 turns, PS Gen 4: random(2,6))
    key, sleep_key = rng_utils.split(key)
    sleep_dur = rng_utils.sleep_roll(sleep_key)
    new_sleep = jnp.where(apply_sleep, sleep_dur,
                           state.sides_team_sleep_turns[side, idx])
    new_status_arr = state.sides_team_status.at[side, idx].set(new_status)
    new_sleep_arr  = state.sides_team_sleep_turns.at[side, idx].set(new_sleep)
    state = state._replace(sides_team_status=new_status_arr,
                            sides_team_sleep_turns=new_sleep_arr)

    # ------------------------------------------------------------------
    # Perish Song countdown: decrement counter each turn, faint at 0.
    # Switching out clears the volatile (handled by switch_out/clear_volatiles).
    # ------------------------------------------------------------------
    perish_active = has_volatile(state, side, idx, VOL_PERISH)
    perish_count  = state.sides_team_volatile_data[side, idx, VOL_PERISH]
    new_perish_count = jnp.maximum(jnp.int8(0), perish_count - jnp.int8(1))
    perish_expired = perish_active & (new_perish_count == jnp.int8(0))

    # Update counter
    new_data = state.sides_team_volatile_data.at[side, idx, VOL_PERISH].set(
        jnp.where(perish_active, new_perish_count, perish_count)
    )
    state = state._replace(sides_team_volatile_data=new_data)
    # Clear volatile if expired
    state = set_volatile(state, side, idx, VOL_PERISH, perish_active & ~perish_expired)

    # Faint the Pokemon if Perish Song counter hit 0
    perish_faint_hp = jnp.where(perish_expired, jnp.int16(0),
                                 state.sides_team_hp[side, idx])
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[side, idx].set(perish_faint_hp)
    )

    return state, key


# ---------------------------------------------------------------------------
# Entry hazard damage (SwitchIn)
# ---------------------------------------------------------------------------

def apply_entry_hazards(state: BattleState, side: int, tables) -> BattleState:
    """
    Apply entry hazard damage when a Pokemon switches in.
    Handles: Stealth Rock, Spikes, Toxic Spikes, Sticky Web.

    tables is required for Stealth Rock type-effectiveness calculation.
    """
    from pokejax.core.damage import type_effectiveness

    idx   = state.sides_active_idx[side]
    types = state.sides_team_types[side, idx]
    t0    = types[0].astype(jnp.int32)
    t1    = types[1].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side, idx].astype(jnp.int32)

    # ----------------------------------------------------------------
    # Grounded check: Flying-type and (TODO) Levitate/Air Balloon are
    # immune to all ground-level hazards (Spikes, T-Spikes, Sticky Web).
    # Stealth Rock hits everyone regardless.
    # ----------------------------------------------------------------
    is_flying   = (t0 == jnp.int32(TYPE_FLYING)) | (t1 == jnp.int32(TYPE_FLYING))
    # Levitate grants ground immunity (Air Balloon is Gen 5+ only)
    from pokejax.mechanics.abilities import LEVITATE_ID
    ability_id = state.sides_team_ability_id[side, idx].astype(jnp.int32)
    has_levitate = (LEVITATE_ID >= 0) & (ability_id == jnp.int32(LEVITATE_ID))
    is_grounded = ~is_flying & ~has_levitate

    # ----------------------------------------------------------------
    # Stealth Rock: Rock-type effectiveness × max_hp / 8
    # Hits all Pokemon (even airborne ones).
    # ----------------------------------------------------------------
    has_rock = state.sides_side_conditions[side, SC_STEALTHROCK] > jnp.int8(0)
    rock_mult = type_effectiveness(tables, jnp.int32(TYPE_ROCK), t0, t1)
    rock_dmg  = jnp.floor(
        max_hp.astype(jnp.float32) * rock_mult / jnp.float32(8.0)
    ).astype(jnp.int32)
    rock_dmg  = jnp.where(has_rock, jnp.maximum(jnp.int32(1), rock_dmg), jnp.int32(0))

    # ----------------------------------------------------------------
    # Spikes: 1 layer=1/8, 2=1/6, 3=1/4; grounded Pokemon only.
    # ----------------------------------------------------------------
    spike_layers = state.sides_side_conditions[side, SC_SPIKES].astype(jnp.int32)
    spike_denoms = jnp.array([0, 8, 6, 4], dtype=jnp.int32)
    spike_denom  = spike_denoms[jnp.clip(spike_layers, 0, 3)]
    spike_dmg = jnp.where(
        is_grounded & (spike_layers > 0) & (spike_denom > 0),
        jnp.maximum(jnp.int32(1), max_hp // spike_denom),
        jnp.int32(0),
    )

    # ----------------------------------------------------------------
    # Toxic Spikes: grounded non-Poison-type; absorbed by grounded Poison-type.
    # 1 layer → PSN, 2 layers → TOX.
    # ----------------------------------------------------------------
    tspike_layers  = state.sides_side_conditions[side, SC_TOXICSPIKES].astype(jnp.int32)
    is_poison_type = (t0 == jnp.int32(TYPE_POISON)) | (t1 == jnp.int32(TYPE_POISON))
    is_steel_type  = (t0 == jnp.int32(TYPE_STEEL))  | (t1 == jnp.int32(TYPE_STEEL))

    # Grounded Poison-type absorbs the spikes (removes them)
    absorb_tspikes = is_grounded & is_poison_type & (tspike_layers > 0)
    new_tspikes    = jnp.where(absorb_tspikes, jnp.int8(0),
                                state.sides_side_conditions[side, SC_TOXICSPIKES])
    new_sc = state.sides_side_conditions.at[side, SC_TOXICSPIKES].set(new_tspikes)
    state  = state._replace(sides_side_conditions=new_sc)

    # Apply status from toxic spikes (grounded, non-Poison, non-Steel, no existing status)
    # Steel types are immune to poison in all gens.
    can_be_poisoned = ~is_poison_type & ~is_steel_type
    no_status = state.sides_team_status[side, idx] == jnp.int8(STATUS_NONE)
    tspike_status = jnp.where(
        is_grounded & can_be_poisoned & no_status & (tspike_layers >= 2),
        jnp.int8(STATUS_TOX),
        jnp.where(
            is_grounded & can_be_poisoned & no_status & (tspike_layers == 1),
            jnp.int8(STATUS_PSN),
            state.sides_team_status[side, idx],
        ),
    )
    new_status_arr = state.sides_team_status.at[side, idx].set(tspike_status)
    state = state._replace(sides_team_status=new_status_arr)

    # ----------------------------------------------------------------
    # Sticky Web: -1 Speed; grounded Pokemon only.
    # ----------------------------------------------------------------
    has_web   = state.sides_side_conditions[side, SC_STICKYWEB] > jnp.int8(0)
    spe_boost = state.sides_team_boosts[side, idx, 4]  # BOOST_SPE = 4
    new_spe   = jnp.where(
        has_web & is_grounded,
        jnp.maximum(jnp.int8(-6), spe_boost - jnp.int8(1)),
        spe_boost,
    )
    new_boosts = state.sides_team_boosts.at[side, idx, 4].set(new_spe)
    state = state._replace(sides_team_boosts=new_boosts)

    # ----------------------------------------------------------------
    # Apply combined hazard HP damage
    # ----------------------------------------------------------------
    total_dmg = (rock_dmg + spike_dmg).astype(jnp.int16)
    new_hp    = jnp.maximum(jnp.int16(0), state.sides_team_hp[side, idx] - total_dmg)
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr)


# ---------------------------------------------------------------------------
# Side condition turn ticking
# ---------------------------------------------------------------------------

def tick_side_conditions(state: BattleState, side: int) -> BattleState:
    """Decrement turn-based side conditions (screens, tailwind, safeguard, mist)."""
    timed_conds = [SC_REFLECT, SC_LIGHTSCREEN, SC_AURORAVEIL,
                   SC_TAILWIND, SC_SAFEGUARD, SC_MIST]
    sc = state.sides_side_conditions[side]
    for cond_idx in timed_conds:
        val = sc[cond_idx]
        new_val = jnp.where(val > jnp.int8(0), val - jnp.int8(1), val)
        sc = sc.at[cond_idx].set(new_val)
    new_sc = state.sides_side_conditions.at[side].set(sc)
    return state._replace(sides_side_conditions=new_sc)


# ---------------------------------------------------------------------------
# Full residual for one side (called at end of each turn)
# ---------------------------------------------------------------------------

def apply_residual(state: BattleState, side: int,
                   key: jnp.ndarray, cfg) -> tuple[BattleState, jnp.ndarray]:
    """
    Apply all end-of-turn effects for a side's active Pokemon.
    Order (matches Showdown's residual event order):
      1. Weather damage
      2. Leech Seed / Partial Trap / Ingrain
      3. Burn / Poison / Toxic
      4. Sleep counter
      5. Freeze thaw (Gen 4: at end of turn)
      6. Volatile timer decrements
      7. Side condition ticks
    """
    idx = state.sides_active_idx[side]

    # Skip if fainted
    is_alive = ~state.sides_team_fainted[side, idx]

    # 1-3: status and volatile residuals
    state = apply_volatile_residuals(state, side)
    state = apply_burn_residual(state, side, cfg)
    state = apply_poison_residual(state, side)

    # 4: Sleep
    state, key = apply_sleep_residual(state, side, key)

    # 5: Freeze thaw is handled at BeforeMove (check_freeze_before_move).
    #    apply_freeze_residual is a no-op kept for API compatibility.
    state, key = apply_freeze_residual(state, side, key)

    # 6: volatile timers (Yawn uses key for sleep duration roll)
    state, key = decrement_volatile_timers(state, side, key)

    # 7: side conditions
    state = tick_side_conditions(state, side)

    return state, key
