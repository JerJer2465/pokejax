"""
Move effect dispatch for PokeJAX.

Handles all move effects that cannot be encoded in the basic move data table
fields (damage, drain, recoil, secondary status, secondary stat changes, healing).

These effects are read from Tables.move_effects (shape [N_MOVES, 8]) and applied
via branchless jnp.where chains — no lax.switch on move_id needed.

Effect type codes match move_effects_data.ME_* constants.

Call `execute_move_effects` from engine/actions.py after the damage pipeline.
"""

import jax
import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
    SC_REFLECT, SC_LIGHTSCREEN, SC_AURORAVEIL, SC_TAILWIND, SC_SAFEGUARD, SC_MIST,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    TERRAIN_NONE,
    STATUS_NONE, STATUS_SLP,
    VOL_PROTECT, VOL_SUBSTITUTE, VOL_SEEDED, VOL_PARTIALLY_TRAPPED, VOL_YAWN,
    VOL_DESTINYBOND, VOL_DISABLE, VOL_PERISH,
    VOL_CONFUSED, VOL_ATTRACT,
    BOOST_ATK, BOOST_ACC,
    TYPE_GRASS,
)
from pokejax.data.move_effects_data import (
    ME_NONE, ME_SELF_BOOST, ME_FOE_LOWER, ME_HAZARD, ME_SCREEN,
    ME_WEATHER, ME_TERRAIN, ME_TRICK_ROOM, ME_VOLATILE_SELF, ME_VOLATILE_FOE,
    ME_SUBSTITUTE, ME_RAPID_SPIN, ME_ROAR, ME_U_TURN, ME_BATON_PASS,
    ME_RECOVERY, ME_REST, ME_BELLY_DRUM, ME_KNOCK_OFF, ME_PAIN_SPLIT,
    ME_WISH, ME_HEAL_BELL, ME_DISABLE, ME_YAWN, ME_DESTINY_BOND,
    ME_PERISH_SONG, ME_SLEEP_TALK, ME_DEFOG, ME_TRICK, ME_HAZE, ME_TWO_TURN,
    NONE_STAT,
)
from pokejax.mechanics.items import (
    DAMP_ROCK_ID, HEAT_ROCK_ID, ICY_ROCK_ID, SMOOTH_ROCK_ID, LIGHT_CLAY_ID,
)

# Move-effect table field indices
_EFF_TYPE  = 0
_EFF_STAT1 = 1
_EFF_AMT1  = 2
_EFF_STAT2 = 3
_EFF_AMT2  = 4
_EFF_STAT3 = 5
_EFF_AMT3  = 6

# NONE_STAT sentinel as a JAX int32 constant (used in comparisons)
_NONE_STAT_I32 = jnp.int32(NONE_STAT)


# ---------------------------------------------------------------------------
# Internal helpers — all operate on JAX arrays, no Python-level branching
# ---------------------------------------------------------------------------

def _apply_stat_boost(
    state: BattleState,
    side: int,
    slot: jnp.ndarray,
    stat_idx: jnp.ndarray,
    amount: jnp.ndarray,
    do_apply: jnp.ndarray,
    is_foe: bool = False,
) -> BattleState:
    """
    Add `amount` to state.sides_team_boosts[side, slot, stat_idx], clamped to [-6,+6].
    No-op when do_apply is False or stat_idx == NONE_STAT.

    Ability interactions:
      - Simple: doubles the stat change amount.
      - Clear Body / White Smoke: blocks all negative foe-inflicted stat changes.
      - Hyper Cutter: blocks foe-inflicted ATK drops.
      - Keen Eye: blocks foe-inflicted accuracy drops.

    `is_foe` (compile-time bool): True when the stat change is inflicted by the
    opponent (ME_FOE_LOWER, secondary effects). False for self-boosts.
    """
    from pokejax.mechanics.abilities import (
        CLEAR_BODY_ID, WHITE_SMOKE_ID, HYPER_CUTTER_ID, KEEN_EYE_ID, SIMPLE_ID,
    )

    active = do_apply & (stat_idx != _NONE_STAT_I32)
    s_clamped = jnp.clip(stat_idx, 0, 6)

    # Get target's ability
    ability_id = state.sides_team_ability_id[side, slot].astype(jnp.int32)

    # Simple: double all stat changes (both self and foe-inflicted)
    has_simple = (SIMPLE_ID >= 0) & (ability_id == jnp.int32(SIMPLE_ID))
    amount = jnp.where(has_simple, amount * 2, amount)

    if is_foe:
        # Clear Body / White Smoke: block all negative stat changes from foes
        has_clear_body = (CLEAR_BODY_ID >= 0) & (ability_id == jnp.int32(CLEAR_BODY_ID))
        has_white_smoke = (WHITE_SMOKE_ID >= 0) & (ability_id == jnp.int32(WHITE_SMOKE_ID))
        foe_drop = amount < 0
        blocked_by_clear = foe_drop & (has_clear_body | has_white_smoke)

        # Hyper Cutter: block ATK drops from foes
        has_hyper_cutter = (HYPER_CUTTER_ID >= 0) & (ability_id == jnp.int32(HYPER_CUTTER_ID))
        blocked_by_hc = foe_drop & has_hyper_cutter & (s_clamped == jnp.int32(BOOST_ATK))

        # Keen Eye: block accuracy drops from foes
        has_keen_eye = (KEEN_EYE_ID >= 0) & (ability_id == jnp.int32(KEEN_EYE_ID))
        blocked_by_ke = foe_drop & has_keen_eye & (s_clamped == jnp.int32(BOOST_ACC))

        active = active & ~(blocked_by_clear | blocked_by_hc | blocked_by_ke)

    cur = state.sides_team_boosts[side, slot, s_clamped]
    new_val = jnp.clip(cur.astype(jnp.int32) + amount, -6, 6).astype(jnp.int8)
    chosen = jnp.where(active, new_val, cur)
    new_boosts = state.sides_team_boosts.at[side, slot, s_clamped].set(chosen)
    return state._replace(sides_team_boosts=new_boosts)


def _apply_three_stat_boosts(
    state: BattleState,
    side: int,
    slot: jnp.ndarray,
    stat1: jnp.ndarray, amt1: jnp.ndarray,
    stat2: jnp.ndarray, amt2: jnp.ndarray,
    stat3: jnp.ndarray, amt3: jnp.ndarray,
    do_apply: jnp.ndarray,
    is_foe: bool = False,
) -> BattleState:
    """Apply up to three stat boosts (self or foe) conditionally."""
    state = _apply_stat_boost(state, side, slot, stat1, amt1, do_apply, is_foe)
    state = _apply_stat_boost(state, side, slot, stat2, amt2, do_apply, is_foe)
    state = _apply_stat_boost(state, side, slot, stat3, amt3, do_apply, is_foe)
    return state


def _apply_side_condition(
    state: BattleState,
    side: int,
    sc_idx: jnp.ndarray,
    new_val: jnp.ndarray,
    do_apply: jnp.ndarray,
) -> BattleState:
    """Set sides_side_conditions[side, sc_idx] = new_val if do_apply."""
    sc_clamped = jnp.clip(sc_idx, 0, 9)
    cur = state.sides_side_conditions[side, sc_clamped]
    chosen = jnp.where(do_apply, new_val.astype(jnp.int8), cur)
    new_sc = state.sides_side_conditions.at[side, sc_clamped].set(chosen)
    return state._replace(sides_side_conditions=new_sc)


def _apply_volatile_bit(
    state: BattleState,
    side: int,
    slot: jnp.ndarray,
    vol_bit: jnp.ndarray,
    do_apply: jnp.ndarray,
) -> BattleState:
    """Set a volatile bit on state.sides_team_volatiles[side, slot]."""
    mask = jnp.uint32(1) << vol_bit.astype(jnp.uint32)
    cur = state.sides_team_volatiles[side, slot]
    new_vol = jnp.where(do_apply, cur | mask, cur)
    new_vols = state.sides_team_volatiles.at[side, slot].set(new_vol)
    return state._replace(sides_team_volatiles=new_vols)


# ---------------------------------------------------------------------------
# execute_move_effects — main entry point called from actions.py
# ---------------------------------------------------------------------------

def execute_move_effects(
    tables,
    state: BattleState,
    atk_side: int,
    def_side: int,
    move_id: jnp.ndarray,
    cancelled: jnp.ndarray,
    key: jnp.ndarray,
    cfg,
) -> tuple[BattleState, jnp.ndarray]:
    """
    Apply special move effects encoded in Tables.move_effects.

    Called AFTER the damage pipeline so we always have the post-damage state.
    Effects are skipped when `cancelled` is True (move missed / was blocked).

    Handled effects:
      ME_SELF_BOOST    — boost attacker's stats (up to 3 stat/amount pairs)
      ME_FOE_LOWER     — modify foe's stats (up to 3 stat/amount pairs)
      ME_HAZARD        — set entry hazard on foe's side
      ME_SCREEN        — set screen / field condition on own side
      ME_WEATHER       — set weather on the field
      ME_TERRAIN       — set terrain on the field
      ME_TRICK_ROOM    — toggle trick room
      ME_VOLATILE_SELF — set a volatile bit on the attacker
      ME_VOLATILE_FOE  — set a volatile bit on the defender

    Returns: (new_state, key)
    """
    atk_idx = state.sides_active_idx[atk_side]
    def_idx = state.sides_active_idx[def_side]
    mid = move_id.astype(jnp.int32)

    eff = tables.move_effects[mid]   # int16[8]
    effect_type = eff[_EFF_TYPE].astype(jnp.int32)
    stat1 = eff[_EFF_STAT1].astype(jnp.int32)
    amt1  = eff[_EFF_AMT1].astype(jnp.int32)
    stat2 = eff[_EFF_STAT2].astype(jnp.int32)
    amt2  = eff[_EFF_AMT2].astype(jnp.int32)
    stat3 = eff[_EFF_STAT3].astype(jnp.int32)
    amt3  = eff[_EFF_AMT3].astype(jnp.int32)

    # Effects are only applied when the move actually connected
    should_apply = ~cancelled

    # ------------------------------------------------------------------
    # ME_SELF_BOOST: boost attacker's stats
    # ------------------------------------------------------------------
    is_self_boost = should_apply & (effect_type == jnp.int32(ME_SELF_BOOST))
    state = _apply_three_stat_boosts(
        state, atk_side, atk_idx,
        stat1, amt1, stat2, amt2, stat3, amt3,
        is_self_boost,
    )

    # ------------------------------------------------------------------
    # ME_FOE_LOWER: modify foe's stats (amounts are typically negative)
    # ------------------------------------------------------------------
    is_foe_lower = should_apply & (effect_type == jnp.int32(ME_FOE_LOWER))
    state = _apply_three_stat_boosts(
        state, def_side, def_idx,
        stat1, amt1, stat2, amt2, stat3, amt3,
        is_foe_lower,
        is_foe=True,
    )

    # ------------------------------------------------------------------
    # ME_HAZARD: set entry hazard on foe's side
    # stat1 = SC_* index,  amt1 = max layers
    # ------------------------------------------------------------------
    is_hazard = should_apply & (effect_type == jnp.int32(ME_HAZARD))
    sc_clamped = jnp.clip(stat1, 0, 9)
    cur_layers = state.sides_side_conditions[def_side, sc_clamped]
    max_layers = amt1.astype(jnp.int8)
    new_layers = jnp.minimum(max_layers, cur_layers + jnp.int8(1))
    state = _apply_side_condition(state, def_side, stat1, new_layers, is_hazard)

    # ------------------------------------------------------------------
    # ME_SCREEN: set screen / field effect on attacker's own side
    # stat1 = SC_* index,  amt1 = turns
    # Light Clay extends Reflect/Light Screen to 8 turns.
    # ------------------------------------------------------------------
    is_screen = should_apply & (effect_type == jnp.int32(ME_SCREEN))
    sc_clamped_s = jnp.clip(stat1, 0, 9)
    already_up = state.sides_side_conditions[atk_side, sc_clamped_s] > jnp.int8(0)
    atk_item_s = state.sides_team_item_id[atk_side, atk_idx].astype(jnp.int32)
    has_light_clay = (atk_item_s == jnp.int32(LIGHT_CLAY_ID)) & (jnp.int32(LIGHT_CLAY_ID) > 0)
    screen_turns_base = amt1.astype(jnp.int32)
    screen_turns = jnp.where(has_light_clay, jnp.int32(8), screen_turns_base).astype(jnp.int8)
    screen_val = jnp.where(already_up,
                           state.sides_side_conditions[atk_side, sc_clamped_s],
                           screen_turns)
    state = _apply_side_condition(state, atk_side, stat1, screen_val, is_screen)

    # ------------------------------------------------------------------
    # ME_WEATHER: set weather
    # stat1 = weather_id,  amt1 = turns
    # Rock items extend move-set weather to 8 turns:
    #   Damp Rock (Rain), Heat Rock (Sun), Icy Rock (Hail), Smooth Rock (Sand)
    # ------------------------------------------------------------------
    is_weather = should_apply & (effect_type == jnp.int32(ME_WEATHER))
    atk_item_w = state.sides_team_item_id[atk_side, atk_idx].astype(jnp.int32)
    weather_being_set = stat1  # int32 weather_id
    has_damp_rock   = (atk_item_w == jnp.int32(DAMP_ROCK_ID))   & (jnp.int32(DAMP_ROCK_ID)   > 0) & (weather_being_set == jnp.int32(WEATHER_RAIN))
    has_heat_rock   = (atk_item_w == jnp.int32(HEAT_ROCK_ID))   & (jnp.int32(HEAT_ROCK_ID)   > 0) & (weather_being_set == jnp.int32(WEATHER_SUN))
    has_icy_rock    = (atk_item_w == jnp.int32(ICY_ROCK_ID))    & (jnp.int32(ICY_ROCK_ID)    > 0) & (weather_being_set == jnp.int32(WEATHER_HAIL))
    has_smooth_rock = (atk_item_w == jnp.int32(SMOOTH_ROCK_ID)) & (jnp.int32(SMOOTH_ROCK_ID) > 0) & (weather_being_set == jnp.int32(WEATHER_SAND))
    has_extend_rock = has_damp_rock | has_heat_rock | has_icy_rock | has_smooth_rock
    w_turns_extended = jnp.where(has_extend_rock, jnp.int32(8), amt1)
    new_weather = jnp.where(is_weather, stat1.astype(jnp.int8), state.field.weather)
    new_w_turns = jnp.where(is_weather, w_turns_extended.astype(jnp.int8), state.field.weather_turns)
    new_field_w = state.field._replace(
        weather=new_weather,
        weather_turns=new_w_turns,
        weather_max_turns=new_w_turns,
    )
    state = state._replace(field=new_field_w)

    # ------------------------------------------------------------------
    # ME_TERRAIN: set terrain
    # stat1 = terrain_id,  amt1 = turns
    # ------------------------------------------------------------------
    is_terrain = should_apply & (effect_type == jnp.int32(ME_TERRAIN))
    new_terrain = jnp.where(is_terrain, stat1.astype(jnp.int8), state.field.terrain)
    new_t_turns = jnp.where(is_terrain, amt1.astype(jnp.int8), state.field.terrain_turns)
    new_field_t = state.field._replace(terrain=new_terrain, terrain_turns=new_t_turns)
    state = state._replace(field=new_field_t)

    # ------------------------------------------------------------------
    # ME_TRICK_ROOM: toggle trick room (5 turns on, or cancel if active)
    # ------------------------------------------------------------------
    is_trick_room = should_apply & (effect_type == jnp.int32(ME_TRICK_ROOM))
    tr_active = state.field.trick_room > jnp.int8(0)
    new_tr = jnp.where(tr_active, jnp.int8(0), jnp.int8(5))
    new_tr_val = jnp.where(is_trick_room, new_tr, state.field.trick_room)
    new_field_tr = state.field._replace(trick_room=new_tr_val)
    state = state._replace(field=new_field_tr)

    # ------------------------------------------------------------------
    # ME_VOLATILE_SELF: set a volatile bit on the attacker
    # stat1 = vol_bit index
    #
    # Special handling for Protect/Detect (VOL_PROTECT):
    #   Consecutive use has decreasing success rate: 1/(2^n) where n is
    #   the number of consecutive successful uses (Gen 4 mechanics).
    #   volatile_data[VOL_PROTECT] stores the consecutive-use counter.
    # ------------------------------------------------------------------
    is_vol_self = should_apply & (effect_type == jnp.int32(ME_VOLATILE_SELF))

    # Protect consecutive failure check
    is_protect = is_vol_self & (stat1 == jnp.int32(VOL_PROTECT))
    protect_counter = state.sides_team_volatile_data[
        atk_side, atk_idx, VOL_PROTECT
    ].astype(jnp.int32)
    # Success chance = 1 / (2^counter). At counter=0, always succeeds (1/1).
    # At counter=1, 50%. At counter=2, 25%. etc.
    # We cap at counter=4 to avoid extreme values.
    capped_counter = jnp.minimum(protect_counter, jnp.int32(4))
    # Denominator = 2^counter (1, 2, 4, 8, 16)
    denom = jnp.int32(1) << capped_counter
    # Roll: succeed if random value [0, denom) == 0
    key, protect_key = jax.random.split(key)
    roll = jax.random.randint(protect_key, (), 0, denom)
    protect_succeeds = roll == jnp.int32(0)

    # If Protect: gate on the roll result
    # If non-Protect volatile_self: always apply
    vol_self_applies = jnp.where(is_protect, is_protect & protect_succeeds, is_vol_self)
    state = _apply_volatile_bit(state, atk_side, atk_idx, stat1, vol_self_applies)

    # Update protect counter: increment on success, reset to 0 on failure
    new_protect_counter = jnp.where(
        is_protect & protect_succeeds,
        jnp.minimum(protect_counter + jnp.int32(1), jnp.int32(127)).astype(jnp.int8),
        jnp.where(
            is_protect & ~protect_succeeds,
            jnp.int8(0),
            state.sides_team_volatile_data[atk_side, atk_idx, VOL_PROTECT],
        ),
    )
    state = state._replace(
        sides_team_volatile_data=state.sides_team_volatile_data.at[
            atk_side, atk_idx, VOL_PROTECT
        ].set(new_protect_counter)
    )

    # ------------------------------------------------------------------
    # ME_VOLATILE_FOE: set a volatile bit on the defender
    # stat1 = vol_bit index
    # Substitute blocks volatile effects on the defender (Leech Seed, etc.)
    # Own Tempo blocks confusion (VOL_CONFUSED).
    # Oblivious blocks infatuation (VOL_ATTRACT).
    # ------------------------------------------------------------------
    from pokejax.mechanics.abilities import OWN_TEMPO_ID, OBLIVIOUS_ID
    def_has_sub = (state.sides_team_volatiles[def_side, def_idx]
                   & jnp.uint32(1 << VOL_SUBSTITUTE)) != jnp.uint32(0)
    def_ability_mv = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)
    has_own_tempo  = (jnp.int32(OWN_TEMPO_ID) > 0) & (def_ability_mv == jnp.int32(OWN_TEMPO_ID))
    has_oblivious  = (jnp.int32(OBLIVIOUS_ID) > 0) & (def_ability_mv == jnp.int32(OBLIVIOUS_ID))
    blocked_confusion  = has_own_tempo & (stat1 == jnp.int32(VOL_CONFUSED))
    blocked_infatuation = has_oblivious & (stat1 == jnp.int32(VOL_ATTRACT))
    # Leech Seed: Grass types are immune (onTryImmunity in PS)
    def_types_mv = state.sides_team_types[def_side, def_idx]
    def_is_grass = (
        (def_types_mv[0].astype(jnp.int32) == jnp.int32(TYPE_GRASS)) |
        (def_types_mv[1].astype(jnp.int32) == jnp.int32(TYPE_GRASS))
    )
    blocked_seed = def_is_grass & (stat1 == jnp.int32(VOL_SEEDED))
    is_vol_foe = (should_apply & (effect_type == jnp.int32(ME_VOLATILE_FOE))
                  & ~def_has_sub & ~blocked_confusion & ~blocked_infatuation
                  & ~blocked_seed)
    state = _apply_volatile_bit(state, def_side, def_idx, stat1, is_vol_foe)

    # ------------------------------------------------------------------
    # ME_SUBSTITUTE: create substitute at 25% max HP cost
    # ------------------------------------------------------------------
    is_substitute = should_apply & (effect_type == jnp.int32(ME_SUBSTITUTE))
    # In Pokemon Showdown, Substitute fails if one already exists
    already_has_sub = (state.sides_team_volatiles[atk_side, atk_idx]
                       & jnp.uint32(1 << VOL_SUBSTITUTE)) != jnp.uint32(0)
    atk_hp     = state.sides_team_hp[atk_side, atk_idx].astype(jnp.int32)
    atk_max_hp = state.sides_team_max_hp[atk_side, atk_idx].astype(jnp.int32)
    sub_cost   = jnp.maximum(jnp.int32(1), atk_max_hp // 4)
    can_sub    = is_substitute & (atk_hp > sub_cost) & ~already_has_sub
    # Deduct HP cost
    new_hp_sub = jnp.where(can_sub,
                           (atk_hp - sub_cost).astype(jnp.int16),
                           state.sides_team_hp[atk_side, atk_idx])
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[atk_side, atk_idx].set(new_hp_sub)
    )
    # Set substitute volatile bit
    state = _apply_volatile_bit(state, atk_side, atk_idx,
                                 jnp.int32(VOL_SUBSTITUTE), can_sub)
    # Store substitute HP in volatile_data (sub HP = 25% of max HP, clamped to int8)
    # We store raw HP clamped to [1, 127] since int8 max is 127.
    sub_hp_val = jnp.where(can_sub,
                            jnp.clip(sub_cost, 1, 127).astype(jnp.int8),
                            state.sides_team_volatile_data[atk_side, atk_idx, VOL_SUBSTITUTE])
    state = state._replace(
        sides_team_volatile_data=state.sides_team_volatile_data.at[
            atk_side, atk_idx, VOL_SUBSTITUTE
        ].set(sub_hp_val)
    )

    # ------------------------------------------------------------------
    # ME_RAPID_SPIN: remove hazards from own side
    # ------------------------------------------------------------------
    is_rapid_spin = should_apply & (effect_type == jnp.int32(ME_RAPID_SPIN))
    for sc_idx in [SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB]:
        cur_sc = state.sides_side_conditions[atk_side, sc_idx]
        new_sc = jnp.where(is_rapid_spin, jnp.int8(0), cur_sc)
        state = state._replace(
            sides_side_conditions=state.sides_side_conditions.at[atk_side, sc_idx].set(new_sc)
        )
    # Also remove Leech Seed and Partial Trap from the active mon
    for vol_bit in [VOL_SEEDED, VOL_PARTIALLY_TRAPPED]:
        mask = jnp.uint32(1) << jnp.uint32(vol_bit)
        cur_vol = state.sides_team_volatiles[atk_side, atk_idx]
        new_vol = jnp.where(is_rapid_spin, cur_vol & ~mask, cur_vol)
        state = state._replace(
            sides_team_volatiles=state.sides_team_volatiles.at[atk_side, atk_idx].set(new_vol)
        )

    # ------------------------------------------------------------------
    # ME_DEFOG: clear hazards from both sides, clear screens from foe
    # ------------------------------------------------------------------
    is_defog = should_apply & (effect_type == jnp.int32(ME_DEFOG))
    # Clear foe hazards + screens
    for sc_idx in [SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
                   SC_REFLECT, SC_LIGHTSCREEN, SC_AURORAVEIL]:
        cur_sc = state.sides_side_conditions[def_side, sc_idx]
        new_sc = jnp.where(is_defog, jnp.int8(0), cur_sc)
        state = state._replace(
            sides_side_conditions=state.sides_side_conditions.at[def_side, sc_idx].set(new_sc)
        )
    # Clear own hazards
    for sc_idx in [SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB]:
        cur_sc = state.sides_side_conditions[atk_side, sc_idx]
        new_sc = jnp.where(is_defog, jnp.int8(0), cur_sc)
        state = state._replace(
            sides_side_conditions=state.sides_side_conditions.at[atk_side, sc_idx].set(new_sc)
        )

    # ------------------------------------------------------------------
    # ME_ROAR: force foe to switch (handled in actions.py after effects)
    # We just set a flag here; actual switch handled by caller.
    # For now, we store ME_ROAR in a way the caller can detect.
    # ------------------------------------------------------------------
    # (No state change here — caller checks effect_type for ME_ROAR)

    # ------------------------------------------------------------------
    # ME_U_TURN: user switches out after damage (handled by caller)
    # ------------------------------------------------------------------
    # (No state change here — caller checks effect_type for ME_U_TURN)

    # ------------------------------------------------------------------
    # ME_BATON_PASS: switch passing boosts (handled by caller)
    # ------------------------------------------------------------------
    # (No state change here — caller checks effect_type for ME_BATON_PASS)

    # ------------------------------------------------------------------
    # ME_RECOVERY: heal num/den of max HP
    # stat1 = numerator, amt1 = denominator
    # ------------------------------------------------------------------
    is_recovery = should_apply & (effect_type == jnp.int32(ME_RECOVERY))
    rec_num = jnp.maximum(jnp.int32(1), stat1)
    rec_den = jnp.maximum(jnp.int32(1), amt1)
    rec_heal = jnp.maximum(jnp.int32(1), atk_max_hp * rec_num // rec_den)
    rec_new_hp = jnp.minimum(atk_max_hp, atk_hp + rec_heal).astype(jnp.int16)
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[atk_side, atk_idx].set(
            jnp.where(is_recovery, rec_new_hp, state.sides_team_hp[atk_side, atk_idx])
        )
    )

    # ------------------------------------------------------------------
    # ME_REST: full heal + sleep for 2 turns
    # ------------------------------------------------------------------
    is_rest = should_apply & (effect_type == jnp.int32(ME_REST))
    rest_hp = jnp.where(is_rest, atk_max_hp.astype(jnp.int16),
                         state.sides_team_hp[atk_side, atk_idx])
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[atk_side, atk_idx].set(rest_hp)
    )
    rest_status = jnp.where(is_rest, jnp.int8(STATUS_SLP),
                             state.sides_team_status[atk_side, atk_idx])
    state = state._replace(
        sides_team_status=state.sides_team_status.at[atk_side, atk_idx].set(rest_status)
    )
    rest_sleep_turns = jnp.where(is_rest, jnp.int8(2),
                                  state.sides_team_sleep_turns[atk_side, atk_idx])
    state = state._replace(
        sides_team_sleep_turns=state.sides_team_sleep_turns.at[atk_side, atk_idx].set(rest_sleep_turns)
    )

    # ------------------------------------------------------------------
    # ME_BELLY_DRUM: maximize Atk (+6) at 50% HP cost
    # ------------------------------------------------------------------
    is_belly_drum = should_apply & (effect_type == jnp.int32(ME_BELLY_DRUM))
    bd_cost = jnp.maximum(jnp.int32(1), atk_max_hp // 2)
    can_bd = is_belly_drum & (atk_hp > bd_cost)
    bd_new_hp = jnp.where(can_bd,
                           (atk_hp - bd_cost).astype(jnp.int16),
                           state.sides_team_hp[atk_side, atk_idx])
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[atk_side, atk_idx].set(bd_new_hp)
    )
    bd_new_atk = jnp.where(can_bd, jnp.int8(6),
                             state.sides_team_boosts[atk_side, atk_idx, BOOST_ATK])
    state = state._replace(
        sides_team_boosts=state.sides_team_boosts.at[atk_side, atk_idx, BOOST_ATK].set(bd_new_atk)
    )

    # ------------------------------------------------------------------
    # ME_KNOCK_OFF: remove foe's item
    # ------------------------------------------------------------------
    is_knock_off = should_apply & (effect_type == jnp.int32(ME_KNOCK_OFF))
    new_foe_item = jnp.where(is_knock_off, jnp.int16(0),
                              state.sides_team_item_id[def_side, def_idx])
    state = state._replace(
        sides_team_item_id=state.sides_team_item_id.at[def_side, def_idx].set(new_foe_item)
    )

    # ------------------------------------------------------------------
    # ME_TRICK: swap items
    # ------------------------------------------------------------------
    is_trick = should_apply & (effect_type == jnp.int32(ME_TRICK))
    atk_item = state.sides_team_item_id[atk_side, atk_idx]
    def_item = state.sides_team_item_id[def_side, def_idx]
    new_atk_item = jnp.where(is_trick, def_item, atk_item)
    new_def_item = jnp.where(is_trick, atk_item, def_item)
    state = state._replace(
        sides_team_item_id=state.sides_team_item_id.at[atk_side, atk_idx].set(new_atk_item)
                                                   .at[def_side, def_idx].set(new_def_item)
    )

    # ------------------------------------------------------------------
    # ME_PAIN_SPLIT: average HP
    # ------------------------------------------------------------------
    is_pain_split = should_apply & (effect_type == jnp.int32(ME_PAIN_SPLIT))
    def_hp = state.sides_team_hp[def_side, def_idx].astype(jnp.int32)
    avg_hp = (atk_hp + def_hp) // 2
    ps_atk_hp = jnp.minimum(avg_hp, atk_max_hp).astype(jnp.int16)
    def_max_hp = state.sides_team_max_hp[def_side, def_idx].astype(jnp.int32)
    ps_def_hp = jnp.minimum(avg_hp, def_max_hp).astype(jnp.int16)
    state = state._replace(
        sides_team_hp=state.sides_team_hp
            .at[atk_side, atk_idx].set(jnp.where(is_pain_split, ps_atk_hp,
                                                    state.sides_team_hp[atk_side, atk_idx]))
            .at[def_side, def_idx].set(jnp.where(is_pain_split, ps_def_hp,
                                                    state.sides_team_hp[def_side, def_idx]))
    )

    # ------------------------------------------------------------------
    # ME_HEAL_BELL: cure team status
    # ------------------------------------------------------------------
    is_heal_bell = should_apply & (effect_type == jnp.int32(ME_HEAL_BELL))
    # Cure all 6 team members
    for i in range(6):
        cur_status = state.sides_team_status[atk_side, i]
        new_status = jnp.where(is_heal_bell, jnp.int8(STATUS_NONE), cur_status)
        state = state._replace(
            sides_team_status=state.sides_team_status.at[atk_side, i].set(new_status)
        )

    # ------------------------------------------------------------------
    # ME_DISABLE: disable foe's last used move
    # ------------------------------------------------------------------
    is_disable = should_apply & (effect_type == jnp.int32(ME_DISABLE))
    foe_last_move = state.sides_team_last_move_id[def_side, def_idx]
    # Find which move slot matches and disable it
    for slot in range(4):
        slot_move = state.sides_team_move_ids[def_side, def_idx, slot]
        slot_matches = (slot_move == foe_last_move) & (foe_last_move >= jnp.int16(0))
        should_disable = is_disable & slot_matches
        cur_disabled = state.sides_team_move_disabled[def_side, def_idx, slot]
        state = state._replace(
            sides_team_move_disabled=state.sides_team_move_disabled
                .at[def_side, def_idx, slot].set(jnp.where(should_disable, True, cur_disabled))
        )
    # Set volatile
    state = _apply_volatile_bit(state, def_side, def_idx,
                                 jnp.int32(VOL_DISABLE), is_disable)

    # ------------------------------------------------------------------
    # ME_YAWN: inflict drowsiness (sleep next turn via volatile)
    # ------------------------------------------------------------------
    is_yawn = should_apply & (effect_type == jnp.int32(ME_YAWN))
    state = _apply_volatile_bit(state, def_side, def_idx,
                                 jnp.int32(VOL_YAWN), is_yawn)

    # ------------------------------------------------------------------
    # ME_DESTINY_BOND: set volatile on self
    # ------------------------------------------------------------------
    is_db = should_apply & (effect_type == jnp.int32(ME_DESTINY_BOND))
    state = _apply_volatile_bit(state, atk_side, atk_idx,
                                 jnp.int32(VOL_DESTINYBOND), is_db)

    # ------------------------------------------------------------------
    # ME_HAZE: reset all stat boosts
    # ------------------------------------------------------------------
    is_haze = should_apply & (effect_type == jnp.int32(ME_HAZE))
    for side in [atk_side, def_side]:
        idx = state.sides_active_idx[side]
        for b in range(7):
            cur_b = state.sides_team_boosts[side, idx, b]
            new_b = jnp.where(is_haze, jnp.int8(0), cur_b)
            state = state._replace(
                sides_team_boosts=state.sides_team_boosts.at[side, idx, b].set(new_b)
            )

    # ------------------------------------------------------------------
    # ME_WISH: store wish data (simplified: heal immediately for RL)
    # A proper implementation would store turn+heal in side_conditions
    # and resolve at end of next turn. Simplified as immediate 50% heal.
    # ------------------------------------------------------------------
    is_wish = should_apply & (effect_type == jnp.int32(ME_WISH))
    wish_heal = jnp.maximum(jnp.int32(1), atk_max_hp // 2)
    wish_new_hp = jnp.minimum(atk_max_hp, atk_hp + wish_heal).astype(jnp.int16)
    state = state._replace(
        sides_team_hp=state.sides_team_hp.at[atk_side, atk_idx].set(
            jnp.where(is_wish, wish_new_hp, state.sides_team_hp[atk_side, atk_idx])
        )
    )

    # ------------------------------------------------------------------
    # ME_PERISH_SONG: set VOL_PERISH on BOTH sides' active Pokemon with
    # counter = 3. Does NOT apply if the target already has VOL_PERISH.
    # ------------------------------------------------------------------
    is_perish = should_apply & (effect_type == jnp.int32(ME_PERISH_SONG))

    for ps_side in [atk_side, def_side]:
        ps_idx = state.sides_active_idx[ps_side]
        already_perish = (state.sides_team_volatiles[ps_side, ps_idx]
                          & jnp.uint32(1 << VOL_PERISH)) != jnp.uint32(0)
        apply_perish = is_perish & ~already_perish
        new_vols = jnp.where(
            apply_perish,
            state.sides_team_volatiles[ps_side, ps_idx] | jnp.uint32(1 << VOL_PERISH),
            state.sides_team_volatiles[ps_side, ps_idx],
        )
        new_data = jnp.where(
            apply_perish,
            jnp.int8(3),
            state.sides_team_volatile_data[ps_side, ps_idx, VOL_PERISH],
        )
        state = state._replace(
            sides_team_volatiles=state.sides_team_volatiles.at[ps_side, ps_idx].set(new_vols),
            sides_team_volatile_data=state.sides_team_volatile_data.at[ps_side, ps_idx, VOL_PERISH].set(new_data),
        )

    # ME_SLEEP_TALK: noop for now (too complex / niche)

    return state, key
