"""
8-step hit pipeline (Showdown's trySpreadMoveHit).

Each step runs branchlessly and passes results to the next.
Cancelled hits still run all steps but produce 0 damage.

Step 1: Invulnerability (Fly, Dig, Phantom Force semi-invulnerability)
Step 2: TryHit       (Protect, absorption abilities)
Step 3: Type immunity (Ghost vs Normal, Electric vs Ground, etc.)
Step 4: TryImmunity  (Powder immunity, Prankster/Dark immunity)
Step 5: Accuracy roll
Step 6: BreakProtect (remove protection effects)
Step 7: [StealBoosts - Spectral Thief; Gen 8+, skipped for now]
Step 8: MoveHitLoop  (actual damage, secondary effects)
"""

import jax
import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    VOL_CHARGING, VOL_SUBSTITUTE, CATEGORY_STATUS,
    TYPE_NONE, STATUS_FRZ, STATUS_NONE,
    WEATHER_SAND, WEATHER_HAIL,
    CATEGORY_PHYSICAL,
)
from pokejax.core.damage import (
    compute_damage, type_effectiveness, apply_damage,
    MF_ACCURACY, MF_TYPE, MF_CATEGORY, MF_BASE_POWER,
    MF_TARGET, MF_MULTIHIT_MIN, MF_MULTIHIT_MAX,
    MF_SEC_CHANCE, MF_SEC_STATUS, MF_SEC_BOOST_STAT, MF_SEC_BOOST_AMT,
    MF_DRAIN_NUM, MF_DRAIN_DEN,
    MF_RECOIL_NUM, MF_RECOIL_DEN, MF_HEAL_NUM, MF_HEAL_DEN,
    MF_FLAGS_LO, FLAG_DEFROST,
)
from pokejax.core.state import set_fainted
from pokejax.core.damage import apply_heal, fraction_of_max_hp
from pokejax.mechanics.events import (
    run_event_try_hit, run_event_try_hit_state,
    run_event_modify_atk, run_event_modify_spa,
    run_event_modify_def, run_event_modify_spd,
    run_event_base_power, run_event_modify_damage,
    run_event_contact_punish,
)
from pokejax.mechanics import conditions as cond
from pokejax.core import rng as rng_utils

# Move target codes (matches extractor._TARGET_MAP)
_TARGET_SELF      = 1   # move targets the user (Recover, Swords Dance, etc.)
_TARGET_ALL       = 8   # all Pokemon on field
_TARGET_ALLY_TEAM = 9   # user's team
_TARGET_FOE_SIDE  = 10  # foe's side (Spikes, Stealth Rock, etc.)
_TARGET_ALLY_SIDE = 11  # user's side (Reflect, Light Screen, etc.)

# Accuracy sentinel: 101 = always hits
ALWAYS_HITS_SENTINEL = 101


def step1_invulnerability(state: BattleState,
                           atk_side: int, def_side: int,
                           move_id: jnp.ndarray) -> jnp.ndarray:
    """
    Step 1: Is the target invulnerable?
    Returns cancelled: bool (True = move misses due to semi-invulnerable).
    """
    def_idx = state.sides_active_idx[def_side]
    # Defender is in semi-invulnerable turn (Fly/Dig/etc.)
    is_charging = (state.sides_team_volatiles[def_side, def_idx]
                   & jnp.uint32(1 << VOL_CHARGING)) != jnp.uint32(0)
    return is_charging  # True = cancelled (target is invulnerable)


def step2_try_hit(state: BattleState,
                   atk_side: int, atk_idx: jnp.ndarray,
                   def_side: int, def_idx: jnp.ndarray,
                   move_id: jnp.ndarray,
                   cancelled: jnp.ndarray,
                   move_type: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, BattleState]:
    """
    Step 2: TryHit - can the move hit at all?
    Checks Protect-like effects and absorption abilities.
    Also fires the state-returning TryHit event (absorption heals, Motor Drive, etc.).

    move_type must be the actual move type (from tables.moves[move_id, MF_TYPE]),
    NOT a placeholder — it is forwarded to run_event_try_hit_state so that
    Water Absorb / Volt Absorb / Sap Sipper etc. can check whether they trigger.

    Returns (relay, cancelled, new_state).
    """
    relay = jnp.bool_(True)
    relay, new_cancel = run_event_try_hit(
        relay, state, atk_side, atk_idx, def_side, def_idx, move_id
    )
    # State-mutating absorption effects (heal on absorb, stat boosts on immunity).
    # Gate on ~cancelled so we don't fire absorption side-effects for moves that
    # were already blocked by invulnerability or an earlier step.
    state = run_event_try_hit_state(
        state, atk_side, atk_idx, def_side, def_idx, move_id, move_type
    )
    return relay, cancelled | new_cancel, state


def step3_type_immunity(tables, state: BattleState,
                         def_side: int, def_idx: jnp.ndarray,
                         move_id: jnp.ndarray,
                         cancelled: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Step 3: Type immunity check.
    Returns (effectiveness, cancelled).
    If effectiveness == 0.0, the move is cancelled (immune).
    """
    move_type = tables.moves[move_id.astype(jnp.int32), MF_TYPE].astype(jnp.int32)
    def_types = state.sides_team_types[def_side, def_idx]
    t0 = def_types[0].astype(jnp.int32)
    t1 = def_types[1].astype(jnp.int32)

    effectiveness = type_effectiveness(tables, move_type, t0, t1)
    immune = effectiveness == jnp.float32(0.0)
    return effectiveness, cancelled | immune


def step5_accuracy(tables, state: BattleState,
                    atk_side: int, atk_idx: jnp.ndarray,
                    def_side: int, def_idx: jnp.ndarray,
                    move_id: jnp.ndarray,
                    cancelled: jnp.ndarray,
                    key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Step 5: Accuracy roll.
    Returns (cancelled, new_key).
    """
    from pokejax.mechanics.abilities import (
        NO_GUARD_ID, COMPOUND_EYES_ID, HUSTLE_ID,
        SAND_VEIL_ID, SNOW_CLOAK_ID, TANGLED_FEET_ID,
    )

    accuracy = tables.moves[move_id.astype(jnp.int32), MF_ACCURACY].astype(jnp.int32)
    always_hits = accuracy >= jnp.int32(ALWAYS_HITS_SENTINEL)

    # Ability IDs
    atk_ability = state.sides_team_ability_id[atk_side, atk_idx].astype(jnp.int32)
    def_ability = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)

    # No Guard: both sides' moves always hit
    no_guard = (
        ((NO_GUARD_ID >= 0) & (atk_ability == jnp.int32(NO_GUARD_ID))) |
        ((NO_GUARD_ID >= 0) & (def_ability == jnp.int32(NO_GUARD_ID)))
    )

    # Accuracy boost
    atk_acc_boost = state.sides_team_boosts[atk_side, atk_idx, 5]  # BOOST_ACC
    # Evasion boost
    def_eva_boost = state.sides_team_boosts[def_side, def_idx, 6]  # BOOST_EVA

    # Net accuracy stage = attacker acc boost - defender evasion boost, clamped to [-6,6]
    net_boost = jnp.clip(
        atk_acc_boost.astype(jnp.int32) - def_eva_boost.astype(jnp.int32),
        -6, 6
    )
    # Accuracy multiplier from stages
    num = jnp.maximum(3, 3 + net_boost).astype(jnp.float32)
    den = jnp.maximum(3, 3 - net_boost).astype(jnp.float32)
    acc_mult = num / den

    # Compound Eyes: x1.3 accuracy
    compound_eyes = (COMPOUND_EYES_ID >= 0) & (atk_ability == jnp.int32(COMPOUND_EYES_ID))
    acc_mult = jnp.where(compound_eyes, acc_mult * 1.3, acc_mult)

    # Hustle: x0.8 accuracy for physical moves
    move_category = tables.moves[move_id.astype(jnp.int32), MF_CATEGORY].astype(jnp.int8)
    hustle = (HUSTLE_ID >= 0) & (atk_ability == jnp.int32(HUSTLE_ID))
    is_physical = move_category == CATEGORY_PHYSICAL
    acc_mult = jnp.where(hustle & is_physical, acc_mult * 0.8, acc_mult)

    # Sand Veil: 0.8x evasion in sandstorm (defender)
    sand_veil = (SAND_VEIL_ID >= 0) & (def_ability == jnp.int32(SAND_VEIL_ID)) & (state.field.weather == WEATHER_SAND)
    acc_mult = jnp.where(sand_veil, acc_mult * 0.8, acc_mult)

    # Snow Cloak: 0.8x evasion in hail (defender)
    snow_cloak = (SNOW_CLOAK_ID >= 0) & (def_ability == jnp.int32(SNOW_CLOAK_ID)) & (state.field.weather == WEATHER_HAIL)
    acc_mult = jnp.where(snow_cloak, acc_mult * 0.8, acc_mult)

    # Tangled Feet: 0.5x accuracy when confused (defender)
    from pokejax.types import VOL_CONFUSED
    def_confused = (state.sides_team_volatiles[def_side, def_idx]
                    & jnp.uint32(1 << VOL_CONFUSED)) != jnp.uint32(0)
    tangled_feet = (TANGLED_FEET_ID >= 0) & (def_ability == jnp.int32(TANGLED_FEET_ID)) & def_confused
    acc_mult = jnp.where(tangled_feet, acc_mult * 0.5, acc_mult)

    effective_acc = jnp.floor(accuracy.astype(jnp.float32) * acc_mult).astype(jnp.int32)
    effective_acc = jnp.clip(effective_acc, 1, 100)

    key, subkey = rng_utils.split(key)
    hits = rng_utils.accuracy_roll(subkey, effective_acc)
    missed = ~hits & ~always_hits & ~no_guard

    return cancelled | missed, key


def step8_move_hit_loop(tables, state: BattleState,
                         atk_side: int, atk_idx: jnp.ndarray,
                         def_side: int, def_idx: jnp.ndarray,
                         move_id: jnp.ndarray,
                         effectiveness: jnp.ndarray,
                         cancelled: jnp.ndarray,
                         key: jnp.ndarray,
                         cfg,
                         atk_relay: jnp.ndarray = jnp.float32(1.0),
                         def_relay: jnp.ndarray = jnp.float32(1.0),
                         bp_relay: jnp.ndarray = jnp.float32(1.0),
                         damage_relay: jnp.ndarray = jnp.float32(1.0),
                         guts: jnp.ndarray = jnp.bool_(False),
                         adaptability: jnp.ndarray = jnp.bool_(False),
                         ) -> tuple[BattleState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Step 8: Deal damage (and secondary effects) for one hit.
    Multi-hit moves call this multiple times in actions.py.

    Returns: (new_state, total_damage_dealt, is_crit, new_key)
    """
    # Skip all if cancelled
    key, dmg_key = rng_utils.split(key)

    # ---- Substitute check ----
    # Check if defender has a Substitute up
    def_has_sub = (state.sides_team_volatiles[def_side, def_idx]
                   & jnp.uint32(1 << VOL_SUBSTITUTE)) != jnp.uint32(0)
    sub_hp = state.sides_team_volatile_data[def_side, def_idx, VOL_SUBSTITUTE].astype(jnp.int32)

    new_key, dmg, is_crit, eff = compute_damage(
        state, tables, atk_side, def_side, move_id, dmg_key,
        atk_relay=atk_relay, def_relay=def_relay,
        bp_relay=bp_relay, damage_relay=damage_relay,
        guts=guts, adaptability=adaptability,
        crit_multiplier=cfg.crit_damage_multiplier,
    )
    # If cancelled or immune (effectiveness=0), damage = 0
    dmg = jnp.where(cancelled | (effectiveness == jnp.float32(0.0)),
                     jnp.int32(0), dmg)

    # ---- Substitute damage absorption ----
    # If defender has Substitute, damage goes to the sub, not the real Pokemon.
    # If the sub breaks, excess damage is NOT passed through (Showdown Gen 4 behavior).
    new_sub_hp = jnp.maximum(jnp.int32(0), sub_hp - dmg)
    sub_broke = def_has_sub & (new_sub_hp <= jnp.int32(0)) & (dmg > jnp.int32(0))
    # Update sub HP (clear to 0 if broken)
    new_sub_data = jnp.where(def_has_sub & ~cancelled,
                              new_sub_hp.astype(jnp.int8),
                              state.sides_team_volatile_data[def_side, def_idx, VOL_SUBSTITUTE])
    state = state._replace(
        sides_team_volatile_data=state.sides_team_volatile_data.at[
            def_side, def_idx, VOL_SUBSTITUTE
        ].set(new_sub_data)
    )
    # Clear substitute volatile if broken
    sub_clear_mask = ~jnp.uint32(1 << VOL_SUBSTITUTE)
    new_vols = jnp.where(sub_broke,
                          state.sides_team_volatiles[def_side, def_idx] & sub_clear_mask,
                          state.sides_team_volatiles[def_side, def_idx])
    state = state._replace(
        sides_team_volatiles=state.sides_team_volatiles.at[def_side, def_idx].set(new_vols)
    )
    # Zero out real damage if sub absorbed it
    real_dmg = jnp.where(def_has_sub & ~cancelled, jnp.int32(0), dmg)

    # Capture pre-damage HP for Focus Sash check
    pre_def_hp  = state.sides_team_hp[def_side, def_idx].astype(jnp.int32)
    pre_def_max = state.sides_team_max_hp[def_side, def_idx].astype(jnp.int32)

    # Apply damage to defender (0 if sub absorbed it)
    state = apply_damage(state, def_side, def_idx, real_dmg)

    # Focus Sash: survive lethal hit at full HP (consume item)
    from pokejax.mechanics.items import FOCUS_SASH_ID
    def_item = state.sides_team_item_id[def_side, def_idx].astype(jnp.int32)
    has_sash  = (FOCUS_SASH_ID >= 0) & (def_item == jnp.int32(FOCUS_SASH_ID))
    was_full  = pre_def_hp >= pre_def_max
    would_faint = (state.sides_team_hp[def_side, def_idx] <= jnp.int16(0))
    sash_saves  = has_sash & was_full & would_faint & ~cancelled
    # Clamp HP to 1 and consume the item
    sash_hp_arr = state.sides_team_hp.at[def_side, def_idx].set(
        jnp.where(sash_saves, jnp.int16(1), state.sides_team_hp[def_side, def_idx])
    )
    sash_item_arr = state.sides_team_item_id.at[def_side, def_idx].set(
        jnp.where(sash_saves, jnp.int16(0), state.sides_team_item_id[def_side, def_idx])
    )
    state = state._replace(sides_team_hp=sash_hp_arr, sides_team_item_id=sash_item_arr)

    # Sturdy: survive lethal hit at full HP (ability, not consumed)
    from pokejax.mechanics.abilities import STURDY_ID
    def_ability_st = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)
    has_sturdy = (STURDY_ID >= 0) & (def_ability_st == jnp.int32(STURDY_ID))
    would_faint_now = (state.sides_team_hp[def_side, def_idx] <= jnp.int16(0))
    sturdy_saves = has_sturdy & was_full & would_faint_now & ~cancelled
    sturdy_hp_arr = state.sides_team_hp.at[def_side, def_idx].set(
        jnp.where(sturdy_saves, jnp.int16(1), state.sides_team_hp[def_side, def_idx])
    )
    state = state._replace(sides_team_hp=sturdy_hp_arr)

    # Drain (e.g., Drain Punch, Giga Drain)
    drain_num = tables.moves[move_id.astype(jnp.int32), MF_DRAIN_NUM].astype(jnp.int32)
    drain_den = tables.moves[move_id.astype(jnp.int32), MF_DRAIN_DEN].astype(jnp.int32)
    has_drain = (drain_num > jnp.int32(0)) & (drain_den > jnp.int32(0))
    drain_heal = jnp.where(
        has_drain,
        jnp.maximum(jnp.int32(1), dmg * drain_num // drain_den),
        jnp.int32(0)
    )
    state = apply_heal(state, atk_side, atk_idx, drain_heal.astype(jnp.int16))

    # Recoil (e.g., Brave Bird, Flare Blitz)
    # Rock Head: prevents recoil damage
    from pokejax.mechanics.abilities import ROCK_HEAD_ID
    atk_ability_rh = state.sides_team_ability_id[atk_side, atk_idx].astype(jnp.int32)
    has_rock_head = (ROCK_HEAD_ID >= 0) & (atk_ability_rh == jnp.int32(ROCK_HEAD_ID))
    recoil_num = tables.moves[move_id.astype(jnp.int32), MF_RECOIL_NUM].astype(jnp.int32)
    recoil_den = tables.moves[move_id.astype(jnp.int32), MF_RECOIL_DEN].astype(jnp.int32)
    has_recoil = (recoil_num > jnp.int32(0)) & (recoil_den > jnp.int32(0)) & ~has_rock_head
    recoil_dmg = jnp.where(
        has_recoil,
        jnp.maximum(jnp.int32(1), dmg * recoil_num // recoil_den),
        jnp.int32(0)
    )
    state = apply_damage(state, atk_side, atk_idx, recoil_dmg)

    # Self-heal (recovery moves: Recover, Roost, Softboiled, etc.)
    # MF_HEAL_NUM / MF_HEAL_DEN encode the fraction of max HP to restore.
    heal_num = tables.moves[move_id.astype(jnp.int32), MF_HEAL_NUM].astype(jnp.int32)
    heal_den = tables.moves[move_id.astype(jnp.int32), MF_HEAL_DEN].astype(jnp.int32)
    has_heal = (heal_num > jnp.int32(0)) & (heal_den > jnp.int32(0)) & ~cancelled
    max_hp_atk = state.sides_team_max_hp[atk_side, atk_idx].astype(jnp.int32)
    heal_amt = jnp.where(
        has_heal,
        jnp.maximum(jnp.int32(1), max_hp_atk * heal_num // heal_den),
        jnp.int32(0),
    )
    state = apply_heal(state, atk_side, atk_idx, heal_amt.astype(jnp.int16))

    # Secondary effect: status (e.g. Flamethrower 10% burn)
    # Substitute blocks all secondary effects on the defender (Showdown behavior).
    sec_chance = tables.moves[move_id.astype(jnp.int32), MF_SEC_CHANCE].astype(jnp.int32)
    sec_status  = tables.moves[move_id.astype(jnp.int32), MF_SEC_STATUS].astype(jnp.int8)
    has_secondary = (sec_chance > jnp.int32(0)) & (sec_status > jnp.int8(0))

    # Serene Grace: double secondary effect chance
    from pokejax.mechanics.abilities import SERENE_GRACE_ID
    atk_ability_sg = state.sides_team_ability_id[atk_side, atk_idx].astype(jnp.int32)
    has_serene_grace = (SERENE_GRACE_ID >= 0) & (atk_ability_sg == jnp.int32(SERENE_GRACE_ID))
    effective_sec_chance = jnp.where(has_serene_grace,
                                      jnp.minimum(jnp.int32(100), sec_chance * 2),
                                      sec_chance)

    key, sec_key = rng_utils.split(key)
    sec_hits = has_secondary & rng_utils.rand_bool_pct(sec_key, effective_sec_chance) & ~cancelled & ~def_has_sub

    # Apply secondary status if it triggers and target has no status already
    cur_status = state.sides_team_status[def_side, def_idx]
    no_status  = cur_status == jnp.int8(0)
    new_status = jnp.where(sec_hits & no_status, sec_status, cur_status)
    new_status_arr = state.sides_team_status.at[def_side, def_idx].set(new_status)
    state = state._replace(sides_team_status=new_status_arr)

    # Synchronize: reflect BRN/PSN/PAR back to attacker
    from pokejax.mechanics.abilities import SYNCHRONIZE_ID
    from pokejax.types import STATUS_BRN, STATUS_PSN, STATUS_TOX, STATUS_PAR
    def_ability_sync = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)
    has_synchronize = (SYNCHRONIZE_ID >= 0) & (def_ability_sync == jnp.int32(SYNCHRONIZE_ID))
    status_applied = sec_hits & no_status
    # Synchronize only reflects BRN, PSN, PAR (not TOX, SLP, FRZ)
    sync_status = sec_status
    can_sync = (sync_status == jnp.int8(STATUS_BRN)) | (sync_status == jnp.int8(STATUS_PSN)) | (sync_status == jnp.int8(STATUS_PAR))
    atk_no_status = state.sides_team_status[atk_side, atk_idx] == jnp.int8(0)
    do_sync = has_synchronize & status_applied & can_sync & atk_no_status & ~cancelled
    atk_new_status = jnp.where(do_sync, sync_status, state.sides_team_status[atk_side, atk_idx])
    state = state._replace(
        sides_team_status=state.sides_team_status.at[atk_side, atk_idx].set(atk_new_status)
    )

    # Secondary effect: stat change on foe (e.g. Psychic -10% SPD, Crunch -20% DEF)
    sec_boost_stat = tables.moves[move_id.astype(jnp.int32), MF_SEC_BOOST_STAT].astype(jnp.int32)
    sec_boost_amt  = tables.moves[move_id.astype(jnp.int32), MF_SEC_BOOST_AMT].astype(jnp.int32)
    has_sec_boost  = (sec_chance > jnp.int32(0)) & (sec_boost_amt != jnp.int32(0))

    key, sec_b_key = rng_utils.split(key)
    sec_b_hits = has_sec_boost & rng_utils.rand_bool_pct(sec_b_key, effective_sec_chance) & ~cancelled & ~def_has_sub

    # Ability interactions for foe-inflicted secondary stat changes
    from pokejax.mechanics.abilities import (
        CLEAR_BODY_ID, WHITE_SMOKE_ID, HYPER_CUTTER_ID, KEEN_EYE_ID, SIMPLE_ID,
    )
    from pokejax.types import BOOST_ATK, BOOST_ACC
    def_ability = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)

    # Simple: double the stat change
    has_simple = (SIMPLE_ID >= 0) & (def_ability == jnp.int32(SIMPLE_ID))
    sec_boost_amt = jnp.where(has_simple, sec_boost_amt * 2, sec_boost_amt)

    s_clamped = jnp.clip(sec_boost_stat, 0, 6)

    # Clear Body / White Smoke: block all negative foe stat drops
    has_clear_body = (CLEAR_BODY_ID >= 0) & (def_ability == jnp.int32(CLEAR_BODY_ID))
    has_white_smoke = (WHITE_SMOKE_ID >= 0) & (def_ability == jnp.int32(WHITE_SMOKE_ID))
    is_drop = sec_boost_amt < 0
    blocked_clear = is_drop & (has_clear_body | has_white_smoke)

    # Hyper Cutter: block ATK drops from foes
    has_hyper_cutter = (HYPER_CUTTER_ID >= 0) & (def_ability == jnp.int32(HYPER_CUTTER_ID))
    blocked_hc = is_drop & has_hyper_cutter & (s_clamped == jnp.int32(BOOST_ATK))

    # Keen Eye: block accuracy drops from foes
    has_keen_eye = (KEEN_EYE_ID >= 0) & (def_ability == jnp.int32(KEEN_EYE_ID))
    blocked_ke = is_drop & has_keen_eye & (s_clamped == jnp.int32(BOOST_ACC))

    sec_b_hits = sec_b_hits & ~(blocked_clear | blocked_hc | blocked_ke)

    cur_boost  = state.sides_team_boosts[def_side, def_idx, s_clamped]
    new_boost  = jnp.clip(cur_boost.astype(jnp.int32) + sec_boost_amt, -6, 6).astype(jnp.int8)
    chosen_b   = jnp.where(sec_b_hits, new_boost, cur_boost)
    new_boosts = state.sides_team_boosts.at[def_side, def_idx, s_clamped].set(chosen_b)
    state = state._replace(sides_team_boosts=new_boosts)

    # Defrost: moves with the FLAG_DEFROST flag thaw a frozen target on hit.
    # (Fire-type moves and certain other moves carry this flag in the extractor.)
    # Matches Showdown's onHit handler for the FRZ condition.
    flags_lo    = tables.moves[move_id.astype(jnp.int32), MF_FLAGS_LO].astype(jnp.int32)
    has_defrost = (flags_lo & jnp.int32(FLAG_DEFROST)) != jnp.int32(0)
    def_status  = state.sides_team_status[def_side, def_idx]
    def_frozen  = def_status == jnp.int8(STATUS_FRZ)
    do_defrost  = has_defrost & def_frozen & ~cancelled & (dmg > jnp.int32(0))
    new_def_status = jnp.where(do_defrost, jnp.int8(STATUS_NONE), def_status)
    new_status_arr2 = state.sides_team_status.at[def_side, def_idx].set(new_def_status)
    state = state._replace(sides_team_status=new_status_arr2)

    return state, dmg.astype(jnp.int32), is_crit, new_key


def execute_move_hit(tables, state: BattleState,
                      atk_side: int, def_side: int,
                      move_id: jnp.ndarray,
                      key: jnp.ndarray,
                      cfg,
                      ) -> tuple[BattleState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run the full 8-step hit pipeline for one move attempt.

    Returns: (new_state, total_damage, is_crit, new_key, cancelled)

    `cancelled` is returned so execute_move_action can pass it to
    execute_move_effects to suppress effects on missed moves.

    Handles multi-hit moves by repeating step 8 (unrolled for max hits=5).
    """
    atk_idx = state.sides_active_idx[atk_side]
    def_idx = state.sides_active_idx[def_side]
    move_id_i32 = move_id.astype(jnp.int32)

    # ---- Event relay computation (before the multi-hit loop) ----
    # ModifyAtk / ModifySpA: determines attacker's effective stat multiplier
    atk_relay_raw = run_event_modify_atk(
        jnp.float32(1.0), state, atk_side, atk_idx, def_side, def_idx, move_id
    )
    spa_relay_raw = run_event_modify_spa(
        jnp.float32(1.0), state, atk_side, atk_idx, def_side, def_idx, move_id
    )
    move_category = tables.moves[move_id_i32, MF_CATEGORY]
    is_physical_move = (move_category == jnp.int8(0))  # CATEGORY_PHYSICAL = 0
    atk_relay = jnp.where(is_physical_move, atk_relay_raw, spa_relay_raw)

    # BasePower event (Technician, Iron Fist, etc.)
    move_bp_f = tables.moves[move_id_i32, MF_BASE_POWER].astype(jnp.float32)
    bp_relay = run_event_base_power(move_bp_f, state, atk_side, atk_idx, def_side, def_idx, move_id)
    # bp_relay now contains the effective base power (not a multiplier); convert to ratio
    bp_relay_ratio = jnp.where(move_bp_f > jnp.float32(0.0),
                                bp_relay / move_bp_f,
                                jnp.float32(1.0))

    # ModifyDef/SpD: defender's ability may modify defense stats
    def_relay_raw = run_event_modify_def(
        jnp.float32(1.0), state, atk_side, atk_idx, def_side, def_idx, move_id
    )
    spd_relay_raw = run_event_modify_spd(
        jnp.float32(1.0), state, atk_side, atk_idx, def_side, def_idx, move_id
    )
    def_relay_final = jnp.where(is_physical_move, def_relay_raw, spd_relay_raw)

    # ModifyDamage event (Life Orb, Expert Belt, etc.)
    damage_relay = run_event_modify_damage(
        jnp.float32(1.0), state, atk_side, atk_idx, def_side, def_idx, move_id
    )

    # Guts: prevents burn penalty; set by ability population (global from abilities.py)
    from pokejax.mechanics.abilities import GUTS_ID, ADAPTABILITY_ID
    atk_ability_id = state.sides_team_ability_id[atk_side, atk_idx].astype(jnp.int32)
    guts = (atk_ability_id == jnp.int32(GUTS_ID)) & (jnp.int32(GUTS_ID) > 0)
    adaptability = (atk_ability_id == jnp.int32(ADAPTABILITY_ID)) & (jnp.int32(ADAPTABILITY_ID) > 0)

    # Determine if this move bypasses type immunity.
    # Self-targeting and field/side-targeting moves should never be cancelled
    # by the type chart (e.g. Swords Dance vs Ghost, Spikes vs Flying).
    move_target = tables.moves[move_id_i32, MF_TARGET].astype(jnp.int32)
    bypass_type_immunity = (
        (move_target == jnp.int32(_TARGET_SELF))      |
        (move_target == jnp.int32(_TARGET_ALL))        |
        (move_target == jnp.int32(_TARGET_ALLY_TEAM))  |
        (move_target == jnp.int32(_TARGET_FOE_SIDE))   |
        (move_target == jnp.int32(_TARGET_ALLY_SIDE))
    )

    # Compute move type once — used by steps 2, 3, and defrost in step 8.
    move_type_i32 = tables.moves[move_id_i32, MF_TYPE].astype(jnp.int32)

    # Step 1: Invulnerability
    cancelled = step1_invulnerability(state, atk_side, def_side, move_id)

    # Step 2: TryHit (protection/absorption + state-mutating effects).
    # move_type_i32 is forwarded so absorption abilities (Water Absorb, Volt Absorb,
    # Sap Sipper, Motor Drive, etc.) can match on the correct move type.
    _, cancelled, state = step2_try_hit(state, atk_side, atk_idx, def_side, def_idx,
                                         move_id, cancelled, move_type_i32)

    # Step 3: Type immunity — skipped for self/field-targeting moves
    effectiveness, type_cancelled = step3_type_immunity(
        tables, state, def_side, def_idx, move_id, cancelled
    )
    # Scrappy: Normal/Fighting moves hit Ghost types (bypass immunity)
    from pokejax.mechanics.abilities import SCRAPPY_ID
    from pokejax.types import TYPE_NORMAL, TYPE_FIGHTING, TYPE_GHOST
    has_scrappy = (SCRAPPY_ID >= 0) & (atk_ability_id == jnp.int32(SCRAPPY_ID))
    is_normal_or_fighting = (move_type_i32 == jnp.int32(TYPE_NORMAL)) | (move_type_i32 == jnp.int32(TYPE_FIGHTING))
    def_types_sc = state.sides_team_types[def_side, def_idx]
    is_ghost_type = (def_types_sc[0] == jnp.int8(TYPE_GHOST)) | (def_types_sc[1] == jnp.int8(TYPE_GHOST))
    scrappy_bypass = has_scrappy & is_normal_or_fighting & is_ghost_type
    # If Scrappy bypasses, recalculate effectiveness ignoring Ghost type
    # For Ghost/X, Normal is 0× Ghost but hits X; we force minimum 1.0 effectiveness
    effectiveness = jnp.where(scrappy_bypass & (effectiveness == jnp.float32(0.0)),
                               jnp.float32(1.0), effectiveness)
    type_cancelled = jnp.where(scrappy_bypass, cancelled, type_cancelled)

    cancelled = jnp.where(bypass_type_immunity, cancelled, type_cancelled)
    effectiveness = jnp.where(bypass_type_immunity, jnp.float32(1.0), effectiveness)

    # Step 4: TryImmunity — Wonder Guard blocks non-super-effective moves.
    from pokejax.mechanics.abilities import WONDER_GUARD_ID, MOLD_BREAKER_ID
    def_ability_id = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)
    has_wonder_guard = (def_ability_id == jnp.int32(WONDER_GUARD_ID)) & (jnp.int32(WONDER_GUARD_ID) > 0)
    # Mold Breaker on attacker bypasses Wonder Guard
    has_mold_breaker = (atk_ability_id == jnp.int32(MOLD_BREAKER_ID)) & (jnp.int32(MOLD_BREAKER_ID) > 0)
    # Status moves (category 2) bypass Wonder Guard
    is_status_move = (move_category == jnp.int8(2))
    wg_blocks = has_wonder_guard & ~has_mold_breaker & ~is_status_move & (effectiveness <= jnp.float32(1.0))
    cancelled = cancelled | wg_blocks

    # Step 4b: Substitute blocks status moves targeting the defender.
    # In Showdown, most status moves fail against a Substitute (except sound-based).
    def_has_sub = (state.sides_team_volatiles[def_side, def_idx]
                   & jnp.uint32(1 << VOL_SUBSTITUTE)) != jnp.uint32(0)
    is_status_targeting_foe = (
        is_status_move
        & ~bypass_type_immunity  # not self/field/side-targeting
    )
    cancelled = cancelled | (is_status_targeting_foe & def_has_sub)

    # Step 5: Accuracy
    cancelled, key = step5_accuracy(
        tables, state, atk_side, atk_idx, def_side, def_idx, move_id, cancelled, key
    )

    # Step 6: BreakProtect (remove Protect etc. — handled by protect volatile bit)
    # Not separately implemented; Protect is checked in step 2.

    # Step 8: Damage (multi-hit unrolled up to 5 hits)
    multi_min = tables.moves[move_id_i32, MF_MULTIHIT_MIN].astype(jnp.int32)
    multi_max = tables.moves[move_id_i32, MF_MULTIHIT_MAX].astype(jnp.int32)
    is_multihit = multi_max > jnp.int32(1)

    # Roll number of hits
    # Skill Link: always max hits (5) on multi-hit moves
    from pokejax.mechanics.abilities import SKILL_LINK_ID
    has_skill_link = (SKILL_LINK_ID >= 0) & (atk_ability_id == jnp.int32(SKILL_LINK_ID))
    key, hit_key = rng_utils.split(key)
    n_hits = jnp.where(
        is_multihit,
        jnp.where(has_skill_link,
                   jnp.int32(5),
                   rng_utils.multi_hit_roll(hit_key, 2, 5).astype(jnp.int32)),
        jnp.int32(1)
    )

    # Unroll up to 5 hits (max for 2-5 hit moves)
    total_dmg = jnp.int32(0)
    final_crit = jnp.bool_(False)

    for hit_n in range(5):
        # Only apply this hit if hit_n < n_hits and move connects
        do_hit = (~cancelled) & (jnp.int32(hit_n) < n_hits)
        # Pass hit_cancelled=True for extra iterations so drain/heal/secondary
        # effects don't fire on unused hit slots.
        hit_cancelled = cancelled | ~do_hit

        state, hit_dmg, is_crit, key = step8_move_hit_loop(
            tables, state, atk_side, atk_idx, def_side, def_idx,
            move_id, effectiveness, hit_cancelled, key, cfg,
            atk_relay=atk_relay, def_relay=def_relay_final,
            bp_relay=bp_relay_ratio, damage_relay=damage_relay,
            guts=guts, adaptability=adaptability,
        )
        total_dmg = total_dmg + jnp.where(do_hit, hit_dmg, jnp.int32(0))
        final_crit = final_crit | (do_hit & is_crit)

    # Contact punishment (Rough Skin, Static, Flame Body, etc.)
    # Only triggers if: move made contact, was not cancelled, dealt damage
    flags_lo = tables.moves[move_id_i32, MF_FLAGS_LO].astype(jnp.int32)
    from pokejax.core.damage import FLAG_CONTACT
    is_contact = (flags_lo & jnp.int32(FLAG_CONTACT)) != jnp.int32(0)
    should_punish = is_contact & ~cancelled & (total_dmg > jnp.int32(0))
    # Always call (branchless) but noop handler does nothing
    state, key = run_event_contact_punish(
        state, atk_side, atk_idx, def_side, def_idx, key
    )

    return state, total_dmg, final_crit, key, cancelled
