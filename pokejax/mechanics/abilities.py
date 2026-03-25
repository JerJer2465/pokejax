"""
Gen 4 ability table population for PokeJAX.

Relay events (ModifyAtk, BasePower, TryHit, …) are registered by writing scalar
parameters into events.py's JAX array lookup tables — no lax.switch branches.

State-mutating events (SwitchIn, SwitchOut, ResidualState, TryHitState) are
registered by:
  1. Setting an effect-ID in events._AB_*_EFF[ability_id].
  2. Installing the concrete handler at events._*_HANDLERS[effect_id].

After populate_ability_tables(), the module-level IDs GUTS_ID and ADAPTABILITY_ID
are set for use by hit_pipeline.py.
"""

import jax
import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX, STATUS_SLP, STATUS_FRZ, STATUS_PAR,
    WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE,
    TYPE_WATER, TYPE_FIRE, TYPE_ELECTRIC, TYPE_GRASS, TYPE_GROUND,
    TYPE_NORMAL, TYPE_ICE, TYPE_FIGHTING, TYPE_POISON, TYPE_FLYING,
    TYPE_PSYCHIC, TYPE_BUG, TYPE_ROCK, TYPE_GHOST, TYPE_DRAGON,
    TYPE_DARK, TYPE_STEEL,
)
from pokejax.core.damage import (
    apply_heal, apply_damage, fraction_of_max_hp,
)
from pokejax.mechanics import events as ev

# ---------------------------------------------------------------------------
# Module-level ability ID constants (set by populate_ability_tables)
# ---------------------------------------------------------------------------
GUTS_ID           = -1
FLASH_FIRE_ID     = -1
ADAPTABILITY_ID   = -1
WONDER_GUARD_ID   = -1
MOLD_BREAKER_ID   = -1
LEVITATE_ID       = -1
NO_GUARD_ID       = -1
COMPOUND_EYES_ID  = -1
HUSTLE_ID         = -1
SAND_VEIL_ID      = -1
SNOW_CLOAK_ID     = -1
ARENA_TRAP_ID     = -1
SHADOW_TAG_ID     = -1
MAGNET_PULL_ID    = -1
BATTLE_ARMOR_ID   = -1
SHELL_ARMOR_ID    = -1
SNIPER_ID         = -1
SERENE_GRACE_ID   = -1
ROCK_HEAD_ID      = -1
STURDY_ID         = -1
SKILL_LINK_ID     = -1
SCRAPPY_ID        = -1
SUPER_LUCK_ID     = -1
THICK_FAT_ID      = -1
CLEAR_BODY_ID     = -1
WHITE_SMOKE_ID    = -1
HYPER_CUTTER_ID   = -1
KEEN_EYE_ID       = -1
SIMPLE_ID         = -1
UNAWARE_ID        = -1
SYNCHRONIZE_ID    = -1
EARLY_BIRD_ID     = -1
FILTER_ID         = -1
SOLID_ROCK_ID     = -1
STEADFAST_ID      = -1
TANGLED_FEET_ID   = -1
PRESSURE_ID       = -1
OWN_TEMPO_ID      = -1
OBLIVIOUS_ID      = -1


# ---------------------------------------------------------------------------
# Internal helpers (shared by state-mutating handlers)
# ---------------------------------------------------------------------------

def _apply_boost(state: BattleState, side_i32: jnp.ndarray,
                  slot_i32: jnp.ndarray, boost_idx: int,
                  amount: int) -> BattleState:
    cur = state.sides_team_boosts[side_i32, slot_i32, boost_idx]
    new_val = jnp.clip(cur.astype(jnp.int32) + amount, -6, 6).astype(jnp.int8)
    new_boosts = state.sides_team_boosts.at[side_i32, slot_i32, boost_idx].set(new_val)
    return state._replace(sides_team_boosts=new_boosts)


def _set_weather(state: BattleState, weather_id: int, turns: int) -> BattleState:
    new_field = state.field._replace(
        weather=jnp.int8(weather_id),
        weather_turns=jnp.int8(turns),
        weather_max_turns=jnp.int8(turns),
    )
    return state._replace(field=new_field)


def _foe_side(side_i32: jnp.ndarray) -> jnp.ndarray:
    return jnp.int32(1) - side_i32


# ---------------------------------------------------------------------------
# State-mutating SwitchIn handlers
# Registered at events._SWITCH_IN_HANDLERS[EFF_*] by populate_ability_tables().
# ---------------------------------------------------------------------------

def _intimidate_switch_in(state, side_i32, slot_i32):
    foe = _foe_side(side_i32)
    foe_idx = state.sides_active_idx[foe].astype(jnp.int32)
    cur = state.sides_team_boosts[foe, foe_idx, BOOST_ATK]
    new_val = jnp.clip(cur.astype(jnp.int32) - 1, -6, 6).astype(jnp.int8)
    new_boosts = state.sides_team_boosts.at[foe, foe_idx, BOOST_ATK].set(new_val)
    return state._replace(sides_team_boosts=new_boosts)


def _drizzle_switch_in(state, side_i32, slot_i32):
    # Gen 4: weather from abilities is permanent (use 127 = int8 max)
    return _set_weather(state, WEATHER_RAIN, 127)


def _drought_switch_in(state, side_i32, slot_i32):
    return _set_weather(state, WEATHER_SUN, 127)


def _sand_stream_switch_in(state, side_i32, slot_i32):
    return _set_weather(state, WEATHER_SAND, 127)


def _snow_warning_switch_in(state, side_i32, slot_i32):
    return _set_weather(state, WEATHER_HAIL, 127)


def _download_switch_in(state, side_i32, slot_i32):
    foe = _foe_side(side_i32)
    foe_idx = state.sides_active_idx[foe].astype(jnp.int32)
    foe_def = state.sides_team_base_stats[foe, foe_idx, 2].astype(jnp.int32)
    foe_spd = state.sides_team_base_stats[foe, foe_idx, 4].astype(jnp.int32)
    # Showdown: if foe Def > SpD → boost ATK; else → boost SpA
    boost_atk = foe_def > foe_spd
    cur_atk = state.sides_team_boosts[side_i32, slot_i32, BOOST_ATK]
    new_atk = jnp.clip(cur_atk.astype(jnp.int32) + 1, -6, 6).astype(jnp.int8)
    cur_spa = state.sides_team_boosts[side_i32, slot_i32, BOOST_SPA]
    new_spa = jnp.clip(cur_spa.astype(jnp.int32) + 1, -6, 6).astype(jnp.int8)
    chosen_atk = jnp.where(boost_atk, new_atk, cur_atk)
    chosen_spa = jnp.where(boost_atk, cur_spa, new_spa)
    new_boosts = state.sides_team_boosts.at[side_i32, slot_i32, BOOST_ATK].set(chosen_atk)
    new_boosts = new_boosts.at[side_i32, slot_i32, BOOST_SPA].set(chosen_spa)
    return state._replace(sides_team_boosts=new_boosts)


# ---------------------------------------------------------------------------
# State-mutating SwitchOut handlers
# ---------------------------------------------------------------------------

def _natural_cure_switch_out(state, side_i32, slot_i32):
    new_status = state.sides_team_status.at[side_i32, slot_i32].set(jnp.int8(STATUS_NONE))
    return state._replace(sides_team_status=new_status)


# ---------------------------------------------------------------------------
# State-mutating Residual handlers
# ---------------------------------------------------------------------------

def _speed_boost_residual(state, key, side_i32, slot_i32):
    return _apply_boost(state, side_i32, slot_i32, BOOST_SPE, 1), key


def _poison_heal_residual(state, key, side_i32, slot_i32):
    from pokejax.types import STATUS_PSN, STATUS_TOX
    status = state.sides_team_status[side_i32, slot_i32]
    is_poisoned = (status == jnp.int8(STATUS_PSN)) | (status == jnp.int8(STATUS_TOX))
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    heal   = jnp.maximum(jnp.int32(1), max_hp // 8)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(
        jnp.where(is_poisoned, new_hp, state.sides_team_hp[side_i32, slot_i32])
    )
    return state._replace(sides_team_hp=new_hp_arr), key


# ---------------------------------------------------------------------------
# State-mutating TryHit-state handlers
# ---------------------------------------------------------------------------

def _water_absorb_try_hit_state(state, atk_si, atk_idx_i32, def_si, def_idx_i32, move_type_i32):
    is_water = move_type_i32 == jnp.int32(TYPE_WATER)
    hp     = state.sides_team_hp[def_si, def_idx_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[def_si, def_idx_i32].astype(jnp.int32)
    heal   = jnp.maximum(jnp.int32(1), max_hp // 4)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[def_si, def_idx_i32].set(
        jnp.where(is_water, new_hp, state.sides_team_hp[def_si, def_idx_i32])
    )
    return state._replace(sides_team_hp=new_hp_arr)


def _volt_absorb_try_hit_state(state, atk_si, atk_idx_i32, def_si, def_idx_i32, move_type_i32):
    is_electric = move_type_i32 == jnp.int32(TYPE_ELECTRIC)
    hp     = state.sides_team_hp[def_si, def_idx_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[def_si, def_idx_i32].astype(jnp.int32)
    heal   = jnp.maximum(jnp.int32(1), max_hp // 4)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[def_si, def_idx_i32].set(
        jnp.where(is_electric, new_hp, state.sides_team_hp[def_si, def_idx_i32])
    )
    return state._replace(sides_team_hp=new_hp_arr)


def _motor_drive_try_hit_state(state, atk_si, atk_idx_i32, def_si, def_idx_i32, move_type_i32):
    is_electric = move_type_i32 == jnp.int32(TYPE_ELECTRIC)
    cur = state.sides_team_boosts[def_si, def_idx_i32, BOOST_SPE]
    new_val = jnp.clip(cur.astype(jnp.int32) + 1, -6, 6).astype(jnp.int8)
    new_boosts = state.sides_team_boosts.at[def_si, def_idx_i32, BOOST_SPE].set(
        jnp.where(is_electric, new_val, cur)
    )
    return state._replace(sides_team_boosts=new_boosts)


def _sap_sipper_try_hit_state(state, atk_si, atk_idx_i32, def_si, def_idx_i32, move_type_i32):
    is_grass = move_type_i32 == jnp.int32(TYPE_GRASS)
    cur = state.sides_team_boosts[def_si, def_idx_i32, BOOST_ATK]
    new_val = jnp.clip(cur.astype(jnp.int32) + 1, -6, 6).astype(jnp.int8)
    new_boosts = state.sides_team_boosts.at[def_si, def_idx_i32, BOOST_ATK].set(
        jnp.where(is_grass, new_val, cur)
    )
    return state._replace(sides_team_boosts=new_boosts)


def _storm_drain_try_hit_state(state, atk_si, atk_idx_i32, def_si, def_idx_i32, move_type_i32):
    is_water = move_type_i32 == jnp.int32(TYPE_WATER)
    cur = state.sides_team_boosts[def_si, def_idx_i32, BOOST_SPA]
    new_val = jnp.clip(cur.astype(jnp.int32) + 1, -6, 6).astype(jnp.int8)
    new_boosts = state.sides_team_boosts.at[def_si, def_idx_i32, BOOST_SPA].set(
        jnp.where(is_water, new_val, cur)
    )
    return state._replace(sides_team_boosts=new_boosts)


def _lightning_rod_try_hit_state(state, atk_si, atk_idx_i32, def_si, def_idx_i32, move_type_i32):
    is_electric = move_type_i32 == jnp.int32(TYPE_ELECTRIC)
    cur = state.sides_team_boosts[def_si, def_idx_i32, BOOST_SPA]
    new_val = jnp.clip(cur.astype(jnp.int32) + 1, -6, 6).astype(jnp.int8)
    new_boosts = state.sides_team_boosts.at[def_si, def_idx_i32, BOOST_SPA].set(
        jnp.where(is_electric, new_val, cur)
    )
    return state._replace(sides_team_boosts=new_boosts)


def _dry_skin_absorb_try_hit_state(state, atk_si, atk_idx_i32, def_si, def_idx_i32, move_type_i32):
    """Dry Skin: Water moves → heal 25% (immunity handled by TryHit relay)."""
    is_water = move_type_i32 == jnp.int32(TYPE_WATER)
    hp     = state.sides_team_hp[def_si, def_idx_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[def_si, def_idx_i32].astype(jnp.int32)
    heal   = jnp.maximum(jnp.int32(1), max_hp // 4)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[def_si, def_idx_i32].set(
        jnp.where(is_water, new_hp, state.sides_team_hp[def_si, def_idx_i32])
    )
    return state._replace(sides_team_hp=new_hp_arr)


def _flash_fire_try_hit_state(state, atk_si, atk_idx_i32, def_si, def_idx_i32, move_type_i32):
    """Flash Fire: absorbing a Fire move sets VOL_FLASH_FIRE, boosting Fire moves by ×1.5."""
    from pokejax.types import VOL_FLASH_FIRE
    is_fire = move_type_i32 == jnp.int32(TYPE_FIRE)
    ff_mask = jnp.uint32(1 << VOL_FLASH_FIRE)
    old_vols = state.sides_team_volatiles[def_si, def_idx_i32]
    new_vols_val = jnp.where(is_fire, old_vols | ff_mask, old_vols)
    new_vols_arr = state.sides_team_volatiles.at[def_si, def_idx_i32].set(new_vols_val)
    return state._replace(sides_team_volatiles=new_vols_arr)


# ---------------------------------------------------------------------------
# State-mutating SwitchIn handlers (new: Trace, Forecast, Frisk, etc.)
# ---------------------------------------------------------------------------

def _trace_switch_in(state, side_i32, slot_i32):
    """Trace: copy the foe's ability."""
    foe = _foe_side(side_i32)
    foe_idx = state.sides_active_idx[foe].astype(jnp.int32)
    foe_ability = state.sides_team_ability_id[foe, foe_idx]
    new_abilities = state.sides_team_ability_id.at[side_i32, slot_i32].set(foe_ability)
    return state._replace(sides_team_ability_id=new_abilities)


def _forecast_switch_in(state, side_i32, slot_i32):
    """Forecast: Castform type changes based on weather. Simplified noop for now."""
    return state


def _frisk_switch_in(state, side_i32, slot_i32):
    """Frisk: reveals foe's item. No state change needed for RL (perfect info)."""
    return state


def _anticipation_switch_in(state, side_i32, slot_i32):
    """Anticipation: alert if foe has SE moves. No state change for RL."""
    return state


def _forewarn_switch_in(state, side_i32, slot_i32):
    """Forewarn: reveals foe's strongest move. No state change for RL."""
    return state


# ---------------------------------------------------------------------------
# State-mutating SwitchOut handlers (new: Regenerator)
# ---------------------------------------------------------------------------

def _regenerator_switch_out(state, side_i32, slot_i32):
    """Regenerator: heal 1/3 max HP on switch-out."""
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    heal   = jnp.maximum(jnp.int32(1), max_hp // 3)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr)


# ---------------------------------------------------------------------------
# State-mutating Residual handlers (new: Bad Dreams, Dry Skin, Rain Dish, etc.)
# ---------------------------------------------------------------------------

def _bad_dreams_residual(state, key, side_i32, slot_i32):
    """Bad Dreams: each sleeping foe loses 1/8 max HP."""
    foe = _foe_side(side_i32)
    foe_idx = state.sides_active_idx[foe].astype(jnp.int32)
    foe_status = state.sides_team_status[foe, foe_idx]
    is_sleeping = foe_status == jnp.int8(STATUS_SLP)
    foe_hp     = state.sides_team_hp[foe, foe_idx].astype(jnp.int32)
    foe_max_hp = state.sides_team_max_hp[foe, foe_idx].astype(jnp.int32)
    dmg = jnp.maximum(jnp.int32(1), foe_max_hp // 8)
    new_hp = jnp.maximum(jnp.int32(0), foe_hp - dmg).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[foe, foe_idx].set(
        jnp.where(is_sleeping, new_hp, state.sides_team_hp[foe, foe_idx])
    )
    return state._replace(sides_team_hp=new_hp_arr), key


def _dry_skin_residual(state, key, side_i32, slot_i32):
    """Dry Skin: heal 1/8 in rain, lose 1/8 in sun."""
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    weather = state.field.weather
    in_rain = weather == jnp.int8(WEATHER_RAIN)
    in_sun  = weather == jnp.int8(WEATHER_SUN)
    heal = jnp.maximum(jnp.int32(1), max_hp // 8)
    dmg  = jnp.maximum(jnp.int32(1), max_hp // 8)
    new_hp_rain = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_sun  = jnp.maximum(jnp.int32(0), hp - dmg).astype(jnp.int16)
    new_hp = jnp.where(in_rain, new_hp_rain,
             jnp.where(in_sun, new_hp_sun,
                        state.sides_team_hp[side_i32, slot_i32]))
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr), key


def _rain_dish_residual(state, key, side_i32, slot_i32):
    """Rain Dish: heal 1/16 in rain."""
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    in_rain = state.field.weather == jnp.int8(WEATHER_RAIN)
    heal = jnp.maximum(jnp.int32(1), max_hp // 16)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(
        jnp.where(in_rain, new_hp, state.sides_team_hp[side_i32, slot_i32])
    )
    return state._replace(sides_team_hp=new_hp_arr), key


def _ice_body_residual(state, key, side_i32, slot_i32):
    """Ice Body: heal 1/16 in hail."""
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    in_hail = state.field.weather == jnp.int8(WEATHER_HAIL)
    heal = jnp.maximum(jnp.int32(1), max_hp // 16)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(
        jnp.where(in_hail, new_hp, state.sides_team_hp[side_i32, slot_i32])
    )
    return state._replace(sides_team_hp=new_hp_arr), key


def _shed_skin_residual(state, key, side_i32, slot_i32):
    """Shed Skin: 33% chance to cure status each turn (Gen 4 PS: randomChance(1,3))."""
    from pokejax.core import rng as rng_utils
    status = state.sides_team_status[side_i32, slot_i32]
    has_status = status != jnp.int8(STATUS_NONE)
    key, subkey = rng_utils.split(key)
    cured = has_status & rng_utils.rand_bool_pct(subkey, 33)
    new_status = jnp.where(cured, jnp.int8(STATUS_NONE), status)
    new_status_arr = state.sides_team_status.at[side_i32, slot_i32].set(new_status)
    return state._replace(sides_team_status=new_status_arr), key


# ---------------------------------------------------------------------------
# Contact punishment handlers
# ---------------------------------------------------------------------------

def _rough_skin_contact(state, atk_si, atk_idx_i32, def_si, def_idx_i32, key):
    """Rough Skin / Iron Barbs: attacker loses 1/8 max HP on contact."""
    hp     = state.sides_team_hp[atk_si, atk_idx_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[atk_si, atk_idx_i32].astype(jnp.int32)
    dmg = jnp.maximum(jnp.int32(1), max_hp // 8)
    new_hp = jnp.maximum(jnp.int32(0), hp - dmg).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[atk_si, atk_idx_i32].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr), key


def _static_contact(state, atk_si, atk_idx_i32, def_si, def_idx_i32, key):
    """Static: 30% chance to paralyze attacker on contact."""
    from pokejax.core import rng as rng_utils
    key, subkey = rng_utils.split(key)
    triggers = rng_utils.rand_bool_pct(subkey, jnp.int32(30))
    cur_status = state.sides_team_status[atk_si, atk_idx_i32]
    no_status = cur_status == jnp.int8(STATUS_NONE)
    new_status = jnp.where(triggers & no_status, jnp.int8(STATUS_PAR), cur_status)
    new_status_arr = state.sides_team_status.at[atk_si, atk_idx_i32].set(new_status)
    return state._replace(sides_team_status=new_status_arr), key


def _poison_point_contact(state, atk_si, atk_idx_i32, def_si, def_idx_i32, key):
    """Poison Point: 30% chance to poison attacker on contact."""
    from pokejax.core import rng as rng_utils
    key, subkey = rng_utils.split(key)
    triggers = rng_utils.rand_bool_pct(subkey, jnp.int32(30))
    cur_status = state.sides_team_status[atk_si, atk_idx_i32]
    no_status = cur_status == jnp.int8(STATUS_NONE)
    # Check poison type immunity
    atk_types = state.sides_team_types[atk_si, atk_idx_i32]
    is_poison_type = (atk_types[0] == jnp.int8(TYPE_POISON)) | (atk_types[1] == jnp.int8(TYPE_POISON))
    is_steel_type = (atk_types[0] == jnp.int8(TYPE_STEEL)) | (atk_types[1] == jnp.int8(TYPE_STEEL))
    can_poison = ~is_poison_type & ~is_steel_type
    new_status = jnp.where(triggers & no_status & can_poison, jnp.int8(STATUS_PSN), cur_status)
    new_status_arr = state.sides_team_status.at[atk_si, atk_idx_i32].set(new_status)
    return state._replace(sides_team_status=new_status_arr), key


def _flame_body_contact(state, atk_si, atk_idx_i32, def_si, def_idx_i32, key):
    """Flame Body: 30% chance to burn attacker on contact."""
    from pokejax.core import rng as rng_utils
    key, subkey = rng_utils.split(key)
    triggers = rng_utils.rand_bool_pct(subkey, jnp.int32(30))
    cur_status = state.sides_team_status[atk_si, atk_idx_i32]
    no_status = cur_status == jnp.int8(STATUS_NONE)
    # Fire types are immune to burn
    atk_types = state.sides_team_types[atk_si, atk_idx_i32]
    is_fire_type = (atk_types[0] == jnp.int8(TYPE_FIRE)) | (atk_types[1] == jnp.int8(TYPE_FIRE))
    new_status = jnp.where(triggers & no_status & ~is_fire_type, jnp.int8(STATUS_BRN), cur_status)
    new_status_arr = state.sides_team_status.at[atk_si, atk_idx_i32].set(new_status)
    return state._replace(sides_team_status=new_status_arr), key


def _effect_spore_contact(state, atk_si, atk_idx_i32, def_si, def_idx_i32, key):
    """Effect Spore: 30% chance to inflict PSN/PAR/SLP (10% each) on contact."""
    from pokejax.core import rng as rng_utils
    key, subkey = rng_utils.split(key)
    triggers = rng_utils.rand_bool_pct(subkey, jnp.int32(30))
    cur_status = state.sides_team_status[atk_si, atk_idx_i32]
    no_status = cur_status == jnp.int8(STATUS_NONE)
    # Grass types are immune to Effect Spore
    atk_types = state.sides_team_types[atk_si, atk_idx_i32]
    is_grass_type = (atk_types[0] == jnp.int8(TYPE_GRASS)) | (atk_types[1] == jnp.int8(TYPE_GRASS))
    # Pick which status: use a second roll (0=PSN, 1=PAR, 2=SLP)
    key, pick_key = rng_utils.split(key)
    roll = jax.random.randint(pick_key, (), 0, 3)
    chosen_status = jnp.where(roll == jnp.int32(0), jnp.int8(STATUS_PSN),
                    jnp.where(roll == jnp.int32(1), jnp.int8(STATUS_PAR),
                              jnp.int8(STATUS_SLP)))
    new_status = jnp.where(triggers & no_status & ~is_grass_type, chosen_status, cur_status)
    new_status_arr = state.sides_team_status.at[atk_si, atk_idx_i32].set(new_status)
    return state._replace(sides_team_status=new_status_arr), key


def _cute_charm_contact(state, atk_si, atk_idx_i32, def_si, def_idx_i32, key):
    """Cute Charm: 30% chance to infatuate attacker. Simplified noop for RL."""
    # Infatuation requires opposite genders and is uncommon in Gen 4 randbats.
    # Simplified as noop for now.
    return state, key


# ---------------------------------------------------------------------------
# Ability parameter registry
#
# Keys map to events.py array updates in populate_ability_tables():
#   modify_atk_mult  → ev._AB_ATK_MULT[id]      (float32 constant multiplier)
#   modify_atk_cond  → ev._AB_ATK_COND[id]      (int8 condition code)
#   modify_spa_mult  → ev._AB_SPA_MULT[id]
#   modify_spa_cond  → ev._AB_SPA_COND[id]
#   bp_cond          → ev._AB_BP_COND[id]
#   bp_type          → ev._AB_BP_TYPE[id]        (int8 triggering type)
#   modify_dmg_mult  → ev._AB_DMG_MULT[id]
#   tryhit_immune    → ev._AB_TRYHIT_IMMUNE[id]  (int8 type, -1 = none)
#   tryhit_state_eff → ev._AB_TRYHIT_STATE_EFF[id]  (int8 effect ID)
#   switch_in_eff    → ev._AB_SWITCH_IN_EFF[id]
#   switch_out_eff   → ev._AB_SWITCH_OUT_EFF[id]
#   residual_eff     → ev._AB_RESIDUAL_EFF[id]
# ---------------------------------------------------------------------------

ABILITY_HANDLERS = {
    # ==== Stat multiplier abilities (Tier 1 relay) ====

    # ---- ModifyAtk (constant) ----
    "Huge Power":    {"modify_atk_mult": 2.0},
    "Pure Power":    {"modify_atk_mult": 2.0},
    "Hustle":        {"modify_atk_mult": 1.5},
    "Flower Gift":   {"modify_atk_cond": ev.ABCOND_ATK_SUN},  # sun → ×1.5 Atk (also SpD relay)

    # ---- ModifyAtk (conditional) ----
    "Guts":          {"modify_atk_cond": ev.ABCOND_ATK_GUTS},

    # ---- ModifyDef (conditional) ----
    "Marvel Scale":  {"modify_def_cond": ev.ABCOND_DEF_MARVEL_SCALE},

    # ---- ModifySpD (conditional) ----
    # Flower Gift SpD boost handled via modify_spd_cond
    # (Flower Gift also has modify_atk_mult above)

    # ---- ModifySpa ----
    "Solar Power":   {"modify_spa_cond": ev.ABCOND_SPA_SOLAR_POWER},

    # ---- BasePower conditions ----
    "Technician":    {"bp_cond": ev.ABCOND_BP_TECHNICIAN},
    "Iron Fist":     {"bp_cond": ev.ABCOND_BP_IRON_FIST},
    "Reckless":      {"bp_cond": ev.ABCOND_BP_RECKLESS},

    # ---- Starter trio (type + low HP) ----
    "Blaze":         {"bp_cond": ev.ABCOND_BP_STARTER, "bp_type": TYPE_FIRE},
    "Torrent":       {"bp_cond": ev.ABCOND_BP_STARTER, "bp_type": TYPE_WATER},
    "Overgrow":      {"bp_cond": ev.ABCOND_BP_STARTER, "bp_type": TYPE_GRASS},
    "Swarm":         {"bp_cond": ev.ABCOND_BP_STARTER, "bp_type": TYPE_BUG},

    # ---- ModifyDamage (conditional) ----
    # Sniper: ×1.5 crit multiplier — handled via SNIPER_ID in compute_damage
    # Tinted Lens: ×2 on NVE — handled via ability damage condition
    "Tinted Lens":   {"modify_dmg_cond": ev.ABCOND_DMG_TINTED_LENS},

    # ---- Thick Fat: handled via THICK_FAT_ID in compute_damage ----

    # ==== Speed modifiers (Tier 1 relay) ====
    "Swift Swim":    {"speed_cond": ev.ABCOND_SPEED_RAIN},
    "Chlorophyll":   {"speed_cond": ev.ABCOND_SPEED_SUN},

    # ==== Absorption / immunity (TryHit bool + state) ====
    "Water Absorb":  {"tryhit_immune": TYPE_WATER, "tryhit_state_eff": ev.EFF_WATER_ABSORB},
    "Volt Absorb":   {"tryhit_immune": TYPE_ELECTRIC, "tryhit_state_eff": ev.EFF_VOLT_ABSORB},
    "Flash Fire":    {"tryhit_immune": TYPE_FIRE,
                      "tryhit_state_eff": ev.EFF_FLASH_FIRE,
                      "bp_cond": ev.ABCOND_BP_FLASH_FIRE},
    "Levitate":      {"tryhit_immune": TYPE_GROUND},
    "Motor Drive":   {"tryhit_immune": TYPE_ELECTRIC, "tryhit_state_eff": ev.EFF_MOTOR_DRIVE},
    "Sap Sipper":    {"tryhit_immune": TYPE_GRASS, "tryhit_state_eff": ev.EFF_SAP_SIPPER},
    "Storm Drain":   {"tryhit_immune": TYPE_WATER, "tryhit_state_eff": ev.EFF_STORM_DRAIN},
    "Lightning Rod": {"tryhit_immune": TYPE_ELECTRIC, "tryhit_state_eff": ev.EFF_LIGHTNING_ROD},
    "Dry Skin":      {"tryhit_immune": TYPE_WATER, "tryhit_state_eff": ev.EFF_DRY_SKIN_ABSORB,
                      "residual_eff": ev.EFF_DRY_SKIN_RES},

    # ==== Status immunity (Tier 1 relay) ====
    "Immunity":      {"status_immune": STATUS_PSN},   # also covers TOX
    "Limber":        {"status_immune": STATUS_PAR},
    "Insomnia":      {"status_immune": STATUS_SLP},
    "Vital Spirit":  {"status_immune": STATUS_SLP},
    "Magma Armor":   {"status_immune": STATUS_FRZ},
    "Water Veil":    {"status_immune": STATUS_BRN},
    "Own Tempo":     {},  # confusion immunity (not a status code; handled in volatile check)
    "Oblivious":     {},  # infatuation immunity (handled in volatile check)

    # ==== Contact punishment (Tier 2 state-mutating) ====
    "Rough Skin":    {"contact_eff": ev.EFF_ROUGH_SKIN},
    # Iron Barbs is Gen 5 only; omitted for Gen 4
    "Static":        {"contact_eff": ev.EFF_STATIC_CONTACT},
    "Poison Point":  {"contact_eff": ev.EFF_POISON_POINT},
    "Flame Body":    {"contact_eff": ev.EFF_FLAME_BODY},
    "Effect Spore":  {"contact_eff": ev.EFF_EFFECT_SPORE},
    "Cute Charm":    {"contact_eff": ev.EFF_CUTE_CHARM},

    # ==== Switch-in effects (Tier 2) ====
    "Intimidate":    {"switch_in_eff": ev.EFF_INTIMIDATE},
    "Drizzle":       {"switch_in_eff": ev.EFF_DRIZZLE},
    "Drought":       {"switch_in_eff": ev.EFF_DROUGHT},
    "Sand Stream":   {"switch_in_eff": ev.EFF_SAND_STREAM},
    "Snow Warning":  {"switch_in_eff": ev.EFF_SNOW_WARNING},
    "Download":      {"switch_in_eff": ev.EFF_DOWNLOAD},
    "Trace":         {"switch_in_eff": ev.EFF_TRACE},
    "Forecast":      {"switch_in_eff": ev.EFF_FORECAST},
    "Frisk":         {"switch_in_eff": ev.EFF_FRISK},
    "Anticipation":  {"switch_in_eff": ev.EFF_ANTICIPATION},
    "Forewarn":      {"switch_in_eff": ev.EFF_FOREWARN},

    # ==== Switch-out effects (Tier 2) ====
    "Natural Cure":  {"switch_out_eff": ev.EFF_NATURAL_CURE},
    # Regenerator is Gen 5 only; omitted for Gen 4

    # ==== Residual (end-of-turn state, Tier 2) ====
    "Speed Boost":   {"residual_eff": ev.EFF_SPEED_BOOST},
    "Poison Heal":   {"residual_eff": ev.EFF_POISON_HEAL},
    "Bad Dreams":    {"residual_eff": ev.EFF_BAD_DREAMS},
    "Rain Dish":     {"residual_eff": ev.EFF_RAIN_DISH},
    "Ice Body":      {"residual_eff": ev.EFF_ICE_BODY},
    "Shed Skin":     {"residual_eff": ev.EFF_SHED_SKIN},
    # Dry Skin residual is set above with its TryHit entry

    # ==== Passive abilities (handled via ID checks in specific engine files) ====
    # Adaptability — ADAPTABILITY_ID in hit_pipeline.py (STAB 2.0×)
    # Arena Trap, Magnet Pull, Shadow Tag — trapping in action_mask.py
    # Battle Armor, Shell Armor — BATTLE_ARMOR_ID/SHELL_ARMOR_ID in damage.py (prevent crits)
    # Clear Body, White Smoke — CLEAR_BODY_ID/WHITE_SMOKE_ID in moves.py (prevent foe stat drops)
    # Compound Eyes — COMPOUND_EYES_ID in hit_pipeline.py (×1.3 accuracy)
    # Early Bird — EARLY_BIRD_ID in conditions.py (halve sleep turns)
    # Filter, Solid Rock — FILTER_ID/SOLID_ROCK_ID in damage.py (×0.75 SE damage)
    # Hyper Cutter — HYPER_CUTTER_ID in moves.py (prevent ATK drops from foes)
    # Keen Eye — KEEN_EYE_ID in moves.py (prevent accuracy drops from foes)
    # Mold Breaker — MOLD_BREAKER_ID in events.py (bypass defender ability)
    # No Guard — NO_GUARD_ID in hit_pipeline.py (always hit)
    # Pressure — PRESSURE_ID in actions.py (extra PP deduction)
    # Rock Head — ROCK_HEAD_ID in hit_pipeline.py (no recoil)
    # Scrappy — SCRAPPY_ID in hit_pipeline.py (Normal/Fighting hit Ghost)
    # Serene Grace — SERENE_GRACE_ID in hit_pipeline.py (double secondary chance)
    # Simple — SIMPLE_ID in moves.py (double stat changes)
    # Skill Link — SKILL_LINK_ID in hit_pipeline.py (multi-hit always 5)
    # Sniper — SNIPER_ID in damage.py (×3 crit multiplier)
    # Steadfast — STEADFAST_ID in conditions.py (flinch → +1 SPE)
    # Sturdy — STURDY_ID in hit_pipeline.py (survive OHKO at full HP)
    # Super Luck — SUPER_LUCK_ID in damage.py (+1 crit stage)
    # Synchronize — SYNCHRONIZE_ID in hit_pipeline.py (reflect status)
    # Tangled Feet — TANGLED_FEET_ID in hit_pipeline.py (evasion when confused)
    # Thick Fat — THICK_FAT_ID in damage.py (halve Fire/Ice attack)
    # Unaware — UNAWARE_ID in damage.py (ignore foe boosts)
    # Damp — too niche for Gen 4 randbats (prevent Self-Destruct/Explosion)
    # Gluttony — no pinch berries in Gen 4 randbats
}


# ---------------------------------------------------------------------------
# populate_ability_tables
# ---------------------------------------------------------------------------

def populate_ability_tables(ability_name_to_id: dict, tables=None) -> None:
    """
    Install ability parameters into events.py relay arrays and handler lists.

    Must be called BEFORE any jax.jit compilation of run_event_* functions.

    Args:
        ability_name_to_id: dict mapping ability display name → integer ID
        tables: optional Tables namedtuple (stored as ev._TABLES_REF for move/type data)
    """
    global GUTS_ID, FLASH_FIRE_ID, ADAPTABILITY_ID, WONDER_GUARD_ID, MOLD_BREAKER_ID
    global LEVITATE_ID, NO_GUARD_ID, COMPOUND_EYES_ID, HUSTLE_ID
    global SAND_VEIL_ID, SNOW_CLOAK_ID
    global ARENA_TRAP_ID, SHADOW_TAG_ID, MAGNET_PULL_ID
    global BATTLE_ARMOR_ID, SHELL_ARMOR_ID, SNIPER_ID, SERENE_GRACE_ID
    global ROCK_HEAD_ID, STURDY_ID, SKILL_LINK_ID, SCRAPPY_ID
    global SUPER_LUCK_ID, THICK_FAT_ID
    global CLEAR_BODY_ID, WHITE_SMOKE_ID, HYPER_CUTTER_ID, KEEN_EYE_ID
    global SIMPLE_ID, UNAWARE_ID, SYNCHRONIZE_ID, EARLY_BIRD_ID
    global FILTER_ID, SOLID_ROCK_ID, STEADFAST_ID, TANGLED_FEET_ID, PRESSURE_ID
    global OWN_TEMPO_ID, OBLIVIOUS_ID

    if tables is not None:
        ev._TABLES_REF = tables

    def _get(name):
        return ability_name_to_id.get(name, -1)

    GUTS_ID           = _get("Guts")
    FLASH_FIRE_ID     = _get("Flash Fire")
    ADAPTABILITY_ID   = _get("Adaptability")
    WONDER_GUARD_ID   = _get("Wonder Guard")
    MOLD_BREAKER_ID   = _get("Mold Breaker")
    LEVITATE_ID       = _get("Levitate")
    NO_GUARD_ID       = _get("No Guard")
    COMPOUND_EYES_ID  = _get("Compound Eyes")
    if COMPOUND_EYES_ID < 0:
        COMPOUND_EYES_ID = _get("Compoundeyes")
    HUSTLE_ID         = _get("Hustle")
    SAND_VEIL_ID      = _get("Sand Veil")
    SNOW_CLOAK_ID     = _get("Snow Cloak")
    ARENA_TRAP_ID     = _get("Arena Trap")
    SHADOW_TAG_ID     = _get("Shadow Tag")
    MAGNET_PULL_ID    = _get("Magnet Pull")
    BATTLE_ARMOR_ID   = _get("Battle Armor")
    SHELL_ARMOR_ID    = _get("Shell Armor")
    SNIPER_ID         = _get("Sniper")
    SERENE_GRACE_ID   = _get("Serene Grace")
    ROCK_HEAD_ID      = _get("Rock Head")
    STURDY_ID         = _get("Sturdy")
    SKILL_LINK_ID     = _get("Skill Link")
    SCRAPPY_ID        = _get("Scrappy")
    SUPER_LUCK_ID     = _get("Super Luck")
    THICK_FAT_ID      = _get("Thick Fat")
    CLEAR_BODY_ID     = _get("Clear Body")
    WHITE_SMOKE_ID    = _get("White Smoke")
    HYPER_CUTTER_ID   = _get("Hyper Cutter")
    KEEN_EYE_ID       = _get("Keen Eye")
    SIMPLE_ID         = _get("Simple")
    UNAWARE_ID        = _get("Unaware")
    SYNCHRONIZE_ID    = _get("Synchronize")
    EARLY_BIRD_ID     = _get("Early Bird")
    FILTER_ID         = _get("Filter")
    SOLID_ROCK_ID     = _get("Solid Rock")
    STEADFAST_ID      = _get("Steadfast")
    TANGLED_FEET_ID   = _get("Tangled Feet")
    PRESSURE_ID       = _get("Pressure")
    OWN_TEMPO_ID      = _get("Own Tempo")
    OBLIVIOUS_ID      = _get("Oblivious")

    # --- Install state-mutating handlers into the small handler lists ---
    # SwitchIn
    ev._SWITCH_IN_HANDLERS[ev.EFF_INTIMIDATE]   = _intimidate_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_DRIZZLE]      = _drizzle_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_DROUGHT]      = _drought_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_SAND_STREAM]  = _sand_stream_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_SNOW_WARNING] = _snow_warning_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_DOWNLOAD]     = _download_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_TRACE]        = _trace_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_FORECAST]     = _forecast_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_FRISK]        = _frisk_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_ANTICIPATION] = _anticipation_switch_in
    ev._SWITCH_IN_HANDLERS[ev.EFF_FOREWARN]     = _forewarn_switch_in

    # SwitchOut
    ev._SWITCH_OUT_HANDLERS[ev.EFF_NATURAL_CURE] = _natural_cure_switch_out
    # Regenerator is Gen 5 only — not registered for Gen 4

    # Ability Residual
    ev._AB_RESIDUAL_HANDLERS[ev.EFF_SPEED_BOOST]  = _speed_boost_residual
    ev._AB_RESIDUAL_HANDLERS[ev.EFF_POISON_HEAL]  = _poison_heal_residual
    ev._AB_RESIDUAL_HANDLERS[ev.EFF_BAD_DREAMS]   = _bad_dreams_residual
    ev._AB_RESIDUAL_HANDLERS[ev.EFF_DRY_SKIN_RES] = _dry_skin_residual
    ev._AB_RESIDUAL_HANDLERS[ev.EFF_RAIN_DISH]    = _rain_dish_residual
    ev._AB_RESIDUAL_HANDLERS[ev.EFF_ICE_BODY]     = _ice_body_residual
    ev._AB_RESIDUAL_HANDLERS[ev.EFF_SHED_SKIN]    = _shed_skin_residual

    # TryHit State
    ev._TRYHIT_STATE_HANDLERS[ev.EFF_WATER_ABSORB]    = _water_absorb_try_hit_state
    ev._TRYHIT_STATE_HANDLERS[ev.EFF_VOLT_ABSORB]     = _volt_absorb_try_hit_state
    ev._TRYHIT_STATE_HANDLERS[ev.EFF_MOTOR_DRIVE]     = _motor_drive_try_hit_state
    ev._TRYHIT_STATE_HANDLERS[ev.EFF_SAP_SIPPER]      = _sap_sipper_try_hit_state
    ev._TRYHIT_STATE_HANDLERS[ev.EFF_STORM_DRAIN]     = _storm_drain_try_hit_state
    ev._TRYHIT_STATE_HANDLERS[ev.EFF_LIGHTNING_ROD]   = _lightning_rod_try_hit_state
    ev._TRYHIT_STATE_HANDLERS[ev.EFF_DRY_SKIN_ABSORB] = _dry_skin_absorb_try_hit_state
    ev._TRYHIT_STATE_HANDLERS[ev.EFF_FLASH_FIRE]      = _flash_fire_try_hit_state

    # Contact punishment
    ev._CONTACT_HANDLERS[ev.EFF_ROUGH_SKIN]      = _rough_skin_contact
    ev._CONTACT_HANDLERS[ev.EFF_STATIC_CONTACT]  = _static_contact
    ev._CONTACT_HANDLERS[ev.EFF_POISON_POINT]    = _poison_point_contact
    ev._CONTACT_HANDLERS[ev.EFF_FLAME_BODY]      = _flame_body_contact
    ev._CONTACT_HANDLERS[ev.EFF_EFFECT_SPORE]    = _effect_spore_contact
    ev._CONTACT_HANDLERS[ev.EFF_CUTE_CHARM]      = _cute_charm_contact

    # --- Update per-ability relay arrays for each known ability ---
    for ability_name, params in ABILITY_HANDLERS.items():
        ability_id = ability_name_to_id.get(ability_name, -1)
        if ability_id < 0 or ability_id >= ev._MAX_ABILITIES:
            continue

        if "modify_atk_mult" in params:
            ev._AB_ATK_MULT = ev._AB_ATK_MULT.at[ability_id].set(
                jnp.float32(params["modify_atk_mult"]))

        if "modify_atk_cond" in params:
            ev._AB_ATK_COND = ev._AB_ATK_COND.at[ability_id].set(
                jnp.int8(params["modify_atk_cond"]))

        if "modify_def_cond" in params:
            ev._AB_DEF_COND = ev._AB_DEF_COND.at[ability_id].set(
                jnp.int8(params["modify_def_cond"]))

        if "modify_spd_cond" in params:
            ev._AB_SPD_COND = ev._AB_SPD_COND.at[ability_id].set(
                jnp.int8(params["modify_spd_cond"]))

        if "modify_spa_mult" in params:
            ev._AB_SPA_MULT = ev._AB_SPA_MULT.at[ability_id].set(
                jnp.float32(params["modify_spa_mult"]))

        if "modify_spa_cond" in params:
            ev._AB_SPA_COND = ev._AB_SPA_COND.at[ability_id].set(
                jnp.int8(params["modify_spa_cond"]))

        if "bp_cond" in params:
            ev._AB_BP_COND = ev._AB_BP_COND.at[ability_id].set(
                jnp.int8(params["bp_cond"]))

        if "bp_type" in params:
            ev._AB_BP_TYPE = ev._AB_BP_TYPE.at[ability_id].set(
                jnp.int8(params["bp_type"]))

        if "modify_dmg_mult" in params:
            ev._AB_DMG_MULT = ev._AB_DMG_MULT.at[ability_id].set(
                jnp.float32(params["modify_dmg_mult"]))

        if "modify_dmg_cond" in params:
            ev._AB_DMG_COND = ev._AB_DMG_COND.at[ability_id].set(
                jnp.int8(params["modify_dmg_cond"]))

        if "speed_cond" in params:
            ev._AB_SPEED_COND = ev._AB_SPEED_COND.at[ability_id].set(
                jnp.int8(params["speed_cond"]))

        if "status_immune" in params:
            ev._AB_STATUS_IMMUNE = ev._AB_STATUS_IMMUNE.at[ability_id].set(
                jnp.int8(params["status_immune"]))

        if "contact_eff" in params:
            ev._AB_CONTACT_EFF = ev._AB_CONTACT_EFF.at[ability_id].set(
                jnp.int8(params["contact_eff"]))

        if "tryhit_immune" in params:
            ev._AB_TRYHIT_IMMUNE = ev._AB_TRYHIT_IMMUNE.at[ability_id].set(
                jnp.int8(params["tryhit_immune"]))

        if "tryhit_state_eff" in params:
            ev._AB_TRYHIT_STATE_EFF = ev._AB_TRYHIT_STATE_EFF.at[ability_id].set(
                jnp.int8(params["tryhit_state_eff"]))

        if "switch_in_eff" in params:
            ev._AB_SWITCH_IN_EFF = ev._AB_SWITCH_IN_EFF.at[ability_id].set(
                jnp.int8(params["switch_in_eff"]))

        if "switch_out_eff" in params:
            ev._AB_SWITCH_OUT_EFF = ev._AB_SWITCH_OUT_EFF.at[ability_id].set(
                jnp.int8(params["switch_out_eff"]))

        if "residual_eff" in params:
            ev._AB_RESIDUAL_EFF = ev._AB_RESIDUAL_EFF.at[ability_id].set(
                jnp.int8(params["residual_eff"]))
