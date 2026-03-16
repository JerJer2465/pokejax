"""
Weather and terrain management.

Handles:
  - Weather residual damage (sandstorm, hail)
  - Weather/terrain turn ticking and expiry
  - Field pseudoweather ticking (trick room, gravity, etc.)

All functions are pure / branchless.
"""

import jax.numpy as jnp

from pokejax.types import (
    BattleState, FieldState,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    TERRAIN_NONE,
    TYPE_ROCK, TYPE_GROUND, TYPE_STEEL, TYPE_ICE, TYPE_FIRE, TYPE_WATER,
)
from pokejax.core.damage import fraction_of_max_hp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_type(state: BattleState, side: int, type_id: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    types = state.sides_team_types[side, idx]
    return (types[0] == jnp.int8(type_id)) | (types[1] == jnp.int8(type_id))


def _is_grounded(state: BattleState, side: int) -> jnp.ndarray:
    """
    Returns True if the active Pokemon is grounded (affected by terrain/spikes).
    Checks Flying-type and Levitate ability.
    """
    from pokejax.types import TYPE_FLYING
    from pokejax.mechanics.abilities import LEVITATE_ID
    idx = state.sides_active_idx[side]
    types = state.sides_team_types[side, idx]
    is_flying = (types[0] == jnp.int8(TYPE_FLYING)) | (types[1] == jnp.int8(TYPE_FLYING))
    ability_id = state.sides_team_ability_id[side, idx].astype(jnp.int32)
    has_levitate = (LEVITATE_ID >= 0) & (ability_id == jnp.int32(LEVITATE_ID))
    return ~is_flying & ~has_levitate


# ---------------------------------------------------------------------------
# Weather residual damage
# ---------------------------------------------------------------------------

def apply_weather_residual(state: BattleState, side: int) -> BattleState:
    """
    Apply sandstorm and hail damage at end of turn.

    Sandstorm: 1/16 max HP; exempt: Rock, Ground, Steel types.
    Hail (Gen 4-8): 1/16 max HP; exempt: Ice type.
    """
    weather = state.field.weather
    idx = state.sides_active_idx[side]

    is_sand = weather == jnp.int8(WEATHER_SAND)
    is_hail = weather == jnp.int8(WEATHER_HAIL)

    type0 = state.sides_team_types[side, idx][0].astype(jnp.int32)
    type1 = state.sides_team_types[side, idx][1].astype(jnp.int32)

    # Sand immunity: Rock, Ground, Steel
    sand_immune = (
        (type0 == jnp.int32(TYPE_ROCK))   | (type1 == jnp.int32(TYPE_ROCK))  |
        (type0 == jnp.int32(TYPE_GROUND)) | (type1 == jnp.int32(TYPE_GROUND))|
        (type0 == jnp.int32(TYPE_STEEL))  | (type1 == jnp.int32(TYPE_STEEL))
    )
    # Hail immunity: Ice
    hail_immune = (type0 == jnp.int32(TYPE_ICE)) | (type1 == jnp.int32(TYPE_ICE))

    takes_sand = is_sand & ~sand_immune
    takes_hail = is_hail & ~hail_immune

    dmg_16 = fraction_of_max_hp(state, side, idx, 1, 16)

    total_dmg = jnp.where(takes_sand | takes_hail, dmg_16, jnp.int32(0))

    new_hp = jnp.maximum(
        jnp.int16(0),
        state.sides_team_hp[side, idx] - total_dmg.astype(jnp.int16)
    )
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr)


# ---------------------------------------------------------------------------
# Weather / terrain / pseudoweather ticking
# ---------------------------------------------------------------------------

def tick_weather(state: BattleState) -> BattleState:
    """Decrement weather turn counter and clear on expiry."""
    turns = state.field.weather_turns
    # 0 means either: no weather, or permanent weather (e.g. from Drizzle w/ no timer)
    # Positive turns: decrement
    active = (state.field.weather != jnp.int8(WEATHER_NONE)) & (turns > jnp.int8(0))
    new_turns = jnp.where(active, turns - jnp.int8(1), turns)
    expired = active & (new_turns == jnp.int8(0))
    new_weather = jnp.where(expired, jnp.int8(WEATHER_NONE), state.field.weather)

    new_field = state.field._replace(weather=new_weather, weather_turns=new_turns)
    return state._replace(field=new_field)


def tick_terrain(state: BattleState) -> BattleState:
    """Decrement terrain turn counter and clear on expiry."""
    turns = state.field.terrain_turns
    active = (state.field.terrain != jnp.int8(TERRAIN_NONE)) & (turns > jnp.int8(0))
    new_turns = jnp.where(active, turns - jnp.int8(1), turns)
    expired = active & (new_turns == jnp.int8(0))
    new_terrain = jnp.where(expired, jnp.int8(TERRAIN_NONE), state.field.terrain)

    new_field = state.field._replace(terrain=new_terrain, terrain_turns=new_turns)
    return state._replace(field=new_field)


def tick_pseudoweather(state: BattleState) -> BattleState:
    """Tick Trick Room, Gravity, Magic Room, Wonder Room."""
    f = state.field

    def _tick(val: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(jnp.int8(0), val - jnp.int8(1))

    new_field = f._replace(
        trick_room=_tick(f.trick_room),
        gravity=_tick(f.gravity),
        magic_room=_tick(f.magic_room),
        wonder_room=_tick(f.wonder_room),
    )
    return state._replace(field=new_field)


def apply_field_residual(state: BattleState) -> BattleState:
    """Apply all field-level residual effects (weather damage for both sides)."""
    state = apply_weather_residual(state, 0)
    state = apply_weather_residual(state, 1)
    return state


def tick_all_field_timers(state: BattleState) -> BattleState:
    """Tick weather, terrain, and pseudoweather timers. Called at end of each turn."""
    state = tick_weather(state)
    state = tick_terrain(state)
    state = tick_pseudoweather(state)
    return state


# ---------------------------------------------------------------------------
# Weather / terrain setters (used by moves like Rain Dance, Sunny Day, etc.)
# ---------------------------------------------------------------------------

def set_weather(state: BattleState, weather: jnp.ndarray,
                turns: int = 5) -> BattleState:
    """Set weather with a turn duration. 0 = permanent."""
    new_field = state.field._replace(
        weather=weather,
        weather_turns=jnp.int8(turns),
        weather_max_turns=jnp.int8(turns),
    )
    return state._replace(field=new_field)


def set_terrain(state: BattleState, terrain: jnp.ndarray,
                turns: int = 5) -> BattleState:
    new_field = state.field._replace(
        terrain=terrain,
        terrain_turns=jnp.int8(turns),
    )
    return state._replace(field=new_field)


def set_trick_room(state: BattleState, turns: int = 5) -> BattleState:
    """Toggle Trick Room. If already active, end it; otherwise start 5 turns."""
    currently_active = state.field.trick_room > jnp.int8(0)
    new_turns = jnp.where(currently_active, jnp.int8(0), jnp.int8(turns))
    new_field = state.field._replace(trick_room=new_turns)
    return state._replace(field=new_field)


# ---------------------------------------------------------------------------
# Speed modifier from weather (used in priority/speed calc)
# ---------------------------------------------------------------------------

def weather_speed_modifier(state: BattleState, side: int,
                            ability_id: jnp.ndarray,
                            ABILITY_SWIFT_SWIM: int = -1,
                            ABILITY_CHLOROPHYLL: int = -2,
                            ABILITY_SAND_RUSH: int = -3,
                            ABILITY_SLUSH_RUSH: int = -4) -> jnp.ndarray:
    """
    Return speed multiplier from weather-boosting abilities (float32).
    Placeholder IDs will be replaced once ability IDs are extracted from Showdown data.
    """
    weather = state.field.weather

    rain_boost  = (ability_id == jnp.int16(ABILITY_SWIFT_SWIM))  & (weather == jnp.int8(WEATHER_RAIN))
    sun_boost   = (ability_id == jnp.int16(ABILITY_CHLOROPHYLL)) & (weather == jnp.int8(WEATHER_SUN))
    sand_boost  = (ability_id == jnp.int16(ABILITY_SAND_RUSH))   & (weather == jnp.int8(WEATHER_SAND))
    hail_boost  = (ability_id == jnp.int16(ABILITY_SLUSH_RUSH))  & (weather == jnp.int8(WEATHER_HAIL))

    any_boost = rain_boost | sun_boost | sand_boost | hail_boost
    return jnp.where(any_boost, jnp.float32(2.0), jnp.float32(1.0))


# ---------------------------------------------------------------------------
# Grassy terrain healing (residual)
# ---------------------------------------------------------------------------

def apply_terrain_residual(state: BattleState, side: int) -> BattleState:
    """Grassy Terrain: heal grounded Pokemon by 1/16 max HP per turn."""
    from pokejax.types import TERRAIN_GRASSY
    idx = state.sides_active_idx[side]
    is_grassy = state.field.terrain == jnp.int8(TERRAIN_GRASSY)
    grounded = _is_grounded(state, side)

    heal = fraction_of_max_hp(state, side, idx, 1, 16)
    heal = jnp.where(is_grassy & grounded, heal, jnp.int32(0))

    cur_hp = state.sides_team_hp[side, idx]
    max_hp = state.sides_team_max_hp[side, idx]
    new_hp = jnp.minimum(max_hp, cur_hp + heal.astype(jnp.int16))
    new_hp_arr = state.sides_team_hp.at[side, idx].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr)
