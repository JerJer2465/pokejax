"""
BattleState constructors and helper accessors.

All functions are pure: they take state + arguments and return new state.
Accessors return slices of the batched arrays in BattleState.
"""

import jax.numpy as jnp
import numpy as np

from pokejax.types import (
    BattleState, FieldState, RevealState,
    MAX_TEAM_SIZE, MAX_MOVES, MAX_SIDE_CONDS, MAX_VOLATILES, N_BOOSTS,
    STATUS_NONE, WEATHER_NONE, TERRAIN_NONE,
)


# ---------------------------------------------------------------------------
# Helpers to read one Pokemon's data from the batched arrays
# ---------------------------------------------------------------------------

def get_hp(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_hp[side, slot]

def get_max_hp(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_max_hp[side, slot]

def get_status(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_status[side, slot]

def get_boosts(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_boosts[side, slot]  # shape (7,)

def get_boost(state: BattleState, side: int, slot: int, boost_idx: int) -> jnp.ndarray:
    return state.sides_team_boosts[side, slot, boost_idx]

def get_types(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_types[side, slot]  # shape (2,)

def get_ability(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_ability_id[side, slot]

def get_item(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_item_id[side, slot]

def get_base_stats(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_base_stats[side, slot]  # shape (6,)

def get_move_ids(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_move_ids[side, slot]  # shape (4,)

def get_move_pp(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_move_pp[side, slot]  # shape (4,)

def get_move_disabled(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_move_disabled[side, slot]  # shape (4,)

def get_volatiles(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_volatiles[side, slot]  # uint32 scalar

def get_volatile_data(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_volatile_data[side, slot]  # shape (MAX_VOLATILES,)

def get_active_idx(state: BattleState, side: int) -> jnp.ndarray:
    return state.sides_active_idx[side]

def get_active_hp(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_hp[side, idx]

def get_active_status(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_status[side, idx]

def get_active_ability(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_ability_id[side, idx]

def get_active_item(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_item_id[side, idx]

def get_active_boosts(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_boosts[side, idx]

def get_active_types(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_types[side, idx]

def get_active_base_stats(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_base_stats[side, idx]

def get_active_level(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_level[side, idx]

def get_active_weight_hg(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_weight_hg[side, idx]

def get_active_volatiles(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_volatiles[side, idx]

def get_active_volatile_data(state: BattleState, side: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return state.sides_team_volatile_data[side, idx]

def get_side_condition(state: BattleState, side: int, cond_idx: int) -> jnp.ndarray:
    return state.sides_side_conditions[side, cond_idx]

def has_volatile(state: BattleState, side: int, slot: int, vol_bit: int) -> jnp.ndarray:
    """Check if a volatile status bit is set (branchless)."""
    mask = jnp.uint32(1 << vol_bit)
    return (state.sides_team_volatiles[side, slot] & mask) != jnp.uint32(0)

def has_active_volatile(state: BattleState, side: int, vol_bit: int) -> jnp.ndarray:
    idx = state.sides_active_idx[side]
    return has_volatile(state, side, idx, vol_bit)


# ---------------------------------------------------------------------------
# Mutators (return new BattleState via _replace on NamedTuples)
# ---------------------------------------------------------------------------
# NamedTuple._replace is shallow-copy; JAX arrays are immutable so this is fine.
# We use index updates: new_arr = arr.at[idx].set(val)

def set_hp(state: BattleState, side: int, slot: int, hp: jnp.ndarray) -> BattleState:
    new_hp = state.sides_team_hp.at[side, slot].set(hp)
    return state._replace(sides_team_hp=new_hp)

def set_status(state: BattleState, side: int, slot: int,
               status: jnp.ndarray, turns: jnp.ndarray = jnp.int8(0)) -> BattleState:
    new_status = state.sides_team_status.at[side, slot].set(status)
    new_turns  = state.sides_team_status_turns.at[side, slot].set(turns)
    return state._replace(sides_team_status=new_status, sides_team_status_turns=new_turns)

def set_boost(state: BattleState, side: int, slot: int,
              boost_idx: int, value: jnp.ndarray) -> BattleState:
    clamped = jnp.clip(value, jnp.int8(-6), jnp.int8(6)).astype(jnp.int8)
    new_boosts = state.sides_team_boosts.at[side, slot, boost_idx].set(clamped)
    return state._replace(sides_team_boosts=new_boosts)

def add_boost(state: BattleState, side: int, slot: int,
              boost_idx: int, delta: jnp.ndarray) -> BattleState:
    current = state.sides_team_boosts[side, slot, boost_idx]
    return set_boost(state, side, slot, boost_idx, current + delta)

def set_item(state: BattleState, side: int, slot: int, item_id: jnp.ndarray) -> BattleState:
    new_items = state.sides_team_item_id.at[side, slot].set(item_id)
    return state._replace(sides_team_item_id=new_items)

def consume_item(state: BattleState, side: int, slot: int) -> BattleState:
    """Remove held item (set to 0)."""
    return set_item(state, side, slot, jnp.int16(0))

def set_pp(state: BattleState, side: int, slot: int,
           move_slot: int, pp: jnp.ndarray) -> BattleState:
    new_pp = state.sides_team_move_pp.at[side, slot, move_slot].set(pp)
    return state._replace(sides_team_move_pp=new_pp)

def deduct_pp(state: BattleState, side: int, slot: int, move_slot: int,
              amount: jnp.ndarray = jnp.int8(1)) -> BattleState:
    current = state.sides_team_move_pp[side, slot, move_slot]
    new_pp = jnp.maximum(jnp.int8(0), current - amount.astype(jnp.int8))
    return set_pp(state, side, slot, move_slot, new_pp)

def set_volatile(state: BattleState, side: int, slot: int, vol_bit: int,
                 active: bool | jnp.ndarray) -> BattleState:
    mask = jnp.uint32(1 << vol_bit)
    old = state.sides_team_volatiles[side, slot]
    new_v = jnp.where(active, old | mask, old & ~mask)
    new_vols = state.sides_team_volatiles.at[side, slot].set(new_v)
    return state._replace(sides_team_volatiles=new_vols)

def set_volatile_counter(state: BattleState, side: int, slot: int,
                         vol_bit: int, value: jnp.ndarray) -> BattleState:
    new_data = state.sides_team_volatile_data.at[side, slot, vol_bit].set(
        value.astype(jnp.int8)
    )
    return state._replace(sides_team_volatile_data=new_data)

def clear_volatiles(state: BattleState, side: int, slot: int) -> BattleState:
    new_vols = state.sides_team_volatiles.at[side, slot].set(jnp.uint32(0))
    new_data = state.sides_team_volatile_data.at[side, slot].set(
        jnp.zeros(MAX_VOLATILES, dtype=jnp.int8)
    )
    return state._replace(
        sides_team_volatiles=new_vols,
        sides_team_volatile_data=new_data,
    )

def set_fainted(state: BattleState, side: int, slot: int) -> BattleState:
    new_fainted = state.sides_team_fainted.at[side, slot].set(True)
    new_left = state.sides_pokemon_left.at[side].set(
        state.sides_pokemon_left[side] - jnp.int8(1)
    )
    return state._replace(sides_team_fainted=new_fainted, sides_pokemon_left=new_left)

def set_active(state: BattleState, side: int, slot: int) -> BattleState:
    # Clear old active flag, set new one
    old_idx = state.sides_active_idx[side]
    new_is_active = state.sides_team_is_active.at[side, old_idx].set(False)
    new_is_active = new_is_active.at[side, slot].set(True)
    new_active_idx = state.sides_active_idx.at[side].set(jnp.int8(slot))
    return state._replace(
        sides_team_is_active=new_is_active,
        sides_active_idx=new_active_idx,
    )

def set_weather(state: BattleState, weather: jnp.ndarray, turns: jnp.ndarray) -> BattleState:
    new_field = state.field._replace(
        weather=weather,
        weather_turns=turns,
        weather_max_turns=turns,
    )
    return state._replace(field=new_field)

def set_terrain(state: BattleState, terrain: jnp.ndarray, turns: jnp.ndarray) -> BattleState:
    new_field = state.field._replace(terrain=terrain, terrain_turns=turns)
    return state._replace(field=new_field)

def set_trick_room(state: BattleState, turns: jnp.ndarray) -> BattleState:
    new_field = state.field._replace(trick_room=turns)
    return state._replace(field=new_field)

def set_side_condition(state: BattleState, side: int, cond_idx: int,
                       value: jnp.ndarray) -> BattleState:
    new_sc = state.sides_side_conditions.at[side, cond_idx].set(value.astype(jnp.int8))
    return state._replace(sides_side_conditions=new_sc)

def add_side_condition_layer(state: BattleState, side: int, cond_idx: int,
                              max_layers: int) -> BattleState:
    current = state.sides_side_conditions[side, cond_idx]
    new_val = jnp.minimum(current + jnp.int8(1), jnp.int8(max_layers))
    return set_side_condition(state, side, cond_idx, new_val)

def reset_boosts(state: BattleState, side: int, slot: int) -> BattleState:
    new_boosts = state.sides_team_boosts.at[side, slot].set(
        jnp.zeros(N_BOOSTS, dtype=jnp.int8)
    )
    return state._replace(sides_team_boosts=new_boosts)

def set_last_move(state: BattleState, side: int, slot: int,
                  move_id: jnp.ndarray) -> BattleState:
    new_lm = state.sides_team_last_move_id.at[side, slot].set(move_id)
    return state._replace(sides_team_last_move_id=new_lm)


# ---------------------------------------------------------------------------
# Battle state constructor
# ---------------------------------------------------------------------------

def make_reveal_state(state: BattleState) -> RevealState:
    """Create an initial RevealState from a BattleState.

    Semantics of indices:
      revealed_moves[p, s, m]   True  ↔  player p knows that the *opponent's*
                                          team slot s, move slot m has been used.
      revealed_pokemon[p, s]    True  ↔  player p has seen the opponent's slot s.
      revealed_ability[p, s]    True  ↔  player p has seen the opponent's slot s ability.
      revealed_item[p, s]       True  ↔  player p has seen the opponent's slot s item.

    At battle start, each player knows the opponent's lead (slot 0).
    No moves, abilities, or items are revealed yet.
    """
    # Both players see the opponent's lead Pokemon (slot 0 is always lead at start)
    revealed_pokemon = jnp.zeros((2, 6), dtype=jnp.bool_)
    p0_lead = state.sides_active_idx[0]  # typically 0
    p1_lead = state.sides_active_idx[1]  # typically 0
    # P0's lead is visible to P1 (revealed_pokemon[1, p0_lead])
    # P1's lead is visible to P0 (revealed_pokemon[0, p1_lead])
    lead0_mask = (jnp.arange(6) == p0_lead.astype(jnp.int32))
    lead1_mask = (jnp.arange(6) == p1_lead.astype(jnp.int32))
    revealed_pokemon = revealed_pokemon.at[1].set(lead0_mask)
    revealed_pokemon = revealed_pokemon.at[0].set(lead1_mask)

    return RevealState(
        revealed_moves=jnp.zeros((2, 6, 4), dtype=jnp.bool_),
        revealed_pokemon=revealed_pokemon,
        revealed_ability=jnp.zeros((2, 6), dtype=jnp.bool_),
        revealed_item=jnp.zeros((2, 6), dtype=jnp.bool_),
    )


def make_field() -> FieldState:
    return FieldState(
        weather=jnp.int8(0),
        weather_turns=jnp.int8(0),
        weather_max_turns=jnp.int8(0),
        terrain=jnp.int8(0),
        terrain_turns=jnp.int8(0),
        trick_room=jnp.int8(0),
        gravity=jnp.int8(0),
        magic_room=jnp.int8(0),
        wonder_room=jnp.int8(0),
    )


def make_battle_state(
    *,
    # Teams: numpy/python arrays for both sides, shape: (6, ...) per field
    # Each team is a list of 6 pokemon descriptors (dicts or arrays).
    # team_p{i}_{field} shape (6,) or (6, sub_dim)
    p1_species:     np.ndarray,   # int16[6]
    p1_abilities:   np.ndarray,   # int16[6]
    p1_items:       np.ndarray,   # int16[6]
    p1_types:       np.ndarray,   # int8[6, 2]
    p1_base_stats:  np.ndarray,   # int16[6, 6]
    p1_max_hp:      np.ndarray,   # int16[6]
    p1_move_ids:    np.ndarray,   # int16[6, 4]
    p1_move_pp:     np.ndarray,   # int8[6, 4]
    p1_move_max_pp: np.ndarray,   # int8[6, 4]
    p1_levels:      np.ndarray,   # int8[6]
    p1_genders:     np.ndarray,   # int8[6]
    p1_natures:     np.ndarray,   # int8[6]
    p1_weights_hg:  np.ndarray,   # int16[6]
    # same for p2
    p2_species:     np.ndarray,
    p2_abilities:   np.ndarray,
    p2_items:       np.ndarray,
    p2_types:       np.ndarray,
    p2_base_stats:  np.ndarray,
    p2_max_hp:      np.ndarray,
    p2_move_ids:    np.ndarray,
    p2_move_pp:     np.ndarray,
    p2_move_max_pp: np.ndarray,
    p2_levels:      np.ndarray,
    p2_genders:     np.ndarray,
    p2_natures:     np.ndarray,
    p2_weights_hg:  np.ndarray,
    rng_key:        jnp.ndarray,
) -> BattleState:
    """Construct an initial BattleState from team descriptor arrays.

    All teams start at slot 0 active, full HP, no status or volatiles.
    """
    def _stack(a, b):
        return jnp.array(np.stack([a, b], axis=0))

    # 6×4 move arrays
    def _z6_4_i16():
        return jnp.full((2, 6, 4), -1, dtype=jnp.int16)
    def _z6_4_i8():
        return jnp.zeros((2, 6, 4), dtype=jnp.int8)
    def _z6_4_bool():
        return jnp.zeros((2, 6, 4), dtype=jnp.bool_)

    move_ids   = _stack(p1_move_ids,    p2_move_ids)
    move_pp    = _stack(p1_move_pp,     p2_move_pp)
    move_max_pp= _stack(p1_move_max_pp, p2_move_max_pp)

    # Build is_active: only slot 0 is active at start
    is_active = jnp.zeros((2, 6), dtype=jnp.bool_).at[:, 0].set(True)

    return BattleState(
        sides_team_species_id    = _stack(p1_species,   p2_species).astype(jnp.int16),
        sides_team_ability_id    = _stack(p1_abilities, p2_abilities).astype(jnp.int16),
        sides_team_item_id       = _stack(p1_items,     p2_items).astype(jnp.int16),
        sides_team_types         = _stack(p1_types,     p2_types).astype(jnp.int8),
        sides_team_base_stats    = _stack(p1_base_stats,p2_base_stats).astype(jnp.int16),
        sides_team_hp            = _stack(p1_max_hp,    p2_max_hp).astype(jnp.int16),
        sides_team_max_hp        = _stack(p1_max_hp,    p2_max_hp).astype(jnp.int16),
        sides_team_boosts        = jnp.zeros((2, 6, 7), dtype=jnp.int8),
        sides_team_move_ids      = move_ids.astype(jnp.int16),
        sides_team_move_pp       = move_pp.astype(jnp.int8),
        sides_team_move_max_pp   = move_max_pp.astype(jnp.int8),
        sides_team_move_disabled = jnp.zeros((2, 6, 4), dtype=jnp.bool_),
        sides_team_status        = jnp.zeros((2, 6), dtype=jnp.int8),
        sides_team_status_turns  = jnp.zeros((2, 6), dtype=jnp.int8),
        sides_team_sleep_turns   = jnp.zeros((2, 6), dtype=jnp.int8),
        sides_team_volatiles     = jnp.zeros((2, 6), dtype=jnp.uint32),
        sides_team_volatile_data = jnp.zeros((2, 6, MAX_VOLATILES), dtype=jnp.int8),
        sides_team_is_active     = is_active,
        sides_team_fainted       = jnp.zeros((2, 6), dtype=jnp.bool_),
        sides_team_last_move_id  = jnp.full((2, 6), -1, dtype=jnp.int16),
        sides_team_move_this_turn= jnp.zeros((2, 6), dtype=jnp.bool_),
        sides_team_times_attacked= jnp.zeros((2, 6), dtype=jnp.int8),
        sides_team_level         = _stack(p1_levels,    p2_levels).astype(jnp.int8),
        sides_team_gender        = _stack(p1_genders,   p2_genders).astype(jnp.int8),
        sides_team_nature_id     = _stack(p1_natures,   p2_natures).astype(jnp.int8),
        sides_team_weight_hg     = _stack(p1_weights_hg,p2_weights_hg).astype(jnp.int16),
        sides_team_base_species_id=_stack(p1_species,   p2_species).astype(jnp.int16),
        sides_active_idx         = jnp.zeros(2, dtype=jnp.int8),
        sides_pokemon_left       = jnp.full(2, 6, dtype=jnp.int8),
        sides_side_conditions    = jnp.zeros((2, MAX_SIDE_CONDS), dtype=jnp.int8),
        sides_dynamax_turns      = jnp.zeros(2, dtype=jnp.int8),
        field                    = make_field(),
        turn                     = jnp.int16(0),
        finished                 = jnp.bool_(False),
        winner                   = jnp.int8(-1),
        rng_key                  = rng_key,
    )
