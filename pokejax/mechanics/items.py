"""
Gen 4 item table population for PokeJAX.

Relay events (ModifyAtk, ModifySpa, ModifyDamage) are registered by writing scalar
parameters into events.py's JAX array lookup tables.

State-mutating residual events are registered via an effect-ID indirection layer
(events._ITEM_RESIDUAL_EFF[item_id] → small lax.switch handler list).
"""

import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX,
    TYPE_POISON, TYPE_FIRE, TYPE_WATER, TYPE_ELECTRIC, TYPE_GRASS,
    TYPE_ICE, TYPE_FIGHTING, TYPE_GROUND, TYPE_FLYING,
    TYPE_PSYCHIC, TYPE_BUG, TYPE_ROCK, TYPE_GHOST, TYPE_DRAGON,
    TYPE_DARK, TYPE_STEEL, TYPE_NORMAL,
)
from pokejax.mechanics import events as ev

# ---------------------------------------------------------------------------
# Module-level item ID constants (set by populate_item_tables)
# ---------------------------------------------------------------------------
_TABLES = None          # kept for API compat but no longer used by relay handlers

LIFE_ORB_ID     = -1
CHOICE_BAND_ID  = -1
CHOICE_SPECS_ID = -1
CHOICE_SCARF_ID = -1
FOCUS_SASH_ID   = -1


# ---------------------------------------------------------------------------
# State-mutating item residual handlers
# Installed at ev._ITEM_RESIDUAL_HANDLERS[EFF_*] by populate_item_tables().
# ---------------------------------------------------------------------------

def _leftovers_residual_state(state, key, side_i32, slot_i32):
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    heal   = jnp.maximum(jnp.int32(1), max_hp // 16)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr), key


def _black_sludge_residual_state(state, key, side_i32, slot_i32):
    types = state.sides_team_types[side_i32, slot_i32]
    is_poison = (types[0] == jnp.int8(TYPE_POISON)) | (types[1] == jnp.int8(TYPE_POISON))
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    heal   = jnp.maximum(jnp.int32(1), max_hp // 16)
    damage = jnp.maximum(jnp.int32(1), max_hp // 8)
    new_hp_heal = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_dmg  = jnp.maximum(jnp.int32(0), hp - damage).astype(jnp.int16)
    new_hp = jnp.where(is_poison, new_hp_heal, new_hp_dmg)
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr), key


def _sitrus_berry_residual_state(state, key, side_i32, slot_i32):
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    low    = (hp * jnp.int32(2)) < max_hp
    heal   = jnp.maximum(jnp.int32(1), max_hp // 4)
    new_hp = jnp.minimum(max_hp, hp + heal).astype(jnp.int16)
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(
        jnp.where(low, new_hp, state.sides_team_hp[side_i32, slot_i32])
    )
    return state._replace(sides_team_hp=new_hp_arr), key


def _lum_berry_residual_state(state, key, side_i32, slot_i32):
    status = state.sides_team_status[side_i32, slot_i32]
    has_status = status != jnp.int8(STATUS_NONE)
    new_status = jnp.where(has_status, jnp.int8(STATUS_NONE), status)
    new_status_arr = state.sides_team_status.at[side_i32, slot_i32].set(new_status)
    return state._replace(sides_team_status=new_status_arr), key


def _life_orb_residual_state(state, key, side_i32, slot_i32):
    hp     = state.sides_team_hp[side_i32, slot_i32].astype(jnp.int32)
    max_hp = state.sides_team_max_hp[side_i32, slot_i32].astype(jnp.int32)
    recoil = jnp.maximum(jnp.int32(1), max_hp // 10)
    new_hp = jnp.maximum(jnp.int32(0), hp - recoil).astype(jnp.int16)
    alive  = hp > jnp.int32(0)
    new_hp_arr = state.sides_team_hp.at[side_i32, slot_i32].set(
        jnp.where(alive, new_hp, state.sides_team_hp[side_i32, slot_i32])
    )
    return state._replace(sides_team_hp=new_hp_arr), key


def _flame_orb_residual_state(state, key, side_i32, slot_i32):
    """Flame Orb: inflict burn at end of turn if no status."""
    status = state.sides_team_status[side_i32, slot_i32]
    no_status = status == jnp.int8(STATUS_NONE)
    # Fire types are immune to burn
    types = state.sides_team_types[side_i32, slot_i32]
    is_fire = (types[0] == jnp.int8(TYPE_FIRE)) | (types[1] == jnp.int8(TYPE_FIRE))
    new_status = jnp.where(no_status & ~is_fire, jnp.int8(STATUS_BRN), status)
    new_status_arr = state.sides_team_status.at[side_i32, slot_i32].set(new_status)
    return state._replace(sides_team_status=new_status_arr), key


def _toxic_orb_residual_state(state, key, side_i32, slot_i32):
    """Toxic Orb: inflict toxic poison at end of turn if no status."""
    status = state.sides_team_status[side_i32, slot_i32]
    no_status = status == jnp.int8(STATUS_NONE)
    # Poison and Steel types are immune
    types = state.sides_team_types[side_i32, slot_i32]
    is_poison = (types[0] == jnp.int8(TYPE_POISON)) | (types[1] == jnp.int8(TYPE_POISON))
    is_steel = (types[0] == jnp.int8(TYPE_STEEL)) | (types[1] == jnp.int8(TYPE_STEEL))
    can_poison = ~is_poison & ~is_steel
    new_status = jnp.where(no_status & can_poison, jnp.int8(STATUS_TOX), status)
    new_status_arr = state.sides_team_status.at[side_i32, slot_i32].set(new_status)
    return state._replace(sides_team_status=new_status_arr), key


# ---------------------------------------------------------------------------
# Item parameter registry
#
# Keys map to events.py array updates in populate_item_tables():
#   modify_atk_mult  → ev._ITEM_ATK_MULT[id]
#   modify_spa_mult  → ev._ITEM_SPA_MULT[id]
#   modify_dmg_mult  → ev._ITEM_DMG_MULT[id]    (constant multiplier)
#   modify_dmg_cond  → ev._ITEM_DMG_COND[id]    (conditional code)
#   residual_eff     → ev._ITEM_RESIDUAL_EFF[id] (effect ID → handler list slot)
# ---------------------------------------------------------------------------

ITEM_HANDLERS = {
    # ==== Choice items ====
    "Choice Band":  {"modify_atk_mult": 1.5},
    "Choice Specs": {"modify_spa_mult": 1.5},
    "Choice Scarf": {"speed_mult": 1.5},

    # ==== Damage boosters ====
    "Life Orb":     {"modify_dmg_mult": 1.3, "residual_eff": ev.EFF_LIFE_ORB_RESIDUAL},
    "Expert Belt":  {"modify_dmg_cond": ev.ITEMCOND_DMG_EXPERT_BELT},
    "Muscle Band":  {"modify_dmg_cond": ev.ITEMCOND_DMG_MUSCLE_BAND},
    "Wise Glasses": {"modify_dmg_cond": ev.ITEMCOND_DMG_WISE_GLASSES},

    # ==== Type-boosting items (×1.2 to matching type moves) ====
    "Charcoal":       {"type_boost_type": TYPE_FIRE,     "type_boost_mult": 1.2},
    "Mystic Water":   {"type_boost_type": TYPE_WATER,    "type_boost_mult": 1.2},
    "Magnet":         {"type_boost_type": TYPE_ELECTRIC, "type_boost_mult": 1.2},
    "Miracle Seed":   {"type_boost_type": TYPE_GRASS,    "type_boost_mult": 1.2},
    "NeverMeltIce":   {"type_boost_type": TYPE_ICE,      "type_boost_mult": 1.2},
    "Black Belt":     {"type_boost_type": TYPE_FIGHTING, "type_boost_mult": 1.2},
    "Poison Barb":    {"type_boost_type": TYPE_POISON,   "type_boost_mult": 1.2},
    "Soft Sand":      {"type_boost_type": TYPE_GROUND,   "type_boost_mult": 1.2},
    "Sharp Beak":     {"type_boost_type": TYPE_FLYING,   "type_boost_mult": 1.2},
    "Twisted Spoon":  {"type_boost_type": TYPE_PSYCHIC,  "type_boost_mult": 1.2},
    "Silver Powder":  {"type_boost_type": TYPE_BUG,      "type_boost_mult": 1.2},
    "Hard Stone":     {"type_boost_type": TYPE_ROCK,     "type_boost_mult": 1.2},
    "Spell Tag":      {"type_boost_type": TYPE_GHOST,    "type_boost_mult": 1.2},
    "Dragon Fang":    {"type_boost_type": TYPE_DRAGON,   "type_boost_mult": 1.2},
    "BlackGlasses":   {"type_boost_type": TYPE_DARK,     "type_boost_mult": 1.2},
    "Metal Coat":     {"type_boost_type": TYPE_STEEL,    "type_boost_mult": 1.2},
    "Silk Scarf":     {"type_boost_type": TYPE_NORMAL,   "type_boost_mult": 1.2},

    # ==== Plates (×1.2, same as type-boosting but Arceus-specific) ====
    "Flame Plate":    {"type_boost_type": TYPE_FIRE,     "type_boost_mult": 1.2},
    "Splash Plate":   {"type_boost_type": TYPE_WATER,    "type_boost_mult": 1.2},
    "Zap Plate":      {"type_boost_type": TYPE_ELECTRIC, "type_boost_mult": 1.2},
    "Meadow Plate":   {"type_boost_type": TYPE_GRASS,    "type_boost_mult": 1.2},
    "Icicle Plate":   {"type_boost_type": TYPE_ICE,      "type_boost_mult": 1.2},
    "Fist Plate":     {"type_boost_type": TYPE_FIGHTING, "type_boost_mult": 1.2},
    "Toxic Plate":    {"type_boost_type": TYPE_POISON,   "type_boost_mult": 1.2},
    "Earth Plate":    {"type_boost_type": TYPE_GROUND,   "type_boost_mult": 1.2},
    "Sky Plate":      {"type_boost_type": TYPE_FLYING,   "type_boost_mult": 1.2},
    "Mind Plate":     {"type_boost_type": TYPE_PSYCHIC,  "type_boost_mult": 1.2},
    "Insect Plate":   {"type_boost_type": TYPE_BUG,      "type_boost_mult": 1.2},
    "Stone Plate":    {"type_boost_type": TYPE_ROCK,     "type_boost_mult": 1.2},
    "Spooky Plate":   {"type_boost_type": TYPE_GHOST,    "type_boost_mult": 1.2},
    "Draco Plate":    {"type_boost_type": TYPE_DRAGON,   "type_boost_mult": 1.2},
    "Dread Plate":    {"type_boost_type": TYPE_DARK,     "type_boost_mult": 1.2},
    "Iron Plate":     {"type_boost_type": TYPE_STEEL,    "type_boost_mult": 1.2},

    # ==== End-of-turn recovery ====
    "Leftovers":      {"residual_eff": ev.EFF_LEFTOVERS},
    "Black Sludge":   {"residual_eff": ev.EFF_BLACK_SLUDGE},
    "Sitrus Berry":   {"residual_eff": ev.EFF_SITRUS_BERRY},
    "Lum Berry":      {"residual_eff": ev.EFF_LUM_BERRY},

    # ==== Status-inflicting items ====
    "Flame Orb":      {"residual_eff": ev.EFF_FLAME_ORB},
    "Toxic Orb":      {"residual_eff": ev.EFF_TOXIC_ORB},

    # ==== Focus Sash ====
    # Handled in damage application (survive OHKO at full HP).
    # No relay entry needed — checked directly by item ID in damage calc.

    # ==== Duration-extending items ====
    # Light Clay, Damp Rock, Heat Rock, Icy Rock, Smooth Rock
    # These extend screen/weather duration when set. Handled at set-time, not relay.
}


# ---------------------------------------------------------------------------
# populate_item_tables
# ---------------------------------------------------------------------------

def populate_item_tables(item_name_to_id: dict, tables=None) -> None:
    """
    Install item parameters into events.py relay arrays and handler lists.

    Must be called BEFORE any jax.jit compilation of run_event_* functions.
    """
    global _TABLES, LIFE_ORB_ID, CHOICE_BAND_ID, CHOICE_SPECS_ID, CHOICE_SCARF_ID, FOCUS_SASH_ID

    if tables is not None:
        _TABLES = tables

    LIFE_ORB_ID     = item_name_to_id.get("Life Orb", -1)
    CHOICE_BAND_ID  = item_name_to_id.get("Choice Band", -1)
    CHOICE_SPECS_ID = item_name_to_id.get("Choice Specs", -1)
    CHOICE_SCARF_ID = item_name_to_id.get("Choice Scarf", -1)
    FOCUS_SASH_ID   = item_name_to_id.get("Focus Sash", -1)

    # --- Install state-mutating residual handlers ---
    ev._ITEM_RESIDUAL_HANDLERS[ev.EFF_LEFTOVERS]         = _leftovers_residual_state
    ev._ITEM_RESIDUAL_HANDLERS[ev.EFF_BLACK_SLUDGE]      = _black_sludge_residual_state
    ev._ITEM_RESIDUAL_HANDLERS[ev.EFF_SITRUS_BERRY]      = _sitrus_berry_residual_state
    ev._ITEM_RESIDUAL_HANDLERS[ev.EFF_LUM_BERRY]         = _lum_berry_residual_state
    ev._ITEM_RESIDUAL_HANDLERS[ev.EFF_LIFE_ORB_RESIDUAL] = _life_orb_residual_state
    ev._ITEM_RESIDUAL_HANDLERS[ev.EFF_FLAME_ORB]         = _flame_orb_residual_state
    ev._ITEM_RESIDUAL_HANDLERS[ev.EFF_TOXIC_ORB]         = _toxic_orb_residual_state

    # --- Update per-item relay arrays ---
    for item_name, params in ITEM_HANDLERS.items():
        item_id = item_name_to_id.get(item_name, -1)
        if item_id < 0 or item_id >= ev._MAX_ITEMS:
            continue

        if "modify_atk_mult" in params:
            ev._ITEM_ATK_MULT = ev._ITEM_ATK_MULT.at[item_id].set(
                jnp.float32(params["modify_atk_mult"]))

        if "modify_spa_mult" in params:
            ev._ITEM_SPA_MULT = ev._ITEM_SPA_MULT.at[item_id].set(
                jnp.float32(params["modify_spa_mult"]))

        if "modify_dmg_mult" in params:
            ev._ITEM_DMG_MULT = ev._ITEM_DMG_MULT.at[item_id].set(
                jnp.float32(params["modify_dmg_mult"]))

        if "modify_dmg_cond" in params:
            ev._ITEM_DMG_COND = ev._ITEM_DMG_COND.at[item_id].set(
                jnp.int8(params["modify_dmg_cond"]))

        if "speed_mult" in params:
            ev._ITEM_SPEED_MULT = ev._ITEM_SPEED_MULT.at[item_id].set(
                jnp.float32(params["speed_mult"]))

        if "type_boost_type" in params:
            ev._ITEM_TYPE_BOOST_TYPE = ev._ITEM_TYPE_BOOST_TYPE.at[item_id].set(
                jnp.int8(params["type_boost_type"]))

        if "type_boost_mult" in params:
            ev._ITEM_TYPE_BOOST_MULT = ev._ITEM_TYPE_BOOST_MULT.at[item_id].set(
                jnp.float32(params["type_boost_mult"]))

        if "residual_eff" in params:
            ev._ITEM_RESIDUAL_EFF = ev._ITEM_RESIDUAL_EFF.at[item_id].set(
                jnp.int8(params["residual_eff"]))
