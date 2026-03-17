"""
Flattened event dispatch system for PokeJAX.

Architecture: two-tier dispatch.

Tier 1 — Relay events (float / bool results, called multiple times per turn):
  Uses JAX array lookup tables instead of lax.switch, giving O(1) XLA compilation
  regardless of the number of registered abilities/items.
  Each event stores per-ability/item scalar parameters (constant multipliers,
  condition codes, type IDs) in compact jnp arrays that are indexed at runtime.

Tier 2 — State-mutating events (return BattleState, called ≤2x per turn):
  Uses lax.switch over a SMALL handler list (≤7 branches) via an indirection table:
    ability_id → effect_id → handler function
  This replaces the old direct lax.switch over all ability IDs (35 branches),
  reducing per-event compile cost by 5–17×.

Implemented events:
  Relay:  ModifyAtk, ModifySpa, BasePower, ModifyDamage, TryHit, DamagingHit, Residual
  State:  SwitchIn, SwitchOut, ResidualState, TryHitState
"""

from typing import Callable, List
import jax
import jax.numpy as jnp

from pokejax.types import BattleState


# ---------------------------------------------------------------------------
# Table dimensions
# ---------------------------------------------------------------------------

_MAX_ABILITIES = 512  # covers Gen 1-9 (Gen 9 has ~270 abilities); must be set before any JIT
_MAX_ITEMS     = 512  # covers Gen 1-9 (Gen 9 has ~350 items)
_MAX_MOVES_EV  = 10   # kept for API compatibility
_N_WEATHERS    = 5
_N_TERRAINS    = 5
_N_STATUSES    = 7


# ---------------------------------------------------------------------------
# Tables reference
# Set by abilities.py during populate_ability_tables().
# Provides move data (tables.moves) and type chart (for type_effectiveness).
# ---------------------------------------------------------------------------
_TABLES_REF = None   # full Tables namedtuple


# ---------------------------------------------------------------------------
# Ability condition codes (used by run_event_* to select branchless formula)
# ---------------------------------------------------------------------------

# ModifyAtk conditions
ABCOND_ATK_NONE     = jnp.int8(0)
ABCOND_ATK_GUTS     = jnp.int8(1)   # if statused → ×1.5
ABCOND_ATK_SUN      = jnp.int8(2)   # Flower Gift: if sun → ×1.5

# ModifySpa conditions
ABCOND_SPA_NONE        = jnp.int8(0)
ABCOND_SPA_SOLAR_POWER = jnp.int8(1)  # if sun → ×1.5

# BasePower conditions
ABCOND_BP_NONE        = jnp.int8(0)
ABCOND_BP_TECHNICIAN  = jnp.int8(1)   # if bp ≤ 60 → ×1.5
ABCOND_BP_IRON_FIST   = jnp.int8(2)   # if punch_flag → ×1.2
ABCOND_BP_RECKLESS    = jnp.int8(3)   # if recoil → ×1.2
ABCOND_BP_STARTER     = jnp.int8(4)   # if move_type==_AB_BP_TYPE && low_hp → ×1.5

# ModifyDef conditions
ABCOND_DEF_NONE         = jnp.int8(0)
ABCOND_DEF_MARVEL_SCALE = jnp.int8(1)   # if statused → ×1.5 Def

# ModifySpD conditions
ABCOND_SPD_NONE         = jnp.int8(0)
ABCOND_SPD_FLOWER_GIFT  = jnp.int8(1)   # if sun → ×1.5 SpD

# Speed conditions
ABCOND_SPEED_NONE       = jnp.int8(0)
ABCOND_SPEED_RAIN       = jnp.int8(1)   # Swift Swim: rain → ×2 Speed
ABCOND_SPEED_SUN        = jnp.int8(2)   # Chlorophyll: sun → ×2 Speed
ABCOND_SPEED_SAND       = jnp.int8(3)   # Sand Rush: sand → ×2 Speed (Gen 5, but some mods)
ABCOND_SPEED_UNBURDEN   = jnp.int8(4)   # Unburden: lost item → ×2 Speed

# Ability ModifyDamage conditions
ABCOND_DMG_NONE         = jnp.int8(0)
ABCOND_DMG_TINTED_LENS  = jnp.int8(1)   # if not-very-effective → ×2

# Item ModifyDamage conditions
ITEMCOND_DMG_NONE        = jnp.int8(0)
ITEMCOND_DMG_EXPERT_BELT = jnp.int8(1)   # if super-effective → ×1.2
ITEMCOND_DMG_MUSCLE_BAND = jnp.int8(2)   # if physical → ×1.1
ITEMCOND_DMG_WISE_GLASSES = jnp.int8(3)  # if special → ×1.1


# ---------------------------------------------------------------------------
# Tier-1: Relay event arrays
# All initialised to no-op values (1.0 multipliers, -1 immune types, 0 conds).
# Updated by populate_ability_tables / populate_item_tables BEFORE any JIT.
# ---------------------------------------------------------------------------

# ---- ModifyAtk ----
_AB_ATK_MULT      = jnp.ones(_MAX_ABILITIES,  dtype=jnp.float32)
_AB_ATK_COND      = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_AB_SRC_ATK_MULT  = jnp.ones(_MAX_ABILITIES,  dtype=jnp.float32)   # defender ability → atk
_ITEM_ATK_MULT    = jnp.ones(_MAX_ITEMS,       dtype=jnp.float32)

# ---- ModifySpa ----
_AB_SPA_MULT      = jnp.ones(_MAX_ABILITIES,  dtype=jnp.float32)
_AB_SPA_COND      = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_AB_SRC_SPA_MULT  = jnp.ones(_MAX_ABILITIES,  dtype=jnp.float32)
_ITEM_SPA_MULT    = jnp.ones(_MAX_ITEMS,       dtype=jnp.float32)

# ---- BasePower ----
_AB_BP_COND       = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_AB_BP_TYPE       = jnp.full(_MAX_ABILITIES,  -1, dtype=jnp.int8)  # triggering type for starters
_ITEM_BP_MULT     = jnp.ones(_MAX_ITEMS,       dtype=jnp.float32)

# ---- ModifyDamage ----
_AB_DMG_MULT      = jnp.ones(_MAX_ABILITIES,  dtype=jnp.float32)
_AB_DMG_COND      = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_ITEM_DMG_MULT    = jnp.ones(_MAX_ITEMS,       dtype=jnp.float32)
_ITEM_DMG_COND    = jnp.zeros(_MAX_ITEMS,      dtype=jnp.int8)

# ---- ModifyDef ----
_AB_DEF_MULT      = jnp.ones(_MAX_ABILITIES,  dtype=jnp.float32)
_AB_DEF_COND      = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)

# ---- ModifySpD ----
_AB_SPD_MULT      = jnp.ones(_MAX_ABILITIES,  dtype=jnp.float32)
_AB_SPD_COND      = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)

# ---- Speed modifier ----
_AB_SPEED_MULT    = jnp.ones(_MAX_ABILITIES,  dtype=jnp.float32)
_AB_SPEED_COND    = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_ITEM_SPEED_MULT  = jnp.ones(_MAX_ITEMS,      dtype=jnp.float32)

# ---- Status immunity ----
# -1 = no immunity; >= 0 = the STATUS_* code this ability is immune to
_AB_STATUS_IMMUNE = jnp.full(_MAX_ABILITIES, -1, dtype=jnp.int8)

# ---- Type-boosting items ----
# -1 = no type boost; >= 0 = the type this item boosts
_ITEM_TYPE_BOOST_TYPE = jnp.full(_MAX_ITEMS, -1, dtype=jnp.int8)
_ITEM_TYPE_BOOST_MULT = jnp.ones(_MAX_ITEMS,     dtype=jnp.float32)

# ---- TryHit immunity ----
# -1 = no immunity; otherwise = the move type that is absorbed/negated
_AB_TRYHIT_IMMUNE = jnp.full(_MAX_ABILITIES, -1, dtype=jnp.int8)


# ---------------------------------------------------------------------------
# Tier-2: State-mutating effect IDs
# Maps ability_id / item_id → a compact effect_id in a small fixed range.
# The run_event_* functions call lax.switch(effect_id, SMALL_HANDLER_LIST, ...).
# ---------------------------------------------------------------------------

# Effect IDs are defined per-event-type (different enums).

# SwitchIn effect IDs:
EFF_SWITCH_IN_NOOP        = 0
EFF_INTIMIDATE            = 1
EFF_DRIZZLE               = 2
EFF_DROUGHT               = 3
EFF_SAND_STREAM           = 4
EFF_SNOW_WARNING          = 5
EFF_DOWNLOAD              = 6
EFF_TRACE                 = 7
EFF_FORECAST              = 8
EFF_FRISK                 = 9
EFF_ANTICIPATION          = 10
EFF_FOREWARN              = 11
_N_SWITCH_IN_EFFS         = 12  # total slots

# SwitchOut effect IDs:
EFF_SWITCH_OUT_NOOP       = 0
EFF_NATURAL_CURE          = 1
EFF_REGENERATOR           = 2
_N_SWITCH_OUT_EFFS        = 3

# Ability Residual effect IDs:
EFF_AB_RES_NOOP           = 0
EFF_SPEED_BOOST           = 1
EFF_POISON_HEAL           = 2
EFF_BAD_DREAMS            = 3
EFF_DRY_SKIN_RES          = 4
EFF_RAIN_DISH             = 5
EFF_ICE_BODY              = 6
EFF_SHED_SKIN             = 7
_N_AB_RES_EFFS            = 8

# Item Residual effect IDs:
EFF_ITEM_RES_NOOP         = 0
EFF_LEFTOVERS             = 1
EFF_BLACK_SLUDGE          = 2
EFF_SITRUS_BERRY          = 3
EFF_LUM_BERRY             = 4
EFF_LIFE_ORB_RESIDUAL     = 5
EFF_FLAME_ORB             = 6
EFF_TOXIC_ORB             = 7
_N_ITEM_RES_EFFS          = 8

# TryHit-state effect IDs:
EFF_TRYHIT_STATE_NOOP     = 0
EFF_WATER_ABSORB          = 1
EFF_VOLT_ABSORB           = 2
EFF_MOTOR_DRIVE           = 3
EFF_SAP_SIPPER            = 4
EFF_STORM_DRAIN           = 5
EFF_LIGHTNING_ROD         = 6
EFF_DRY_SKIN_ABSORB       = 7
_N_TRYHIT_STATE_EFFS      = 8

# Contact punishment effect IDs (state-mutating):
EFF_CONTACT_NOOP          = 0
EFF_ROUGH_SKIN            = 1
EFF_STATIC_CONTACT        = 2
EFF_POISON_POINT          = 3
EFF_FLAME_BODY            = 4
EFF_EFFECT_SPORE          = 5
EFF_CUTE_CHARM            = 6
_N_CONTACT_EFFS           = 7

# --- Effect ID tables (ability_id → effect_id, item_id → effect_id) ---
_AB_SWITCH_IN_EFF    = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_AB_SWITCH_OUT_EFF   = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_AB_RESIDUAL_EFF     = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_ITEM_RESIDUAL_EFF   = jnp.zeros(_MAX_ITEMS,     dtype=jnp.int8)
_AB_TRYHIT_STATE_EFF = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)
_AB_CONTACT_EFF      = jnp.zeros(_MAX_ABILITIES, dtype=jnp.int8)


# ---------------------------------------------------------------------------
# No-op factories for state-mutating handlers
# ---------------------------------------------------------------------------

def _make_noop_state():
    def _noop(state, side_i32, slot_i32):
        return state
    return _noop


def _make_noop_residual():
    def _noop(state, key, side_i32, slot_i32):
        return state, key
    return _noop


def _make_noop_state_try_hit():
    def _noop(state, atk_side_i32, atk_idx_i32, def_side_i32, def_idx_i32, move_type_i32):
        return state
    return _noop


def _make_noop_contact():
    def _noop(state, atk_side_i32, atk_idx_i32, def_side_i32, def_idx_i32, key):
        return state, key
    return _noop


# ---------------------------------------------------------------------------
# State-mutating handler lists (small, filled by populate_*_tables)
# Each list has _N_*_EFFS slots; slot 0 is always noop.
# ---------------------------------------------------------------------------

_SWITCH_IN_HANDLERS:     List[Callable] = [_make_noop_state()    for _ in range(_N_SWITCH_IN_EFFS)]
_SWITCH_OUT_HANDLERS:    List[Callable] = [_make_noop_state()    for _ in range(_N_SWITCH_OUT_EFFS)]
_AB_RESIDUAL_HANDLERS:   List[Callable] = [_make_noop_residual() for _ in range(_N_AB_RES_EFFS)]
_ITEM_RESIDUAL_HANDLERS: List[Callable] = [_make_noop_residual() for _ in range(_N_ITEM_RES_EFFS)]
_TRYHIT_STATE_HANDLERS:  List[Callable] = [_make_noop_state_try_hit() for _ in range(_N_TRYHIT_STATE_EFFS)]
_CONTACT_HANDLERS:       List[Callable] = [_make_noop_contact()       for _ in range(_N_CONTACT_EFFS)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_ability(aid):
    """Clip a traced ability ID to valid array index."""
    return jnp.clip(aid.astype(jnp.int32), 0, _MAX_ABILITIES - 1)


def _safe_item(iid):
    """Clip a traced item ID to valid array index (0 = no item)."""
    return jnp.clip(iid.astype(jnp.int32), 0, _MAX_ITEMS - 1)


def _has_item(iid):
    return iid.astype(jnp.int32) > 0


# ---------------------------------------------------------------------------
# Tier-1: Relay event implementations
# ---------------------------------------------------------------------------

def run_event_modify_atk(relay: jnp.ndarray, state: BattleState,
                          atk_side: int, atk_idx: jnp.ndarray,
                          def_side: int, def_idx: jnp.ndarray,
                          move_id: jnp.ndarray) -> jnp.ndarray:
    """
    ModifyAtk: returns float32 Attack multiplier.
    Priority: attacker ability → attacker item → defender ability (source mod).
    """
    # 1. Attacker ability — constant multiplier (e.g. Huge Power ×2, Hustle ×1.5)
    aid = _safe_ability(state.sides_team_ability_id[atk_side, atk_idx])
    relay = relay * _AB_ATK_MULT[aid]

    # 1b. Conditional: Guts (cond=1 → ×1.5 when statused)
    from pokejax.types import STATUS_NONE, WEATHER_SUN
    is_statused = state.sides_team_status[atk_side, atk_idx] != jnp.int8(STATUS_NONE)
    atk_cond = _AB_ATK_COND[aid]
    relay = relay * jnp.where(
        (atk_cond == ABCOND_ATK_GUTS) & is_statused,
        jnp.float32(1.5), jnp.float32(1.0)
    )

    # 1c. Conditional: Flower Gift (cond=2 → ×1.5 in sun)
    in_sun = state.field.weather == jnp.int8(WEATHER_SUN)
    relay = relay * jnp.where(
        (atk_cond == ABCOND_ATK_SUN) & in_sun,
        jnp.float32(1.5), jnp.float32(1.0)
    )

    # 2. Attacker item — constant multiplier (e.g. Choice Band ×1.5)
    iid = state.sides_team_item_id[atk_side, atk_idx]
    relay = relay * jnp.where(_has_item(iid), _ITEM_ATK_MULT[_safe_item(iid)], jnp.float32(1.0))

    # 3. Defender ability source modifier (none currently active, table is all 1.0)
    did = _safe_ability(state.sides_team_ability_id[def_side, def_idx])
    relay = relay * _AB_SRC_ATK_MULT[did]

    return relay


def run_event_modify_spa(relay: jnp.ndarray, state: BattleState,
                          atk_side: int, atk_idx: jnp.ndarray,
                          def_side: int, def_idx: jnp.ndarray,
                          move_id: jnp.ndarray) -> jnp.ndarray:
    """ModifySpA: returns float32 Special Attack multiplier."""
    from pokejax.types import WEATHER_SUN

    # 1. Attacker ability
    aid = _safe_ability(state.sides_team_ability_id[atk_side, atk_idx])
    relay = relay * _AB_SPA_MULT[aid]

    # Solar Power: cond=1 → ×1.5 in sun
    in_sun = state.field.weather == jnp.int8(WEATHER_SUN)
    relay = relay * jnp.where(
        (_AB_SPA_COND[aid] == ABCOND_SPA_SOLAR_POWER) & in_sun,
        jnp.float32(1.5), jnp.float32(1.0)
    )

    # 2. Attacker item
    iid = state.sides_team_item_id[atk_side, atk_idx]
    relay = relay * jnp.where(_has_item(iid), _ITEM_SPA_MULT[_safe_item(iid)], jnp.float32(1.0))

    # 3. Defender ability source modifier
    did = _safe_ability(state.sides_team_ability_id[def_side, def_idx])
    relay = relay * _AB_SRC_SPA_MULT[did]

    return relay


def run_event_base_power(relay: jnp.ndarray, state: BattleState,
                          atk_side: int, atk_idx: jnp.ndarray,
                          def_side: int, def_idx: jnp.ndarray,
                          move_id: jnp.ndarray) -> jnp.ndarray:
    """
    BasePower: modifies effective base power before damage formula.
    e.g. Technician, Iron Fist, Reckless, Blaze/Torrent/Overgrow/Swarm.
    """
    from pokejax.core.damage import MF_FLAGS_LO, MF_RECOIL_NUM, MF_TYPE, FLAG_PUNCH

    aid = _safe_ability(state.sides_team_ability_id[atk_side, atk_idx])
    cond = _AB_BP_COND[aid]

    # Technician: if relay ≤ 60 → ×1.5
    relay = relay * jnp.where(
        (cond == ABCOND_BP_TECHNICIAN) & (relay <= jnp.float32(60.0)),
        jnp.float32(1.5), jnp.float32(1.0)
    )

    if _TABLES_REF is not None:
        moves = _TABLES_REF.moves
        mid = move_id.astype(jnp.int32)
        flags_lo   = moves[mid, MF_FLAGS_LO].astype(jnp.int32)
        has_recoil = moves[mid, MF_RECOIL_NUM].astype(jnp.int32) > jnp.int32(0)
        move_type  = moves[mid, MF_TYPE].astype(jnp.int32)
        is_punch   = (flags_lo & jnp.int32(FLAG_PUNCH)) != jnp.int32(0)

        # Iron Fist: punch flag → ×1.2
        relay = relay * jnp.where(
            (cond == ABCOND_BP_IRON_FIST) & is_punch,
            jnp.float32(1.2), jnp.float32(1.0)
        )

        # Reckless: recoil move → ×1.2
        relay = relay * jnp.where(
            (cond == ABCOND_BP_RECKLESS) & has_recoil,
            jnp.float32(1.2), jnp.float32(1.0)
        )

        # Starter trio (Blaze/Torrent/Overgrow/Swarm): matching type & HP ≤ 1/3 → ×1.5
        hp     = state.sides_team_hp[atk_side, atk_idx].astype(jnp.int32)
        max_hp = state.sides_team_max_hp[atk_side, atk_idx].astype(jnp.int32)
        low_hp = (hp * jnp.int32(3)) <= max_hp
        ab_type = _AB_BP_TYPE[aid].astype(jnp.int32)
        relay = relay * jnp.where(
            (cond == ABCOND_BP_STARTER) & (move_type == ab_type) & low_hp,
            jnp.float32(1.5), jnp.float32(1.0)
        )

    # Item BP multiplier (constant, e.g. Metronome — not used in Gen 4 randbats)
    iid = state.sides_team_item_id[atk_side, atk_idx]
    safe_iid = _safe_item(iid)
    relay = relay * jnp.where(_has_item(iid), _ITEM_BP_MULT[safe_iid], jnp.float32(1.0))

    # Type-boosting items (Charcoal, Mystic Water, etc.)
    if _TABLES_REF is not None:
        moves = _TABLES_REF.moves
        mid = move_id.astype(jnp.int32)
        move_type_for_boost = moves[mid, MF_TYPE].astype(jnp.int32)
        item_boost_type = _ITEM_TYPE_BOOST_TYPE[safe_iid].astype(jnp.int32)
        item_boost_mult = _ITEM_TYPE_BOOST_MULT[safe_iid]
        type_matches = (item_boost_type >= jnp.int32(0)) & (move_type_for_boost == item_boost_type)
        relay = relay * jnp.where(
            _has_item(iid) & type_matches,
            item_boost_mult, jnp.float32(1.0)
        )

    return relay


def run_event_modify_damage(relay: jnp.ndarray, state: BattleState,
                             atk_side: int, atk_idx: jnp.ndarray,
                             def_side: int, def_idx: jnp.ndarray,
                             move_id: jnp.ndarray) -> jnp.ndarray:
    """
    ModifyDamage: final multipliers after base calc.
    e.g. Life Orb (×1.3), Expert Belt (×1.2 SE), Muscle Band/Wise Glasses (×1.1).
    """
    from pokejax.core.damage import MF_TYPE, MF_CATEGORY

    # Ability constant modifier (currently unused after moving Sniper/Tinted Lens)
    aid = _safe_ability(state.sides_team_ability_id[atk_side, atk_idx])
    relay = relay * _AB_DMG_MULT[aid]

    # Ability conditional modifier: Tinted Lens (NVE → ×2)
    ab_dmg_cond = _AB_DMG_COND[aid]
    if _TABLES_REF is not None:
        from pokejax.core.damage import type_effectiveness as _te, MF_TYPE as _MF_TYPE
        _mid = move_id.astype(jnp.int32)
        _mt = _TABLES_REF.moves[_mid, _MF_TYPE].astype(jnp.int32)
        _def_idx_i32 = def_idx.astype(jnp.int32)
        _dt = state.sides_team_types[def_side, _def_idx_i32]
        _eff = _te(_TABLES_REF, _mt, _dt[0].astype(jnp.int32), _dt[1].astype(jnp.int32))
        is_nve = _eff < jnp.float32(1.0)
        relay = relay * jnp.where(
            (ab_dmg_cond == ABCOND_DMG_TINTED_LENS) & is_nve,
            jnp.float32(2.0), jnp.float32(1.0)
        )

    # Item constant multiplier (Life Orb = 1.3, others = 1.0)
    iid = state.sides_team_item_id[atk_side, atk_idx]
    safe_iid = _safe_item(iid)
    relay = relay * jnp.where(_has_item(iid), _ITEM_DMG_MULT[safe_iid], jnp.float32(1.0))

    # Item conditional multipliers
    item_cond = _ITEM_DMG_COND[safe_iid]

    if _TABLES_REF is not None:
        moves = _TABLES_REF.moves
        mid = move_id.astype(jnp.int32)
        move_type = moves[mid, MF_TYPE].astype(jnp.int32)
        move_cat  = moves[mid, MF_CATEGORY].astype(jnp.int32)

        # Expert Belt: super-effective → ×1.2
        from pokejax.core.damage import type_effectiveness
        def_idx_i32 = def_idx.astype(jnp.int32)
        def_types = state.sides_team_types[def_side, def_idx_i32]
        eff = type_effectiveness(
            _TABLES_REF,
            move_type,
            def_types[0].astype(jnp.int32),
            def_types[1].astype(jnp.int32),
        )
        is_se = eff > jnp.float32(1.0)
        relay = relay * jnp.where(
            (_has_item(iid)) & (item_cond == ITEMCOND_DMG_EXPERT_BELT) & is_se,
            jnp.float32(1.2), jnp.float32(1.0)
        )

        # Muscle Band: physical → ×1.1
        is_phys = move_cat == jnp.int32(0)
        relay = relay * jnp.where(
            (_has_item(iid)) & (item_cond == ITEMCOND_DMG_MUSCLE_BAND) & is_phys,
            jnp.float32(1.1), jnp.float32(1.0)
        )

        # Wise Glasses: special → ×1.1
        is_spec = move_cat == jnp.int32(1)
        relay = relay * jnp.where(
            (_has_item(iid)) & (item_cond == ITEMCOND_DMG_WISE_GLASSES) & is_spec,
            jnp.float32(1.1), jnp.float32(1.0)
        )

    return relay


def run_event_try_hit(relay: jnp.ndarray, state: BattleState,
                       atk_side: int, atk_idx: jnp.ndarray,
                       def_side: int, def_idx: jnp.ndarray,
                       move_id: jnp.ndarray) -> tuple:
    """
    TryHit: checks type immunities from defender's ability.
    Returns (relay: bool, cancelled: bool).
    relay=False means the move hit is cancelled.
    """
    from pokejax.core.damage import MF_TYPE

    cancelled = jnp.bool_(False)

    # Defender ability immunity type check
    did = _safe_ability(state.sides_team_ability_id[def_side, def_idx])
    immune_type = _AB_TRYHIT_IMMUNE[did].astype(jnp.int32)
    has_immunity = immune_type >= jnp.int32(0)

    # Mold Breaker on attacker bypasses defender's ability-based immunities
    from pokejax.mechanics.abilities import MOLD_BREAKER_ID
    atk_ability = state.sides_team_ability_id[atk_side, atk_idx].astype(jnp.int32)
    has_mold_breaker = (atk_ability == jnp.int32(MOLD_BREAKER_ID)) & (jnp.int32(MOLD_BREAKER_ID) > 0)

    if _TABLES_REF is not None:
        move_type = _TABLES_REF.moves[move_id.astype(jnp.int32), MF_TYPE].astype(jnp.int32)
        type_matches = move_type == immune_type
        ability_cancels = has_immunity & type_matches & ~has_mold_breaker
        cancelled = cancelled | ability_cancels
        relay = jnp.where(ability_cancels, jnp.bool_(False), relay)

    # No item TryHit immunity currently implemented
    return relay, cancelled


def run_event_damaging_hit(relay: jnp.ndarray, state: BattleState,
                            atk_side: int, atk_idx: jnp.ndarray,
                            def_side: int, def_idx: jnp.ndarray,
                            move_id: jnp.ndarray) -> tuple:
    """
    DamagingHit: relay = damage dealt. No active handlers yet.
    Returns (relay, state).
    """
    return relay, state


def run_event_residual(relay: jnp.ndarray, state: BattleState,
                        side: int, slot: jnp.ndarray,
                        move_id: jnp.ndarray = None) -> jnp.ndarray:
    """
    Residual float relay path (currently unused; state-mutating path is primary).
    Returns relay unchanged — no-op shim for API compatibility.
    """
    return relay


# ---------------------------------------------------------------------------
# Back-compat shims: old lax.switch tables that external code may still
# reference (e.g. test_abilities.py). These are all no-op lists.
# The relay run_event_* functions above no longer use them.
# ---------------------------------------------------------------------------
def _make_noop_float():
    def _noop(relay, state, atk_side, atk_idx, def_side, def_idx, move_id):
        return relay
    return _noop

def _make_noop_bool():
    def _noop(relay, state, atk_side, atk_idx, def_side, def_idx, move_id):
        return relay
    return _noop


# ---------------------------------------------------------------------------
# Tier-2: State-mutating event implementations
# ---------------------------------------------------------------------------

def run_event_switch_in(state: BattleState, side: int,
                         slot: jnp.ndarray) -> BattleState:
    """
    SwitchIn: fires ability effect when Pokemon enters.
    e.g. Intimidate, Drizzle, Download.
    lax.switch over _N_SWITCH_IN_EFFS (7) handlers — not 35.
    """
    ability_id = state.sides_team_ability_id[side, slot].astype(jnp.int32)
    safe_ab = _safe_ability(ability_id)
    eff_id = jnp.clip(_AB_SWITCH_IN_EFF[safe_ab].astype(jnp.int32), 0, _N_SWITCH_IN_EFFS - 1)
    side_i32 = jnp.int32(side)
    slot_i32 = slot.astype(jnp.int32)
    return jax.lax.switch(eff_id, _SWITCH_IN_HANDLERS, state, side_i32, slot_i32)


def run_event_switch_out(state: BattleState, side: int,
                          slot: jnp.ndarray) -> BattleState:
    """
    SwitchOut: fires ability effect when Pokemon leaves.
    e.g. Natural Cure.
    lax.switch over _N_SWITCH_OUT_EFFS (2) handlers.
    """
    ability_id = state.sides_team_ability_id[side, slot].astype(jnp.int32)
    safe_ab = _safe_ability(ability_id)
    eff_id = jnp.clip(_AB_SWITCH_OUT_EFF[safe_ab].astype(jnp.int32), 0, _N_SWITCH_OUT_EFFS - 1)
    side_i32 = jnp.int32(side)
    slot_i32 = slot.astype(jnp.int32)
    return jax.lax.switch(eff_id, _SWITCH_OUT_HANDLERS, state, side_i32, slot_i32)


def run_event_residual_state(state: BattleState, key: jnp.ndarray,
                              side: int,
                              slot: jnp.ndarray) -> tuple:
    """
    Residual state-mutating effects (end of turn).
    Ability: Speed Boost, Poison Heal, Shed Skin.  Item: Leftovers, Black Sludge, etc.
    Returns (state, key).
    """
    import jax as _jax
    side_i32 = jnp.int32(side)
    slot_i32 = slot.astype(jnp.int32)

    # Ability residual (may consume key, e.g. Shed Skin 30% roll)
    ability_id = state.sides_team_ability_id[side, slot].astype(jnp.int32)
    safe_ab = _safe_ability(ability_id)
    ab_eff = jnp.clip(_AB_RESIDUAL_EFF[safe_ab].astype(jnp.int32), 0, _N_AB_RES_EFFS - 1)
    key, ab_key = _jax.random.split(key)
    state, ab_key = jax.lax.switch(ab_eff, _AB_RESIDUAL_HANDLERS, state, ab_key, side_i32, slot_i32)

    # Item residual
    item_id = state.sides_team_item_id[side, slot].astype(jnp.int32)
    safe_item_idx = _safe_item(item_id)
    item_eff = jnp.clip(_ITEM_RESIDUAL_EFF[safe_item_idx].astype(jnp.int32), 0, _N_ITEM_RES_EFFS - 1)
    key, item_key = _jax.random.split(key)
    state, item_key = jax.lax.switch(item_eff, _ITEM_RESIDUAL_HANDLERS, state, item_key, side_i32, slot_i32)

    return state, key


def run_event_try_hit_state(state: BattleState,
                             atk_side: int, atk_idx: jnp.ndarray,
                             def_side: int, def_idx: jnp.ndarray,
                             move_id: jnp.ndarray,
                             move_type: jnp.ndarray) -> BattleState:
    """
    TryHit state-mutating: absorption ability side-effects.
    Water Absorb (heal), Motor Drive (+SPE), Sap Sipper (+ATK), etc.
    lax.switch over _N_TRYHIT_STATE_EFFS (7) handlers.
    """
    def_ability = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)
    safe_da = _safe_ability(def_ability)

    # Mold Breaker on attacker suppresses defender's ability effects
    from pokejax.mechanics.abilities import MOLD_BREAKER_ID
    atk_ability = state.sides_team_ability_id[atk_side, atk_idx].astype(jnp.int32)
    has_mold_breaker = (atk_ability == jnp.int32(MOLD_BREAKER_ID)) & (jnp.int32(MOLD_BREAKER_ID) > 0)
    # If Mold Breaker, force eff_id to 0 (no-op handler)
    eff_id_raw = _AB_TRYHIT_STATE_EFF[safe_da].astype(jnp.int32)
    eff_id = jnp.where(has_mold_breaker, jnp.int32(0), jnp.clip(eff_id_raw, 0, _N_TRYHIT_STATE_EFFS - 1))

    atk_side_i32  = jnp.int32(atk_side)
    atk_idx_i32   = atk_idx.astype(jnp.int32)
    def_side_i32  = jnp.int32(def_side)
    def_idx_i32   = def_idx.astype(jnp.int32)
    move_type_i32 = move_type.astype(jnp.int32)
    return jax.lax.switch(
        eff_id, _TRYHIT_STATE_HANDLERS,
        state, atk_side_i32, atk_idx_i32, def_side_i32, def_idx_i32, move_type_i32
    )


def run_event_modify_def(relay: jnp.ndarray, state: BattleState,
                          atk_side: int, atk_idx: jnp.ndarray,
                          def_side: int, def_idx: jnp.ndarray,
                          move_id: jnp.ndarray) -> jnp.ndarray:
    """ModifyDef: returns float32 Defense multiplier for the defender."""
    from pokejax.types import STATUS_NONE

    did = _safe_ability(state.sides_team_ability_id[def_side, def_idx])
    relay = relay * _AB_DEF_MULT[did]

    # Marvel Scale: statused → ×1.5 Def
    is_statused = state.sides_team_status[def_side, def_idx] != jnp.int8(STATUS_NONE)
    relay = relay * jnp.where(
        (_AB_DEF_COND[did] == ABCOND_DEF_MARVEL_SCALE) & is_statused,
        jnp.float32(1.5), jnp.float32(1.0)
    )

    return relay


def run_event_modify_spd(relay: jnp.ndarray, state: BattleState,
                          atk_side: int, atk_idx: jnp.ndarray,
                          def_side: int, def_idx: jnp.ndarray,
                          move_id: jnp.ndarray) -> jnp.ndarray:
    """ModifySpD: returns float32 Special Defense multiplier for the defender."""
    from pokejax.types import WEATHER_SUN, WEATHER_SAND

    did = _safe_ability(state.sides_team_ability_id[def_side, def_idx])
    relay = relay * _AB_SPD_MULT[did]

    # Flower Gift: sun → ×1.5 SpD
    in_sun = state.field.weather == jnp.int8(WEATHER_SUN)
    relay = relay * jnp.where(
        (_AB_SPD_COND[did] == ABCOND_SPD_FLOWER_GIFT) & in_sun,
        jnp.float32(1.5), jnp.float32(1.0)
    )

    # Sand: Rock types get ×1.5 SpD in sandstorm (Gen 4 mechanic)
    from pokejax.types import TYPE_ROCK
    in_sand = state.field.weather == jnp.int8(WEATHER_SAND)
    def_types = state.sides_team_types[def_side, def_idx]
    is_rock = (def_types[0] == jnp.int8(TYPE_ROCK)) | (def_types[1] == jnp.int8(TYPE_ROCK))
    relay = relay * jnp.where(in_sand & is_rock, jnp.float32(1.5), jnp.float32(1.0))

    return relay


def run_event_speed(state: BattleState, side: int,
                     slot: jnp.ndarray) -> jnp.ndarray:
    """
    Speed modifier from abilities and items.
    Returns float32 multiplier applied to effective speed.
    """
    from pokejax.types import WEATHER_RAIN, WEATHER_SUN, WEATHER_SAND

    aid = _safe_ability(state.sides_team_ability_id[side, slot])
    relay = _AB_SPEED_MULT[aid]

    # Weather-conditional speed doubling
    cond = _AB_SPEED_COND[aid]
    weather = state.field.weather

    rain_boost = (cond == ABCOND_SPEED_RAIN) & (weather == jnp.int8(WEATHER_RAIN))
    sun_boost  = (cond == ABCOND_SPEED_SUN) & (weather == jnp.int8(WEATHER_SUN))
    sand_boost = (cond == ABCOND_SPEED_SAND) & (weather == jnp.int8(WEATHER_SAND))
    weather_active = rain_boost | sun_boost | sand_boost
    relay = relay * jnp.where(weather_active, jnp.float32(2.0), jnp.float32(1.0))

    # Item speed multiplier (Choice Scarf = 1.5)
    iid = state.sides_team_item_id[side, slot]
    relay = relay * jnp.where(_has_item(iid), _ITEM_SPEED_MULT[_safe_item(iid)], jnp.float32(1.0))

    return relay


def run_event_check_status_immunity(state: BattleState, side: int,
                                      slot: jnp.ndarray,
                                      status_code: jnp.ndarray) -> jnp.ndarray:
    """
    Check if the Pokemon's ability grants immunity to a status.
    Returns bool: True = immune (status should NOT be applied).
    """
    aid = _safe_ability(state.sides_team_ability_id[side, slot])
    immune_to = _AB_STATUS_IMMUNE[aid]
    has_immunity = immune_to >= jnp.int8(0)
    return has_immunity & (immune_to == status_code)


def run_event_contact_punish(state: BattleState,
                               atk_side: int, atk_idx: jnp.ndarray,
                               def_side: int, def_idx: jnp.ndarray,
                               key: jnp.ndarray) -> tuple:
    """
    Contact punishment: after a contact move hits, the defender's ability
    may inflict damage or status on the attacker.
    Returns (new_state, new_key).
    """
    def_ability = state.sides_team_ability_id[def_side, def_idx].astype(jnp.int32)
    safe_da = _safe_ability(def_ability)
    eff_id = jnp.clip(_AB_CONTACT_EFF[safe_da].astype(jnp.int32), 0, _N_CONTACT_EFFS - 1)
    atk_side_i32 = jnp.int32(atk_side)
    atk_idx_i32  = atk_idx.astype(jnp.int32)
    def_side_i32 = jnp.int32(def_side)
    def_idx_i32  = def_idx.astype(jnp.int32)
    return jax.lax.switch(
        eff_id, _CONTACT_HANDLERS,
        state, atk_side_i32, atk_idx_i32, def_side_i32, def_idx_i32, key
    )
