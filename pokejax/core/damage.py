"""
Damage calculation for PokeJAX.

Implements the Gen 4+ damage formula and all modifier chains,
fully branchless (jnp.where / jnp.select throughout).

Main entry point:
    compute_damage(state, tables, cfg, atk_side, def_side, move_id, rng_key)
        → (damage: int16, crit: bool, effectiveness: float32)

Sub-functions correspond directly to Showdown's getDamage() / modifyDamage() chain
and can be individually tested and verified.
"""

import jax
import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    STAT_ATK, STAT_DEF, STAT_SPA, STAT_SPD, STAT_SPE,
    BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD,
    STATUS_BRN,
    WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS,
    TYPE_FIRE, TYPE_WATER,
)
from pokejax.core.state import (
    get_active_base_stats, get_active_boosts, get_active_types,
    get_active_ability, get_active_item, get_active_status, get_active_level,
)
from pokejax.core import rng as rng_utils

# Move data field indices (matches extractor.py MOVE_FIELDS order)
MF_BASE_POWER   = 0
MF_ACCURACY     = 1
MF_TYPE         = 2
MF_CATEGORY     = 3
MF_PRIORITY     = 4
MF_PP           = 5
MF_TARGET       = 6
MF_FLAGS_LO     = 7
MF_FLAGS_HI     = 8
MF_CRIT_RATIO   = 9
MF_SEC_CHANCE   = 10
MF_SEC_STATUS   = 11
MF_SEC_BOOST_STAT= 12
MF_SEC_BOOST_AMT = 13
MF_DRAIN_NUM    = 14
MF_DRAIN_DEN    = 15
MF_RECOIL_NUM   = 16
MF_RECOIL_DEN   = 17
MF_MULTIHIT_MIN = 18
MF_MULTIHIT_MAX = 19
MF_HEAL_NUM     = 20
MF_HEAL_DEN     = 21

# Move flag bits (stored in MF_FLAGS_LO and MF_FLAGS_HI, packed into uint16)
FLAG_CONTACT    = 1 << 0
FLAG_PROTECT    = 1 << 1
FLAG_MIRROR     = 1 << 2
FLAG_SOUND      = 1 << 3
FLAG_PUNCH      = 1 << 4
FLAG_BITE       = 1 << 5
FLAG_BULLET     = 1 << 6
FLAG_DEFROST    = 1 << 7
FLAG_POWDER     = 1 << 8
FLAG_SNATCH     = 1 << 9
FLAG_HEAL       = 1 << 10
FLAG_RECHARGE   = 1 << 11


# ---------------------------------------------------------------------------
# Stat calculation
# ---------------------------------------------------------------------------

def calc_stat(base_stat: jnp.ndarray, level: jnp.ndarray,
              ev: jnp.ndarray, iv: jnp.ndarray,
              nature_mult: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a non-HP stat value.

    Gen 3+ formula:
        stat = floor((floor((2*base + iv + floor(ev/4)) * level / 100) + 5) * nature)
    """
    inner = (2 * base_stat.astype(jnp.int32)
             + iv.astype(jnp.int32)
             + (ev.astype(jnp.int32) // 4))
    stat = jnp.floor(inner * level.astype(jnp.int32) / 100).astype(jnp.int32) + 5
    return jnp.floor(stat.astype(jnp.float32) * nature_mult).astype(jnp.int32)


def calc_hp(base_hp: jnp.ndarray, level: jnp.ndarray,
            ev: jnp.ndarray, iv: jnp.ndarray) -> jnp.ndarray:
    """
    Compute max HP.

    Gen 3+ formula:
        hp = floor((2*base + iv + floor(ev/4)) * level / 100) + level + 10
    """
    inner = (2 * base_hp.astype(jnp.int32)
             + iv.astype(jnp.int32)
             + (ev.astype(jnp.int32) // 4))
    return (jnp.floor(inner * level.astype(jnp.int32) / 100).astype(jnp.int32)
            + level.astype(jnp.int32) + 10).astype(jnp.int16)


def get_effective_stat(base_stat: jnp.ndarray, boost: jnp.ndarray,
                       tables) -> jnp.ndarray:
    """Apply boost stage to a raw stat value.  Returns float32."""
    mult = tables.get_boost_multiplier(boost)
    return base_stat.astype(jnp.float32) * mult


# ---------------------------------------------------------------------------
# Core damage formula (Showdown: getDamage)
# ---------------------------------------------------------------------------

def base_damage(level: jnp.ndarray, power: jnp.ndarray,
                attack: jnp.ndarray, defense: jnp.ndarray) -> jnp.ndarray:
    """
    Pokemon Gen 4+ base damage formula.

    baseDamage = floor(floor(floor(2*Level/5+2) * Power * Atk / Def) / 50) + 2

    All inputs and output are int32.
    """
    level   = level.astype(jnp.int32)
    power   = power.astype(jnp.int32)
    attack  = attack.astype(jnp.int32)
    defense = defense.astype(jnp.int32)

    step1 = (2 * level) // 5 + 2
    step2 = step1 * power * attack // defense
    return step2 // 50 + 2


# ---------------------------------------------------------------------------
# Modifier chain  (Showdown: modifyDamage)
# ---------------------------------------------------------------------------

def apply_spread_modifier(damage: jnp.ndarray, is_spread: jnp.ndarray) -> jnp.ndarray:
    """0.75x for spread moves hitting multiple targets (doubles only)."""
    scaled = jnp.floor(damage.astype(jnp.float32) * 0.75).astype(jnp.int32)
    return jnp.where(is_spread, scaled, damage)


def apply_weather_modifier(damage: jnp.ndarray, move_type: jnp.ndarray,
                            weather: jnp.ndarray) -> jnp.ndarray:
    """
    Weather damage modifier:
      Sun:  Fire 1.5x, Water 0.5x
      Rain: Water 1.5x, Fire 0.5x
      (Sand/Hail don't affect move damage directly)
    """
    is_fire  = (move_type == jnp.int8(TYPE_FIRE))
    is_water = (move_type == jnp.int8(TYPE_WATER))
    is_sun   = (weather == jnp.int8(WEATHER_SUN))
    is_rain  = (weather == jnp.int8(WEATHER_RAIN))

    boost = jnp.float32(1.5)
    nerf  = jnp.float32(0.5)
    d = damage.astype(jnp.float32)

    # Sun effects
    d = jnp.where(is_sun & is_fire,  jnp.floor(d * boost), d)
    d = jnp.where(is_sun & is_water, jnp.floor(d * nerf),  d)
    # Rain effects
    d = jnp.where(is_rain & is_water, jnp.floor(d * boost), d)
    d = jnp.where(is_rain & is_fire,  jnp.floor(d * nerf),  d)

    return d.astype(jnp.int32)


def apply_crit_modifier(damage: jnp.ndarray, is_crit: jnp.ndarray,
                        crit_multiplier: float = 2.0) -> jnp.ndarray:
    """Apply crit damage bonus.  Pass cfg.crit_damage_multiplier for gen accuracy.
    Gen 4-5: 2.0×  |  Gen 6+: 1.5× (Showdown getDamage / modifyDamage chain).
    """
    crit_mult = jnp.float32(crit_multiplier)
    scaled = jnp.floor(damage.astype(jnp.float32) * crit_mult).astype(jnp.int32)
    return jnp.where(is_crit, scaled, damage)


def apply_random_modifier(damage: jnp.ndarray, roll: jnp.ndarray) -> jnp.ndarray:
    """
    Apply the random damage roll.

    roll: float32 in [0.85, 1.00] from rng_utils.damage_roll().
    Showdown does: damage = floor(damage * roll * 100) // 100  (in effect).
    Actually: damage = floor(damage * (85 + n) / 100) for n in [0,15].
    """
    return jnp.floor(damage.astype(jnp.float32) * roll).astype(jnp.int32)


def apply_stab_modifier(damage: jnp.ndarray, move_type: jnp.ndarray,
                         attacker_types: jnp.ndarray,
                         adaptability: jnp.ndarray) -> jnp.ndarray:
    """
    STAB modifier: 1.5x normally, 2.0x with Adaptability.

    attacker_types: int8[2]
    adaptability:   bool scalar
    """
    has_stab = (move_type == attacker_types[0]) | (move_type == attacker_types[1])
    normal_mult     = jnp.float32(1.5)
    adaptability_mult = jnp.float32(2.0)
    mult = jnp.where(adaptability, adaptability_mult, normal_mult)
    scaled = jnp.floor(damage.astype(jnp.float32) * mult).astype(jnp.int32)
    return jnp.where(has_stab, scaled, damage)


def apply_type_modifier(damage: jnp.ndarray, effectiveness: jnp.ndarray) -> jnp.ndarray:
    """Apply type effectiveness multiplier (0, 0.5, 1.0, 2.0, 4.0)."""
    # Pokemon rounds type effectiveness down (truncate each 2x independently)
    # For simplicity we apply combined multiplier with floor
    scaled = jnp.floor(damage.astype(jnp.float32) * effectiveness).astype(jnp.int32)
    # Immune: if effectiveness == 0, damage should be exactly 0
    return jnp.where(effectiveness == jnp.float32(0.0), jnp.int32(0), scaled)


def apply_burn_modifier(damage: jnp.ndarray, category: jnp.ndarray,
                        status: jnp.ndarray, guts: jnp.ndarray) -> jnp.ndarray:
    """Burn halves physical damage (unless attacker has Guts)."""
    is_burned = (status == jnp.int8(STATUS_BRN))
    is_physical = (category == jnp.int8(CATEGORY_PHYSICAL))
    apply = is_burned & is_physical & ~guts
    scaled = jnp.floor(damage.astype(jnp.float32) * jnp.float32(0.5)).astype(jnp.int32)
    return jnp.where(apply, scaled, damage)


def apply_screen_modifier(damage: jnp.ndarray, category: jnp.ndarray,
                           reflect_active: jnp.ndarray,
                           lightscreen_active: jnp.ndarray,
                           is_crit: jnp.ndarray) -> jnp.ndarray:
    """Reflect / Light Screen halve damage (negated by crits)."""
    is_physical = (category == jnp.int8(CATEGORY_PHYSICAL))
    is_special  = (category == jnp.int8(CATEGORY_SPECIAL))
    screen_up = jnp.where(is_physical, reflect_active, lightscreen_active)
    # Crits bypass screens
    apply = screen_up & ~is_crit
    scaled = jnp.floor(damage.astype(jnp.float32) * jnp.float32(0.5)).astype(jnp.int32)
    return jnp.where(apply, scaled, damage)


# ---------------------------------------------------------------------------
# Type effectiveness helper
# ---------------------------------------------------------------------------

def type_effectiveness(tables, move_type: jnp.ndarray,
                        def_type1: jnp.ndarray,
                        def_type2: jnp.ndarray) -> jnp.ndarray:
    """
    Combined type effectiveness for a move against a dual-typed Pokemon.

    Returns float32: 0.0, 0.25, 0.5, 1.0, 2.0, or 4.0.
    """
    mt  = jnp.asarray(move_type,  dtype=jnp.int32)
    dt1 = jnp.asarray(def_type1, dtype=jnp.int32)
    dt2 = jnp.asarray(def_type2, dtype=jnp.int32)
    eff1 = tables.type_chart[mt, dt1]
    eff2 = tables.type_chart[mt, dt2]
    # Sentinel: def_type2 == 0 means single-type, treat as 1.0
    eff2 = jnp.where(dt2 == jnp.int32(0), jnp.float32(1.0), eff2)
    return eff1 * eff2


# ---------------------------------------------------------------------------
# Crit determination
# ---------------------------------------------------------------------------

def get_crit_stage(move_crit_ratio: jnp.ndarray,
                   attacker_focus_energy: jnp.ndarray,
                   attacker_ability_id: jnp.ndarray,
                   attacker_item_id: jnp.ndarray,
                   # Ability/item IDs that boost crit
                   ABILITY_SUPER_LUCK: int = 0,   # placeholder
                   ITEM_SCOPE_LENS: int = 0,       # placeholder
                   ) -> jnp.ndarray:
    """
    Compute net crit stage (0-4).

    Stage 0: base (1/16)
    Stage 1: high crit moves, or +1 abilities/items  (1/8)
    Stage 2: focus energy or +2 abilities             (1/4)
    Stage 3: +3 (rare)                                (1/3 in Gen 4)
    Stage 4+: always crit
    """
    stage = move_crit_ratio.astype(jnp.int32)
    # Focus Energy: +2 in Gen 3+
    stage = jnp.where(attacker_focus_energy, stage + 2, stage)
    # Scope Lens / Sniper ability etc: +1 (we'll fill IDs from tables later)
    # stage = jnp.where(attacker_item_id == ITEM_SCOPE_LENS, stage + 1, stage)
    return jnp.clip(stage, 0, 4).astype(jnp.int32)


def roll_crit(key: jnp.ndarray, crit_stage: jnp.ndarray,
              will_crit: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Roll for critical hit. Returns (new_key, is_crit bool)."""
    key, subkey = rng_utils.split(key)
    rng_crit = rng_utils.critical_hit_roll(subkey, crit_stage)
    is_crit = rng_crit | will_crit.astype(jnp.bool_)
    return key, is_crit


# ---------------------------------------------------------------------------
# Full damage computation (single hit, no event hooks)
# ---------------------------------------------------------------------------

def compute_damage(
    state: BattleState,
    tables,
    atk_side: int,
    def_side: int,
    move_id: jnp.ndarray,
    key: jnp.ndarray,
    *,
    # Pre-computed overrides (used by event hooks later)
    base_power_override: jnp.ndarray | None = None,
    attack_override: jnp.ndarray | None = None,
    defense_override: jnp.ndarray | None = None,
    is_spread: jnp.ndarray = jnp.bool_(False),
    adaptability: jnp.ndarray = jnp.bool_(False),
    guts: jnp.ndarray = jnp.bool_(False),
    # Event relay multipliers (from ModifyAtk/SpA/Def, BasePower, ModifyDamage events)
    atk_relay: jnp.ndarray = jnp.float32(1.0),
    def_relay: jnp.ndarray = jnp.float32(1.0),
    bp_relay: jnp.ndarray = jnp.float32(1.0),
    damage_relay: jnp.ndarray = jnp.float32(1.0),
    # Gen-specific crit multiplier: 2.0 Gen 4-5, 1.5 Gen 6+ (pass cfg.crit_damage_multiplier)
    crit_multiplier: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute damage for one hit of a move.

    Returns:
        new_key:      updated PRNG key
        damage:       int32 final damage
        is_crit:      bool
        effectiveness: float32 type multiplier
    """
    move_data = tables.moves[move_id.astype(jnp.int32)]

    # Static move attributes
    base_power = move_data[MF_BASE_POWER].astype(jnp.int32)
    move_type  = move_data[MF_TYPE].astype(jnp.int32)
    category   = move_data[MF_CATEGORY]
    crit_ratio = move_data[MF_CRIT_RATIO].astype(jnp.int32)

    # Apply base power override (from events like Iron Fist, technician etc)
    if base_power_override is not None:
        base_power = base_power_override.astype(jnp.int32)

    # Apply BasePower event relay (Technician, Iron Fist, Reckless, etc.)
    base_power = jnp.where(
        bp_relay != jnp.float32(1.0),
        jnp.maximum(jnp.int32(1),
                    jnp.floor(base_power.astype(jnp.float32) * bp_relay).astype(jnp.int32)),
        base_power,
    )

    # Attacker stats
    atk_idx   = state.sides_active_idx[atk_side]
    atk_stats = state.sides_team_base_stats[atk_side, atk_idx]
    atk_boosts= state.sides_team_boosts[atk_side, atk_idx]
    atk_types = state.sides_team_types[atk_side, atk_idx]
    atk_status= state.sides_team_status[atk_side, atk_idx]
    atk_level = state.sides_team_level[atk_side, atk_idx]

    # Defender stats
    def_idx   = state.sides_active_idx[def_side]
    def_types = state.sides_team_types[def_side, def_idx]
    def_stats = state.sides_team_base_stats[def_side, def_idx]
    def_boosts= state.sides_team_boosts[def_side, def_idx]

    # Select attack / defense stat based on category
    is_physical = (category == jnp.int8(CATEGORY_PHYSICAL))
    is_special  = (category == jnp.int8(CATEGORY_SPECIAL))

    # Raw attack stat (boosted)
    raw_atk = jnp.where(is_physical,
                         atk_stats[STAT_ATK].astype(jnp.float32),
                         atk_stats[STAT_SPA].astype(jnp.float32))
    atk_boost = jnp.where(is_physical, atk_boosts[BOOST_ATK], atk_boosts[BOOST_SPA])
    # Crits ignore negative attack boosts
    # (handled below after we know is_crit)
    atk_mult  = tables.get_boost_multiplier(atk_boost)

    # Raw defense stat (boosted)
    raw_def = jnp.where(is_physical,
                         def_stats[STAT_DEF].astype(jnp.float32),
                         def_stats[STAT_SPD].astype(jnp.float32))
    def_boost = jnp.where(is_physical, def_boosts[BOOST_DEF], def_boosts[BOOST_SPD])
    def_mult  = tables.get_boost_multiplier(def_boost)

    attack  = jnp.floor(raw_atk * atk_mult * atk_relay).astype(jnp.int32)
    defense = jnp.maximum(jnp.int32(1),
                          jnp.floor(raw_def * def_mult * def_relay).astype(jnp.int32))

    if attack_override is not None:
        attack = attack_override.astype(jnp.int32)
    if defense_override is not None:
        defense = defense_override.astype(jnp.int32)

    # Crit roll
    focus_energy = (state.sides_team_volatiles[atk_side, atk_idx]
                    & jnp.uint32(1 << 21)) != jnp.uint32(0)  # VOL_FOCUSENERGY = 21
    key, is_crit = roll_crit(key, crit_ratio + jnp.int32(0), focus_energy)

    # Crits: use max(original, boosted) for attack, min(original, boosted) for defense
    raw_atk_int = jnp.floor(raw_atk).astype(jnp.int32)
    raw_def_int = jnp.maximum(jnp.int32(1), jnp.floor(raw_def).astype(jnp.int32))
    attack  = jnp.where(is_crit, jnp.maximum(attack, raw_atk_int), attack)
    defense = jnp.where(is_crit, jnp.minimum(defense, raw_def_int), defense)

    # Base damage
    dmg = base_damage(atk_level, base_power, attack, defense)

    # --- Modifier chain ---

    # 1. Spread
    dmg = apply_spread_modifier(dmg, is_spread)

    # 2. Weather
    dmg = apply_weather_modifier(dmg, jnp.int32(move_type), state.field.weather)

    # 3. Crit (gen-specific multiplier: 2.0 in Gen 4-5, 1.5 in Gen 6+)
    dmg = apply_crit_modifier(dmg, is_crit, crit_multiplier=crit_multiplier)

    # 4. Random roll
    key, roll_key = rng_utils.split(key)
    roll = rng_utils.damage_roll(roll_key)
    dmg = apply_random_modifier(dmg, roll)

    # 5. STAB
    dmg = apply_stab_modifier(dmg, jnp.int32(move_type), atk_types, adaptability)

    # 6. Type effectiveness
    effectiveness = type_effectiveness(
        tables,
        jnp.int32(move_type),
        def_types[0].astype(jnp.int32),
        def_types[1].astype(jnp.int32),
    )
    dmg = apply_type_modifier(dmg, effectiveness)

    # 7. Burn
    dmg = apply_burn_modifier(dmg, category, atk_status, guts)

    # 8. Screens  (Reflect/Light Screen)
    reflect_active = state.sides_side_conditions[def_side, 4] > jnp.int8(0)  # SC_REFLECT
    lightscreen_active = state.sides_side_conditions[def_side, 5] > jnp.int8(0)  # SC_LIGHTSCREEN
    dmg = apply_screen_modifier(dmg, category, reflect_active, lightscreen_active, is_crit)

    # 9. ModifyDamage relay (Life Orb, Expert Belt, Sniper, Tinted Lens, etc.)
    dmg = jnp.where(
        damage_relay != jnp.float32(1.0),
        jnp.floor(dmg.astype(jnp.float32) * damage_relay).astype(jnp.int32),
        dmg,
    )

    # Minimum 1 damage (if move would do damage but rolled 0)
    is_damaging = base_power > jnp.int32(0)
    dmg = jnp.where(is_damaging & (effectiveness > jnp.float32(0.0)),
                    jnp.maximum(jnp.int32(1), dmg),
                    dmg)

    return key, dmg.astype(jnp.int32), is_crit, effectiveness


# ---------------------------------------------------------------------------
# HP change helpers
# ---------------------------------------------------------------------------

def apply_damage(state: BattleState, side: int, slot: int,
                 damage: jnp.ndarray) -> BattleState:
    """Subtract damage from a Pokemon's HP, flooring at 0."""
    current = state.sides_team_hp[side, slot]
    new_hp = jnp.maximum(jnp.int16(0), (current - damage.astype(jnp.int16)))
    new_hp_arr = state.sides_team_hp.at[side, slot].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr)


def apply_heal(state: BattleState, side: int, slot: int,
               heal_amount: jnp.ndarray) -> BattleState:
    """Add HP up to max_hp."""
    current = state.sides_team_hp[side, slot]
    max_hp  = state.sides_team_max_hp[side, slot]
    new_hp  = jnp.minimum(max_hp, current + heal_amount.astype(jnp.int16))
    new_hp_arr = state.sides_team_hp.at[side, slot].set(new_hp)
    return state._replace(sides_team_hp=new_hp_arr)


def fraction_of_max_hp(state: BattleState, side: int, slot: int,
                        numerator: int, denominator: int) -> jnp.ndarray:
    """Compute numerator/denominator * max_hp as int32 (minimum 1)."""
    max_hp = state.sides_team_max_hp[side, slot].astype(jnp.int32)
    return jnp.maximum(jnp.int32(1), max_hp * numerator // denominator)


def is_fainted(state: BattleState, side: int, slot: int) -> jnp.ndarray:
    return state.sides_team_hp[side, slot] <= jnp.int16(0)
