"""
NamedTuple state definitions for PokeJAX.

All state is represented as nested NamedTuples of JAX arrays so the entire
battle state is a valid JAX pytree — enabling jit, vmap, and lax.scan.

Array shapes are FIXED at compile time. Sentinel values:
  -1  = unused / absent (IDs)
   0  = no item / no extra type (IDs where 0 is a sentinel)
"""

from typing import NamedTuple
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Integer constants (used as static values, not JAX arrays)
# ---------------------------------------------------------------------------

MAX_TEAM_SIZE   = 6
MAX_MOVES       = 4
MAX_ACTIVE      = 1   # singles; doubles would be 2
MAX_VOLATILES   = 32  # padded volatile counter slots
MAX_SIDE_CONDS  = 10  # spikes, toxicspikes, stealthrock, stickyweb,
                       # reflect, lightscreen, auroraveil, tailwind,
                       # safeguard, mist

N_STATS    = 6   # hp, atk, def, spa, spd, spe
N_BOOSTS   = 7   # atk, def, spa, spd, spe, accuracy, evasion
N_TYPES    = 19  # 18 named types + sentinel "???" (index 0)

# ---------------------------------------------------------------------------
# Status codes  (stored as int8)
# ---------------------------------------------------------------------------
STATUS_NONE  = jnp.int8(0)
STATUS_BRN   = jnp.int8(1)  # burn
STATUS_PSN   = jnp.int8(2)  # poison
STATUS_TOX   = jnp.int8(3)  # badly poisoned (toxic)
STATUS_SLP   = jnp.int8(4)  # sleep
STATUS_FRZ   = jnp.int8(5)  # freeze
STATUS_PAR   = jnp.int8(6)  # paralysis

# ---------------------------------------------------------------------------
# Weather codes  (stored as int8)
# ---------------------------------------------------------------------------
WEATHER_NONE  = jnp.int8(0)
WEATHER_SUN   = jnp.int8(1)  # harsh sunlight
WEATHER_RAIN  = jnp.int8(2)
WEATHER_SAND  = jnp.int8(3)  # sandstorm
WEATHER_HAIL  = jnp.int8(4)  # hail / snow (gen 9+)

# ---------------------------------------------------------------------------
# Terrain codes  (stored as int8)
# ---------------------------------------------------------------------------
TERRAIN_NONE     = jnp.int8(0)
TERRAIN_ELECTRIC = jnp.int8(1)
TERRAIN_GRASSY   = jnp.int8(2)
TERRAIN_MISTY    = jnp.int8(3)
TERRAIN_PSYCHIC  = jnp.int8(4)

# ---------------------------------------------------------------------------
# Type indices  (index into type_chart rows/columns)
# ---------------------------------------------------------------------------
TYPE_NONE    = 0   # sentinel / typeless
TYPE_NORMAL  = 1
TYPE_FIRE    = 2
TYPE_WATER   = 3
TYPE_ELECTRIC= 4
TYPE_GRASS   = 5
TYPE_ICE     = 6
TYPE_FIGHTING= 7
TYPE_POISON  = 8
TYPE_GROUND  = 9
TYPE_FLYING  = 10
TYPE_PSYCHIC = 11
TYPE_BUG     = 12
TYPE_ROCK    = 13
TYPE_GHOST   = 14
TYPE_DRAGON  = 15
TYPE_DARK    = 16
TYPE_STEEL   = 17
TYPE_FAIRY   = 18  # Gen 6+; in Gen 4/5 this slot is unused

TYPE_NAMES = [
    "???", "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

# ---------------------------------------------------------------------------
# Move category codes  (stored as int8)
# ---------------------------------------------------------------------------
CATEGORY_PHYSICAL = jnp.int8(0)
CATEGORY_SPECIAL  = jnp.int8(1)
CATEGORY_STATUS   = jnp.int8(2)

# ---------------------------------------------------------------------------
# Gender codes  (stored as int8)
# ---------------------------------------------------------------------------
GENDER_GENDERLESS = jnp.int8(0)
GENDER_MALE       = jnp.int8(1)
GENDER_FEMALE     = jnp.int8(2)

# ---------------------------------------------------------------------------
# Stat indices  (into base_stats / boosts arrays)
# ---------------------------------------------------------------------------
STAT_HP  = 0
STAT_ATK = 1
STAT_DEF = 2
STAT_SPA = 3
STAT_SPD = 4
STAT_SPE = 5

BOOST_ATK = 0
BOOST_DEF = 1
BOOST_SPA = 2
BOOST_SPD = 3
BOOST_SPE = 4
BOOST_ACC = 5
BOOST_EVA = 6

# ---------------------------------------------------------------------------
# Side condition indices  (into side_conditions array)
# ---------------------------------------------------------------------------
SC_SPIKES      = 0  # layer count (0-3)
SC_TOXICSPIKES = 1  # layer count (0-2)
SC_STEALTHROCK = 2  # 0 or 1
SC_STICKYWEB   = 3  # 0 or 1
SC_REFLECT     = 4  # turns remaining
SC_LIGHTSCREEN = 5  # turns remaining
SC_AURORAVEIL  = 6  # turns remaining
SC_TAILWIND    = 7  # turns remaining
SC_SAFEGUARD   = 8  # turns remaining
SC_MIST        = 9  # turns remaining

# ---------------------------------------------------------------------------
# Volatile status bit indices  (into pokemon.volatiles uint32 bitmask)
# The volatile_data[i] field holds per-volatile counter/state.
# ---------------------------------------------------------------------------
VOL_CONFUSED      = 0   # data = turns remaining
VOL_FLINCH        = 1
VOL_PARTIALLY_TRAPPED = 2  # data = turns remaining
VOL_SEEDED        = 3   # leech seed
VOL_SUBSTITUTE    = 4   # data = substitute HP (stored as fraction * 255)
VOL_PROTECT       = 5   # data = consecutive protect count (for fail rate)
VOL_ENCORE        = 6   # data = turns remaining
VOL_TAUNT         = 7   # data = turns remaining
VOL_TORMENT       = 8
VOL_DISABLE       = 9   # data = turns remaining
VOL_ENDURE        = 10
VOL_MAGICCOAT     = 11
VOL_SNATCH        = 12
VOL_INGRAIN       = 13
VOL_AQUARINGTARGET= 14
VOL_HEALBLOCK     = 15  # data = turns remaining
VOL_EMBARGO       = 16  # data = turns remaining
VOL_CHARGING      = 17  # two-turn move charging (fly/dig/dive etc.)
VOL_RECHARGING    = 18  # must recharge (hyper beam etc.)
VOL_LOCKEDMOVE    = 19  # data = turns remaining (outrage etc.)
VOL_CHOICELOCK    = 20  # locked to a move by Choice item; data = move slot
VOL_FOCUSENERGY   = 21
VOL_MINIMIZED     = 22
VOL_CURSE         = 23  # ghost curse
VOL_NIGHTMARE     = 24
VOL_ATTRACT       = 25  # data = side of infatuating Pokemon
VOL_YAWN          = 26  # data = turns until sleep
VOL_DESTINYBOND   = 27
VOL_GRUDGE        = 28
VOL_POWEROFALOOFONE = 29  # placeholder for future
VOL_PERISH        = 30  # perish song counter (data = turns remaining, faints at 0)
VOL_FLASH_FIRE    = 31  # Flash Fire boost active (set when absorbing a Fire move)

# ---------------------------------------------------------------------------
# State NamedTuples
# ---------------------------------------------------------------------------

class PokemonState(NamedTuple):
    """Per-Pokemon battle state. All scalars are 0-d JAX arrays."""

    # Identity
    species_id:      jnp.ndarray  # int16 scalar — index into species table, -1 = empty
    ability_id:      jnp.ndarray  # int16 scalar
    item_id:         jnp.ndarray  # int16 scalar, 0 = no item

    # Typing (2 slots; second = TYPE_NONE if single-type)
    types:           jnp.ndarray  # int8[2]

    # Base stats (computed once at team gen, never modified)
    base_stats:      jnp.ndarray  # int16[6]  — hp, atk, def, spa, spd, spe

    # Current/max HP
    hp:              jnp.ndarray  # int16 scalar
    max_hp:          jnp.ndarray  # int16 scalar

    # Stat boosts  [-6, +6] encoded as int8
    boosts:          jnp.ndarray  # int8[7]   — atk, def, spa, spd, spe, acc, eva

    # Moves (4 slots)
    move_ids:        jnp.ndarray  # int16[4]  — -1 = empty
    move_pp:         jnp.ndarray  # int8[4]
    move_max_pp:     jnp.ndarray  # int8[4]
    move_disabled:   jnp.ndarray  # bool[4]

    # Non-volatile status
    status:          jnp.ndarray  # int8 scalar
    status_turns:    jnp.ndarray  # int8 scalar — toxic counter or misc duration
    sleep_turns:     jnp.ndarray  # int8 scalar — sleep duration (0 when not asleep)

    # Volatile statuses: bitmask presence + per-volatile counters
    volatiles:       jnp.ndarray  # uint32 scalar — bit i set ↔ volatile i active
    volatile_data:   jnp.ndarray  # int8[MAX_VOLATILES] — per-volatile counter/state

    # Combat tracking
    is_active:       jnp.ndarray  # bool scalar
    fainted:         jnp.ndarray  # bool scalar
    last_move_id:    jnp.ndarray  # int16 scalar — -1 if none
    move_this_turn:  jnp.ndarray  # bool scalar — has moved this turn
    times_attacked:  jnp.ndarray  # int8 scalar — times hit this turn

    # Static attributes (set at team gen)
    level:           jnp.ndarray  # int8 scalar
    gender:          jnp.ndarray  # int8 scalar
    nature_id:       jnp.ndarray  # int8 scalar  (0-24)
    weight_hg:       jnp.ndarray  # int16 scalar — weight in hectograms (kg * 10)

    # Forme tracking
    base_species_id: jnp.ndarray  # int16 scalar — original species (pre-transform)


class SideState(NamedTuple):
    """Per-player (side) battle state."""

    # Team storage — each field is shape (MAX_TEAM_SIZE, *pokemon_field_shape)
    # i.e., we store a "batch" of PokemonState along axis 0
    # Access: get_pokemon(side, idx) returns a PokemonState with scalar fields
    team_species_id:    jnp.ndarray  # int16[6]
    team_ability_id:    jnp.ndarray  # int16[6]
    team_item_id:       jnp.ndarray  # int16[6]
    team_types:         jnp.ndarray  # int8[6, 2]
    team_base_stats:    jnp.ndarray  # int16[6, 6]
    team_hp:            jnp.ndarray  # int16[6]
    team_max_hp:        jnp.ndarray  # int16[6]
    team_boosts:        jnp.ndarray  # int8[6, 7]
    team_move_ids:      jnp.ndarray  # int16[6, 4]
    team_move_pp:       jnp.ndarray  # int8[6, 4]
    team_move_max_pp:   jnp.ndarray  # int8[6, 4]
    team_move_disabled: jnp.ndarray  # bool[6, 4]
    team_status:        jnp.ndarray  # int8[6]
    team_status_turns:  jnp.ndarray  # int8[6]
    team_sleep_turns:   jnp.ndarray  # int8[6]
    team_volatiles:     jnp.ndarray  # uint32[6]
    team_volatile_data: jnp.ndarray  # int8[6, MAX_VOLATILES]
    team_is_active:     jnp.ndarray  # bool[6]
    team_fainted:       jnp.ndarray  # bool[6]
    team_last_move_id:  jnp.ndarray  # int16[6]
    team_move_this_turn:jnp.ndarray  # bool[6]
    team_times_attacked:jnp.ndarray  # int8[6]
    team_level:         jnp.ndarray  # int8[6]
    team_gender:        jnp.ndarray  # int8[6]
    team_nature_id:     jnp.ndarray  # int8[6]
    team_weight_hg:     jnp.ndarray  # int16[6]
    team_base_species_id: jnp.ndarray  # int16[6]

    # Active Pokemon index
    active_idx:         jnp.ndarray  # int8 scalar

    # Number of non-fainted Pokemon remaining
    pokemon_left:       jnp.ndarray  # int8 scalar

    # Side conditions: layer count or turns remaining per condition
    side_conditions:    jnp.ndarray  # int8[MAX_SIDE_CONDS]

    # Dynamax flag (Gen 8+); 0 = not dynamaxed, >0 = turns remaining
    dynamax_turns:      jnp.ndarray  # int8 scalar


class FieldState(NamedTuple):
    """Global battlefield state."""

    weather:           jnp.ndarray  # int8 scalar
    weather_turns:     jnp.ndarray  # int8 scalar — 0 = permanent / inactive
    weather_max_turns: jnp.ndarray  # int8 scalar — for reset on weather change

    terrain:           jnp.ndarray  # int8 scalar
    terrain_turns:     jnp.ndarray  # int8 scalar

    trick_room:        jnp.ndarray  # int8 scalar — turns remaining, 0 = off
    gravity:           jnp.ndarray  # int8 scalar
    magic_room:        jnp.ndarray  # int8 scalar
    wonder_room:       jnp.ndarray  # int8 scalar


class BattleState(NamedTuple):
    """Top-level battle state. The root pytree passed through jit/vmap/scan."""

    # Two sides: each field has a leading dimension of 2
    # e.g., sides_active_idx[0] = P1's active index
    sides_team_species_id:    jnp.ndarray  # int16[2, 6]
    sides_team_ability_id:    jnp.ndarray  # int16[2, 6]
    sides_team_item_id:       jnp.ndarray  # int16[2, 6]
    sides_team_types:         jnp.ndarray  # int8[2, 6, 2]
    sides_team_base_stats:    jnp.ndarray  # int16[2, 6, 6]
    sides_team_hp:            jnp.ndarray  # int16[2, 6]
    sides_team_max_hp:        jnp.ndarray  # int16[2, 6]
    sides_team_boosts:        jnp.ndarray  # int8[2, 6, 7]
    sides_team_move_ids:      jnp.ndarray  # int16[2, 6, 4]
    sides_team_move_pp:       jnp.ndarray  # int8[2, 6, 4]
    sides_team_move_max_pp:   jnp.ndarray  # int8[2, 6, 4]
    sides_team_move_disabled: jnp.ndarray  # bool[2, 6, 4]
    sides_team_status:        jnp.ndarray  # int8[2, 6]
    sides_team_status_turns:  jnp.ndarray  # int8[2, 6]
    sides_team_sleep_turns:   jnp.ndarray  # int8[2, 6]
    sides_team_volatiles:     jnp.ndarray  # uint32[2, 6]
    sides_team_volatile_data: jnp.ndarray  # int8[2, 6, MAX_VOLATILES]
    sides_team_is_active:     jnp.ndarray  # bool[2, 6]
    sides_team_fainted:       jnp.ndarray  # bool[2, 6]
    sides_team_last_move_id:  jnp.ndarray  # int16[2, 6]
    sides_team_move_this_turn:jnp.ndarray  # bool[2, 6]
    sides_team_times_attacked:jnp.ndarray  # int8[2, 6]
    sides_team_level:         jnp.ndarray  # int8[2, 6]
    sides_team_gender:        jnp.ndarray  # int8[2, 6]
    sides_team_nature_id:     jnp.ndarray  # int8[2, 6]
    sides_team_weight_hg:     jnp.ndarray  # int16[2, 6]
    sides_team_base_species_id: jnp.ndarray  # int16[2, 6]

    sides_active_idx:         jnp.ndarray  # int8[2]
    sides_pokemon_left:       jnp.ndarray  # int8[2]
    sides_side_conditions:    jnp.ndarray  # int8[2, MAX_SIDE_CONDS]
    sides_dynamax_turns:      jnp.ndarray  # int8[2]

    # Number of full turns each Pokemon has been active (incremented at end of turn).
    # Speed Boost skips activation on turn 0 (the turn the Pokemon switched in).
    # Reset to 0 on switch-in.
    sides_team_active_turns:  jnp.ndarray  # int8[2, 6]

    # Tracks last physical/special damage taken by the ACTIVE Pokemon this turn
    # Used for Counter (2x last phys), Mirror Coat (2x last spec), Metal Burst (1.5x last)
    sides_last_dmg_phys:      jnp.ndarray  # int16[2]
    sides_last_dmg_spec:      jnp.ndarray  # int16[2]

    # Wish: pending delayed heal per side.
    # sides_wish_turns[s] = 0: no wish pending; 1: heals this end-of-turn.
    # sides_wish_hp[s]: amount to heal (half max HP of the Pokemon that used Wish).
    sides_wish_turns:         jnp.ndarray  # int8[2]
    sides_wish_hp:            jnp.ndarray  # int16[2]

    field:                    FieldState

    # Turn counter and result
    turn:                     jnp.ndarray  # int16 scalar
    finished:                 jnp.ndarray  # bool scalar
    winner:                   jnp.ndarray  # int8 scalar — -1=none, 0=P1, 1=P2, 2=draw

    # JAX PRNG key
    rng_key:                  jnp.ndarray  # uint32[2]


# ---------------------------------------------------------------------------
# Information-masking state
# Tracks what each player has *seen* from the opponent (and themselves).
# Updated by execute_turn via jnp.where — fully JIT/vmap compatible.
#
# Axis 0: player perspective (0=P1 sees, 1=P2 sees)
# Axis 1: team slot index (0-5)
# Axis 2 (moves only): move slot (0-3)
#
# A Pokemon's active slot is always "revealed" to the opponent once it
# has been sent out (revealed_pokemon).  Moves are revealed only when
# the opponent actually uses them.  Ability/item are revealed only when
# triggered.
# ---------------------------------------------------------------------------

class RevealState(NamedTuple):
    """Per-player visibility of opponent (and own) battle information."""

    # Whether each move has been seen by each player.
    # revealed_moves[p, slot, move] = True  ↔  player p has seen that move.
    # Own moves are always True at init; opponent moves start False.
    revealed_moves:   jnp.ndarray  # bool[2, 6, 4]

    # Whether each team slot has been revealed (sent out at least once).
    # Active Pokemon on both sides are True at init.
    revealed_pokemon: jnp.ndarray  # bool[2, 6]

    # Whether each Pokemon's ability has been triggered/revealed.
    revealed_ability: jnp.ndarray  # bool[2, 6]

    # Whether each Pokemon's item has been triggered/revealed.
    revealed_item:    jnp.ndarray  # bool[2, 6]
