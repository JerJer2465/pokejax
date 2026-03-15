"""
JAX observation builder for PokeJAX RL.

Converts (BattleState, RevealState, player, tables) → token arrays for
PokeTransformer, matching the PokemonShowdownClaude format exactly.

Token layout (N_TOKENS=15):
  Token 0:     FIELD      (84-dim float, rest 0)
  Tokens 1–6:  OWN_TEAM[0–5]   (slot 0 = active)
  Tokens 7–12: OPP_TEAM[0–5]   (slot 0 = active; unrevealed = masked)
  Token 13:    ACTOR query     (all zeros)
  Token 14:    CRITIC query    (all zeros)

Output dict keys:
  "int_ids"    : jnp.ndarray (15, 8)      species, 4×move, ability, item, last_move
  "float_feats": jnp.ndarray (15, 394)
  "legal_mask" : jnp.ndarray (10,)         bool

Information masking via RevealState:
  Opponent's unrevealed move  → UNKNOWN_MOVE_IDX, pp_frac=1.0, is_known=0.0
  Opponent's unrevealed poke  → all features zeroed (unknown slot)
  Opponent's unrevealed ability / item → UNKNOWN index

All operations are branchless jnp.where so this function is fully
jit- and vmap-compatible (player must be a Python-level constant).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pokejax.types import (
    BattleState, RevealState,
    STATUS_NONE, STATUS_BRN, STATUS_PSN, STATUS_TOX,
    STATUS_SLP, STATUS_FRZ, STATUS_PAR,
    SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
    SC_REFLECT, SC_LIGHTSCREEN, SC_TAILWIND, SC_SAFEGUARD,
    VOL_CONFUSED, VOL_ATTRACT, VOL_SEEDED, VOL_CURSE,
    VOL_AQUARINGTARGET, VOL_INGRAIN, VOL_TAUNT, VOL_ENCORE,
    VOL_FLINCH, VOL_EMBARGO, VOL_HEALBLOCK,
    VOL_PARTIALLY_TRAPPED, VOL_SUBSTITUTE, VOL_YAWN,
    VOL_FOCUSENERGY, VOL_CHARGING, VOL_RECHARGING,
    VOL_TORMENT, VOL_NIGHTMARE, VOL_DESTINYBOND, VOL_GRUDGE, VOL_PERISH,
    WEATHER_NONE, WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_HAIL,
    TERRAIN_NONE, TERRAIN_ELECTRIC, TERRAIN_GRASSY, TERRAIN_MISTY, TERRAIN_PSYCHIC,
    MAX_VOLATILES,
)

# ---------------------------------------------------------------------------
# Constants (must match PokemonShowdownClaude exactly)
# ---------------------------------------------------------------------------

INT_IDS_PER_TOKEN  = 8
N_TOKENS           = 15
N_TEAM_SLOTS       = 6
N_MOVES_PER_MON    = 4
N_ACTIONS          = 10   # 4 moves + 6 switches

UNKNOWN_SPECIES_IDX  = 0
UNKNOWN_MOVE_IDX     = 0
EMPTY_MOVE_IDX       = 1
UNKNOWN_ABILITY_IDX  = 0
UNKNOWN_ITEM_IDX     = 0

# pokejax type index 0 = TYPE_NONE sentinel; PS uses 1-based types starting at Normal=1
# PS type chart: 0=Normal, 1=Fire, …, 17=Fairy (18 types)
# pokejax types: 0=sentinel, 1=Normal, 2=Fire, …, 18=Fairy
# Subtract 1 to convert pokejax → PS (clamp at 0 for sentinel)
_N_PS_TYPES = 18  # as in PokemonShowdownClaude

# ---------------------------------------------------------------------------
# Volatile mapping: pokejax bit index → PS volatile_index (27 dims)
# -1 means no mapping (bit not represented in the 27-dim PS encoding)
#
# PokemonShowdownClaude VOLATILE_STATUS_LIST (N_VOLATILE=27):
#  0:confusion 1:infatuation 2:leechseed 3:curse 4:aquaring 5:ingrain
#  6:taunt 7:encore 8:flinch 9:embargo 10:healblock 11:magnetrise
#  12:partiallytrapped 13:perishsong 14:powertrick 15:substitute 16:yawn
#  17:focusenergy 18:charge 19:stockpile 20:torment 21:nightmare
#  22:imprison 23:mustrecharge 24:twoturnmove 25:destinybond 26:grudge
# ---------------------------------------------------------------------------
N_VOLATILE = 27

_VOL_MAP = np.full(MAX_VOLATILES, -1, dtype=np.int8)
_VOL_MAP[VOL_CONFUSED]          = 0   # confusion
_VOL_MAP[VOL_ATTRACT]           = 1   # infatuation
_VOL_MAP[VOL_SEEDED]            = 2   # leechseed
_VOL_MAP[VOL_CURSE]             = 3   # curse
_VOL_MAP[VOL_AQUARINGTARGET]    = 4   # aquaring
_VOL_MAP[VOL_INGRAIN]           = 5   # ingrain
_VOL_MAP[VOL_TAUNT]             = 6   # taunt
_VOL_MAP[VOL_ENCORE]            = 7   # encore
_VOL_MAP[VOL_FLINCH]            = 8   # flinch
_VOL_MAP[VOL_EMBARGO]           = 9   # embargo
_VOL_MAP[VOL_HEALBLOCK]         = 10  # healblock
_VOL_MAP[VOL_PARTIALLY_TRAPPED] = 12  # partiallytrapped
_VOL_MAP[VOL_SUBSTITUTE]        = 15  # substitute
_VOL_MAP[VOL_YAWN]              = 16  # yawn
_VOL_MAP[VOL_FOCUSENERGY]       = 17  # focusenergy
_VOL_MAP[VOL_CHARGING]          = 18  # charge/twoturnmove
_VOL_MAP[VOL_RECHARGING]        = 23  # mustrecharge
_VOL_MAP[VOL_TORMENT]           = 20  # torment
_VOL_MAP[VOL_NIGHTMARE]         = 21  # nightmare
_VOL_MAP[VOL_DESTINYBOND]       = 25  # destinybond
_VOL_MAP[VOL_GRUDGE]            = 26  # grudge

# Precompute as JAX constant array (shape [32], int8, value = PS volatile idx or -1)
_VOL_MAP_JAX = jnp.array(_VOL_MAP, dtype=jnp.int8)

# Build a [32, 27] boolean matrix: _VOL_ONEHOT[bit, ps_idx] = 1 if mapped
_VOL_ONEHOT = np.zeros((MAX_VOLATILES, N_VOLATILE), dtype=np.float32)
for _bit, _ps in enumerate(_VOL_MAP):
    if _ps >= 0:
        _VOL_ONEHOT[_bit, _ps] = 1.0
_VOL_ONEHOT_JAX = jnp.array(_VOL_ONEHOT, dtype=jnp.float32)  # [32, 27]

# ---------------------------------------------------------------------------
# Float feature dimensions (must match PokemonShowdownClaude _OFF dict exactly)
# ---------------------------------------------------------------------------

FLOAT_DIM_PER_POKEMON = 394
FIELD_DIM = 84

# Per-pokemon offsets (accumulated from _MON_OFFSET_SPEC in PS obs_builder)
_OFF_HP_FRAC    = 0
_OFF_HP_BIN     = 1    # 10 dims
_OFF_BASE_STATS = 11   # 6 dims
_OFF_BOOSTS     = 17   # 91 dims  (7 stats × 13 levels)
_OFF_STATUS     = 108  # 7 dims
_OFF_VOLATILE   = 115  # 27 dims
_OFF_TYPE1      = 142  # 18 dims
_OFF_TYPE2      = 160  # 18 dims
_OFF_IS_FAINTED = 178  # 1 dim
_OFF_IS_ACTIVE  = 179  # 1 dim
_OFF_SLOT       = 180  # 6 dims
_OFF_IS_OWN     = 186  # 1 dim
_OFF_MOVES      = 187  # 4×45 = 180 dims
_OFF_SLEEP_BIN  = 367  # 4 dims
_OFF_REST_BIN   = 371  # 3 dims
_OFF_SUB_FRAC   = 374  # 1 dim
_OFF_FORCE_TRAP = 375  # 1 dim
_OFF_MOV_DIS    = 376  # 4 dims
_OFF_CONF_BIN   = 380  # 4 dims
_OFF_TAUNT      = 384  # 1 dim
_OFF_ENCORE     = 385  # 1 dim
_OFF_YAWN       = 386  # 1 dim
_OFF_LEVEL      = 387  # 1 dim
_OFF_PERISH_BIN = 388  # 4 dims
_OFF_PROTECT    = 392  # 1 dim
_OFF_LOCKED_MOV = 393  # 1 dim

assert _OFF_LOCKED_MOV + 1 == FLOAT_DIM_PER_POKEMON, \
    f"Float dim mismatch: expected {FLOAT_DIM_PER_POKEMON}"

# Per-move sub-offsets within the 45-dim block
_MOFF_BP_BIN   = 0   # 8 dims
_MOFF_ACC_BIN  = 8   # 6 dims
_MOFF_TYPE     = 14  # 18 dims
_MOFF_CAT      = 32  # 3 dims
_MOFF_PRI      = 35  # 8 dims
_MOFF_PP       = 43  # 1 dim
_MOFF_KNOWN    = 44  # 1 dim

# Field offsets
_FOFF_WEATHER      = 0   # 5 dims
_FOFF_WT_TURNS     = 5   # 8 dims
_FOFF_PSEUDO       = 13  # 5 dims  (trick_room, gravity, wonder_room, ...)
_FOFF_TR_TURNS     = 18  # 4 dims
_FOFF_HAZARDS_OWN  = 22  # 7 dims
_FOFF_HAZARDS_OPP  = 29  # 7 dims
_FOFF_SCREENS_OWN  = 36  # 6 dims
_FOFF_SCREENS_OPP  = 42  # 6 dims
_FOFF_TURN_BIN     = 48  # 10 dims
_FOFF_FAINTED      = 58  # 2 dims
_FOFF_TOXIC_OWN    = 60  # 5 dims
_FOFF_TOXIC_OPP    = 65  # 5 dims
_FOFF_TAILWIND     = 70  # 2 dims
_FOFF_WISH         = 72  # 2 dims  (pokejax has no wish yet → 0)
_FOFF_SAFEGUARD    = 74  # 2 dims
_FOFF_MIST         = 76  # 2 dims
_FOFF_LUCKY_CHANT  = 78  # 2 dims  (pokejax has no lucky_chant → 0)
_FOFF_GRAVITY_TURNS = 80 # 4 dims

assert _FOFF_GRAVITY_TURNS + 4 == FIELD_DIM, \
    f"Field dim mismatch: expected {FIELD_DIM}"

# ---------------------------------------------------------------------------
# Binning threshold arrays (precomputed as JAX constants)
# ---------------------------------------------------------------------------

_BP_THRESHOLDS  = jnp.array([0, 1, 41, 61, 81, 101, 121, 151], dtype=jnp.int32)
_ACC_THRESHOLDS = jnp.array([0, 50, 70, 80, 90, 100], dtype=jnp.int32)
_HP_THRESHOLDS  = jnp.array([0, 10, 20, 33, 50, 66, 75, 88, 100], dtype=jnp.int32)
                  # 0–9%, 10–19%, 20–32%, 33–49%, 50–65%, 66–74%, 75–87%, 88–99%, 100%
_TURN_THRESHOLDS = jnp.array([1, 2, 4, 6, 9, 13, 18, 25, 35], dtype=jnp.int32)
# priority range: -3 to +4 (8 bins), encoded as priority+3 clamped to 0..7
_PRIORITY_MIN = -3


# ---------------------------------------------------------------------------
# Pure JAX helper: one-hot from index
# ---------------------------------------------------------------------------

def _onehot(idx: jnp.ndarray, n: int) -> jnp.ndarray:
    """n-dim one-hot from scalar int index (out-of-range → all zeros)."""
    return (jnp.arange(n, dtype=jnp.int32) == idx.astype(jnp.int32)).astype(jnp.float32)


def _bin_idx(val: jnp.ndarray, thresholds: jnp.ndarray) -> jnp.ndarray:
    """Return number of thresholds <= val, minus 1, clamped to [0, len-1]."""
    idx = jnp.sum(val.astype(jnp.int32) >= thresholds).astype(jnp.int32) - 1
    return jnp.clip(idx, 0, thresholds.shape[0] - 1)


def _bin_onehot(val: jnp.ndarray, thresholds: jnp.ndarray) -> jnp.ndarray:
    """One-hot bin encoding of val."""
    n = thresholds.shape[0]
    return _onehot(_bin_idx(val, thresholds), n)


# ---------------------------------------------------------------------------
# Move feature encoder (returns 45-dim float vector + int move id)
# ---------------------------------------------------------------------------

def _encode_move(
    move_id: jnp.ndarray,   # int16 scalar  (from state.sides_team_move_ids)
    pp: jnp.ndarray,        # int8 scalar
    max_pp: jnp.ndarray,    # int8 scalar
    is_known: jnp.ndarray,  # bool scalar
    tables,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encode one move slot into (int_id scalar, float_feats 45-dim).

    Unknown/unrevealed moves are encoded as zeros + pp_frac=1.0 + is_known=0.
    Empty move slots (move_id < 0) similarly zeroed.
    """
    # Clamp move_id for safe lookup (tables.moves has move 0 = sentinel/splash)
    safe_id = jnp.clip(move_id.astype(jnp.int32), 0, tables.moves.shape[0] - 1)
    row = tables.moves[safe_id]          # int16[22]

    bp       = row[0].astype(jnp.int32)  # base power
    acc      = row[1].astype(jnp.int32)  # accuracy (101 = always hits → bin 0)
    type_id  = row[2].astype(jnp.int32)  # pokejax type index (1-based)
    category = row[3].astype(jnp.int32)  # 0=physical, 1=special, 2=status
    priority = row[4].astype(jnp.int32)  # -3..+4

    # Convert pokejax type index → PS type index (PS uses 0-based: Normal=0)
    ps_type = jnp.clip(type_id - 1, 0, _N_PS_TYPES - 1)

    bp_bin  = _bin_onehot(bp, _BP_THRESHOLDS)   # 8 dims
    # Accuracy: 101 → always hits (treat as bin 0 = bypass accuracy)
    acc_clamped = jnp.where(acc > 100, jnp.int32(0), acc)
    acc_bin = _bin_onehot(acc_clamped, _ACC_THRESHOLDS)  # 6 dims
    type_oh = _onehot(ps_type, _N_PS_TYPES)              # 18 dims
    cat_oh  = _onehot(category, 3)                        # 3 dims
    pri_idx = jnp.clip(priority - _PRIORITY_MIN, 0, 7)
    pri_oh  = _onehot(pri_idx, 8)                         # 8 dims

    pp_safe   = jnp.maximum(pp.astype(jnp.int32), 0)
    max_pp_safe = jnp.maximum(max_pp.astype(jnp.int32), 1)
    pp_frac = pp_safe.astype(jnp.float32) / max_pp_safe.astype(jnp.float32)

    # Build full 45-dim vector for a known move
    known_feats = jnp.concatenate([
        bp_bin, acc_bin, type_oh, cat_oh, pri_oh,
        jnp.array([pp_frac, 1.0], dtype=jnp.float32),
    ])  # shape (45,)

    # Unknown / masked move: zeros except pp_frac=1.0, is_known=0.0
    unknown_feats = jnp.zeros(45, dtype=jnp.float32).at[_MOFF_PP].set(1.0)

    feats = jnp.where(is_known, known_feats, unknown_feats)

    # Integer id: masked if unknown
    valid_move = (move_id >= jnp.int16(0)) & is_known
    int_id = jnp.where(valid_move, safe_id, jnp.int32(UNKNOWN_MOVE_IDX))

    return int_id, feats


# ---------------------------------------------------------------------------
# Pokemon token encoder
# ---------------------------------------------------------------------------

def _encode_pokemon(
    state: BattleState,
    s: int,           # side: Python int (compile-time)
    slot: int,        # team slot: Python int (compile-time)
    is_own: bool,     # Python bool (compile-time)
    is_active: bool,  # Python bool (compile-time)
    player: int,      # observer: Python int (compile-time)
    reveal: RevealState,
    tables,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encode one Pokemon slot into (int_ids [8], float_feats [394]).

    For opponent slots: features are masked via RevealState.
    For own slots: always fully visible.
    Returns (int_ids, float_feats).
    """
    opp = 1 - player  # Python int

    # Whether the observer can see this Pokemon at all
    if is_own:
        is_known_poke = jnp.bool_(True)
    else:
        # revealed_pokemon[player, slot] means player can see opp slot `slot`
        is_known_poke = reveal.revealed_pokemon[player, slot]

    # ---- Integer IDs ----
    species_id = state.sides_team_species_id[s, slot].astype(jnp.int32)
    ability_id = state.sides_team_ability_id[s, slot].astype(jnp.int32)
    item_id    = state.sides_team_item_id[s, slot].astype(jnp.int32)
    last_move  = state.sides_team_last_move_id[s, slot].astype(jnp.int32)

    if not is_own:
        is_known_ability = reveal.revealed_ability[player, slot]
        is_known_item    = reveal.revealed_item[player, slot]
        ability_id = jnp.where(is_known_ability, ability_id,
                               jnp.int32(UNKNOWN_ABILITY_IDX))
        item_id    = jnp.where(is_known_item, item_id,
                               jnp.int32(UNKNOWN_ITEM_IDX))
        # last_move is only visible if the Pokemon has been seen
        last_move  = jnp.where(is_known_poke, last_move, jnp.int32(0))
    else:
        is_known_ability = jnp.bool_(True)
        is_known_item    = jnp.bool_(True)

    species_id = jnp.where(is_known_poke, species_id,
                           jnp.int32(UNKNOWN_SPECIES_IDX))

    # Encode each move slot
    move_int_ids = []
    move_feats_list = []
    for m in range(N_MOVES_PER_MON):
        mid = state.sides_team_move_ids[s, slot, m]
        pp  = state.sides_team_move_pp[s, slot, m]
        mxpp = state.sides_team_move_max_pp[s, slot, m]
        if is_own:
            m_known = (mid >= jnp.int16(0))  # always known if own
        else:
            m_known = reveal.revealed_moves[player, slot, m] & is_known_poke
        mid_int, mfeats = _encode_move(mid, pp, mxpp, m_known, tables)
        move_int_ids.append(mid_int)
        move_feats_list.append(mfeats)

    int_ids = jnp.array([
        jnp.where(is_known_poke, species_id.astype(jnp.int32), jnp.int32(UNKNOWN_SPECIES_IDX)),
        move_int_ids[0].astype(jnp.int32),
        move_int_ids[1].astype(jnp.int32),
        move_int_ids[2].astype(jnp.int32),
        move_int_ids[3].astype(jnp.int32),
        ability_id.astype(jnp.int32),
        item_id.astype(jnp.int32),
        jnp.where(is_known_poke, jnp.maximum(last_move, jnp.int32(0)), jnp.int32(0)).astype(jnp.int32),
    ], dtype=jnp.int32)

    # ---- Float features ----
    buf = jnp.zeros(FLOAT_DIM_PER_POKEMON, dtype=jnp.float32)

    hp     = state.sides_team_hp[s, slot].astype(jnp.float32)
    max_hp = state.sides_team_max_hp[s, slot].astype(jnp.float32)
    max_hp_safe = jnp.maximum(max_hp, jnp.float32(1.0))
    hp_frac = hp / max_hp_safe

    # hp_fraction (1)
    buf = buf.at[_OFF_HP_FRAC].set(hp_frac)

    # hp_bin (10): convert to percentage 0-100
    hp_pct  = jnp.clip((hp_frac * 100.0).astype(jnp.int32), 0, 100)
    hp_idx  = _bin_idx(hp_pct, _HP_THRESHOLDS)
    buf = buf.at[_OFF_HP_BIN + hp_idx].set(1.0)

    # base_stats (6) normalized /255
    bst = state.sides_team_base_stats[s, slot].astype(jnp.float32)
    buf = buf.at[_OFF_BASE_STATS : _OFF_BASE_STATS + 6].set(bst / 255.0)

    # stat_boosts (91 = 7 stats × 13 levels)
    boosts = state.sides_team_boosts[s, slot].astype(jnp.int32)  # [7]
    for i in range(7):
        boost_idx = jnp.clip(boosts[i] + 6, 0, 12)
        buf = buf.at[_OFF_BOOSTS + i * 13 + boost_idx].set(1.0)

    # status (7)
    status = state.sides_team_status[s, slot].astype(jnp.int32)
    buf = buf.at[_OFF_STATUS + jnp.clip(status, 0, 6)].set(1.0)

    # volatile multi-hot (27): use precomputed bit→PS-idx mapping
    vols = state.sides_team_volatiles[s, slot]  # uint32 scalar
    # Expand bitmask to [32] float: bit_i = (vols >> i) & 1
    bits = jnp.array([((vols >> jnp.uint32(i)) & jnp.uint32(1)).astype(jnp.float32)
                      for i in range(MAX_VOLATILES)])  # [32]
    vol_multihot = jnp.matmul(bits, _VOL_ONEHOT_JAX)  # [27], values 0 or 1
    buf = buf.at[_OFF_VOLATILE : _OFF_VOLATILE + N_VOLATILE].set(
        jnp.minimum(vol_multihot, 1.0)
    )

    # type1, type2 (18 each) — pokejax type is 1-based, PS uses 0-based
    types = state.sides_team_types[s, slot]  # int8[2]
    ps_type1 = jnp.clip(types[0].astype(jnp.int32) - 1, 0, _N_PS_TYPES - 1)
    ps_type2 = jnp.clip(types[1].astype(jnp.int32) - 1, 0, _N_PS_TYPES - 1)
    buf = buf.at[_OFF_TYPE1 + ps_type1].set(1.0)
    # type2 only set if not the sentinel (type=0 → ps_type2=-1→clamped to 0, but
    # we need to check the original value)
    has_type2 = types[1].astype(jnp.int32) > 0
    buf = buf.at[_OFF_TYPE2 + ps_type2].set(jnp.where(has_type2, 1.0, 0.0))

    # is_fainted (1), is_active (1)
    fainted = state.sides_team_fainted[s, slot].astype(jnp.float32)
    buf = buf.at[_OFF_IS_FAINTED].set(fainted)
    buf = buf.at[_OFF_IS_ACTIVE].set(jnp.float32(1.0 if is_active else 0.0))

    # slot one-hot (6)
    buf = buf.at[_OFF_SLOT + slot].set(1.0)

    # is_own (1)
    buf = buf.at[_OFF_IS_OWN].set(jnp.float32(1.0 if is_own else 0.0))

    # moves (4 × 45)
    for m in range(N_MOVES_PER_MON):
        moff = _OFF_MOVES + m * 45
        buf = buf.at[moff : moff + 45].set(move_feats_list[m])

    # sleep_turns bin (4)
    sleep_t = state.sides_team_sleep_turns[s, slot].astype(jnp.int32)
    buf = buf.at[_OFF_SLEEP_BIN + jnp.clip(sleep_t, 0, 3)].set(1.0)

    # rest_turns bin (3) — pokejax doesn't separate rest turns; use sleep_turns
    # when status==SLP (rest turns are just sleep turns capped at 2)
    rest_t = jnp.where(status == STATUS_SLP.astype(jnp.int32),
                       jnp.clip(sleep_t, 0, 2), jnp.int32(0))
    buf = buf.at[_OFF_REST_BIN + rest_t].set(1.0)

    # substitute_health fraction (1)
    # VOL_SUBSTITUTE data stored as fraction of max_hp * 255
    sub_data = state.sides_team_volatile_data[s, slot, VOL_SUBSTITUTE].astype(jnp.int32)
    sub_frac = sub_data.astype(jnp.float32) / 255.0
    has_sub = (vols & jnp.uint32(1 << VOL_SUBSTITUTE)) != jnp.uint32(0)
    buf = buf.at[_OFF_SUB_FRAC].set(jnp.where(has_sub, sub_frac, 0.0))

    # force_trapped (1) — pokejax uses partially_trapped volatile
    force_trap = (vols & jnp.uint32(1 << VOL_PARTIALLY_TRAPPED)) != jnp.uint32(0)
    buf = buf.at[_OFF_FORCE_TRAP].set(force_trap.astype(jnp.float32))

    # move_disabled (4)
    disabled = state.sides_team_move_disabled[s, slot]  # bool[4]
    buf = buf.at[_OFF_MOV_DIS : _OFF_MOV_DIS + 4].set(disabled.astype(jnp.float32))

    # confusion_bin (4)
    conf_t = state.sides_team_volatile_data[s, slot, VOL_CONFUSED].astype(jnp.int32)
    buf = buf.at[_OFF_CONF_BIN + jnp.clip(conf_t, 0, 3)].set(1.0)

    # taunt, encore, yawn flags
    has_taunt  = (vols & jnp.uint32(1 << VOL_TAUNT))  != jnp.uint32(0)
    has_encore = (vols & jnp.uint32(1 << VOL_ENCORE)) != jnp.uint32(0)
    has_yawn   = (vols & jnp.uint32(1 << VOL_YAWN))   != jnp.uint32(0)
    buf = buf.at[_OFF_TAUNT].set(has_taunt.astype(jnp.float32))
    buf = buf.at[_OFF_ENCORE].set(has_encore.astype(jnp.float32))
    buf = buf.at[_OFF_YAWN].set(has_yawn.astype(jnp.float32))

    # level_normalized (1)
    level = state.sides_team_level[s, slot].astype(jnp.float32)
    buf = buf.at[_OFF_LEVEL].set(level / 100.0)

    # perish_count bin (4) — one-hot [0=none, 1, 2, 3]
    has_perish = (state.sides_team_volatiles[s, slot] & jnp.uint32(1 << VOL_PERISH)) != jnp.uint32(0)
    perish_cnt = jnp.where(has_perish,
                            state.sides_team_volatile_data[s, slot, VOL_PERISH].astype(jnp.int32),
                            jnp.int32(0))
    perish_bin = jnp.zeros(4, dtype=jnp.float32).at[jnp.clip(perish_cnt, 0, 3)].set(1.0)
    buf = buf.at[_OFF_PERISH_BIN:_OFF_PERISH_BIN + 4].set(perish_bin)

    # protect_count_normalized (1)
    prot_data = state.sides_team_volatile_data[s, slot, 5].astype(jnp.int32)  # VOL_PROTECT data
    buf = buf.at[_OFF_PROTECT].set(
        jnp.minimum(prot_data.astype(jnp.float32), 4.0) / 4.0
    )

    # locked_move (1)
    from pokejax.types import VOL_LOCKEDMOVE, VOL_CHOICELOCK
    has_lock = (vols & jnp.uint32((1 << VOL_LOCKEDMOVE) | (1 << VOL_CHOICELOCK))) != jnp.uint32(0)
    buf = buf.at[_OFF_LOCKED_MOV].set(has_lock.astype(jnp.float32))

    # --- Mask ALL features if the Pokemon is unknown to the observer ---
    buf = jnp.where(is_known_poke, buf, jnp.zeros_like(buf))

    return int_ids, buf


# ---------------------------------------------------------------------------
# Field token encoder
# ---------------------------------------------------------------------------

def _encode_field(state: BattleState, player: int) -> jnp.ndarray:
    """Encode the field token into a FIELD_DIM (84) float vector.

    The field token float_feats is FIELD_DIM dims; the rest of the 394-dim
    slot is zero-padded.
    """
    buf = jnp.zeros(FIELD_DIM, dtype=jnp.float32)

    opp = 1 - player
    own = player

    # Weather one-hot (5)
    weather = state.field.weather.astype(jnp.int32)
    buf = buf.at[_FOFF_WEATHER + jnp.clip(weather, 0, 4)].set(1.0)

    # Weather turns bin (8): 0..7 turns remaining
    wt = jnp.clip(state.field.weather_turns.astype(jnp.int32), 0, 7)
    buf = buf.at[_FOFF_WT_TURNS + wt].set(1.0)

    # Pseudo-weather (5): trick room, gravity, wonder room (bits 0, 1, 2)
    trick_room = state.field.trick_room.astype(jnp.int32) > 0
    gravity    = state.field.gravity.astype(jnp.int32) > 0
    wonder_room = state.field.wonder_room.astype(jnp.int32) > 0
    buf = buf.at[_FOFF_PSEUDO].set(trick_room.astype(jnp.float32))
    buf = buf.at[_FOFF_PSEUDO + 1].set(gravity.astype(jnp.float32))
    buf = buf.at[_FOFF_PSEUDO + 2].set(wonder_room.astype(jnp.float32))

    # Trick room turns bin (4)
    tr_t = jnp.clip(state.field.trick_room.astype(jnp.int32), 0, 3)
    buf = buf.at[_FOFF_TR_TURNS + tr_t].set(1.0)

    # Hazards own / opp (7 each): sr(1) + spikes(3) + tspikes(2) + web(1)
    def _fill_hazards(side_idx: int, foff: int):
        sc = state.sides_side_conditions[side_idx]
        sr   = sc[SC_STEALTHROCK].astype(jnp.int32) > 0
        sp   = sc[SC_SPIKES].astype(jnp.int32)
        ts   = sc[SC_TOXICSPIKES].astype(jnp.int32)
        web  = sc[SC_STICKYWEB].astype(jnp.int32) > 0
        return jnp.array([
            sr.astype(jnp.float32),
            (sp >= 1).astype(jnp.float32),
            (sp >= 2).astype(jnp.float32),
            (sp >= 3).astype(jnp.float32),
            (ts >= 1).astype(jnp.float32),
            (ts >= 2).astype(jnp.float32),
            web.astype(jnp.float32),
        ], dtype=jnp.float32)

    buf = buf.at[_FOFF_HAZARDS_OWN : _FOFF_HAZARDS_OWN + 7].set(_fill_hazards(own, 0))
    buf = buf.at[_FOFF_HAZARDS_OPP : _FOFF_HAZARDS_OPP + 7].set(_fill_hazards(opp, 0))

    # Screens own / opp (6 each): [ls_flag, ls_turns/5, ref_flag, ref_turns/5, 0, 0]
    def _fill_screens(side_idx: int):
        sc  = state.sides_side_conditions[side_idx]
        ls  = sc[SC_REFLECT].astype(jnp.int32)   # note: SC_LIGHTSCREEN not SC_REFLECT
        ref = sc[SC_LIGHTSCREEN].astype(jnp.int32)
        # Actually SC_REFLECT=4, SC_LIGHTSCREEN=5 from types.py
        ls_real  = state.sides_side_conditions[side_idx, SC_LIGHTSCREEN].astype(jnp.int32)
        ref_real = state.sides_side_conditions[side_idx, SC_REFLECT].astype(jnp.int32)
        return jnp.array([
            (ls_real > 0).astype(jnp.float32),
            jnp.minimum(ls_real.astype(jnp.float32) / 5.0, 1.0),
            (ref_real > 0).astype(jnp.float32),
            jnp.minimum(ref_real.astype(jnp.float32) / 5.0, 1.0),
            0.0, 0.0,
        ], dtype=jnp.float32)

    buf = buf.at[_FOFF_SCREENS_OWN : _FOFF_SCREENS_OWN + 6].set(_fill_screens(own))
    buf = buf.at[_FOFF_SCREENS_OPP : _FOFF_SCREENS_OPP + 6].set(_fill_screens(opp))

    # Turn number bin (10)
    turn = state.turn.astype(jnp.int32)
    turn_idx = jnp.sum(turn >= _TURN_THRESHOLDS).astype(jnp.int32)
    turn_idx = jnp.clip(turn_idx, 0, 9)
    buf = buf.at[_FOFF_TURN_BIN + turn_idx].set(1.0)

    # Fainted counts (2)
    own_fainted = state.sides_team_fainted[own].astype(jnp.float32).sum()
    opp_fainted = state.sides_team_fainted[opp].astype(jnp.float32).sum()
    buf = buf.at[_FOFF_FAINTED].set(own_fainted / 6.0)
    buf = buf.at[_FOFF_FAINTED + 1].set(opp_fainted / 6.0)

    # Toxic count bins (5 each): 0, 1, 2, 3-4, 5+
    def _toxic_bin(side_idx: int):
        active_idx = state.sides_active_idx[side_idx]
        toxic_count = state.sides_team_status_turns[side_idx, active_idx].astype(jnp.int32)
        is_tox = state.sides_team_status[side_idx, active_idx].astype(jnp.int32) == 3  # STATUS_TOX
        count = jnp.where(is_tox, toxic_count, jnp.int32(0))
        idx = jnp.where(count <= 0, jnp.int32(0),
              jnp.where(count == 1, jnp.int32(1),
              jnp.where(count == 2, jnp.int32(2),
              jnp.where(count <= 4, jnp.int32(3), jnp.int32(4)))))
        return idx

    buf = buf.at[_FOFF_TOXIC_OWN + _toxic_bin(own)].set(1.0)
    buf = buf.at[_FOFF_TOXIC_OPP + _toxic_bin(opp)].set(1.0)

    # Tailwind flags (2)
    buf = buf.at[_FOFF_TAILWIND].set(
        (state.sides_side_conditions[own, SC_TAILWIND] > 0).astype(jnp.float32))
    buf = buf.at[_FOFF_TAILWIND + 1].set(
        (state.sides_side_conditions[opp, SC_TAILWIND] > 0).astype(jnp.float32))

    # Wish (2): not implemented → zeros

    # Safeguard flags (2)
    buf = buf.at[_FOFF_SAFEGUARD].set(
        (state.sides_side_conditions[own, SC_SAFEGUARD] > 0).astype(jnp.float32))
    buf = buf.at[_FOFF_SAFEGUARD + 1].set(
        (state.sides_side_conditions[opp, SC_SAFEGUARD] > 0).astype(jnp.float32))

    # Mist (2): SC_MIST index 9
    from pokejax.types import SC_MIST
    buf = buf.at[_FOFF_MIST].set(
        (state.sides_side_conditions[own, SC_MIST] > 0).astype(jnp.float32))
    buf = buf.at[_FOFF_MIST + 1].set(
        (state.sides_side_conditions[opp, SC_MIST] > 0).astype(jnp.float32))

    # Lucky chant (2): not implemented → zeros

    # Gravity turns bin (4)
    grav_t = jnp.clip(state.field.gravity.astype(jnp.int32), 0, 3)
    buf = buf.at[_FOFF_GRAVITY_TURNS + grav_t].set(1.0)

    return buf


# ---------------------------------------------------------------------------
# Legal action mask
# ---------------------------------------------------------------------------

def _build_legal_mask(state: BattleState, player: int) -> jnp.ndarray:
    """Return float32[10] legal action mask for player.

    Actions 0-3: use move slot i.  Legal if pp > 0 and not disabled.
    Actions 4-9: switch to team slot i.  Legal if alive, not active, not finished.
    """
    s = player
    active_idx = state.sides_active_idx[s]

    # Move actions (4)
    pp   = state.sides_team_move_pp[s, active_idx]   # int8[4]
    dis  = state.sides_team_move_disabled[s, active_idx]  # bool[4]
    move_legal = (pp > jnp.int8(0)) & (~dis)

    # Switch actions (6)
    fainted    = state.sides_team_fainted[s]          # bool[6]
    is_active  = state.sides_team_is_active[s]        # bool[6]
    switch_legal = (~fainted) & (~is_active)

    mask = jnp.concatenate([
        move_legal.astype(jnp.float32),
        switch_legal.astype(jnp.float32),
    ])

    # If no legal moves at all (shouldn't happen), allow move 0
    has_any = mask.any()
    mask = jnp.where(has_any, mask, jnp.zeros(10, dtype=jnp.float32).at[0].set(1.0))
    return mask


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_obs(
    state: BattleState,
    reveal: RevealState,
    player: int,    # compile-time Python int (0 or 1)
    tables,
) -> dict:
    """
    Build the 15-token observation for `player`.

    Returns dict with:
      "int_ids"    : jnp.ndarray (15, 8)
      "float_feats": jnp.ndarray (15, 394)
      "legal_mask" : jnp.ndarray (10,)
    """
    opp = 1 - player

    int_ids_list   = []
    float_feats_list = []

    # Token 0: field
    field_buf = jnp.zeros(FLOAT_DIM_PER_POKEMON, dtype=jnp.float32)
    field_buf = field_buf.at[:FIELD_DIM].set(_encode_field(state, player))
    int_ids_list.append(jnp.zeros(INT_IDS_PER_TOKEN, dtype=jnp.int32))
    float_feats_list.append(field_buf)

    # Tokens 1-6: own team (side = player)
    for slot in range(N_TEAM_SLOTS):
        is_active_dyn = state.sides_team_is_active[player, slot]
        ii, ff = _encode_pokemon(
            state, player, slot, is_own=True, is_active=False,  # is_active overridden below
            player=player, reveal=reveal, tables=tables,
        )
        # Override is_active feature in float_feats (branchless, JAX-compatible)
        ff = ff.at[_OFF_IS_ACTIVE].set(is_active_dyn.astype(jnp.float32))
        int_ids_list.append(ii)
        float_feats_list.append(ff)

    # Tokens 7-12: opp team (side = opp)
    for slot in range(N_TEAM_SLOTS):
        is_active_dyn = state.sides_team_is_active[opp, slot]
        ii, ff = _encode_pokemon(
            state, opp, slot, is_own=False, is_active=False,  # is_active overridden below
            player=player, reveal=reveal, tables=tables,
        )
        ff = ff.at[_OFF_IS_ACTIVE].set(is_active_dyn.astype(jnp.float32))
        int_ids_list.append(ii)
        float_feats_list.append(ff)

    # Tokens 13-14: actor/critic queries — all zeros
    for _ in range(2):
        int_ids_list.append(jnp.zeros(INT_IDS_PER_TOKEN, dtype=jnp.int32))
        float_feats_list.append(jnp.zeros(FLOAT_DIM_PER_POKEMON, dtype=jnp.float32))

    int_ids    = jnp.stack(int_ids_list, axis=0)    # (15, 8)
    float_feats = jnp.stack(float_feats_list, axis=0)  # (15, 394)
    legal_mask = _build_legal_mask(state, player)       # (10,)

    return {
        "int_ids":    int_ids,
        "float_feats": float_feats,
        "legal_mask":  legal_mask,
    }


# Alias used by tests and external callers
build_observation = build_obs
