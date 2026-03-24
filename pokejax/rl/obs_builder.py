"""
JAX observation builder for PokeJAX RL — VECTORIZED version.

Converts (BattleState, RevealState, player, tables) → token arrays for
PokeTransformer, matching the PokemonShowdownClaude format exactly.

PERFORMANCE: All 12 Pokemon are encoded simultaneously using batched
jax.nn.one_hot + concatenation instead of Python for-loops with scalar
buf.at[].set(). This reduces the XLA trace graph by ~10x.

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

All operations are branchless jnp.where so this function is fully
jit- and vmap-compatible (player must be a Python-level constant).
"""

from __future__ import annotations

import jax
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
    VOL_LOCKEDMOVE, VOL_CHOICELOCK,
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

_N_PS_TYPES = 18

# ---------------------------------------------------------------------------
# Volatile mapping: pokejax bit index → PS volatile_index (27 dims)
# ---------------------------------------------------------------------------
N_VOLATILE = 27

_VOL_MAP = np.full(MAX_VOLATILES, -1, dtype=np.int8)
_VOL_MAP[VOL_CONFUSED]          = 0
_VOL_MAP[VOL_ATTRACT]           = 1
_VOL_MAP[VOL_SEEDED]            = 2
_VOL_MAP[VOL_CURSE]             = 3
_VOL_MAP[VOL_AQUARINGTARGET]    = 4
_VOL_MAP[VOL_INGRAIN]           = 5
_VOL_MAP[VOL_TAUNT]             = 6
_VOL_MAP[VOL_ENCORE]            = 7
_VOL_MAP[VOL_FLINCH]            = 8
_VOL_MAP[VOL_EMBARGO]           = 9
_VOL_MAP[VOL_HEALBLOCK]         = 10
_VOL_MAP[VOL_PARTIALLY_TRAPPED] = 12
_VOL_MAP[VOL_SUBSTITUTE]        = 15
_VOL_MAP[VOL_YAWN]              = 16
_VOL_MAP[VOL_FOCUSENERGY]       = 17
_VOL_MAP[VOL_CHARGING]          = 18
_VOL_MAP[VOL_RECHARGING]        = 23
_VOL_MAP[VOL_TORMENT]           = 20
_VOL_MAP[VOL_NIGHTMARE]         = 21
_VOL_MAP[VOL_DESTINYBOND]       = 25
_VOL_MAP[VOL_GRUDGE]            = 26

_VOL_MAP_JAX = jnp.array(_VOL_MAP, dtype=jnp.int8)

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

# Per-pokemon offsets (for reference/testing — the vectorized encoder uses
# concatenation order instead of scattered writes)
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
_FOFF_PSEUDO       = 13  # 5 dims
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
_FOFF_WISH         = 72  # 2 dims
_FOFF_SAFEGUARD    = 74  # 2 dims
_FOFF_MIST         = 76  # 2 dims
_FOFF_LUCKY_CHANT  = 78  # 2 dims
_FOFF_GRAVITY_TURNS = 80 # 4 dims

assert _FOFF_GRAVITY_TURNS + 4 == FIELD_DIM, \
    f"Field dim mismatch: expected {FIELD_DIM}"

# ---------------------------------------------------------------------------
# Binning threshold arrays (precomputed as JAX constants)
# ---------------------------------------------------------------------------

_BP_THRESHOLDS  = jnp.array([0, 1, 41, 61, 81, 101, 121, 151], dtype=jnp.int32)
_ACC_THRESHOLDS = jnp.array([0, 50, 70, 80, 90, 100], dtype=jnp.int32)
_HP_THRESHOLDS  = jnp.array([0, 10, 20, 33, 50, 66, 75, 88, 100], dtype=jnp.int32)
_TURN_THRESHOLDS = jnp.array([1, 2, 4, 6, 9, 13, 18, 25, 35], dtype=jnp.int32)
_PRIORITY_MIN = -3

# Precomputed constants
_SLOT_ONEHOT = jnp.eye(6, dtype=jnp.float32)        # (6, 6)
_BIT_SHIFTS  = jnp.arange(MAX_VOLATILES, dtype=jnp.uint32)  # (32,)


# ---------------------------------------------------------------------------
# Vectorized binning helpers (operate on arrays, not scalars)
# ---------------------------------------------------------------------------

def _bin_idx_vec(vals: jnp.ndarray, thresholds: jnp.ndarray) -> jnp.ndarray:
    """Vectorized bin index: vals (...,) → indices (...,)."""
    idx = jnp.sum(vals[..., None].astype(jnp.int32) >= thresholds, axis=-1) - 1
    return jnp.clip(idx, 0, thresholds.shape[0] - 1)


# ---------------------------------------------------------------------------
# Vectorized Pokemon batch encoder
# ---------------------------------------------------------------------------

def _encode_pokemon_batch(
    state: BattleState,
    side: int,          # Python int (compile-time): which side's team
    is_own: bool,       # Python bool (compile-time)
    player: int,        # Python int (compile-time): observer
    reveal: RevealState,
    tables,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encode all 6 Pokemon of one side at once.

    Returns:
        int_ids:    (6, 8) int32
        float_feats: (6, 394) float32
    """
    N = N_TEAM_SLOTS  # 6

    # --- Gather raw data for all 6 slots (all shape (6, ...)) ---
    species_ids = state.sides_team_species_id[side]       # (6,) int16
    ability_ids = state.sides_team_ability_id[side]       # (6,) int16
    item_ids    = state.sides_team_item_id[side]          # (6,) int16
    last_moves  = state.sides_team_last_move_id[side]     # (6,) int16
    hp          = state.sides_team_hp[side]               # (6,) int16
    max_hp      = state.sides_team_max_hp[side]           # (6,) int16
    base_stats  = state.sides_team_base_stats[side]       # (6, 6) int16
    boosts      = state.sides_team_boosts[side]           # (6, 7) int8
    status      = state.sides_team_status[side]           # (6,) int8
    volatiles   = state.sides_team_volatiles[side]        # (6,) uint32
    types       = state.sides_team_types[side]            # (6, 2) int8
    fainted     = state.sides_team_fainted[side]          # (6,) bool
    is_active   = state.sides_team_is_active[side]        # (6,) bool
    level       = state.sides_team_level[side]            # (6,) int8
    move_ids    = state.sides_team_move_ids[side]         # (6, 4) int16
    move_pp     = state.sides_team_move_pp[side]          # (6, 4) int8
    move_max_pp = state.sides_team_move_max_pp[side]      # (6, 4) int8
    move_disabled = state.sides_team_move_disabled[side]  # (6, 4) bool
    sleep_turns = state.sides_team_sleep_turns[side]      # (6,) int8
    volatile_data = state.sides_team_volatile_data[side]  # (6, 32) int8

    # --- Masking for opponent's unknown pokemon ---
    if is_own:
        is_known_poke = jnp.ones(N, dtype=jnp.bool_)
        is_known_moves = move_ids >= jnp.int16(0)  # (6, 4)
    else:
        is_known_poke = reveal.revealed_pokemon[player]  # (6,)
        is_known_moves = reveal.revealed_moves[player] & is_known_poke[:, None]  # (6, 4)
        is_known_ability = reveal.revealed_ability[player]  # (6,)
        is_known_item = reveal.revealed_item[player]  # (6,)
        ability_ids = jnp.where(is_known_ability, ability_ids, jnp.int16(UNKNOWN_ABILITY_IDX))
        item_ids = jnp.where(is_known_item, item_ids, jnp.int16(UNKNOWN_ITEM_IDX))
        last_moves = jnp.where(is_known_poke, last_moves, jnp.int16(0))

    species_ids = jnp.where(is_known_poke, species_ids, jnp.int16(UNKNOWN_SPECIES_IDX))

    # ===== INTEGER IDS (6, 8) =====
    safe_move_ids = jnp.clip(move_ids.astype(jnp.int32), 0, tables.moves.shape[0] - 1)
    valid_moves = (move_ids >= jnp.int16(0)) & is_known_moves
    move_int_ids = jnp.where(valid_moves, safe_move_ids, jnp.int32(UNKNOWN_MOVE_IDX))  # (6, 4)

    safe_species = species_ids.astype(jnp.int32)
    safe_ability = ability_ids.astype(jnp.int32)
    safe_item = item_ids.astype(jnp.int32)
    safe_last = jnp.where(is_known_poke, jnp.maximum(last_moves.astype(jnp.int32), 0), 0)

    int_ids = jnp.stack([
        safe_species,
        move_int_ids[:, 0], move_int_ids[:, 1],
        move_int_ids[:, 2], move_int_ids[:, 3],
        safe_ability, safe_item, safe_last,
    ], axis=-1)  # (6, 8)

    # ===== FLOAT FEATURES (6, 394) — build via concatenation =====

    # hp_frac (1)
    hp_f = hp.astype(jnp.float32)
    max_hp_f = jnp.maximum(max_hp.astype(jnp.float32), 1.0)
    hp_frac = (hp_f / max_hp_f)[:, None]  # (6, 1)

    # hp_bin (10) — one-hot
    hp_pct = jnp.clip((hp_frac[:, 0] * 100.0).astype(jnp.int32), 0, 100)
    hp_bin_idx = _bin_idx_vec(hp_pct, _HP_THRESHOLDS)
    hp_bin = jax.nn.one_hot(hp_bin_idx, 10, dtype=jnp.float32)  # (6, 10)

    # base_stats (6) normalized
    bst = base_stats.astype(jnp.float32) / 255.0  # (6, 6)

    # boosts (91 = 7×13) — vectorized one-hot
    boost_idx = jnp.clip(boosts.astype(jnp.int32) + 6, 0, 12)  # (6, 7)
    boost_oh = jax.nn.one_hot(boost_idx, 13, dtype=jnp.float32)  # (6, 7, 13)
    boost_flat = boost_oh.reshape(N, 91)  # (6, 91)

    # status (7) — one-hot
    status_idx = jnp.clip(status.astype(jnp.int32), 0, 6)
    status_oh = jax.nn.one_hot(status_idx, 7, dtype=jnp.float32)  # (6, 7)

    # volatile (27) — vectorized bit extraction + matmul
    bits = ((volatiles[:, None].astype(jnp.uint32) >> _BIT_SHIFTS[None, :]) &
            jnp.uint32(1)).astype(jnp.float32)  # (6, 32)
    vol_multihot = jnp.minimum(bits @ _VOL_ONEHOT_JAX, 1.0)  # (6, 27)

    # type1 (18) — one-hot
    ps_type1 = jnp.clip(types[:, 0].astype(jnp.int32) - 1, 0, _N_PS_TYPES - 1)
    type1_oh = jax.nn.one_hot(ps_type1, _N_PS_TYPES, dtype=jnp.float32)  # (6, 18)

    # type2 (18) — one-hot, zeroed if no second type
    ps_type2 = jnp.clip(types[:, 1].astype(jnp.int32) - 1, 0, _N_PS_TYPES - 1)
    has_type2 = (types[:, 1].astype(jnp.int32) > 0)[:, None]
    type2_oh = jax.nn.one_hot(ps_type2, _N_PS_TYPES, dtype=jnp.float32) * has_type2  # (6, 18)

    # is_fainted (1)
    fainted_f = fainted.astype(jnp.float32)[:, None]  # (6, 1)

    # is_active (1)
    is_active_f = is_active.astype(jnp.float32)[:, None]  # (6, 1)

    # slot one-hot (6) — precomputed identity matrix
    slot_oh = _SLOT_ONEHOT  # (6, 6)

    # is_own (1)
    is_own_f = jnp.full((N, 1), 1.0 if is_own else 0.0, dtype=jnp.float32)

    # ---- Moves (4×45 = 180) — vectorized across all 24 moves ----
    flat_move_ids = move_ids.reshape(-1)  # (24,)
    flat_pp = move_pp.reshape(-1)  # (24,)
    flat_max_pp = move_max_pp.reshape(-1)  # (24,)
    flat_known = is_known_moves.reshape(-1)  # (24,)

    safe_ids = jnp.clip(flat_move_ids.astype(jnp.int32), 0, tables.moves.shape[0] - 1)
    rows = tables.moves[safe_ids]  # (24, >=5)

    bp = rows[:, 0].astype(jnp.int32)
    acc = rows[:, 1].astype(jnp.int32)
    type_id = rows[:, 2].astype(jnp.int32)
    category = rows[:, 3].astype(jnp.int32)
    priority = rows[:, 4].astype(jnp.int32)

    ps_mtype = jnp.clip(type_id - 1, 0, _N_PS_TYPES - 1)

    # bp_bin (8)
    bp_bin_idx = _bin_idx_vec(bp, _BP_THRESHOLDS)
    bp_bin = jax.nn.one_hot(bp_bin_idx, 8, dtype=jnp.float32)  # (24, 8)

    # acc_bin (6)
    acc_clamped = jnp.where(acc > 100, 0, acc)
    acc_bin_idx = _bin_idx_vec(acc_clamped, _ACC_THRESHOLDS)
    acc_bin = jax.nn.one_hot(acc_bin_idx, 6, dtype=jnp.float32)  # (24, 6)

    # move type (18)
    mtype_oh = jax.nn.one_hot(ps_mtype, _N_PS_TYPES, dtype=jnp.float32)  # (24, 18)

    # category (3)
    cat_oh = jax.nn.one_hot(jnp.clip(category, 0, 2), 3, dtype=jnp.float32)  # (24, 3)

    # priority (8)
    pri_idx = jnp.clip(priority - _PRIORITY_MIN, 0, 7)
    pri_oh = jax.nn.one_hot(pri_idx, 8, dtype=jnp.float32)  # (24, 8)

    # pp_frac (1)
    pp_safe = jnp.maximum(flat_pp.astype(jnp.int32), 0)
    max_pp_safe = jnp.maximum(flat_max_pp.astype(jnp.int32), 1)
    pp_frac_m = (pp_safe.astype(jnp.float32) / max_pp_safe.astype(jnp.float32))[:, None]

    # is_known (1)
    known_f = flat_known.astype(jnp.float32)[:, None]

    # Assemble known move features (24, 45)
    known_feats = jnp.concatenate([
        bp_bin, acc_bin, mtype_oh, cat_oh, pri_oh, pp_frac_m, known_f,
    ], axis=-1)  # (24, 45)

    # Unknown move: zeros except pp_frac=1.0
    unknown_feats = jnp.zeros((24, 45), dtype=jnp.float32).at[:, _MOFF_PP].set(1.0)

    move_feats = jnp.where(flat_known[:, None], known_feats, unknown_feats)  # (24, 45)
    move_feats = move_feats.reshape(N, 4 * 45)  # (6, 180)

    # ---- Remaining scalar features ----

    # sleep_bin (4)
    sleep_t = jnp.clip(sleep_turns.astype(jnp.int32), 0, 3)
    sleep_bin = jax.nn.one_hot(sleep_t, 4, dtype=jnp.float32)  # (6, 4)

    # rest_bin (3)
    rest_t = jnp.where(
        status.astype(jnp.int32) == STATUS_SLP.astype(jnp.int32),
        jnp.clip(sleep_turns.astype(jnp.int32), 0, 2),
        jnp.int32(0),
    )
    rest_bin = jax.nn.one_hot(rest_t, 3, dtype=jnp.float32)  # (6, 3)

    # sub_frac (1) — substitute HP stored as raw HP clamped to int8 [1,127]
    # Normalize by max_hp/4 (sub HP = 25% of max HP when created)
    sub_data_raw = volatile_data[:, VOL_SUBSTITUTE].astype(jnp.float32)
    sub_max_hp = jnp.maximum(1.0, jnp.floor(max_hp.astype(jnp.float32) / 4.0))
    has_sub = (volatiles & jnp.uint32(1 << VOL_SUBSTITUTE)) != jnp.uint32(0)
    sub_frac = jnp.where(has_sub, sub_data_raw / sub_max_hp, 0.0)[:, None]  # (6, 1)

    # force_trap (1)
    force_trap = ((volatiles & jnp.uint32(1 << VOL_PARTIALLY_TRAPPED)) !=
                  jnp.uint32(0)).astype(jnp.float32)[:, None]

    # mov_dis (4)
    mov_dis = move_disabled.astype(jnp.float32)  # (6, 4)

    # conf_bin (4)
    conf_t = jnp.clip(volatile_data[:, VOL_CONFUSED].astype(jnp.int32), 0, 3)
    conf_bin = jax.nn.one_hot(conf_t, 4, dtype=jnp.float32)  # (6, 4)

    # taunt, encore, yawn (1 each)
    has_taunt = ((volatiles & jnp.uint32(1 << VOL_TAUNT)) !=
                 jnp.uint32(0)).astype(jnp.float32)[:, None]
    has_encore = ((volatiles & jnp.uint32(1 << VOL_ENCORE)) !=
                  jnp.uint32(0)).astype(jnp.float32)[:, None]
    has_yawn = ((volatiles & jnp.uint32(1 << VOL_YAWN)) !=
                jnp.uint32(0)).astype(jnp.float32)[:, None]

    # level (1)
    level_norm = (level.astype(jnp.float32) / 100.0)[:, None]  # (6, 1)

    # perish_bin (4)
    has_perish = (volatiles & jnp.uint32(1 << VOL_PERISH)) != jnp.uint32(0)
    perish_cnt = jnp.where(
        has_perish, volatile_data[:, VOL_PERISH].astype(jnp.int32), 0,
    )
    perish_bin = jax.nn.one_hot(jnp.clip(perish_cnt, 0, 3), 4, dtype=jnp.float32)

    # protect (1)
    prot_data = volatile_data[:, 5].astype(jnp.float32)  # VOL_PROTECT data
    protect = (jnp.minimum(prot_data, 4.0) / 4.0)[:, None]

    # locked_mov (1)
    lock_mask = jnp.uint32((1 << VOL_LOCKEDMOVE) | (1 << VOL_CHOICELOCK))
    has_lock = ((volatiles & lock_mask) != jnp.uint32(0)).astype(jnp.float32)[:, None]

    # ===== Concatenate ALL segments in exact offset order =====
    float_feats = jnp.concatenate([
        hp_frac,       # 0:   1 dim
        hp_bin,        # 1:  10 dims
        bst,           # 11:  6 dims
        boost_flat,    # 17: 91 dims
        status_oh,     # 108: 7 dims
        vol_multihot,  # 115:27 dims
        type1_oh,      # 142:18 dims
        type2_oh,      # 160:18 dims
        fainted_f,     # 178: 1 dim
        is_active_f,   # 179: 1 dim
        slot_oh,       # 180: 6 dims
        is_own_f,      # 186: 1 dim
        move_feats,    # 187:180 dims
        sleep_bin,     # 367: 4 dims
        rest_bin,      # 371: 3 dims
        sub_frac,      # 374: 1 dim
        force_trap,    # 375: 1 dim
        mov_dis,       # 376: 4 dims
        conf_bin,      # 380: 4 dims
        has_taunt,     # 384: 1 dim
        has_encore,    # 385: 1 dim
        has_yawn,      # 386: 1 dim
        level_norm,    # 387: 1 dim
        perish_bin,    # 388: 4 dims
        protect,       # 392: 1 dim
        has_lock,      # 393: 1 dim
    ], axis=-1)  # (6, 394)

    # Mask ALL features for unknown opponent pokemon
    if not is_own:
        known_mask = is_known_poke[:, None].astype(jnp.float32)  # (6, 1)
        float_feats = float_feats * known_mask

    return int_ids, float_feats


# ---------------------------------------------------------------------------
# Field token encoder
# ---------------------------------------------------------------------------

def _encode_field(state: BattleState, player: int) -> jnp.ndarray:
    """Encode the field token into a FIELD_DIM (84) float vector."""
    opp = 1 - player
    own = player

    from pokejax.types import SC_MIST

    # Build all segments and concatenate (instead of scattered writes)

    # Weather one-hot (5)
    weather = state.field.weather.astype(jnp.int32)
    weather_oh = jax.nn.one_hot(jnp.clip(weather, 0, 4), 5, dtype=jnp.float32)

    # Weather turns bin (8)
    wt = jnp.clip(state.field.weather_turns.astype(jnp.int32), 0, 7)
    wt_oh = jax.nn.one_hot(wt, 8, dtype=jnp.float32)

    # Pseudo-weather flags (5): trick room, gravity, wonder room, 0, 0
    trick_room = (state.field.trick_room.astype(jnp.int32) > 0).astype(jnp.float32)
    gravity = (state.field.gravity.astype(jnp.int32) > 0).astype(jnp.float32)
    wonder_room = (state.field.wonder_room.astype(jnp.int32) > 0).astype(jnp.float32)
    pseudo = jnp.array([trick_room, gravity, wonder_room, 0.0, 0.0])

    # Trick room turns bin (4)
    tr_t = jnp.clip(state.field.trick_room.astype(jnp.int32), 0, 3)
    tr_oh = jax.nn.one_hot(tr_t, 4, dtype=jnp.float32)

    # Hazards helper
    def _hazards(side_idx):
        sc = state.sides_side_conditions[side_idx]
        sr = sc[SC_STEALTHROCK].astype(jnp.int32) > 0
        sp = sc[SC_SPIKES].astype(jnp.int32)
        ts = sc[SC_TOXICSPIKES].astype(jnp.int32)
        web = sc[SC_STICKYWEB].astype(jnp.int32) > 0
        return jnp.array([
            sr.astype(jnp.float32),
            (sp >= 1).astype(jnp.float32), (sp >= 2).astype(jnp.float32),
            (sp >= 3).astype(jnp.float32),
            (ts >= 1).astype(jnp.float32), (ts >= 2).astype(jnp.float32),
            web.astype(jnp.float32),
        ])

    hazards_own = _hazards(own)   # (7,)
    hazards_opp = _hazards(opp)   # (7,)

    # Screens helper
    def _screens(side_idx):
        ls_real = state.sides_side_conditions[side_idx, SC_LIGHTSCREEN].astype(jnp.int32)
        ref_real = state.sides_side_conditions[side_idx, SC_REFLECT].astype(jnp.int32)
        return jnp.array([
            (ls_real > 0).astype(jnp.float32),
            jnp.minimum(ls_real.astype(jnp.float32) / 5.0, 1.0),
            (ref_real > 0).astype(jnp.float32),
            jnp.minimum(ref_real.astype(jnp.float32) / 5.0, 1.0),
            0.0, 0.0,
        ])

    screens_own = _screens(own)  # (6,)
    screens_opp = _screens(opp)  # (6,)

    # Turn number bin (10)
    turn = state.turn.astype(jnp.int32)
    turn_idx = jnp.clip(jnp.sum(turn >= _TURN_THRESHOLDS), 0, 9)
    turn_oh = jax.nn.one_hot(turn_idx, 10, dtype=jnp.float32)

    # Fainted counts (2)
    own_fainted = state.sides_team_fainted[own].astype(jnp.float32).sum() / 6.0
    opp_fainted = state.sides_team_fainted[opp].astype(jnp.float32).sum() / 6.0
    fainted_v = jnp.array([own_fainted, opp_fainted])

    # Toxic count bins (5 each)
    def _toxic_bin(side_idx):
        active_idx = state.sides_active_idx[side_idx]
        toxic_count = state.sides_team_status_turns[side_idx, active_idx].astype(jnp.int32)
        is_tox = state.sides_team_status[side_idx, active_idx].astype(jnp.int32) == 3
        count = jnp.where(is_tox, toxic_count, 0)
        idx = jnp.where(count <= 0, 0,
              jnp.where(count == 1, 1,
              jnp.where(count == 2, 2,
              jnp.where(count <= 4, 3, 4))))
        return jax.nn.one_hot(idx, 5, dtype=jnp.float32)

    toxic_own = _toxic_bin(own)  # (5,)
    toxic_opp = _toxic_bin(opp)  # (5,)

    # Tailwind (2)
    tw_own = (state.sides_side_conditions[own, SC_TAILWIND] > 0).astype(jnp.float32)
    tw_opp = (state.sides_side_conditions[opp, SC_TAILWIND] > 0).astype(jnp.float32)
    tailwind = jnp.array([tw_own, tw_opp])

    # Wish (2) — not implemented
    wish = jnp.zeros(2, dtype=jnp.float32)

    # Safeguard (2)
    sg_own = (state.sides_side_conditions[own, SC_SAFEGUARD] > 0).astype(jnp.float32)
    sg_opp = (state.sides_side_conditions[opp, SC_SAFEGUARD] > 0).astype(jnp.float32)
    safeguard = jnp.array([sg_own, sg_opp])

    # Mist (2)
    mist_own = (state.sides_side_conditions[own, SC_MIST] > 0).astype(jnp.float32)
    mist_opp = (state.sides_side_conditions[opp, SC_MIST] > 0).astype(jnp.float32)
    mist = jnp.array([mist_own, mist_opp])

    # Lucky chant (2) — not implemented
    lucky = jnp.zeros(2, dtype=jnp.float32)

    # Gravity turns bin (4)
    grav_t = jnp.clip(state.field.gravity.astype(jnp.int32), 0, 3)
    grav_oh = jax.nn.one_hot(grav_t, 4, dtype=jnp.float32)

    # Concatenate all field segments
    field_buf = jnp.concatenate([
        weather_oh,    # 0:  5
        wt_oh,         # 5:  8
        pseudo,        # 13: 5
        tr_oh,         # 18: 4
        hazards_own,   # 22: 7
        hazards_opp,   # 29: 7
        screens_own,   # 36: 6
        screens_opp,   # 42: 6
        turn_oh,       # 48:10
        fainted_v,     # 58: 2
        toxic_own,     # 60: 5
        toxic_opp,     # 65: 5
        tailwind,      # 70: 2
        wish,          # 72: 2
        safeguard,     # 74: 2
        mist,          # 76: 2
        lucky,         # 78: 2
        grav_oh,       # 80: 4
    ])  # (84,)

    return field_buf


# ---------------------------------------------------------------------------
# Legal action mask
# ---------------------------------------------------------------------------

def _build_legal_mask(state: BattleState, player: int, tables) -> jnp.ndarray:
    """Return float32[10] legal action mask for player.

    Only masks moves that are truly illegal (no PP, disabled, choice-locked, etc.).
    Moves that would fail (hazards at max layers, status vs Substitute) are NOT
    masked — the model should learn these waste a turn, matching Pokemon Showdown
    behavior where they are selectable but fail.
    """
    s = player
    opp = 1 - player
    active_idx = state.sides_active_idx[s]

    pp  = state.sides_team_move_pp[s, active_idx]            # int8[4]
    dis = state.sides_team_move_disabled[s, active_idx]       # bool[4]
    move_legal = (pp > jnp.int8(0)) & (~dis)

    # NOTE: We intentionally do NOT mask hazard moves at max layers or status
    # moves vs Substitute here. The model should learn from the engine that
    # these moves fail/waste a turn, rather than being shielded from them.
    # This avoids train/test distribution mismatch with Pokemon Showdown,
    # which allows these moves to be selected (they just fail).

    fainted_arr = state.sides_team_fainted[s]                 # bool[6]
    is_active_arr = state.sides_team_is_active[s]             # bool[6]
    switch_legal = (~fainted_arr) & (~is_active_arr)

    mask = jnp.concatenate([
        move_legal.astype(jnp.float32),
        switch_legal.astype(jnp.float32),
    ])

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

    # Token 0: field (84-dim padded to 394)
    field_raw = _encode_field(state, player)  # (84,)
    field_pad = jnp.zeros(FLOAT_DIM_PER_POKEMON - FIELD_DIM, dtype=jnp.float32)
    field_feats = jnp.concatenate([field_raw, field_pad])  # (394,)
    field_int = jnp.zeros(INT_IDS_PER_TOKEN, dtype=jnp.int32)

    # Tokens 1-6: own team
    own_int, own_float = _encode_pokemon_batch(
        state, player, is_own=True, player=player, reveal=reveal, tables=tables,
    )  # (6, 8), (6, 394)

    # Tokens 7-12: opp team
    opp_int, opp_float = _encode_pokemon_batch(
        state, opp, is_own=False, player=player, reveal=reveal, tables=tables,
    )  # (6, 8), (6, 394)

    # Tokens 13-14: actor/critic queries (zeros)
    query_int = jnp.zeros((2, INT_IDS_PER_TOKEN), dtype=jnp.int32)
    query_float = jnp.zeros((2, FLOAT_DIM_PER_POKEMON), dtype=jnp.float32)

    # Stack all 15 tokens
    int_ids = jnp.concatenate([
        field_int[None],    # (1, 8)
        own_int,            # (6, 8)
        opp_int,            # (6, 8)
        query_int,          # (2, 8)
    ], axis=0)  # (15, 8)

    float_feats = jnp.concatenate([
        field_feats[None],  # (1, 394)
        own_float,          # (6, 394)
        opp_float,          # (6, 394)
        query_float,        # (2, 394)
    ], axis=0)  # (15, 394)

    legal_mask = _build_legal_mask(state, player, tables)  # (10,)

    return {
        "int_ids":    int_ids,
        "float_feats": float_feats,
        "legal_mask":  legal_mask,
    }


# Alias used by tests and external callers
build_observation = build_obs
