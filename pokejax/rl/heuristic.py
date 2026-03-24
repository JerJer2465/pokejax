"""
Pure JAX heuristic opponent + random opponent for BC training.

All functions are fully JIT/vmap compatible — no Python control flow,
no numpy, no string matching. Uses pre-computed move category arrays
built from the engine's Tables at module-init time.

Actions: 0-3 = moves, 4-9 = switch to team slot 0-5.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.types import (
    BattleState,
    CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS,
    STATUS_NONE,
    SC_STEALTHROCK, SC_SPIKES, SC_TOXICSPIKES,
    BOOST_ATK, BOOST_SPA,
)
from pokejax.data.tables import Tables
from pokejax.env.action_mask import get_action_mask


# Move data column indices
MF_BASE_POWER = 0
MF_ACCURACY = 1
MF_TYPE = 2
MF_CATEGORY = 3
MF_PRIORITY = 4

# Status move name sets (for building lookup arrays)
_SLEEP_MOVES = {
    "sleeppowder", "spore", "hypnosis", "lovelykiss", "grasswhistle",
    "sing", "darkvoid", "yawn",
}
_PARA_MOVES = {"thunderwave", "stunspore", "glare", "bodyslam"}
_TOXIC_MOVES = {"toxic"}
_BURN_MOVES = {"willowisp"}
_RECOVERY_MOVES = {
    "recover", "softboiled", "roost", "slackoff", "moonlight",
    "morningsun", "synthesis", "wish", "milkdrink", "rest",
}
_SETUP_MOVES = {
    "swordsdance", "dragondance", "nastyplot", "calmmind",
    "bulkup", "agility", "rockpolish", "curse", "quiverdance",
    "growth", "howl", "meditate", "sharpen",
}
_FIXED_DAMAGE_MOVES = {"seismictoss", "nightshade"}


def _normalize(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


# ---------------------------------------------------------------------------
# Pre-computed move category lookup arrays (built once per Tables instance)
# ---------------------------------------------------------------------------

def _build_move_categories(tables: Tables):
    """Build boolean JAX arrays: is_sleep[move_id], is_para[move_id], etc.

    No global caching — called inside JAX-traced functions (jax.lax.scan),
    so cached jnp arrays from a previous trace become stale tracers and
    cause UnexpectedTracerError. The computation is cheap (string lookups)
    and JAX's XLA compilation cache handles the real deduplication.
    """
    n = len(tables.move_names)
    cats = {}
    for cat_name, name_set in [
        ("sleep", _SLEEP_MOVES), ("para", _PARA_MOVES),
        ("toxic", _TOXIC_MOVES), ("burn", _BURN_MOVES),
        ("recovery", _RECOVERY_MOVES), ("setup", _SETUP_MOVES),
        ("fixed_dmg", _FIXED_DAMAGE_MOVES),
    ]:
        arr = np.zeros(n, dtype=np.float32)
        for i, mname in enumerate(tables.move_names):
            if _normalize(mname) in name_set:
                arr[i] = 1.0
        cats[cat_name] = jnp.array(arr)

    # Specific hazard move IDs
    sr_id = 0
    spikes_id = 0
    tspikes_id = 0
    for i, mname in enumerate(tables.move_names):
        nm = _normalize(mname)
        if nm == "stealthrock":
            sr_id = i
        elif nm == "spikes":
            spikes_id = i
        elif nm == "toxicspikes":
            tspikes_id = i
    cats["sr_id"] = jnp.int32(sr_id)
    cats["spikes_id"] = jnp.int32(spikes_id)
    cats["tspikes_id"] = jnp.int32(tspikes_id)

    return cats


# ---------------------------------------------------------------------------
# Core scoring functions (pure JAX)
# ---------------------------------------------------------------------------

def _type_eff(atk_type, def_type1, def_type2, type_chart):
    """Type effectiveness multiplier (scalar)."""
    eff1 = type_chart[atk_type, def_type1]
    eff2 = jnp.where(def_type2 > 0, type_chart[atk_type, def_type2], 1.0)
    return eff1 * eff2


def _estimate_damage(bp, move_type, category, accuracy,
                     atk_types, atk_stats, def_types, def_stats,
                     type_chart):
    """Rough damage score for a damaging move. Returns scalar float."""
    # STAB
    stab = jnp.where(
        (move_type == atk_types[0]) | ((atk_types[1] > 0) & (move_type == atk_types[1])),
        1.5, 1.0,
    )

    # Type effectiveness
    eff = _type_eff(move_type, def_types[0], def_types[1], type_chart)

    # Stat ratio based on category
    atk_stat = jnp.where(
        category == CATEGORY_PHYSICAL,
        jnp.maximum(atk_stats[1].astype(jnp.float32), 1.0),  # ATK
        jnp.maximum(atk_stats[3].astype(jnp.float32), 1.0),  # SPA
    )
    def_stat = jnp.where(
        category == CATEGORY_PHYSICAL,
        jnp.maximum(def_stats[2].astype(jnp.float32), 1.0),  # DEF
        jnp.maximum(def_stats[4].astype(jnp.float32), 1.0),  # SPD
    )

    acc = jnp.where(accuracy > 100, 100.0, accuracy.astype(jnp.float32))
    score = bp.astype(jnp.float32) * (atk_stat / def_stat) * stab * eff * (acc / 100.0)

    # Zero score for status moves (bp=0 already → score=0, but be explicit)
    return jnp.where(category == CATEGORY_STATUS, 0.0, score)


def _score_status_move(move_id, accuracy, own_hp_frac, own_boost_atk, own_boost_spa,
                       opp_status, opp_speed, opp_atk, opp_sc,
                       turn, move_type, def_types, type_chart, cats):
    """Score a status (bp=0) move. Returns scalar float."""
    acc_f = jnp.where(accuracy > 100, 1.0, accuracy.astype(jnp.float32) / 100.0)
    no_status = (opp_status == STATUS_NONE)

    # Stealth Rock
    sr_score = jnp.where(
        (move_id == cats["sr_id"]) & (opp_sc[SC_STEALTHROCK] == 0),
        jnp.where(turn <= 3, 250.0, 150.0),
        0.0,
    )

    # Spikes
    spikes_score = jnp.where(
        (move_id == cats["spikes_id"]) & (opp_sc[SC_SPIKES] < 3),
        jnp.where(turn <= 4, 180.0, 100.0),
        0.0,
    )

    # Toxic Spikes
    tspikes_score = jnp.where(
        (move_id == cats["tspikes_id"]) & (opp_sc[SC_TOXICSPIKES] < 2),
        jnp.where(turn <= 4, 160.0, 80.0),
        0.0,
    )

    hazard_score = jnp.maximum(sr_score, jnp.maximum(spikes_score, tspikes_score))

    # Sleep
    sleep_score = jnp.where(
        (cats["sleep"][move_id] > 0.5) & no_status,
        300.0 * acc_f,
        0.0,
    )

    # Paralysis
    para_score = jnp.where(
        (cats["para"][move_id] > 0.5) & no_status,
        jnp.where(opp_speed > 90, 200.0, 120.0) * acc_f,
        0.0,
    )

    # Toxic
    toxic_score = jnp.where(
        (cats["toxic"][move_id] > 0.5) & no_status,
        180.0 * acc_f,
        0.0,
    )

    # Burn
    burn_score = jnp.where(
        (cats["burn"][move_id] > 0.5) & no_status,
        jnp.where(opp_atk > 90, 180.0, 100.0) * acc_f,
        0.0,
    )

    # Setup
    boost_sum = own_boost_atk + own_boost_spa
    setup_score = jnp.where(
        (cats["setup"][move_id] > 0.5) & (boost_sum < 2) & (own_hp_frac > 0.6),
        jnp.where(turn <= 5, 200.0, 120.0),
        0.0,
    )

    # Recovery
    recovery_score = jnp.where(
        (cats["recovery"][move_id] > 0.5) & (own_hp_frac < 0.5),
        150.0,
        0.0,
    )

    # Fixed damage (Seismic Toss, Night Shade) — check type immunity
    eff = _type_eff(move_type, def_types[0], def_types[1], type_chart)
    fixed_score = jnp.where(
        (cats["fixed_dmg"][move_id] > 0.5) & (eff > 0.0),
        100.0,
        0.0,
    )

    # Default: small positive for any other status move
    default_score = 20.0

    # Take the max of all category scores (move belongs to exactly one category)
    all_scores = jnp.array([
        hazard_score, sleep_score, para_score, toxic_score, burn_score,
        setup_score, recovery_score, fixed_score, default_score,
    ])
    return jnp.max(all_scores)


# ---------------------------------------------------------------------------
# Main heuristic action (pure JAX, JIT-compatible)
# ---------------------------------------------------------------------------

def heuristic_action(state: BattleState, side: int, tables: Tables,
                     rng_key: jnp.ndarray) -> jnp.ndarray:
    """
    Pure JAX heuristic action selection. Fully JIT/vmap compatible.

    Args:
        state: BattleState (JAX arrays)
        side: 0 or 1 (Python int, compile-time constant)
        tables: Tables with moves, type_chart (Python object, compile-time)
        rng_key: JAX PRNG key (unused currently, reserved for stochastic tie-breaking)

    Returns:
        int32 scalar: action index 0-9
    """
    cats = _build_move_categories(tables)
    type_chart = tables.type_chart  # float32[19, 19]
    moves_table = tables.moves      # int16[N_MOVES, 22]

    mask = get_action_mask(state, side)  # bool[10]
    opp = 1 - side

    active_idx = state.sides_active_idx[side]
    opp_active_idx = state.sides_active_idx[opp]

    # Own active info
    own_types = state.sides_team_types[side, active_idx]        # int8[2]
    own_stats = state.sides_team_base_stats[side, active_idx]   # int16[6]
    own_hp = state.sides_team_hp[side, active_idx].astype(jnp.float32)
    own_max_hp = jnp.maximum(state.sides_team_max_hp[side, active_idx].astype(jnp.float32), 1.0)
    own_hp_frac = own_hp / own_max_hp
    own_boosts = state.sides_team_boosts[side, active_idx]      # int8[7]

    # Opponent active info
    opp_types = state.sides_team_types[opp, opp_active_idx]
    opp_stats = state.sides_team_base_stats[opp, opp_active_idx]
    opp_status = state.sides_team_status[opp, opp_active_idx].astype(jnp.int32)
    opp_speed = opp_stats[5].astype(jnp.float32)
    opp_atk = opp_stats[1].astype(jnp.float32)
    opp_sc = state.sides_side_conditions[opp]

    turn = state.turn.astype(jnp.int32)

    n_moves = moves_table.shape[0]

    # --- Score moves 0-3 ---
    scores = jnp.full(10, -1e9, dtype=jnp.float32)

    for m in range(4):  # unrolled by JIT
        move_id = state.sides_team_move_ids[side, active_idx, m].astype(jnp.int32)
        safe_id = jnp.clip(move_id, 0, n_moves - 1)
        row = moves_table[safe_id]

        bp = row[MF_BASE_POWER].astype(jnp.int32)
        move_type = row[MF_TYPE].astype(jnp.int32)
        category = row[MF_CATEGORY].astype(jnp.int32)
        accuracy = row[MF_ACCURACY].astype(jnp.int32)

        # Damaging move score
        dmg_score = _estimate_damage(
            bp, move_type, category, accuracy,
            own_types, own_stats, opp_types, opp_stats, type_chart,
        )

        # Status move score
        status_score = _score_status_move(
            safe_id, accuracy, own_hp_frac,
            own_boosts[BOOST_ATK].astype(jnp.float32),
            own_boosts[BOOST_SPA].astype(jnp.float32),
            opp_status, opp_speed, opp_atk, opp_sc,
            turn, move_type, opp_types, type_chart, cats,
        )

        move_score = jnp.where(bp > 0, dmg_score, status_score)
        # Ensure invalid moves get -inf
        move_score = jnp.where(mask[m] & (move_id >= 0), move_score, -1e9)
        scores = scores.at[m].set(move_score)

    best_move_score = jnp.max(scores[:4])

    # --- Score switches 4-9 ---
    for slot in range(6):  # unrolled by JIT
        slot_hp = state.sides_team_hp[side, slot].astype(jnp.float32)
        slot_max_hp = jnp.maximum(state.sides_team_max_hp[side, slot].astype(jnp.float32), 1.0)
        slot_hp_frac = slot_hp / slot_max_hp

        slot_types = state.sides_team_types[side, slot]
        slot_stats = state.sides_team_base_stats[side, slot]

        # Best outgoing damage from this bench mon
        best_dmg = jnp.float32(0.0)
        for sm in range(4):  # unrolled
            mid = state.sides_team_move_ids[side, slot, sm].astype(jnp.int32)
            safe_mid = jnp.clip(mid, 0, n_moves - 1)
            mrow = moves_table[safe_mid]
            mbp = mrow[MF_BASE_POWER].astype(jnp.int32)
            mtype = mrow[MF_TYPE].astype(jnp.int32)
            mcat = mrow[MF_CATEGORY].astype(jnp.int32)
            macc = mrow[MF_ACCURACY].astype(jnp.int32)
            d = _estimate_damage(
                mbp, mtype, mcat, macc,
                slot_types, slot_stats, opp_types, opp_stats, type_chart,
            )
            d = jnp.where(mid >= 0, d, 0.0)
            best_dmg = jnp.maximum(best_dmg, d)

        # Worst incoming damage from opponent
        worst_inc = jnp.float32(0.0)
        for om in range(4):  # unrolled
            oid = state.sides_team_move_ids[opp, opp_active_idx, om].astype(jnp.int32)
            safe_oid = jnp.clip(oid, 0, n_moves - 1)
            orow = moves_table[safe_oid]
            obp = orow[MF_BASE_POWER].astype(jnp.int32)
            otype = orow[MF_TYPE].astype(jnp.int32)
            ocat = orow[MF_CATEGORY].astype(jnp.int32)
            oacc = orow[MF_ACCURACY].astype(jnp.int32)
            inc = _estimate_damage(
                obp, otype, ocat, oacc,
                opp_types, opp_stats, slot_types, slot_stats, type_chart,
            )
            inc = jnp.where(oid >= 0, inc, 0.0)
            worst_inc = jnp.maximum(worst_inc, inc)

        switch_score = (best_dmg * 1.2 - worst_inc * 0.3) * slot_hp_frac
        # Skip if HP too low
        switch_score = jnp.where(slot_hp_frac < 0.15, -1e9, switch_score)
        # Ensure illegal switches get -inf
        switch_score = jnp.where(mask[4 + slot], switch_score, -1e9)
        scores = scores.at[4 + slot].set(switch_score)

    best_switch_score = jnp.max(scores[4:])

    # --- Decision: move vs switch ---
    # Switch if best_move_score <= 0 or switch_score > move_score * 2
    should_switch = (
        ((best_move_score <= 0) & (best_switch_score > -1e8)) |
        ((best_switch_score > best_move_score * 2.0) & (best_switch_score > -1e8))
    )

    # Override: if switch is chosen but no switch is legal, use best move
    any_switch_legal = mask[4:].any()
    any_move_legal = mask[:4].any()
    should_switch = should_switch & any_switch_legal

    # Select action
    move_action = jnp.argmax(scores[:4]).astype(jnp.int32)
    switch_action = (jnp.argmax(scores[4:]) + 4).astype(jnp.int32)

    action = jnp.where(should_switch, switch_action, move_action)

    # Fallback: if selected action is not legal, pick first legal
    fallback = jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)
    action = jnp.where(mask[action], action, fallback)

    return action


# ---------------------------------------------------------------------------
# MaxPower action: always pick highest base power move (pure JAX)
# ---------------------------------------------------------------------------

def maxpower_action(state: BattleState, side: int, tables: Tables,
                    rng_key: jnp.ndarray) -> jnp.ndarray:
    """
    Pick the highest-damage legal move, ignoring type effectiveness.
    Falls back to random legal action if no moves are legal.
    Forces the model to learn defensive play and switching.
    """
    moves_table = tables.moves
    type_chart = tables.type_chart
    mask = get_action_mask(state, side)
    active_idx = state.sides_active_idx[side]
    opp = 1 - side
    opp_active_idx = state.sides_active_idx[opp]

    own_types = state.sides_team_types[side, active_idx]
    opp_types = state.sides_team_types[opp, opp_active_idx]

    scores = jnp.full(10, -1e9, dtype=jnp.float32)
    n_moves = moves_table.shape[0]

    for m in range(4):
        move_id = state.sides_team_move_ids[side, active_idx, m].astype(jnp.int32)
        safe_id = jnp.clip(move_id, 0, n_moves - 1)
        row = moves_table[safe_id]
        bp = row[MF_BASE_POWER].astype(jnp.float32)
        move_type = row[MF_TYPE].astype(jnp.int32)

        # STAB bonus
        stab = jnp.where(
            (move_type == own_types[0]) | ((own_types[1] > 0) & (move_type == own_types[1])),
            1.5, 1.0,
        )
        # Type effectiveness
        eff = _type_eff(move_type, opp_types[0], opp_types[1], type_chart)

        score = bp * stab * eff
        score = jnp.where(mask[m] & (move_id >= 0), score, -1e9)
        scores = scores.at[m].set(score)

    # No switching — always attacks
    action = jnp.argmax(scores[:4]).astype(jnp.int32)
    # Fallback to any legal action if no move is legal
    fallback = jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)
    action = jnp.where(mask[action], action, fallback)
    return action


# ---------------------------------------------------------------------------
# Random action (pure JAX, JIT/vmap compatible)
# ---------------------------------------------------------------------------

def random_action(state: BattleState, side: int, rng_key: jnp.ndarray) -> jnp.ndarray:
    """
    Pick a uniformly random legal action. Fully JIT/vmap compatible.

    Returns int32 scalar: action index 0-9.
    """
    mask = get_action_mask(state, side)  # bool[10]
    # Use categorical sampling with uniform logits over legal actions
    logits = jnp.where(mask, 0.0, -1e9)
    return jax.random.categorical(rng_key, logits).astype(jnp.int32)
