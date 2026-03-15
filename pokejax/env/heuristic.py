"""
Smart heuristic opponent for BC teacher.

Operates on raw JAX BattleState + Tables — no dict conversion needed.
Runs on CPU (Python), NOT inside jax.jit. Used only during BC data collection.

Actions: 0-3 = moves, 4-9 = switch to team slot 0-5.
"""

import numpy as np
from pokejax.types import (
    BattleState,
    CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS,
    STATUS_NONE,
    SC_STEALTHROCK, SC_SPIKES, SC_TOXICSPIKES,
    VOL_SUBSTITUTE,
    BOOST_ATK, BOOST_SPA,
)
from pokejax.data.tables import Tables
from pokejax.env.action_mask import get_action_mask


# Move data column indices (from extractor.py MOVE_FIELDS layout)
MF_BASE_POWER = 0
MF_ACCURACY = 1
MF_TYPE = 2
MF_CATEGORY = 3
MF_PRIORITY = 4
MF_PP = 5

# Move effect columns
ME_EFFECT_TYPE = 0

# Effect type codes for status moves (from move_effects_data.py)
EFFECT_HAZARD = 1
EFFECT_SCREEN = 2
EFFECT_WEATHER = 3
EFFECT_SELF_BOOST = 4
EFFECT_VOLATILE = 5
EFFECT_STATUS = 6

# Well-known move names for status scoring
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
_SCREEN_MOVES = {"reflect", "lightscreen", "auroraveil"}
_HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes"}
_FIXED_DAMAGE_MOVES = {"seismictoss", "nightshade"}


def _state_to_numpy(state: BattleState) -> dict:
    """Bulk-convert relevant BattleState fields to numpy once.
    Avoids repeated device→host transfers.
    """
    return {
        'active_idx': np.array(state.sides_active_idx),           # int8[2]
        'types': np.array(state.sides_team_types),                # int8[2,6,2]
        'base_stats': np.array(state.sides_team_base_stats),      # int16[2,6,6]
        'hp': np.array(state.sides_team_hp),                      # int16[2,6]
        'max_hp': np.array(state.sides_team_max_hp),              # int16[2,6]
        'boosts': np.array(state.sides_team_boosts),              # int8[2,6,7]
        'status': np.array(state.sides_team_status),              # int8[2,6]
        'fainted': np.array(state.sides_team_fainted),            # bool[2,6]
        'move_ids': np.array(state.sides_team_move_ids),          # int16[2,6,4]
        'side_conditions': np.array(state.sides_side_conditions), # int8[2,10]
        'turn': int(state.turn),
    }


def smart_heuristic_action(
    state: BattleState,
    side: int,
    tables: Tables,
    _np_cache: dict = None,
) -> int:
    """
    Type-aware heuristic that considers STAB, type effectiveness, accuracy,
    status moves, setup moves, and switches to better matchups.

    Returns an action index (0-9).

    Pass _np_cache from _state_to_numpy() to avoid repeated device→host transfers.
    """
    mask = np.array(get_action_mask(state, side))
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return 0

    move_actions = [a for a in legal if a < 4]
    switch_actions = [a for a in legal if a >= 4]

    # Use cache or convert
    s = _np_cache if _np_cache is not None else _state_to_numpy(state)

    active_idx = int(s['active_idx'][side])
    opp_side = 1 - side
    opp_active_idx = int(s['active_idx'][opp_side])
    turn = s['turn']

    # Active mon info
    own_types = s['types'][side, active_idx]
    own_stats = s['base_stats'][side, active_idx]
    own_hp = int(s['hp'][side, active_idx])
    own_max_hp = int(s['max_hp'][side, active_idx])
    own_hp_frac = own_hp / max(own_max_hp, 1)
    own_boosts = s['boosts'][side, active_idx]

    # Opponent active info
    opp_types = s['types'][opp_side, opp_active_idx]
    opp_stats = s['base_stats'][opp_side, opp_active_idx]
    opp_hp = int(s['hp'][opp_side, opp_active_idx])
    opp_max_hp = int(s['max_hp'][opp_side, opp_active_idx])
    opp_status = int(s['status'][opp_side, opp_active_idx])

    # Opponent's side conditions
    opp_sc = s['side_conditions'][opp_side]

    # --- Score each legal move ---
    best_move_a = None
    best_move_score = -1.0

    for a in move_actions:
        move_id = int(s['move_ids'][side, active_idx, a])
        if move_id < 0:
            continue
        move_data = _get_np_tables(tables)['moves'][move_id]  # int16[MOVE_FIELDS]
        bp = int(move_data[MF_BASE_POWER])
        move_type = int(move_data[MF_TYPE])
        category = int(move_data[MF_CATEGORY])
        accuracy = int(move_data[MF_ACCURACY])
        if accuracy > 100:
            accuracy = 100  # never-miss sentinel

        move_name = tables.move_names[move_id].lower().replace(" ", "").replace("-", "") if move_id < len(tables.move_names) else ""

        if bp > 0:
            score = _estimate_damage(
                bp, move_type, category, accuracy,
                own_types, own_stats,
                opp_types, opp_stats,
                tables,
            )
            # Fixed damage moves
            if move_name in _FIXED_DAMAGE_MOVES:
                eff = _type_eff(move_type, opp_types, tables)
                score = 100.0 if eff > 0 else 0.0
        else:
            score = _score_status_move(
                move_name, move_type, accuracy,
                own_hp_frac, own_boosts, own_types,
                opp_status, opp_stats, opp_sc,
                turn, tables,
            )

        if score > best_move_score:
            best_move_score = score
            best_move_a = a

    # --- Score switching options ---
    best_switch_a = None
    best_switch_score = -1.0

    for a in switch_actions:
        slot = a - 4
        slot_hp = int(s['hp'][side, slot])
        slot_max_hp = int(s['max_hp'][side, slot])
        if slot_max_hp <= 0:
            continue
        hp_frac = slot_hp / max(slot_max_hp, 1)
        if hp_frac < 0.15:
            continue

        slot_types = s['types'][side, slot]
        slot_stats = s['base_stats'][side, slot]

        # Best damage this mon can do to opponent
        best_dmg = 0.0
        for m in range(4):
            mid = int(s['move_ids'][side, slot, m])
            if mid < 0:
                continue
            md = _get_np_tables(tables)['moves'][mid]
            mbp = int(md[MF_BASE_POWER])
            if mbp <= 0:
                continue
            dmg = _estimate_damage(
                mbp, int(md[MF_TYPE]), int(md[MF_CATEGORY]),
                min(int(md[MF_ACCURACY]), 100),
                slot_types, slot_stats,
                opp_types, opp_stats, tables,
            )
            best_dmg = max(best_dmg, dmg)

        # Worst incoming damage from opponent
        worst_incoming = 0.0
        for m in range(4):
            mid = int(s['move_ids'][opp_side, opp_active_idx, m])
            if mid < 0:
                continue
            md = _get_np_tables(tables)['moves'][mid]
            mbp = int(md[MF_BASE_POWER])
            if mbp <= 0:
                continue
            inc = _estimate_damage(
                mbp, int(md[MF_TYPE]), int(md[MF_CATEGORY]),
                min(int(md[MF_ACCURACY]), 100),
                opp_types, opp_stats,
                slot_types, slot_stats, tables,
            )
            worst_incoming = max(worst_incoming, inc)

        switch_score = (best_dmg * 1.2 - worst_incoming * 0.3) * hp_frac
        if switch_score > best_switch_score:
            best_switch_score = switch_score
            best_switch_a = a

    # --- Decision: move vs switch ---
    should_switch = False
    if best_switch_a is not None and best_move_a is not None:
        if best_move_score <= 0:
            should_switch = True
        elif best_switch_score > best_move_score * 2.0:
            should_switch = True

    if should_switch and best_switch_a is not None:
        return int(best_switch_a)
    if best_move_a is not None:
        return int(best_move_a)
    if switch_actions:
        return int(switch_actions[0])
    return int(legal[0])


_cached_np_tables = {}

def _get_np_tables(tables: Tables) -> dict:
    """Cache numpy versions of tables for fast CPU access."""
    tid = id(tables)
    if tid not in _cached_np_tables:
        _cached_np_tables[tid] = {
            'type_chart': np.array(tables.type_chart),
            'moves': np.array(tables.moves),
        }
    return _cached_np_tables[tid]


def _type_eff(atk_type: int, def_types: np.ndarray, tables: Tables) -> float:
    """Type effectiveness: atk_type vs defender's dual types."""
    tc = _get_np_tables(tables)['type_chart']
    eff = float(tc[atk_type, int(def_types[0])])
    if int(def_types[1]) != 0:  # second type exists
        eff *= float(tc[atk_type, int(def_types[1])])
    return eff


def _estimate_damage(
    bp: int,
    move_type: int,
    category: int,
    accuracy: int,
    atk_types: np.ndarray,
    atk_stats: np.ndarray,
    def_types: np.ndarray,
    def_stats: np.ndarray,
    tables: Tables,
) -> float:
    """Rough damage estimate for move scoring."""
    # STAB
    stab = 1.5 if (move_type == int(atk_types[0]) or
                    (int(atk_types[1]) != 0 and move_type == int(atk_types[1]))) else 1.0

    # Type effectiveness
    eff = _type_eff(move_type, def_types, tables)

    # Atk/Def ratio based on category
    if category == int(CATEGORY_PHYSICAL):
        atk_stat = max(int(atk_stats[1]), 1)  # ATK
        def_stat = max(int(def_stats[2]), 1)  # DEF
    elif category == int(CATEGORY_SPECIAL):
        atk_stat = max(int(atk_stats[3]), 1)  # SPA
        def_stat = max(int(def_stats[4]), 1)  # SPD
    else:
        return 0.0  # Status move, no damage

    score = bp * (atk_stat / def_stat) * stab * eff * (accuracy / 100.0)
    return score


def _score_status_move(
    move_name: str,
    move_type: int,
    accuracy: int,
    own_hp_frac: float,
    own_boosts: np.ndarray,
    own_types: np.ndarray,
    opp_status: int,
    opp_stats: np.ndarray,
    opp_sc: np.ndarray,
    turn: int,
    tables: Tables,
) -> float:
    """Score a status (bp=0) move strategically."""
    # --- Entry hazards ---
    if move_name == "stealthrock":
        if int(opp_sc[SC_STEALTHROCK]) == 0:
            return 250.0 if turn <= 3 else 150.0
        return 0.0

    if move_name == "spikes":
        layers = int(opp_sc[SC_SPIKES])
        if layers < 3:
            return 180.0 if turn <= 4 else 100.0
        return 0.0

    if move_name == "toxicspikes":
        layers = int(opp_sc[SC_TOXICSPIKES])
        if layers < 2:
            return 160.0 if turn <= 4 else 80.0
        return 0.0

    # --- Sleep moves (highest value) ---
    if move_name in _SLEEP_MOVES:
        if opp_status != int(STATUS_NONE):
            return 0.0
        return 300.0 * (accuracy / 100.0)

    # --- Paralysis ---
    if move_name in _PARA_MOVES:
        if opp_status != int(STATUS_NONE):
            return 0.0
        opp_speed = max(int(opp_stats[5]), 1)
        return (200.0 if opp_speed > 90 else 120.0) * (accuracy / 100.0)

    # --- Toxic ---
    if move_name in _TOXIC_MOVES:
        if opp_status != int(STATUS_NONE):
            return 0.0
        opp_hp_frac_approx = 1.0  # We don't have exact opp HP frac but assume high
        return 180.0 * (accuracy / 100.0)

    # --- Burn ---
    if move_name in _BURN_MOVES:
        if opp_status != int(STATUS_NONE):
            return 0.0
        opp_atk = max(int(opp_stats[1]), 1)
        return (180.0 if opp_atk > 90 else 100.0) * (accuracy / 100.0)

    # --- Setup moves ---
    if move_name in _SETUP_MOVES:
        atk_boost = int(own_boosts[BOOST_ATK]) + int(own_boosts[BOOST_SPA])
        if atk_boost < 2 and own_hp_frac > 0.6:
            return 200.0 if turn <= 5 else 120.0
        return 0.0

    # --- Recovery ---
    if move_name in _RECOVERY_MOVES:
        if own_hp_frac < 0.5:
            return 150.0
        return 0.0

    # --- Substitute ---
    if move_name == "substitute":
        if own_hp_frac > 0.3:
            return 80.0
        return 0.0

    # --- Protect ---
    if move_name == "protect" or move_name == "detect":
        return 30.0

    # --- Screens ---
    if move_name in _SCREEN_MOVES:
        return 120.0 if turn <= 3 else 60.0

    # --- Taunt ---
    if move_name == "taunt":
        return 100.0

    # Default small positive
    return 20.0


def random_action(state: BattleState, side: int) -> int:
    """Pick a random legal action (for BC opponent)."""
    mask = np.array(get_action_mask(state, side))
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return 0
    return int(np.random.choice(legal))
