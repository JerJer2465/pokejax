"""
JAX-native team sampler for PokeJAX.

Loads a pre-generated team pool (from scripts/generate_team_pool.py) and
samples teams at reset time using JAX PRNG. Fully JIT/vmap compatible.

Usage:
    pool = load_team_pool("data/team_pool.npz")
    team0, team1 = sample_teams(pool, key)  # int16[6, FIELDS], int16[6, FIELDS]
"""

import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp


# Field indices into the team pool array (matches generate_team_pool.py)
TF_SPECIES_ID = 0
TF_ABILITY_ID = 1
TF_ITEM_ID    = 2
TF_TYPE1      = 3
TF_TYPE2      = 4
TF_BASE_HP    = 5
TF_BASE_ATK   = 6
TF_BASE_DEF   = 7
TF_BASE_SPA   = 8
TF_BASE_SPD   = 9
TF_BASE_SPE   = 10
TF_MAX_HP     = 11
TF_MOVE_ID_0  = 12
TF_MOVE_ID_1  = 13
TF_MOVE_ID_2  = 14
TF_MOVE_ID_3  = 15
TF_MOVE_PP_0  = 16
TF_MOVE_PP_1  = 17
TF_MOVE_PP_2  = 18
TF_MOVE_PP_3  = 19
TF_LEVEL      = 20
TF_WEIGHT_HG  = 21


def load_team_pool(path: str | None = None) -> jnp.ndarray:
    """
    Load pre-generated team pool as a JAX array.

    Args:
        path: path to team_pool.npz. If None, looks in pokejax/data/team_pool.npz.

    Returns:
        int16[N_TEAMS, 6, FIELDS_PER_MON] JAX array
    """
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / 'data' / 'team_pool.npz')

    data = np.load(path, allow_pickle=True)
    teams = data['teams']  # int16[N, 6, F]
    return jnp.array(teams, dtype=jnp.int16)


def sample_teams(pool: jnp.ndarray, key: jax.random.PRNGKey):
    """
    Sample two teams from the pool using JAX PRNG.

    Args:
        pool: int16[N_TEAMS, 6, FIELDS_PER_MON] — the team pool
        key: JAX PRNG key

    Returns:
        (team0, team1): each int16[6, FIELDS_PER_MON]
    """
    k1, k2 = jax.random.split(key)
    n_teams = pool.shape[0]
    idx0 = jax.random.randint(k1, (), 0, n_teams)
    idx1 = jax.random.randint(k2, (), 0, n_teams)
    return pool[idx0], pool[idx1]


def team_to_state_arrays(team: jnp.ndarray) -> dict:
    """
    Convert a team array into the dict format expected by make_battle_state.

    Args:
        team: int16[6, FIELDS_PER_MON]

    Returns:
        dict with keys matching make_battle_state kwargs:
            species, abilities, items, types, base_stats, max_hp,
            move_ids, move_pp, move_max_pp, levels, genders, natures, weights_hg
    """
    n = 6
    species   = team[:, TF_SPECIES_ID]
    abilities = team[:, TF_ABILITY_ID]
    items     = team[:, TF_ITEM_ID]

    types = jnp.stack([team[:, TF_TYPE1], team[:, TF_TYPE2]], axis=1).astype(jnp.int8)

    base_stats = jnp.stack([
        team[:, TF_BASE_HP],
        team[:, TF_BASE_ATK],
        team[:, TF_BASE_DEF],
        team[:, TF_BASE_SPA],
        team[:, TF_BASE_SPD],
        team[:, TF_BASE_SPE],
    ], axis=1)

    max_hp = team[:, TF_MAX_HP]

    move_ids = jnp.stack([
        team[:, TF_MOVE_ID_0],
        team[:, TF_MOVE_ID_1],
        team[:, TF_MOVE_ID_2],
        team[:, TF_MOVE_ID_3],
    ], axis=1)

    move_pp = jnp.stack([
        team[:, TF_MOVE_PP_0],
        team[:, TF_MOVE_PP_1],
        team[:, TF_MOVE_PP_2],
        team[:, TF_MOVE_PP_3],
    ], axis=1).astype(jnp.int8)

    move_max_pp = move_pp  # same for randbats

    levels = team[:, TF_LEVEL].astype(jnp.int8)
    genders = jnp.zeros(n, dtype=jnp.int8)
    natures = jnp.zeros(n, dtype=jnp.int8)
    weights_hg = team[:, TF_WEIGHT_HG]

    return {
        'species': species, 'abilities': abilities, 'items': items,
        'types': types, 'base_stats': base_stats, 'max_hp': max_hp,
        'move_ids': move_ids, 'move_pp': move_pp, 'move_max_pp': move_max_pp,
        'levels': levels, 'genders': genders, 'natures': natures,
        'weights_hg': weights_hg,
    }
