"""
Static lookup tables for PokeJAX.

Loaded once at import time from .npy files and stored as jnp.arrays.
These are DeviceArrays on the accelerator — no host-device transfer at step time.

Tables are generation-specific: call load_tables(gen) to get a Tables object.
The Tables object is NOT a JAX pytree; it holds constant lookup arrays that
are captured in the closure of jit-compiled functions via `static_argnums`
or simply as module-level globals after `init(gen)`.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import jax.numpy as jnp

from pokejax.data.extractor import (
    _build_type_chart,
    _build_nature_table,
    NATURE_NAMES,
    SPECIES_FIELDS,
    MOVE_FIELDS,
    ITEM_FIELDS,
)
from pokejax.data.move_effects_data import (
    build_move_effects_table,
    MOVE_EFFECT_FIELDS,
)
from pokejax.types import N_TYPES, TYPE_NAMES


@dataclass
class Tables:
    """All static lookup tables for one generation."""

    gen: int

    # Type chart: float32[N_TYPES, N_TYPES]
    # type_chart[atk_type, def_type] = damage multiplier
    type_chart: jnp.ndarray

    # Nature multipliers: float32[25, 5]
    # natures[nature_id, stat_col] where stat_col ∈ {atk,def,spa,spd,spe} = 0..4
    natures: jnp.ndarray

    # Species data: int16[N_SPECIES, SPECIES_FIELDS]
    species: jnp.ndarray
    species_names: List[str]
    species_name_to_id: Dict[str, int]

    # Move data: int16[N_MOVES, MOVE_FIELDS]
    moves: jnp.ndarray
    move_names: List[str]
    move_name_to_id: Dict[str, int]

    # Move effects: int16[N_MOVES, MOVE_EFFECT_FIELDS]
    # Encodes self-boosts, hazards, screens, weather, volatile effects, etc.
    # Fields: [effect_type, stat1, amt1, stat2, amt2, stat3, amt3, flags]
    move_effects: jnp.ndarray

    # Ability name → ID mapping (populated from Showdown or empty dict)
    ability_name_to_id: Dict[str, int]
    ability_names: List[str]

    # Item data: int16[N_ITEMS, ITEM_FIELDS] + name → ID mapping
    items: jnp.ndarray
    item_names: List[str]
    item_name_to_id: Dict[str, int]

    # Boost multiplier table: float32[13] indexed by boost+6 (range -6..+6 → 0..12)
    # Values: [2/8, 2/7, 2/6, 2/5, 2/4, 2/3, 2/2, 3/2, 4/2, 5/2, 6/2, 7/2, 8/2]
    boost_multipliers: jnp.ndarray

    # Accuracy boost multiplier table: same indexing as boost_multipliers
    # Evasion uses the inverse.
    acc_multipliers: jnp.ndarray

    def get_type_effectiveness(self, atk_type: int | jnp.ndarray,
                               def_type1: int | jnp.ndarray,
                               def_type2: int | jnp.ndarray) -> jnp.ndarray:
        """Combined effectiveness against a dual-typed defender."""
        eff1 = self.type_chart[atk_type, def_type1]
        eff2 = self.type_chart[atk_type, def_type2]
        # If def_type2 == 0 (sentinel) treat as 1.0
        eff2 = jnp.where(def_type2 == 0, jnp.float32(1.0), eff2)
        return eff1 * eff2

    def get_boost_multiplier(self, boost: jnp.ndarray) -> jnp.ndarray:
        """Return stat multiplier for a boost stage in [-6, +6]."""
        return self.boost_multipliers[boost + 6]

    def get_acc_multiplier(self, boost: jnp.ndarray) -> jnp.ndarray:
        """Return accuracy multiplier for a boost stage."""
        return self.acc_multipliers[boost + 6]

    def get_evasion_multiplier(self, boost: jnp.ndarray) -> jnp.ndarray:
        """Return evasion multiplier — inverse of accuracy boosts."""
        return self.acc_multipliers[-boost + 6]


def _make_boost_table() -> jnp.ndarray:
    """Build boost multiplier array indexed by boost+6."""
    # Pokemon stat boost formula: max(2, 2+stage) / max(2, 2-stage)
    stages = np.arange(-6, 7, dtype=np.float32)
    nums = np.maximum(2.0, 2.0 + stages)
    dens = np.maximum(2.0, 2.0 - stages)
    return jnp.array(nums / dens, dtype=jnp.float32)


def _make_acc_table() -> jnp.ndarray:
    """Accuracy multipliers: 3/3, 3/4, 3/5, ... at negative; ... 9/3 at +6."""
    # Pokemon accuracy formula: (3+stage)/3 for positive, 3/(3-stage) for negative...
    # Actually simpler: same formula as stat boosts
    stages = np.arange(-6, 7, dtype=np.float32)
    nums = np.maximum(3.0, 3.0 + stages)
    dens = np.maximum(3.0, 3.0 - stages)
    return jnp.array(nums / dens, dtype=jnp.float32)


def _load_or_generate_species(gen_dir: Optional[Path], gen: int
                               ) -> tuple:
    """Load species data from disk or return minimal placeholder."""
    if gen_dir and (gen_dir / "species.npy").exists():
        data = np.load(gen_dir / "species.npy")
        with open(gen_dir / "species_name_to_id.pkl", "rb") as f:
            name_to_id = pickle.load(f)
        with open(gen_dir / "species_names.pkl", "rb") as f:
            names = pickle.load(f)
        return data, name_to_id, names

    # Minimal placeholder: 1 species (Bulbasaur-like stats)
    n = 1
    data = np.zeros((n, SPECIES_FIELDS), dtype=np.int16)
    data[0] = [45, 49, 49, 65, 65, 45, 5, 8, 69, 0, 0, 0]  # Bulbasaur
    names = ["Bulbasaur"]
    name_to_id = {"Bulbasaur": 0}
    return data, name_to_id, names


def _load_or_generate_moves(gen_dir: Optional[Path], gen: int
                             ) -> tuple:
    """Load move data from disk or return minimal placeholder."""
    if gen_dir and (gen_dir / "moves.npy").exists():
        data = np.load(gen_dir / "moves.npy")
        with open(gen_dir / "move_name_to_id.pkl", "rb") as f:
            name_to_id = pickle.load(f)
        with open(gen_dir / "move_names.pkl", "rb") as f:
            names = pickle.load(f)
        return data, name_to_id, names

    # Minimal placeholder: Tackle + Growl
    n = 2
    data = np.zeros((n, MOVE_FIELDS), dtype=np.int16)
    # Tackle: 40 bp, 100 acc, Normal, Physical, 0 priority, 35 pp, normal target
    data[0] = [40, 100, 1, 0, 0, 35, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Growl: 0 bp, 100 acc, Normal, Status, 0 priority, 40 pp, allAdjacentFoes
    data[1] = [0,  100, 1, 2, 0, 40, 5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    names = ["Tackle", "Growl"]
    name_to_id = {"Tackle": 0, "Growl": 1}
    return data, name_to_id, names


def _load_or_generate_abilities(gen_dir: Optional[Path]) -> tuple:
    """Load ability name mapping from disk or return empty mapping."""
    if gen_dir and (gen_dir / "ability_name_to_id.pkl").exists():
        with open(gen_dir / "ability_name_to_id.pkl", "rb") as f:
            name_to_id = pickle.load(f)
        with open(gen_dir / "ability_names.pkl", "rb") as f:
            names = pickle.load(f)
        return name_to_id, names
    return {}, []


def _load_or_generate_items(gen_dir: Optional[Path]) -> tuple:
    """Load item data from disk or return empty placeholder."""
    if gen_dir and (gen_dir / "items.npy").exists():
        data = np.load(gen_dir / "items.npy")
        with open(gen_dir / "item_name_to_id.pkl", "rb") as f:
            name_to_id = pickle.load(f)
        with open(gen_dir / "item_names.pkl", "rb") as f:
            names = pickle.load(f)
        return data, name_to_id, names
    # Minimal placeholder: Leftovers + Life Orb
    n = 2
    data = np.zeros((n, ITEM_FIELDS), dtype=np.int16)
    names = ["Leftovers", "Life Orb"]
    name_to_id = {"Leftovers": 0, "Life Orb": 1}
    return data, name_to_id, names


def load_tables(gen: int, showdown_path: Optional[str] = None) -> Tables:
    """
    Load all static tables for the given generation.

    If .npy files exist in pokejax/data/gen{N}/, loads from disk.
    Otherwise uses hard-coded type chart and minimal placeholders.
    Pass showdown_path to auto-extract if files are missing.
    """
    gen_dir = Path(__file__).parent / f"gen{gen}"
    gen_dir_exists = gen_dir.exists() and (gen_dir / "type_chart.npy").exists()

    if not gen_dir_exists and showdown_path:
        from pokejax.data.extractor import extract
        print(f"Extracting gen {gen} data from {showdown_path}...")
        extract(Path(showdown_path), gen, gen_dir)
        gen_dir_exists = True

    # Type chart
    if gen_dir_exists and (gen_dir / "type_chart.npy").exists():
        type_chart_np = np.load(gen_dir / "type_chart.npy")
    else:
        type_chart_np = _build_type_chart(gen)

    # Nature table
    if gen_dir_exists and (gen_dir / "natures.npy").exists():
        natures_np = np.load(gen_dir / "natures.npy")
    else:
        natures_np = _build_nature_table()

    # Species + moves + abilities + items
    opt_gen_dir = gen_dir if gen_dir_exists else None
    species_data, species_name_to_id, species_names = _load_or_generate_species(opt_gen_dir, gen)
    move_data, move_name_to_id, move_names = _load_or_generate_moves(opt_gen_dir, gen)
    ability_name_to_id, ability_names = _load_or_generate_abilities(opt_gen_dir)
    item_data, item_name_to_id, item_names = _load_or_generate_items(opt_gen_dir)

    # Build move effects table from hardcoded Gen 4 data
    move_effects_np = build_move_effects_table(move_name_to_id, len(move_data))

    tables = Tables(
        gen=gen,
        type_chart=jnp.array(type_chart_np),
        natures=jnp.array(natures_np),
        species=jnp.array(species_data),
        species_names=species_names,
        species_name_to_id=species_name_to_id,
        moves=jnp.array(move_data),
        move_names=move_names,
        move_name_to_id=move_name_to_id,
        move_effects=jnp.array(move_effects_np),
        ability_name_to_id=ability_name_to_id,
        ability_names=ability_names,
        items=jnp.array(item_data),
        item_names=item_names,
        item_name_to_id=item_name_to_id,
        boost_multipliers=_make_boost_table(),
        acc_multipliers=_make_acc_table(),
    )

    # Populate ability and item dispatch tables
    from pokejax.mechanics.abilities import populate_ability_tables
    from pokejax.mechanics.items import populate_item_tables
    populate_ability_tables(ability_name_to_id, tables)
    populate_item_tables(item_name_to_id, tables)

    return tables
