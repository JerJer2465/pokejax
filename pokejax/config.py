"""
Generation configuration for PokeJAX.

GenConfig is a plain Python dataclass (NOT a JAX array) — it holds compile-time
constants that are passed as `static_argnums` to jax.jit.  One compiled
function exists per generation.

Usage:
    cfg = GenConfig.for_gen(4)
    step_fn = jax.jit(step, static_argnums=(2,))
    new_state = step_fn(state, actions, cfg)
"""

from dataclasses import dataclass
from typing import ClassVar, Dict


@dataclass(frozen=True)
class GenConfig:
    """Compile-time constants for a specific generation."""

    gen: int

    # -----------------------------------------------------------------------
    # Type system
    # -----------------------------------------------------------------------
    has_fairy_type: bool       # Gen 6+
    n_types: int               # 18 for Gen 1-5, 19 for Gen 6+

    # -----------------------------------------------------------------------
    # Mechanics flags
    # -----------------------------------------------------------------------
    # Status
    sleep_resets_on_switch: bool   # False in Gen 4; True in Gen 5+
    freeze_thaw_chance: int        # percentage (20 in Gen 1-4, 20 in Gen 5+)
    burn_damage_denom: int         # 16 (1/16 HP per turn, Gen 1 was 1/8)

    # Stat calculation
    has_special_split: bool        # True Gen 2+ (separate SpA/SpD)
    critical_hit_ratio_stages: int # stages 1-4 for Gen 4-5; different in Gen 1-2

    # Speed
    paralysis_speed_divisor: int   # 4 (Gen 4-6) → later 2 (Gen 7+)
    paralysis_full_para_chance: int # 25% (Gen 5+), 30% (Gen 1-4)

    # Field
    has_terrain: bool              # Gen 6+
    weather_turns_default: int     # 5 for Gen 4-5, 8 for natural; but moves set 5/8
    weather_turns_with_rock_item: int  # 8 (sandstorm with smooth rock, etc.)

    # Moves
    has_physical_special_split: bool  # True Gen 4+ (False = Gen 1-3 use type-based)
    flinch_check_before_move: bool    # True Gen 4+ (Gen 1-3 after)

    # Items / abilities
    has_abilities: bool            # Gen 3+
    has_held_items: bool           # Gen 2+
    has_berries: bool              # Gen 2+

    # Damage formula gen-specific parameters
    crit_damage_multiplier: float  # 2.0 in Gen 4-5; 1.5 in Gen 6+ (Showdown: Battle.py)

    # Gen-specific mechanics
    has_dynamax: bool              # Gen 8 only
    has_mega_evolution: bool       # Gen 6-7
    has_z_moves: bool              # Gen 7
    has_terastallization: bool     # Gen 9+

    # Entity counts (determines dispatch table sizes)
    n_species: int
    n_moves: int
    n_abilities: int
    n_items: int
    n_natures: int                 # always 25 (Gen 3+), 0 before

    # -----------------------------------------------------------------------
    # Registry
    # -----------------------------------------------------------------------
    _registry: ClassVar[Dict[int, "GenConfig"]] = {}

    @classmethod
    def for_gen(cls, gen: int) -> "GenConfig":
        if gen not in cls._registry:
            raise ValueError(f"No GenConfig registered for Gen {gen}. "
                             f"Available: {sorted(cls._registry)}")
        return cls._registry[gen]

    @classmethod
    def _register(cls, cfg: "GenConfig") -> None:
        cls._registry[cfg.gen] = cfg


# ---------------------------------------------------------------------------
# Gen 4 (Diamond/Pearl/Platinum/HeartGold/SoulSilver)
# ---------------------------------------------------------------------------
_GEN4 = GenConfig(
    gen=4,
    has_fairy_type=False,
    n_types=18,
    sleep_resets_on_switch=False,
    freeze_thaw_chance=20,
    burn_damage_denom=8,       # Gen 4: 1/8 HP per turn (changed to 1/16 in Gen 7)
    has_special_split=True,
    critical_hit_ratio_stages=4,
    paralysis_speed_divisor=4,
    paralysis_full_para_chance=25,
    has_terrain=False,
    weather_turns_default=5,
    weather_turns_with_rock_item=8,
    has_physical_special_split=True,
    flinch_check_before_move=True,
    has_abilities=True,
    has_held_items=True,
    has_berries=True,
    crit_damage_multiplier=2.0,  # Gen 4-5: 2× crit damage (Showdown getDamage)
    has_dynamax=False,
    has_mega_evolution=False,
    has_z_moves=False,
    has_terastallization=False,
    n_species=493,
    n_moves=467,
    n_abilities=123,
    n_items=200,   # approximate battle-relevant items
    n_natures=25,
)
GenConfig._register(_GEN4)

# ---------------------------------------------------------------------------
# Gen 5 (Black/White/B2W2)
# ---------------------------------------------------------------------------
_GEN5 = GenConfig(
    gen=5,
    has_fairy_type=False,
    n_types=18,
    sleep_resets_on_switch=True,
    freeze_thaw_chance=20,
    burn_damage_denom=8,
    has_special_split=True,
    critical_hit_ratio_stages=4,
    paralysis_speed_divisor=4,
    paralysis_full_para_chance=25,
    has_terrain=False,
    weather_turns_default=5,
    weather_turns_with_rock_item=8,
    has_physical_special_split=True,
    flinch_check_before_move=True,
    has_abilities=True,
    has_held_items=True,
    has_berries=True,
    crit_damage_multiplier=2.0,  # Gen 5: still 2× (same as Gen 4)
    has_dynamax=False,
    has_mega_evolution=False,
    has_z_moves=False,
    has_terastallization=False,
    n_species=649,
    n_moves=559,
    n_abilities=164,
    n_items=220,
    n_natures=25,
)
GenConfig._register(_GEN5)

# ---------------------------------------------------------------------------
# Gen 6 (X/Y/ORAS)
# ---------------------------------------------------------------------------
_GEN6 = GenConfig(
    gen=6,
    has_fairy_type=True,
    n_types=19,
    sleep_resets_on_switch=True,
    freeze_thaw_chance=20,
    burn_damage_denom=16,
    has_special_split=True,
    critical_hit_ratio_stages=3,
    paralysis_speed_divisor=4,
    paralysis_full_para_chance=25,
    has_terrain=True,
    weather_turns_default=5,
    weather_turns_with_rock_item=8,
    has_physical_special_split=True,
    flinch_check_before_move=True,
    has_abilities=True,
    has_held_items=True,
    has_berries=True,
    crit_damage_multiplier=1.5,  # Gen 6+: 1.5× crit (Showdown getDamage change)
    has_dynamax=False,
    has_mega_evolution=True,
    has_z_moves=False,
    has_terastallization=False,
    n_species=721,
    n_moves=621,
    n_abilities=191,
    n_items=240,
    n_natures=25,
)
GenConfig._register(_GEN6)

# ---------------------------------------------------------------------------
# Gen 7 (Sun/Moon/USUM)
# ---------------------------------------------------------------------------
_GEN7 = GenConfig(
    gen=7,
    has_fairy_type=True,
    n_types=19,
    sleep_resets_on_switch=True,
    freeze_thaw_chance=20,
    burn_damage_denom=16,
    has_special_split=True,
    critical_hit_ratio_stages=3,
    paralysis_speed_divisor=2,
    paralysis_full_para_chance=25,
    has_terrain=True,
    weather_turns_default=5,
    weather_turns_with_rock_item=8,
    has_physical_special_split=True,
    flinch_check_before_move=True,
    has_abilities=True,
    has_held_items=True,
    has_berries=True,
    crit_damage_multiplier=1.5,
    has_dynamax=False,
    has_mega_evolution=False,  # not in competitive Gen 7 randbats
    has_z_moves=True,
    has_terastallization=False,
    n_species=807,
    n_moves=728,
    n_abilities=233,
    n_items=260,
    n_natures=25,
)
GenConfig._register(_GEN7)

# ---------------------------------------------------------------------------
# Gen 8 (Sword/Shield)
# ---------------------------------------------------------------------------
_GEN8 = GenConfig(
    gen=8,
    has_fairy_type=True,
    n_types=19,
    sleep_resets_on_switch=True,
    freeze_thaw_chance=20,
    burn_damage_denom=16,
    has_special_split=True,
    critical_hit_ratio_stages=3,
    paralysis_speed_divisor=2,
    paralysis_full_para_chance=25,
    has_terrain=True,
    weather_turns_default=5,
    weather_turns_with_rock_item=8,
    has_physical_special_split=True,
    flinch_check_before_move=True,
    has_abilities=True,
    has_held_items=True,
    has_berries=True,
    crit_damage_multiplier=1.5,
    has_dynamax=True,
    has_mega_evolution=False,
    has_z_moves=False,
    has_terastallization=False,
    n_species=898,
    n_moves=841,
    n_abilities=267,
    n_items=280,
    n_natures=25,
)
GenConfig._register(_GEN8)
