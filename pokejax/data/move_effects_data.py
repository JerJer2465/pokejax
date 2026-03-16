"""
Hardcoded Gen 4 move effects data.

Move effects that cannot be encoded in the basic move table fields
(damage, drain, recoil, secondary status, secondary stat changes, healing)
are stored here, keyed by move display name.

At table-load time these are resolved to move IDs via name_to_id and stored
as a jnp.int16 array of shape (N_MOVES, MOVE_EFFECT_FIELDS).

Effect table layout  (MOVE_EFFECT_FIELDS = 8):
  [0] effect_type   — see ME_* constants below
  [1] stat1         — primary stat/param (BOOST_* for boosts, SC_* for hazards, etc.)
  [2] amount1       — magnitude (boost amount, turns, max layers)
  [3] stat2         — second stat (NONE_STAT if unused)
  [4] amount2
  [5] stat3         — third stat (NONE_STAT if unused)
  [6] amount3
  [7] flags         — reserved / future use
"""

import numpy as np

# ---------------------------------------------------------------------------
# Effect type codes
# ---------------------------------------------------------------------------
ME_NONE          = 0   # no special effect beyond basic table fields
ME_SELF_BOOST    = 1   # boost attacker's stats (stat1/amt1, stat2/amt2, stat3/amt3)
ME_FOE_LOWER     = 2   # lower/boost foe's stats (same fields as SELF_BOOST)
ME_HAZARD        = 3   # set entry hazard on foe's side (stat1=SC_idx, amt1=max_layers)
ME_SCREEN        = 4   # set screen/field effect on own side (stat1=SC_idx, amt1=turns)
ME_WEATHER       = 5   # set weather (stat1=weather_id, amt1=turns)
ME_TERRAIN       = 6   # set terrain (stat1=terrain_id, amt1=turns) — Gen 6+
ME_TRICK_ROOM    = 7   # toggle trick room
ME_VOLATILE_SELF = 8   # apply volatile bitmask to self (stat1=vol_bit)
ME_VOLATILE_FOE  = 9   # apply volatile bitmask to foe (stat1=vol_bit)
ME_SUBSTITUTE    = 10  # create substitute at 25% max HP cost
ME_RAPID_SPIN    = 11  # remove hazards + volatile trapping from own side
ME_ROAR          = 12  # force foe to switch to random alive mon
ME_U_TURN        = 13  # damage + user switches out (auto-switch to first alive)
ME_BATON_PASS    = 14  # switch out, passing boosts and select volatiles
ME_RECOVERY      = 15  # heal 50% max HP (stat1=numerator, amt1=denominator)
ME_REST          = 16  # full heal + sleep for 2 turns
ME_BELLY_DRUM    = 17  # maximize Atk at 50% HP cost
ME_KNOCK_OFF     = 18  # remove foe's item
ME_PAIN_SPLIT    = 19  # average HP between attacker and defender
ME_WISH          = 20  # delayed heal: heals 50% max HP at end of next turn
ME_HEAL_BELL     = 21  # cure team status
ME_DISABLE       = 22  # disable foe's last used move
ME_YAWN          = 23  # inflict drowsiness → sleep next turn
ME_DESTINY_BOND  = 24  # if user faints, foe faints too
ME_PERISH_SONG   = 25  # both sides get 3-turn KO timer
ME_SLEEP_TALK    = 26  # use random move while asleep (simplified as noop)
ME_DEFOG         = 27  # clear hazards + screens from both sides
ME_TRICK         = 28  # swap items between attacker and defender
ME_HAZE          = 29  # reset all stat boosts to 0
ME_TWO_TURN      = 30  # two-turn move: charge turn (set VOL_CHARGING) then release

# Number of fields per move in the effect table
MOVE_EFFECT_FIELDS = 8

# Sentinel: "no stat / not used"
NONE_STAT = 15

# ---------------------------------------------------------------------------
# Stat boost indices (must match BOOST_* in types.py)
# ---------------------------------------------------------------------------
_BOOST_ATK = 0
_BOOST_DEF = 1
_BOOST_SPA = 2
_BOOST_SPD = 3
_BOOST_SPE = 4
_BOOST_ACC = 5
_BOOST_EVA = 6

# ---------------------------------------------------------------------------
# Side condition indices (must match SC_* in types.py)
# ---------------------------------------------------------------------------
_SC_SPIKES      = 0
_SC_TOXICSPIKES = 1
_SC_STEALTHROCK = 2
_SC_STICKYWEB   = 3
_SC_REFLECT     = 4
_SC_LIGHTSCREEN = 5
_SC_AURORAVEIL  = 6
_SC_TAILWIND    = 7
_SC_SAFEGUARD   = 8
_SC_MIST        = 9

# ---------------------------------------------------------------------------
# Weather codes (must match WEATHER_* in types.py)
# ---------------------------------------------------------------------------
_WEATHER_SUN  = 1
_WEATHER_RAIN = 2
_WEATHER_SAND = 3
_WEATHER_HAIL = 4

# ---------------------------------------------------------------------------
# Terrain codes (must match TERRAIN_* in types.py)
# ---------------------------------------------------------------------------
_TERRAIN_ELECTRIC = 1
_TERRAIN_GRASSY   = 2
_TERRAIN_MISTY    = 3
_TERRAIN_PSYCHIC  = 4

# ---------------------------------------------------------------------------
# Volatile bit indices (must match VOL_* in types.py)
# ---------------------------------------------------------------------------
_VOL_CONFUSED   = 0
_VOL_SEEDED     = 3
_VOL_PROTECT    = 5
_VOL_ENCORE     = 6
_VOL_TAUNT      = 7
_VOL_TORMENT    = 8
_VOL_ENDURE     = 10
_VOL_INGRAIN    = 13
_VOL_HEALBLOCK  = 15
_VOL_FOCUSENERGY  = 21
_VOL_MINIMIZED    = 22
_VOL_CURSE        = 23
_VOL_NIGHTMARE    = 24
_VOL_ATTRACT      = 25
_VOL_YAWN         = 26
_VOL_DESTINYBOND  = 27
_VOL_SUBSTITUTE   = 4
_VOL_DISABLE      = 9

# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------

def _boost(s1, a1, s2=NONE_STAT, a2=0, s3=NONE_STAT, a3=0):
    """Self stat boost effect."""
    return (ME_SELF_BOOST, s1, a1, s2, a2, s3, a3, 0)

def _lower(s1, a1, s2=NONE_STAT, a2=0, s3=NONE_STAT, a3=0):
    """Foe stat lower effect (amounts should be negative)."""
    return (ME_FOE_LOWER, s1, a1, s2, a2, s3, a3, 0)

def _hazard(sc_idx, max_layers):
    """Entry hazard on foe's side."""
    return (ME_HAZARD, sc_idx, max_layers, NONE_STAT, 0, NONE_STAT, 0, 0)

def _screen(sc_idx, turns=5):
    """Screen / field effect on own side."""
    return (ME_SCREEN, sc_idx, turns, NONE_STAT, 0, NONE_STAT, 0, 0)

def _weather(weather_id, turns=5):
    """Set weather."""
    return (ME_WEATHER, weather_id, turns, NONE_STAT, 0, NONE_STAT, 0, 0)

def _terrain(terrain_id, turns=5):
    """Set terrain."""
    return (ME_TERRAIN, terrain_id, turns, NONE_STAT, 0, NONE_STAT, 0, 0)

def _volatile_self(vol_bit):
    """Apply volatile status to self."""
    return (ME_VOLATILE_SELF, vol_bit, 0, NONE_STAT, 0, NONE_STAT, 0, 0)

def _volatile_foe(vol_bit):
    """Apply volatile status to foe."""
    return (ME_VOLATILE_FOE, vol_bit, 0, NONE_STAT, 0, NONE_STAT, 0, 0)

def _simple_effect(me_type):
    """Simple effect with no parameters."""
    return (me_type, NONE_STAT, 0, NONE_STAT, 0, NONE_STAT, 0, 0)

def _recovery(num=1, den=2):
    """Recovery move: heal num/den of max HP."""
    return (ME_RECOVERY, num, den, NONE_STAT, 0, NONE_STAT, 0, 0)

# ---------------------------------------------------------------------------
# Hardcoded Gen 4 move effects
# Keys are Showdown display names (matching Tables.move_names).
# Values are 8-tuples: (effect_type, stat1, amt1, stat2, amt2, stat3, amt3, flags)
# ---------------------------------------------------------------------------

GEN4_MOVE_EFFECTS = {
    # ---- Self-stat boost moves ----
    "Swords Dance":   _boost(_BOOST_ATK, 2),
    "Nasty Plot":     _boost(_BOOST_SPA, 2),
    "Calm Mind":      _boost(_BOOST_SPA, 1, _BOOST_SPD, 1),
    "Dragon Dance":   _boost(_BOOST_ATK, 1, _BOOST_SPE, 1),
    "Agility":        _boost(_BOOST_SPE, 2),
    "Bulk Up":        _boost(_BOOST_ATK, 1, _BOOST_DEF, 1),
    "Rock Polish":    _boost(_BOOST_SPE, 2),
    "Cosmic Power":   _boost(_BOOST_DEF, 1, _BOOST_SPD, 1),
    "Iron Defense":   _boost(_BOOST_DEF, 2),
    "Barrier":        _boost(_BOOST_DEF, 2),
    "Acid Armor":     _boost(_BOOST_DEF, 2),
    "Amnesia":        _boost(_BOOST_SPD, 2),
    "Growth":         _boost(_BOOST_SPA, 1),
    "Double Team":    _boost(_BOOST_EVA, 1),
    "Minimize":       _boost(_BOOST_EVA, 2),
    "Meditate":       _boost(_BOOST_ATK, 1),
    "Howl":           _boost(_BOOST_ATK, 1),
    "Sharpen":        _boost(_BOOST_ATK, 1),
    "Harden":         _boost(_BOOST_DEF, 1),
    "Withdraw":       _boost(_BOOST_DEF, 1),
    "Defense Curl":   _boost(_BOOST_DEF, 1),
    "Charge":         _boost(_BOOST_SPD, 1),
    "Stockpile":      _boost(_BOOST_DEF, 1, _BOOST_SPD, 1),
    # Curse (non-Ghost): +ATK +DEF -SPE (Ghost version handled separately via Curse's onHit)
    "Curse":          _boost(_BOOST_ATK, 1, _BOOST_DEF, 1, _BOOST_SPE, -1),
    # Gen 5+ (included for forward compatibility)
    "Quiver Dance":   _boost(_BOOST_SPA, 1, _BOOST_SPD, 1, _BOOST_SPE, 1),
    "Shell Smash":    _boost(_BOOST_ATK, 2, _BOOST_SPA, 2, _BOOST_SPE, 2),  # also -DEF/-SPD
    "Work Up":        _boost(_BOOST_ATK, 1, _BOOST_SPA, 1),
    "Coil":           _boost(_BOOST_ATK, 1, _BOOST_DEF, 1),
    "Hone Claws":     _boost(_BOOST_ATK, 1, _BOOST_ACC, 1),
    "Autotomize":     _boost(_BOOST_SPE, 2),
    "Shift Gear":     _boost(_BOOST_ATK, 1, _BOOST_SPE, 2),
    "Victory Dance":  _boost(_BOOST_ATK, 1, _BOOST_DEF, 1, _BOOST_SPE, 1),
    "Clangorous Soul": _boost(_BOOST_ATK, 1, _BOOST_DEF, 1, _BOOST_SPA, 1),

    # ---- Foe stat lower moves ----
    "Growl":         _lower(_BOOST_ATK, -1),
    "Tail Whip":     _lower(_BOOST_DEF, -1),
    "Sand Attack":   _lower(_BOOST_ACC, -1),
    "String Shot":   _lower(_BOOST_SPE, -2),  # Gen 6+ is -2, Gen 4-5 is -1
    "Screech":       _lower(_BOOST_DEF, -2),
    "Scary Face":    _lower(_BOOST_SPE, -2),
    "Feather Dance": _lower(_BOOST_ATK, -2),
    "Charm":         _lower(_BOOST_ATK, -2),
    "Fake Tears":    _lower(_BOOST_SPD, -2),
    "Metal Sound":   _lower(_BOOST_SPD, -2),
    "Cotton Spore":  _lower(_BOOST_SPE, -2),
    "Leer":          _lower(_BOOST_DEF, -1),
    "Sweet Scent":   _lower(_BOOST_EVA, -1),
    "Smokescreen":   _lower(_BOOST_ACC, -1),
    "Tickle":        _lower(_BOOST_ATK, -1, _BOOST_DEF, -1),
    "Captivate":     _lower(_BOOST_SPA, -2),
    "Noble Roar":    _lower(_BOOST_ATK, -1, _BOOST_SPA, -1),
    "Confide":       _lower(_BOOST_SPA, -1),
    "Eerie Impulse": _lower(_BOOST_SPA, -2),
    "Venom Drench":  _lower(_BOOST_ATK, -1, _BOOST_SPA, -1, _BOOST_SPE, -1),
    "Topsy-Turvy":   _lower(_BOOST_ATK, 0),   # complex, placeholder
    "Tearful Look":  _lower(_BOOST_ATK, -1, _BOOST_SPA, -1),

    # ---- Entry hazard moves ----
    "Spikes":        _hazard(_SC_SPIKES, 3),
    "Toxic Spikes":  _hazard(_SC_TOXICSPIKES, 2),
    "Stealth Rock":  _hazard(_SC_STEALTHROCK, 1),
    "Sticky Web":    _hazard(_SC_STICKYWEB, 1),

    # ---- Screen / field condition moves ----
    "Reflect":       _screen(_SC_REFLECT, 5),
    "Light Screen":  _screen(_SC_LIGHTSCREEN, 5),
    "Aurora Veil":   _screen(_SC_AURORAVEIL, 5),
    "Safeguard":     _screen(_SC_SAFEGUARD, 5),
    "Mist":          _screen(_SC_MIST, 5),
    "Tailwind":      _screen(_SC_TAILWIND, 3),   # 3 turns in Gen 4-5

    # ---- Weather-setting moves ----
    "Sunny Day":     _weather(_WEATHER_SUN, 5),
    "Rain Dance":    _weather(_WEATHER_RAIN, 5),
    "Sandstorm":     _weather(_WEATHER_SAND, 5),
    "Hail":          _weather(_WEATHER_HAIL, 5),

    # ---- Terrain-setting moves (Gen 6+) ----
    "Electric Terrain": _terrain(_TERRAIN_ELECTRIC, 5),
    "Grassy Terrain":   _terrain(_TERRAIN_GRASSY, 5),
    "Misty Terrain":    _terrain(_TERRAIN_MISTY, 5),
    "Psychic Terrain":  _terrain(_TERRAIN_PSYCHIC, 5),

    # ---- Trick Room ----
    "Trick Room": (ME_TRICK_ROOM, NONE_STAT, 0, NONE_STAT, 0, NONE_STAT, 0, 0),

    # ---- Volatile status: self ----
    "Protect":       _volatile_self(_VOL_PROTECT),
    "Detect":        _volatile_self(_VOL_PROTECT),
    "Endure":        _volatile_self(_VOL_ENDURE),
    "Focus Energy":  _volatile_self(_VOL_FOCUSENERGY),
    "Ingrain":       _volatile_self(_VOL_INGRAIN),

    # ---- Volatile status: foe ----
    "Leech Seed":    _volatile_foe(_VOL_SEEDED),
    "Encore":        _volatile_foe(_VOL_ENCORE),
    "Taunt":         _volatile_foe(_VOL_TAUNT),
    "Torment":       _volatile_foe(_VOL_TORMENT),
    "Attract":       _volatile_foe(_VOL_ATTRACT),
    "Heal Block":    _volatile_foe(_VOL_HEALBLOCK),
    "Nightmare":     _volatile_foe(_VOL_NIGHTMARE),
    "CurseGhost":    _volatile_foe(_VOL_CURSE),  # Ghost-type Curse (keyed separately to avoid overwriting non-ghost Curse boost)

    # ---- Substitute ----
    "Substitute":    _simple_effect(ME_SUBSTITUTE),

    # ---- Hazard removal ----
    "Rapid Spin":    _simple_effect(ME_RAPID_SPIN),
    "Defog":         _simple_effect(ME_DEFOG),

    # ---- Phazing (force switch) ----
    "Roar":          _simple_effect(ME_ROAR),
    "Whirlwind":     _simple_effect(ME_ROAR),
    "Dragon Tail":   _simple_effect(ME_ROAR),
    "Circle Throw":  _simple_effect(ME_ROAR),

    # ---- Self-switching moves ----
    "U-turn":        _simple_effect(ME_U_TURN),
    "Volt Switch":   _simple_effect(ME_U_TURN),
    "Baton Pass":    _simple_effect(ME_BATON_PASS),

    # ---- Recovery moves (heal 50% max HP) ----
    "Recover":       _recovery(1, 2),
    "Softboiled":    _recovery(1, 2),
    "Milk Drink":    _recovery(1, 2),
    "Slack Off":     _recovery(1, 2),
    "Roost":         _recovery(1, 2),
    "Wish":          _simple_effect(ME_WISH),
    "Moonlight":     _recovery(1, 2),  # weather-dependent in full impl, simplified
    "Morning Sun":   _recovery(1, 2),
    "Synthesis":     _recovery(1, 2),

    # ---- Rest ----
    "Rest":          _simple_effect(ME_REST),

    # ---- Belly Drum ----
    "Belly Drum":    _simple_effect(ME_BELLY_DRUM),

    # ---- Item removal ----
    "Knock Off":     _simple_effect(ME_KNOCK_OFF),
    "Trick":         _simple_effect(ME_TRICK),
    "Switcheroo":    _simple_effect(ME_TRICK),
    "Thief":         _simple_effect(ME_KNOCK_OFF),  # simplified: just removes item

    # ---- Pain Split ----
    "Pain Split":    _simple_effect(ME_PAIN_SPLIT),

    # ---- Heal Bell / Aromatherapy ----
    "Heal Bell":     _simple_effect(ME_HEAL_BELL),
    "Aromatherapy":  _simple_effect(ME_HEAL_BELL),

    # ---- Disable / Yawn / Destiny Bond / Perish Song ----
    "Disable":       _simple_effect(ME_DISABLE),
    "Yawn":          _simple_effect(ME_YAWN),
    "Destiny Bond":  _simple_effect(ME_DESTINY_BOND),
    "Perish Song":   _simple_effect(ME_PERISH_SONG),
    "Sleep Talk":    _simple_effect(ME_SLEEP_TALK),

    # ---- Haze ----
    "Haze":          _simple_effect(ME_HAZE),

    # ---- Two-turn moves (charge turn + release turn) ----
    # User sets VOL_CHARGING on turn 1 (no damage), then attacks on turn 2.
    "Fly":           _simple_effect(ME_TWO_TURN),
    "Dig":           _simple_effect(ME_TWO_TURN),
    "Dive":          _simple_effect(ME_TWO_TURN),
    "Bounce":        _simple_effect(ME_TWO_TURN),
    "Sky Attack":    _simple_effect(ME_TWO_TURN),
    "Razor Wind":    _simple_effect(ME_TWO_TURN),
    "Skull Bash":    _simple_effect(ME_TWO_TURN),
    "Solar Beam":    _simple_effect(ME_TWO_TURN),
    "Ice Burn":      _simple_effect(ME_TWO_TURN),
    "Freeze Shock":  _simple_effect(ME_TWO_TURN),
    "Shadow Force":  _simple_effect(ME_TWO_TURN),
}


def build_move_effects_table(name_to_id: dict, n_moves: int) -> np.ndarray:
    """
    Build the move_effects array from GEN4_MOVE_EFFECTS.

    Args:
        name_to_id: dict mapping move display name → row index in move table
        n_moves:    total number of moves

    Returns:
        int16 array of shape (n_moves, MOVE_EFFECT_FIELDS)
    """
    table = np.zeros((n_moves, MOVE_EFFECT_FIELDS), dtype=np.int16)
    # Initialise all stat-index fields to NONE_STAT sentinel
    table[:, 1] = NONE_STAT  # stat1
    table[:, 3] = NONE_STAT  # stat2
    table[:, 5] = NONE_STAT  # stat3

    for name, effect in GEN4_MOVE_EFFECTS.items():
        if name in name_to_id:
            idx = name_to_id[name]
            if 0 <= idx < n_moves:
                table[idx] = list(effect)

    return table
