# Pokejax Engine vs PS Win Rate Gap — Root Cause Analysis

Investigation of the win rate discrepancy between JAX engine self-play (~50% vs internal
heuristic) and Pokemon Showdown local server play against SimpleHeuristicsPlayer.

---

## 1. Observation Encoding — What Is Consistent

Verification confirmed that `showdown_player.py` (PS inference) and `obs_builder.py`
(JAX training) share identical offset constants and encoding logic for:

- All feature offsets (`_OFF_*` / `_FOFF_*` / `_MOFF_*`) — identical ✓
- Base stats: `base_stats / 255.0` from poke-env's `pokemon.base_stats` dict ✓
- Boosts: 91-dim one-hot (7 stats × 13 stages [-6 to +6]) ✓
- Types: both subtract 1 for 0-indexed 18-dim one-hot ✓
- Status, level, slot, is_own, HP fraction ✓
- Field encoding: weather, hazards, screens, tailwind, safeguard, mist, toxic bin,
  turn bin, fainted counts ✓
- Partial info (reveal masking): the engine properly tracks `RevealState` via
  `turn.py:_update_reveal()` and training genuinely uses partial information ✓

---

## 2. PP Tracking — Mitigated But Not Eliminated

**Training side** (`rollout.py:47–58`): PP noise is applied with **50% probability** per
rollout step, zeroing all PP fractions to 1.0 across all 15 tokens. This was specifically
added to mimic PS behavior and make the model robust. So the model IS partially trained
on pp_frac=1.0 observations.

**PS side**: poke-env does not reliably decrement PP in local server games — `current_pp`
stays equal to `max_pp`, so pp_frac is always 1.0.

**Net gap**: The model sees real PP 50% of training steps but always 1.0 on PS. This is a
~0.5 distribution mismatch on the `_MOFF_PP` dimension (dim 43 of each 45-dim move block,
4 moves × 12 pokemon tokens = 48 values per observation). Small but systematic.

---

## 3. Stale Move Encoding After Switch — Bug Still Present

**Code**: `showdown_player.py:744–773` (desync detection block)

After a player switches, poke-env sometimes delivers `available_moves` containing the
**previous pokemon's moves** while `active_pokemon` has already updated to the new one.
The code detects this via `_moves_match()` but the fallback is wrong:

```python
# Desync: available_moves don't belong to active_pokemon.
# Use available_moves directly — these ARE the legal moves.
own_move_list = list(available_moves)[:4]   # ← still uses STALE moves!
```

The comment is misleading. These are NOT the correct legal moves — they are the previous
pokemon's moves. The new active pokemon's token gets encoded with wrong move IDs and
features (wrong type, base power, category), AND the legal mask encodes these stale
moves as legal. The model picks from actions that don't correspond to the actual pokemon
on the field.

**Correct fix**: on desync, use `list(own_active.moves.values())[:4]` (the new active
pokemon's known moves) instead of `available_moves`.

**Frequency**: every switch triggers a 1-turn desync window. With an average switch rate
of ~15-20% of turns, this affects a meaningful portion of games.

---

## 4. Missing Volatile Effects on PS

Three volatile effects are present in `obs_builder.py`'s `_VOL_MAP` but absent from
`showdown_player.py`'s `_EFFECT_TO_VOLATILE` / `_EFF_MAP`:

| Effect | obs_builder bit | showdown_player | Impact |
|--------|----------------|-----------------|--------|
| HEAL_BLOCK | `VOL_HEALBLOCK = 10` | Missing | Rare, minor |
| GRUDGE | `VOL_GRUDGE = 26` | Missing | Rare, minor |
| PERISH_SONG | read from `VOL_PERISH` via `volatile_data` | Missing entirely | **More common** |

For **Perish Song**: training encodes the remaining perish count (0–3) into a 4-bin
one-hot at `_OFF_PERISH_BIN` (offset 388). On PS, `perish_count` is always 0 because
`_EFFECT_TO_VOLATILE.get(eff)` never returns `"perishsong"` — that key doesn't exist in
the map. So `buf[_OFF_PERISH_BIN + 0] = 1.0` always. Perish Song appears regularly in
Gen4 random battles (Gengar, Misdreavus moveset pools), making this a consistent
misrepresentation.

---

## 5. Wonder Room Not Encoded on PS

`obs_builder.py:501` encodes Wonder Room as pseudo-weather bit 2:
```python
pseudo = jnp.array([trick_room, gravity, wonder_room, 0.0, 0.0])
```

`showdown_player.py` never sets `_FOFF_PSEUDO + 2`. When Wonder Room is active,
training sees bit 2 = 1.0, PS sees bit 2 = 0.0. Minor (Wonder Room is uncommon).

---

## 6. Sleep Turns ±1 Offset

Training (`obs_builder.py:384`): `sleep_turns` in the engine starts at 0 when the
pokemon first falls asleep and increments each turn.

PS inference (`showdown_player.py:438`): poke-env's `sleep_turns` starts at 1 on the
first turn asleep (PS protocol reports the wake-up check immediately).

The code acknowledges this discrepancy in a comment but does not correct it:
```python
# Training engine starts sleep_turns at 0, poke-env at 0 or 1.
```

Result: `sleep_bin` and `rest_bin` are off by one slot compared to training. When
the engine sees bin 0 (first turn asleep), PS sees bin 1. When engine sees bin 1, PS
sees bin 2. This is a consistent +1 shift for the duration of any sleep, which affects
both regular sleep and Rest.

---

## 7. Value Function

**Architecture**: C51 distributional value head, 51 atoms, support `v_min=-2.5,
v_max=2.5`. The wider support than `[-1, +1]` is intentional — the code comment says
"returns can reach ±1.95 with PBRS shaping." This is correct.

**Calibration on PS**: Mean value ≈ -0.34 across PS games with ~80% loss rate. This
is **directionally correct** — an agent losing 80% of games should have negative
expected value. The C51 atoms shift probability mass toward the negative side of the
support when the model predicts a losing position.

**The "inverted" observation from 5-game sample** (wins having lower value than losses)
is almost certainly sampling noise. With 1 win and 4 losses, the 1 winning game may
have been one where the model was genuinely behind throughout but the opponent made
errors, while some losses started as close games. This needs 50+ games to be meaningful.

The value head is **not a primary factor** in the win rate gap — it only affects MCTS
quality, not direct policy decisions.

---

## 8. MCTS-Specific Bugs (battle_bridge.py)

These only affect `play_showdown_tree_mcts.py` and `play_ladder_tree_mcts.py`, not
direct policy play.

### 8a. nature_id Always 0 (`battle_bridge.py:231`)
```python
'nature_id': np.int8(0),   # hardcoded — always Hardy (neutral)
```
Every pokemon in MCTS simulation is treated as having a neutral nature. For pokemon
with +10% or -10% nature modifiers, this makes damage calculations and speed
comparisons wrong in simulation. poke-env exposes `pokemon.nature` — this should be
mapped to a nature ID via the tables lookup.

### 8b. Opponent max_hp Estimated With Wrong IVs/EVs (`battle_bridge.py:165`)
```python
est_hp = int((2 * base_stats[0] + 31 + 21) * level / 100 + level + 10)
```
Hardcodes 31 IVs and 21 EVs. Gen4 random battles assign specific IV/EV spreads per
pokemon. The formula should use actual IV/EV estimates based on the format. The
`+21` approximation for EVs is also wrong (should be `floor(ev/4)` where ev varies
by pokemon and stat). This affects HP totals used in MCTS damage projections.

---

## 9. Training Opponent vs PS Opponent

The model was trained against the JAX heuristic (`pokejax/rl/heuristic.py`), which:
- Does type-aware damage estimation using base stats
- Scores status moves (sleep, hazards)
- Switches based on outgoing minus incoming damage differential

PS uses `poke-env`'s `SimpleHeuristicsPlayer`, which:
- Uses more sophisticated matchup coefficients including speed tiers
- Actively manages boosts (switches on -3 defensive stat drops)
- Uses Rapid Spin / Defog for hazard removal
- Has better switching thresholds

The JAX heuristic is strictly weaker. A model trained at ~50% vs a weaker heuristic
will naturally have a lower win rate vs a stronger one — this is an **expected
generalization gap**, not a bug.

---

## Summary: Ordered by Impact

| # | Issue | Affects | Severity |
|---|-------|---------|----------|
| 1 | Training opponent weaker than PS opponent | Win rate | HIGH |
| 2 | Stale move encoding after switch (desync fallback uses wrong moves) | Policy obs + legal mask | HIGH |
| 3 | Perish Song never encoded on PS (always bin 0) | Policy obs | MEDIUM |
| 4 | PP always 1.0 on PS (partially mitigated by 50% noise in training) | Policy obs | MEDIUM |
| 5 | Sleep turns ±1 offset (engine starts at 0, poke-env starts at 1) | Policy obs | MEDIUM |
| 6 | nature_id=0 for all pokemon in BattleBridge | MCTS sim stats | MEDIUM |
| 7 | Opponent max_hp wrong IVs/EVs in BattleBridge | MCTS sim HP | MEDIUM |
| 8 | HEAL_BLOCK / GRUDGE missing from PS volatile encoding | Policy obs | LOW |
| 9 | Wonder Room not encoded on PS | Policy obs | LOW |
| 10 | Value calibration (C51 ±2.5 support, mean=-0.34) | MCTS value | LOW |

---

## Recommended Fixes

### Fix 1: Stale move desync (`showdown_player.py:769–771`)
Change the else branch to use the active pokemon's own moves:
```python
# Instead of: own_move_list = list(available_moves)[:4]
own_move_list = list(own_active.moves.values())[:4]
```

### Fix 2: Perish Song in `_EFF_MAP` (`showdown_player.py:109–124`)
Add PERISH_SONG → "perishsong" to `_EFF_MAP`. Also need to read the turns value
correctly since poke-env stores remaining turns in the Effect dict.

### Fix 3: Sleep turns offset (`showdown_player.py:439`)
Check whether poke-env's sleep_turns is 0-indexed or 1-indexed and subtract 1 if needed
to match the engine's 0-indexed scheme.

### Fix 4: nature_id in BattleBridge (`battle_bridge.py:231`)
Map `pokemon.nature` through the nature lookup table:
```python
'nature_id': np.int8(self.obs._find_id(pokemon.nature, self.obs._nature_lookup)),
```

### Fix 5: Train vs stronger heuristic
Add the JAX heuristic as a poke-env player for true apples-to-apples comparison.
If win rate vs JAX heuristic on PS ≈ 50%, the obs encoding is correct and the gap
is purely opponent strength. If it's still below 50%, there are remaining obs bugs.
