# Analysis: 14% Win Rate on PS vs 50% in Training

## Executive Summary

The model achieves ~50% vs the internal JAX heuristic during training but only ~14% vs
poke-env's SimpleHeuristicsPlayer on Pokemon Showdown. This gap has **three root causes**
(ordered by impact):

1. **Different opponents** (accounts for ~20% of the gap)
2. **Stale observation data after switches** (accounts for ~10% of the gap)
3. **Observation distribution shift** (accounts for the remaining ~5%)

---

## Diagnostic Results (5 games, 207 turns)

| Metric | Value |
|--------|-------|
| Win rate | 20% (1W/4L) |
| Total turns | 207 |
| Anomalies detected | 13 |
| Empty available_moves turns | 69 (33.3%) |
| Move mismatch turns (stale data) | 33 (15.9%) |
| Fallback actions (model choice invalid) | 24 (11.6%) |
| Species ID unmapped | 0 (all species resolve correctly) |
| PP tracking | **Always 1.0** (poke-env doesn't track PP in local games) |
| Near-uniform policy | 1/207 (0.5%) |
| Mean confidence | 82.1% |

### Value Estimates
- Mean: -0.34, Std: 0.47
- **WIN games: mean value = -0.62** (model thinks it's losing when winning)
- **LOSS games: mean value = -0.25** (model thinks it's doing OK when losing)
- Value predictions are **not correlated with outcomes**

### Action Distribution
- Move actions: 64.3%
- Switch actions: 30.0%
- Fallback (invalid action): 11.6%

---

## Root Cause Analysis

### 1. Different Opponents (~20% gap)

**Training heuristic** (`pokejax/env/heuristic.py` / `pokejax/rl/heuristic.py`):
- Simple type-aware damage estimation
- Basic status move scoring (sleep: 300pts, stealth rock: 250pts)
- Damage-based switching (outgoing - incoming)
- Runs inside JAX engine with full state access

**poke-env SimpleHeuristicsPlayer** (`poke_env.player.baselines`):
- More sophisticated matchup estimation with speed tier coefficients
- Explicit stat boost management (switches on -3 def/spd/atk/spa)
- Entry hazard setup AND removal (Rapid Spin, Defog)
- Better switching logic (matchup-based with thresholds)
- Dynamax strategic integration

SimpleHeuristicsPlayer is a **materially stronger opponent** than the training heuristic.
This alone accounts for a significant portion of the gap. A 50% win rate vs a weaker
opponent would naturally drop against a stronger one.

### 2. Stale Move Data After Switches (~10% gap)

**Critical finding**: After every switch, poke-env delivers 1-2 turns where
`available_moves` contains the **previous pokemon's moves** instead of the new active
pokemon's moves.

Observed pattern:
```
Turn N:   CHOSE switch to Heatran
Turn N+1: Active: Heatran, available_moves: [closecombat, stoneedge, ...] ← Primeape's moves!
          [WARN] Move mismatch: poke-env active=heatran, moves match=primeape
Turn N+2: Active: Heatran, available_moves: [explosion, flamethrower, ...] ← Correct!
```

**Impact**: On 16% of turns, the model:
- Sees wrong move features for the active token (wrong types, base powers, categories)
- Has an incorrect legal mask
- May choose a move that doesn't belong to the active pokemon
- Falls back to a random/first legal action

This is a **poke-env state update race condition** that affects every game.

### 3. Observation Distribution Shift (~5% gap)

**PP tracking**: PP is always 1.0 in poke-env local games. During training, the model
sees PP decrease as moves are used. This removes an information channel the model
relies on.

**Forced switch turns**: 33% of turns have empty `available_moves`. Most are legitimate
forced switches (62/69), but 7 turns have both empty moves AND switches (team preview
or race condition), forcing fallback to action 0.

**Opponent information**: During training (self-play), the model sees full opponent state.
On PS, it only sees what has been revealed. Opponent tokens start as zeros and fill in
as pokemon are revealed. This is handled correctly by the bridge, but the distribution
is fundamentally different.

**Value calibration**: The model's value estimates show mean=-0.34 across all turns.
This persistent pessimism suggests the observation distribution on PS maps to a region
of state space that the model associates with losing positions.

---

## Specific Bugs Found

### Bug 1: Move mismatch after switch
**File**: `pokejax/players/showdown_player.py` (ObsBridge.build_obs)
**Cause**: poke-env's `battle.available_moves` is stale for 1-2 turns after a switch
**Impact**: Wrong move encoding, wrong legal mask, invalid action selection (16% of turns)
**Fix**: Cache the real move list per pokemon species and use it when a mismatch is
detected, rather than relying on `available_moves`.

### Bug 2: Empty available_moves on 7 turns
**File**: `pokejax/players/showdown_player.py` (_choose_move_impl)
**Cause**: poke-env calls choose_move before state is fully populated
**Impact**: Fallback to action 0 (essentially random)
**Fix**: Return `choose_default_move()` immediately when both moves and switches are
empty, rather than going through the full observation pipeline.

### Bug 3: Legal mask includes active pokemon for switching
**File**: `pokejax/players/showdown_player.py` (ObsBridge.build_obs)
**Cause**: After switch, poke-env's `available_switches` still includes the now-active
pokemon
**Impact**: Safety check catches this but it indicates stale state (caught 56 times)
**Fix**: Already handled by the safety check, but could return early to avoid encoding
wrong data.

---

## Recommendations

### Short-term fixes (should improve to ~25-30%)

1. **Fix move mismatch**: Cache each pokemon's real moves when first seen (from
   `pokemon.moves` dict). On mismatch, use the cached moves for the actual active
   pokemon instead of the stale `available_moves`.

2. **Handle empty state**: Return `choose_default_move()` immediately when poke-env
   provides no options, skipping the full obs/model pipeline.

3. **Evaluate vs the same opponent**: Add the JAX heuristic as a poke-env opponent
   so you can compare apples-to-apples. If the model gets ~50% vs the JAX heuristic
   on PS, the obs encoding is correct and the gap is purely opponent strength.

### Medium-term improvements (should improve to ~40-50%)

4. **Train against SimpleHeuristicsPlayer**: Include poke-env's heuristic as an
   opponent during PPO training (via periodic PS eval games that feed back into
   the training loop).

5. **Add observation noise during training**: Randomly mask PP information, add noise
   to opponent HP fractions, and occasionally shuffle move order to make the policy
   robust to the kind of noise seen in the PS bridge.

6. **Train with partial information**: Add a reveal mask during training that mimics
   what poke-env would show (only reveal opponent pokemon/moves as they appear).

### Long-term (to surpass heuristic)

7. **MCTS integration**: Use `play_showdown_tree_mcts.py` which does multi-turn
   lookahead and should significantly boost performance regardless of obs issues.

8. **Fine-tune on PS games**: Collect replays from PS games and fine-tune the policy
   on the real observation distribution (domain adaptation).

9. **Bigger model + more training**: The 2.2M parameter model with 1100 PPO updates
   may not have sufficient capacity/training to learn all the nuances. Scale up.
