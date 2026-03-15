# PokeJAX Training Plan: BC → Self-Play PPO

## Context

**Goal**: Train a Gen 4 Random Battle bot that can compete on the Pokemon Showdown ladder.

**Current state**: Engine (99.99% action mask, 95.4% winner agreement with PS), observation builder (15 tokens × 394 dims), PokeTransformer (5.67M params, C51 value), PPO + rollout infrastructure — all complete and tested.

**SOTA reference**: ps-ppo achieved >1900 Elo on Gen 9 ladder using BC warm-start → PPO self-play. Wang (MIT 2024) hit 1756 Glicko on Gen 4 Random Battles with PPO + MCTS.

**Key insight**: BC before self-play is critical. Random init PPO leads to policy collapse. PokemonShowdownClaude confirms this pipeline works.

---

## Phase 1: Behavioral Cloning from Heuristic (BC)

### 1A. Implement Smart Heuristic Opponent

**File**: `pokejax/env/heuristic.py`

Port `smart_heuristic_opponent` from PokemonShowdownClaude (`pokebot/env/poke_engine_env.py:1231-1315`) to work with JAX BattleState:
- Type-aware damage estimation (base power × STAB × effectiveness × atk/def)
- Status move scoring (Stealth Rock 250pts, sleep 300pts, setup gated by HP/boosts)
- Smart switching (matchup scoring, switch when losing badly)
- Target: ~80%+ win rate vs random moves

This runs on **CPU** (Python, not JIT) — it's only used during BC data collection, not inside the rollout scan.

### 1B. BC Data Collection

**File**: `pokejax/rl/bc.py`

Collect (observation, expert_action) pairs by running battles:
- Player 1: heuristic opponent (expert teacher)
- Player 2: random opponent (diverse states)
- Per turn: build observation via `build_obs()`, record heuristic's chosen action
- Collect ~500K-1M transitions (stored as numpy arrays)
- Action distribution: categorize moves vs switches for monitoring

**Alternative**: Extract from existing Showdown battle logs (`data/showdown_battles.jsonl`). These are real human games, potentially stronger than heuristic. But they require reconstructing BattleState from logs (complex). **Decision**: Use heuristic first (simpler), add human replay BC later if needed.

### 1C. BC Training Loop

**File**: `scripts/train_bc.py`

Simple supervised learning:
```
for batch in dataset:
    obs = (int_ids, float_feats, legal_mask)
    log_probs, _, _ = model(obs)
    loss = cross_entropy(log_probs, expert_action)
    grads = jax.grad(loss)(params)
    params = optax.apply_updates(params, optimizer.update(grads))
```

**Hyperparams** (from PokemonShowdownClaude):
- LR: 3e-4 with cosine decay to 1e-5
- Batch size: 1024
- Gradient clip: 0.5
- Train for ~1M steps or until convergence

**Eval metrics**:
- Action agreement with heuristic (target: >60%)
- Win rate vs random (target: >70%)
- Win rate vs simple heuristic (target: >40%)

**Checkpoint**: Save best model params as `checkpoints/bc_init.pkl`

---

## Phase 2: PPO Self-Play Fine-Tuning

### 2A. Load BC Checkpoint

**File**: Modify `scripts/train_ppo.py`

- Add `--bc-init checkpoints/bc_init.pkl` flag
- Load BC params → inject into TrainState (fresh optimizer state)
- This replaces random initialization

### 2B. Opponent Mix (Curriculum)

**File**: Modify `pokejax/rl/rollout.py`

During self-play rollout, mix opponents:
- **Stage 1** (0–50M steps): 50% random + 50% heuristic
- **Stage 2** (50M–150M): 100% self-play (shared model)
- **Stage 3** (150M+): 20% heuristic anchor + 20% latest + 60% pool

The heuristic anchor prevents catastrophic forgetting. Pool is FIFO deque of 20 past checkpoints.

**Implementation note**: Since rollout runs inside `lax.scan` (JIT), opponent selection must be branchless. For Stage 1/2, we can simply have the rollout always use the shared model (symmetric self-play) — the BC warm-start ensures it doesn't collapse. Heuristic mixing requires running heuristic outside the scan (CPU callback or separate collection).

**Simpler approach for v1**: Just do symmetric self-play with BC warm-start. Add opponent pool later if needed.

### 2C. Reward Shaping

Current reward in `pokejax/env/pokejax_env.py`:
- Win: +1.0, Lose: -1.0
- Per-turn: HP advantage delta × 0.1

This is already good. Consider adding:
- Per-faint bonus: +0.1 per opponent faint, -0.1 per own faint (ps-ppo uses this)
- Anneal shaping reward to 0 over training (so final policy optimizes pure win rate)

### 2D. Checkpointing

**File**: `pokejax/utils/checkpoints.py`

- Save params + optimizer state every 100 PPO updates
- Keep last 20 checkpoints (for opponent pool)
- Use `orbax` or simple pickle

### 2E. Monitoring & Evaluation

**File**: Modify `pokejax/rl/self_play.py`

Track per update:
- PPO loss components (policy, value, entropy)
- Win rate of current model (from rollout episode outcomes)
- Average episode length
- Action distribution entropy

Periodic eval (every 200 updates):
- 100 games vs heuristic → win rate
- 100 games vs random → win rate

---

## Phase 3: Ladder Deployment (Future)

Not in scope for now, but the path:
- Connect to Pokemon Showdown via websocket
- Translate observations from PS format → `build_obs()` format
- Run model inference → select action → send to PS
- Optional: MCTS at test time (+60 Elo per Wang)

---

## Implementation Order

| Step | What | Files | Effort |
|------|------|-------|--------|
| 1 | Heuristic opponent | `pokejax/env/heuristic.py` | ~2h |
| 2 | BC data collection | `pokejax/rl/bc.py` | ~1h |
| 3 | BC training script | `scripts/train_bc.py` | ~1h |
| 4 | Train BC, eval | Run script | ~1h compute |
| 5 | Add `--bc-init` to PPO | `scripts/train_ppo.py` | ~30m |
| 6 | Checkpointing | `pokejax/utils/checkpoints.py` | ~30m |
| 7 | Run PPO self-play | Run script | ~3h compute |
| 8 | Eval & iterate | Monitor, tune | Ongoing |

**Start with Steps 1-4.** Verify BC model beats random before touching PPO.
