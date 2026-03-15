# Questions & Notes for User

## Questions

1. **Farfetch'd unmapped**: The team pool generator can't map "Farfetch'd" to an engine species ID due to the apostrophe in the name. This affects 1/296 species. Should I add a special case or is this acceptable to skip?

2. **BC data size**: Currently collecting 500K transitions by default. PokemonShowdownClaude uses 1-2M. Should we go bigger?

3. **BC epochs**: Default is 10 passes over the dataset. More epochs risk overfitting. Is 10 good or should we tune?

4. **Team pool vs real Showdown teams**: The team pool samples from gen4randombattle.json roles/movesets. It may not perfectly match what Showdown actually generates (e.g., EV spreads, nature interactions). If BC performance is low, we may need to use Showdown-generated teams from the battle logs instead.

5. **Heuristic quality**: The heuristic uses base stats for damage estimation (not computed battle stats with nature/EVs). This is a simplification. If BC accuracy is low, we may need a more accurate damage estimator.

## Status

- Heuristic opponent: implemented (`pokejax/env/heuristic.py`)
- BC data collection + training: implemented (`pokejax/rl/bc.py`)
- BC training script: implemented (`scripts/train_bc.py`)
- Eval script: implemented (`scripts/eval_bc.py`)
- PPO with BC init + checkpointing: updated (`scripts/train_ppo.py`, `pokejax/rl/self_play.py`)
- Team pool: generated (50K teams, `data/team_pool.npz`)
- BC data: collected 50K transitions at 44 trans/s (`data/bc_data_50k.npz`)
- **BC training complete** (10 epochs on 50K data):
  - Final accuracy: 95.5% (action agreement with heuristic teacher)
  - Loss: 0.14
  - vs Random: **100% win rate** (20 games)
  - vs Heuristic: **25% win rate** (20 games) — expected since trained vs random opponent
- Next step: PPO self-play with BC warm-start (`--bc-init checkpoints/bc_final.pkl`)

## Differential Test Results (latest)

- MaxHP: 100% (1200/1200)
- Action mask: 99.99% (12979/12980)
- Boost: 99.05% (84703/85519)
- Status: 95.79% (RNG-dependent)
- Winner: 95.38% (62/65 finished games)
- HP violations: 0
