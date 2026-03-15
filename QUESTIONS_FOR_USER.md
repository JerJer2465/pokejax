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
- PPO with BC init + checkpointing: updated (`scripts/train_ppo.py`, `pokejax/rl/self_play.py`)
- Team pool generation: running (50K teams)
- BC training: pending team pool completion

## Differential Test Results (latest)

- MaxHP: 100% (1200/1200)
- Action mask: 99.99% (12979/12980)
- Boost: 99.05% (84703/85519)
- Status: 95.79% (RNG-dependent)
- Winner: 95.38% (62/65 finished games)
- HP violations: 0
