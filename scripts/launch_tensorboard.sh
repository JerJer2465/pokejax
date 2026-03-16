#!/bin/bash
# Launch TensorBoard to monitor PPO training.
# Run from WSL: bash scripts/launch_tensorboard.sh
#
# Access at: http://localhost:6006
# From Windows: http://localhost:6006

LOGDIR="${1:-runs}"
PORT="${2:-6006}"

echo "Starting TensorBoard on port $PORT..."
echo "Log directory: $LOGDIR"
echo "Access at: http://localhost:$PORT"
echo ""

~/.local/bin/tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT"
