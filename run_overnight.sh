#!/bin/bash
# FINAL OVERNIGHT EXPERIMENTAL RUN
# =================================
# Task 1: Expand T5 cache (50 → 100 samples) - ~2 hours
# Task 2: Train 3 syntactic ablations - ~9 hours
# Total: ~11 hours
#
# Usage:
#   chmod +x run_overnight.sh
#   nohup ./run_overnight.sh > overnight.log 2>&1 &
#   tail -f overnight.log  # to monitor

echo "=========================================="
echo "FINAL EXPERIMENTAL RUN - STARTING"
echo "Time: $(date)"
echo "=========================================="

cd "$(dirname "$0")"

# Task 1: Expand T5 cache (2 hours)
echo ""
echo "[1/2] Expanding T5 cache to 100 samples..."
echo "Time: $(date)"
python experiments/expand_t5_cache.py
echo "T5 cache expansion done at: $(date)"

# Task 2: Train ablations (9 hours)
echo ""
echo "[2/2] Training syntactic ablations..."
echo "Time: $(date)"
python experiments/train_syntactic_ablations.py
echo "Ablation training done at: $(date)"

echo ""
echo "=========================================="
echo "OVERNIGHT RUN COMPLETE"
echo "Time: $(date)"
echo "=========================================="
echo ""
echo "Next: run in the morning:"
echo "  python experiments/eval_ablations.py"
echo ""
echo "Checklist:"
echo "  1. wc -l data/eval_adversarial_cache.jsonl  (should be ~100)"
echo "  2. ls -la models/dann_pos_only.pth"
echo "  3. ls -la models/dann_function_only.pth"
echo "  4. ls -la models/dann_readability_only.pth"
