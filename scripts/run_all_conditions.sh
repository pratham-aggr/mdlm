#!/bin/bash
set -e

GPU=${1:-0}
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Running on GPU $GPU"
echo "Started: $(date)"

for COND in A B C D E F G H; do
    RESULTS="results/condition_${COND}/results.jsonl"
    
    # count completed problems
    if [ -f "$RESULTS" ]; then
        N=$(wc -l < "$RESULTS")
        if [ "$N" -ge 200 ]; then
            echo "Skipping condition $COND — already complete ($N problems)"
            continue
        else
            echo "Resuming condition $COND — $N/200 problems done"
        fi
    fi

    echo "========================================"
    echo "Starting condition $COND: $(date)"

    CUDA_VISIBLE_DEVICES=$GPU python -u scripts/run_condition.py \
        --condition $COND \
        --no_hidden_states \
        2>&1 | tee logs/condition_${COND}.log

    echo "Condition $COND done: $(date)"
    grep "Accuracy:" logs/condition_${COND}.log
done

echo "========================================"
echo "All conditions complete: $(date)"
echo ""
echo "=== FINAL SUMMARY ==="
for COND in A B C D E F G H; do
    if [ -f "results/condition_${COND}/results.jsonl" ]; then
        N=$(wc -l < "results/condition_${COND}/results.jsonl")
        ACC=$(grep "Accuracy:" logs/condition_${COND}.log 2>/dev/null | tail -1 || echo "see results file")
        echo "  Condition $COND ($N problems): $ACC"
    else
        echo "  Condition $COND: not run"
    fi
done