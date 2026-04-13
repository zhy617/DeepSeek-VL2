#!/bin/bash
set -eu

# ============================================================
# Phase 2c: REAP Pruning + Bridge Variant
# B10: REAP 50% prune + eval
# B11: REAP 75% prune + eval
# B12: REAP+Bridge 50% prune + eval
# B13: REAP+Bridge 75% prune + eval
# ============================================================

cd /root/DeepSeek-VL2
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/fsas/models/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache}"
export PYTHONPATH="/root/DeepSeek-VL2:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

BASELINES="$HOME/fsas/vlm/deepseek-vl2-bridge/results/baselines"
RESULTS="$HOME/fsas/vlm/deepseek-vl2-bridge/results"
BASE_MM_JSON="$RESULTS/step5_base_mm_full/lmms_eval_results.json"
BASE_TEXT_JSON="$RESULTS/step5_base_lm_eval_full/lm_eval_results.json"
LOGDIR="$HOME/fsas/vlm/deepseek-vl2-bridge/logs"
REAP_SCRIPT="mystle/experiments/reap_prune.py"
EVAL_SCRIPT="mystle/experiments/evaluate.py"

mkdir -p "$LOGDIR"

log_step() {
    echo ""
    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "========================================"
}

run_eval() {
    local backend=$1
    local pretrained=$2
    local run_id=$3
    local baseline_json=$4
    local batch_size=$5

    log_step "EVAL: $run_id (backend=$backend)"

    if [ -f "$RESULTS/$run_id/retention.json" ]; then
        echo "SKIP: $run_id already has retention.json"
        return 0
    fi

    python "$EVAL_SCRIPT" \
        --backend "$backend" \
        --pretrained "$pretrained" \
        --run-id "$run_id" \
        --baseline-json "$baseline_json" \
        --batch-size "$batch_size" \
        --dtype bfloat16
}

echo "============================================================"
echo "Phase 2c: REAP Pruning Chain"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- B10: REAP 50% ----
log_step "B10: REAP 50% prune"
if [ ! -f "$BASELINES/reap_keep_0p5/merged_model/config.json" ]; then
    python "$REAP_SCRIPT" \
        --compression-keep 0.5 \
        --output-dir "$RESULTS" \
        --probe-count 256 \
        --smoke-forward
else
    echo "SKIP: REAP 50% checkpoint already exists"
fi

run_eval mm "$BASELINES/reap_keep_0p5/merged_model" \
    "batch_b10_reap_0p50_mm" "$BASE_MM_JSON" 1
run_eval text "$BASELINES/reap_keep_0p5/merged_model" \
    "batch_b10_reap_0p50_text" "$BASE_TEXT_JSON" 4

# ---- B11: REAP 75% ----
log_step "B11: REAP 75% prune"
if [ ! -f "$BASELINES/reap_keep_0p75/merged_model/config.json" ]; then
    python "$REAP_SCRIPT" \
        --compression-keep 0.75 \
        --output-dir "$RESULTS" \
        --probe-count 256 \
        --smoke-forward
else
    echo "SKIP: REAP 75% checkpoint already exists"
fi

run_eval mm "$BASELINES/reap_keep_0p75/merged_model" \
    "batch_b11_reap_0p75_mm" "$BASE_MM_JSON" 1
run_eval text "$BASELINES/reap_keep_0p75/merged_model" \
    "batch_b11_reap_0p75_text" "$BASE_TEXT_JSON" 4

# ---- B12: REAP+Bridge 50% ----
log_step "B12: REAP+Bridge 50% prune"
if [ ! -f "$BASELINES/reap_bridge_keep_0p5/merged_model/config.json" ]; then
    python "$REAP_SCRIPT" \
        --compression-keep 0.5 \
        --bridge \
        --output-dir "$RESULTS" \
        --probe-count 256 \
        --smoke-forward
else
    echo "SKIP: REAP+Bridge 50% checkpoint already exists"
fi

run_eval mm "$BASELINES/reap_bridge_keep_0p5/merged_model" \
    "batch_b12_reap_bridge_0p50_mm" "$BASE_MM_JSON" 1
run_eval text "$BASELINES/reap_bridge_keep_0p5/merged_model" \
    "batch_b12_reap_bridge_0p50_text" "$BASE_TEXT_JSON" 4

# ---- B13: REAP+Bridge 75% ----
log_step "B13: REAP+Bridge 75% prune"
if [ ! -f "$BASELINES/reap_bridge_keep_0p75/merged_model/config.json" ]; then
    python "$REAP_SCRIPT" \
        --compression-keep 0.75 \
        --bridge \
        --output-dir "$RESULTS" \
        --probe-count 256 \
        --smoke-forward
else
    echo "SKIP: REAP+Bridge 75% checkpoint already exists"
fi

run_eval mm "$BASELINES/reap_bridge_keep_0p75/merged_model" \
    "batch_b13_reap_bridge_0p75_mm" "$BASE_MM_JSON" 1
run_eval text "$BASELINES/reap_bridge_keep_0p75/merged_model" \
    "batch_b13_reap_bridge_0p75_text" "$BASE_TEXT_JSON" 4

echo ""
echo "============================================================"
echo "Phase 2c: REAP Pruning Chain COMPLETE"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "Results written to: $RESULTS/batch_b1[0-3]*/"
echo "Next steps: Phase 3 (cross-model) or Paper Writing"
