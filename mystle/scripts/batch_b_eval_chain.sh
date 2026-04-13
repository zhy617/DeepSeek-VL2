#!/bin/bash
set -e

cd /root/DeepSeek-VL2
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/fsas/models/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/fsas/datasets/deepseek-vl2-bridge/hf_datasets_cache}"
export LMMS_EVAL_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export PYTHONPATH="/root/DeepSeek-VL2:/root/DeepSeek-VL2/lm-evaluation-harness:/root/DeepSeek-VL2/lmms-eval:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

BASELINES="$HOME/fsas/vlm/deepseek-vl2-bridge/results/baselines"
RESULTS="$HOME/fsas/vlm/deepseek-vl2-bridge/results"
BASE_MM_JSON="$RESULTS/step5_base_mm_full/lmms_eval_results.json"
BASE_TEXT_JSON="$RESULTS/step5_base_lm_eval_full/lm_eval_results.json"
LOGDIR="$HOME/fsas/vlm/deepseek-vl2-bridge/logs"
mkdir -p "$LOGDIR"

SCRIPT="mystle/experiments/evaluate.py"

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

    log_step "START: $run_id (backend=$backend)"

    if [ -f "$RESULTS/$run_id/retention.json" ]; then
        echo "SKIP: $run_id already has retention.json"
        return 0
    fi

    python "$SCRIPT" \
        --backend "$backend" \
        --pretrained "$pretrained" \
        --run-id "$run_id" \
        --baseline-json "$baseline_json" \
        --batch-size "$batch_size" \
        --dtype bfloat16

    log_step "DONE: $run_id"
}

echo "============================================================"
echo "Batch B Evaluation Chain (reordered: 50% first, 75% second)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ===== PART 1: 50% models (fit on 48G easily, ~31GB each) =====

# B4: HC-SMoE 50% text
run_eval text "$BASELINES/hcsmoe_keep_0p50/merged_model" \
    "batch_b4_hcsmoe_0p50_text" "$BASE_TEXT_JSON" 4

# B5: MergeMoE 50% text
run_eval text "$BASELINES/mergemoe_keep_0p50/merged_model" \
    "batch_b5_mergemoe_0p50_text" "$BASE_TEXT_JSON" 4

# B6: Ours 50% text
run_eval text "$BASELINES/admissibility_keep_0p50/merged_model" \
    "batch_b6_ours_0p50_text" "$BASE_TEXT_JSON" 4

# B14: MC-SMoE 50% multimodal + text
run_eval mm "$BASELINES/mcsmoe_keep_0p5/merged_model" \
    "batch_b14_mcsmoe_0p50_mm" "$BASE_MM_JSON" 1
run_eval text "$BASELINES/mcsmoe_keep_0p5/merged_model" \
    "batch_b14_mcsmoe_0p50_text" "$BASE_TEXT_JSON" 4

# B16: MC-SMoE+Bridge 50% multimodal + text
run_eval mm "$BASELINES/mcsmoe_bridge_keep_0p5/merged_model" \
    "batch_b16_mcsmoe_bridge_0p50_mm" "$BASE_MM_JSON" 1
run_eval text "$BASELINES/mcsmoe_bridge_keep_0p5/merged_model" \
    "batch_b16_mcsmoe_bridge_0p50_text" "$BASE_TEXT_JSON" 4

log_step "PART 1 DONE: All 50% evaluations complete"

# ===== PART 2: 75% models (device_map=auto for offload, ~45GB each) =====

# B7: HC-SMoE 75% text
run_eval text "$BASELINES/hcsmoe_keep_0p75/merged_model" \
    "batch_b7_hcsmoe_0p75_text" "$BASE_TEXT_JSON" 4

# B8: MergeMoE 75% text
run_eval text "$BASELINES/mergemoe_keep_0p75/merged_model" \
    "batch_b8_mergemoe_0p75_text" "$BASE_TEXT_JSON" 4

# B9: Ours 75% text
run_eval text "$BASELINES/admissibility_keep_0p75/merged_model" \
    "batch_b9_ours_0p75_text" "$BASE_TEXT_JSON" 4

# B1: HC-SMoE 75% multimodal
run_eval mm "$BASELINES/hcsmoe_keep_0p75/merged_model" \
    "batch_b1_hcsmoe_0p75_mm" "$BASE_MM_JSON" 1

# B2: MergeMoE 75% multimodal
run_eval mm "$BASELINES/mergemoe_keep_0p75/merged_model" \
    "batch_b2_mergemoe_0p75_mm" "$BASE_MM_JSON" 1

# B3: Ours 75% multimodal
run_eval mm "$BASELINES/admissibility_keep_0p75/merged_model" \
    "batch_b3_ours_0p75_mm" "$BASE_MM_JSON" 1

# B15: MC-SMoE 75% multimodal + text
run_eval mm "$BASELINES/mcsmoe_keep_0p75/merged_model" \
    "batch_b15_mcsmoe_0p75_mm" "$BASE_MM_JSON" 1
run_eval text "$BASELINES/mcsmoe_keep_0p75/merged_model" \
    "batch_b15_mcsmoe_0p75_text" "$BASE_TEXT_JSON" 4

# B17: MC-SMoE+Bridge 75% multimodal + text
run_eval mm "$BASELINES/mcsmoe_bridge_keep_0p75/merged_model" \
    "batch_b17_mcsmoe_bridge_0p75_mm" "$BASE_MM_JSON" 1
run_eval text "$BASELINES/mcsmoe_bridge_keep_0p75/merged_model" \
    "batch_b17_mcsmoe_bridge_0p75_text" "$BASE_TEXT_JSON" 4

echo ""
echo "============================================================"
echo "Batch B Evaluation Chain COMPLETE"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
