#!/usr/bin/env bash
#SBATCH --job-name=nutricoach-n5k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=nutricoach-n5k-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH=.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

food_vision_dir=nut_agent/nutricoach/food_vision
food_vision_mode="${1:-all}"
food_vision_count="${FOOD_VISION_COUNT:-500}"
food_vision_data="$food_vision_dir/data/nutrition5k_$food_vision_count"
case "$food_vision_mode" in
    all|rgb|food_r1|papers|foodlmm|foodlmm_precision|dpf_reproduction) ;;
    *) echo 'Expected all, rgb, food_r1, papers, foodlmm, foodlmm_precision, or dpf_reproduction' >&2; exit 2 ;;
esac

if [[ "$food_vision_mode" == foodlmm_precision ]]; then
    module load cuda/12.4.1
    export MAX_JOBS=4
    food_vision_python="$food_vision_dir/data/foodlmm_env/bin/python"
    food_vision_packed="$food_vision_dir/data/foodlmm_4bit"
    if [[ ! -f "$food_vision_packed/configuration.json" ]]; then
        srun uv run --offline --python "$food_vision_python" python "$food_vision_dir/evaluate_paper_system.py" \
            --method foodlmm --quantization 4bit --export-quantized "$food_vision_packed" --prepare-only \
            --data-dir "$food_vision_data" --output-dir "$food_vision_dir/results/foodlmm_4bit_prepare"
    fi
    srun uv run --offline --python "$food_vision_python" python "$food_vision_dir/evaluate_paper_system.py" \
        --method foodlmm --quantization 4bit --quantized-checkpoint "$food_vision_packed" \
        --data-dir "$food_vision_data" --output-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_foodlmm_4bit"
    srun uv run --offline --python "$food_vision_python" python "$food_vision_dir/evaluate_paper_system.py" \
        --method foodlmm --data-dir "$food_vision_data" \
        --output-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_foodlmm_16bit_profiled"
    if [[ ! -f "$food_vision_dir/results/nutrition5k_${food_vision_count}_dpf_profiled/summary.json" ]]; then
        srun uv run --offline python "$food_vision_dir/evaluate_paper_system.py" \
            --method dpf_reproduction --data-dir "$food_vision_data" \
            --output-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_dpf_profiled"
    fi
    food_vision_runs=(
        "$food_vision_dir/results/nutrition5k_${food_vision_count}_rgb"
        "$food_vision_dir/results/nutrition5k_${food_vision_count}_food_r1"
        "$food_vision_dir/results/nutrition5k_${food_vision_count}_foodlmm_16bit_profiled"
        "$food_vision_dir/results/nutrition5k_${food_vision_count}_foodlmm_4bit"
        "$food_vision_dir/results/nutrition5k_${food_vision_count}_dpf_profiled"
    )
    if [[ -f "$food_vision_dir/results/nutrition5k_${food_vision_count}_foodlmm/foodlmm.json" ]]; then
        food_vision_runs+=("$food_vision_dir/results/nutrition5k_${food_vision_count}_foodlmm")
    fi
    srun uv run --offline python "$food_vision_dir/report_nutrition5k.py" \
        --manifest "$food_vision_data/manifest.json" --model Qwen/Qwen3.8-27B \
        --run-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_qwen_local" "${food_vision_runs[@]}" \
        --output "$food_vision_dir/nutrition5k_${food_vision_count}_qwen_results.json"
    srun uv run --offline python "$food_vision_dir/report_nutrition5k.py" \
        --manifest "$food_vision_data/manifest.json" \
        --run-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_dots_high_api" "${food_vision_runs[@]}" \
        --output "$food_vision_dir/nutrition5k_${food_vision_count}_results.json"
fi

if [[ "$food_vision_mode" == all || "$food_vision_mode" == rgb ]]; then
    srun uv run --offline python "$food_vision_dir/train_rgb_regression.py" \
        --device cuda --batch-size 64 --threads "$OMP_NUM_THREADS"
    srun uv run --offline python "$food_vision_dir/evaluate_nutrition5k.py" \
        --methods rgb_regression --device cuda --count "$food_vision_count" --data-dir "$food_vision_data" \
        --output-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_rgb"
fi
if [[ "$food_vision_mode" == all || "$food_vision_mode" == papers || "$food_vision_mode" == dpf_reproduction ]]; then
    srun uv run --offline python "$food_vision_dir/evaluate_paper_system.py" \
        --method dpf_reproduction --data-dir "$food_vision_data" \
        --output-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_dpf_profiled"
fi
if [[ "$food_vision_mode" == all || "$food_vision_mode" == papers || "$food_vision_mode" == foodlmm ]]; then
    srun uv run --offline --python "$food_vision_dir/data/foodlmm_env/bin/python" \
        python "$food_vision_dir/evaluate_paper_system.py" \
        --method foodlmm --data-dir "$food_vision_data" \
        --output-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_foodlmm_16bit_profiled"
fi
if [[ "$food_vision_mode" == all || "$food_vision_mode" == food_r1 ]]; then
    srun uv run --offline python "$food_vision_dir/evaluate_nutrition5k.py" \
        --methods food_r1 --device cuda --count "$food_vision_count" --data-dir "$food_vision_data" \
        --output-dir "$food_vision_dir/results/nutrition5k_${food_vision_count}_food_r1"
fi
