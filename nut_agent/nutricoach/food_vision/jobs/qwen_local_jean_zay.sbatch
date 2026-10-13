#!/usr/bin/env bash
#SBATCH --job-name=nutricoach-qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=nutricoach-qwen-%j.log

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
unset OPENROUTER_API_KEY GROQ_API_KEY

food_vision_dir=nut_agent/nutricoach/food_vision
food_vision_mode="${1:-all}"
case "$food_vision_mode" in
    all|tests|benchmark|quantized) ;;
    *) echo 'Expected all, tests, benchmark, or quantized' >&2; exit 2 ;;
esac
if [[ "$food_vision_mode" == all || "$food_vision_mode" == tests ]]; then
    export RUN_LOCAL_QWEN_TESTS=1
    food_vision_test_tmp="${SLURM_TMPDIR:-/tmp}/nutricoach-qwen-tests-${SLURM_JOB_ID}"
    trap 'rm -rf "$food_vision_test_tmp"' EXIT
    srun uv run --offline python -m pytest nut_agent/tests/ -q -p no:cacheprovider \
        --junitxml="$food_vision_dir/results/slurm/qwen-tests-${SLURM_JOB_ID}.xml" \
        --basetemp="$food_vision_test_tmp"
fi
if [[ "$food_vision_mode" == quantized ]]; then
    export RUN_LOCAL_QWEN_TESTS=1
    export RUN_LOCAL_QWEN_4BIT_TEST=1
    food_vision_test_tmp="${SLURM_TMPDIR:-/tmp}/nutricoach-qwen-tests-${SLURM_JOB_ID}"
    trap 'rm -rf "$food_vision_test_tmp"' EXIT
    srun uv run --offline python -m pytest \
        nut_agent/tests/test_local_qwen_live.py::test_local_4bit_photo_with_32_gib_gpu_limit \
        -q -s -p no:cacheprovider --basetemp="$food_vision_test_tmp"
fi
if [[ "$food_vision_mode" == all || "$food_vision_mode" == benchmark ]]; then
    srun uv run --offline python "$food_vision_dir/evaluate_nutrition5k.py" \
        --provider local --model Qwen/Qwen3.8-27B --methods vlm_single \
        --reasoning-effort none --device cuda --workers 1 --request-interval 0 \
        --count 500 --data-dir "$food_vision_dir/data/nutrition5k_500" \
        --output-dir "$food_vision_dir/results/nutrition5k_500_qwen_single" \
        --retry-errors
    srun uv run --offline python "$food_vision_dir/report_nutrition5k.py" \
        --model Qwen/Qwen3.8-27B \
        --run-dir "$food_vision_dir/results/nutrition5k_500_qwen_single" \
                  "$food_vision_dir/results/nutrition5k_500_rgb" \
                  "$food_vision_dir/results/nutrition5k_500_food_r1" \
                  "$food_vision_dir/results/nutrition5k_500_foodlmm" \
                  "$food_vision_dir/results/nutrition5k_500_dpf_reproduction" \
        --output "$food_vision_dir/nutrition5k_500_qwen_results.json"
fi
