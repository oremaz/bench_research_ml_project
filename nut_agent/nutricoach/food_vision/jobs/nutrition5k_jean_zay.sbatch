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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

food_vision_dir=nut_agent/nutricoach/food_vision
food_vision_mode="${1:-all}"
case "$food_vision_mode" in
    all|rgb|food_r1) ;;
    *) echo 'Expected all, rgb, or food_r1' >&2; exit 2 ;;
esac

if [[ "$food_vision_mode" == all || "$food_vision_mode" == rgb ]]; then
    srun uv run --offline python "$food_vision_dir/train_rgb_regression.py" \
        --device cuda --batch-size 64 --threads "$OMP_NUM_THREADS"
    srun uv run --offline python "$food_vision_dir/evaluate_nutrition5k.py" \
        --methods rgb_regression --device cuda \
        --output-dir "$food_vision_dir/results/nutrition5k_50_rgb"
fi
if [[ "$food_vision_mode" == all || "$food_vision_mode" == food_r1 ]]; then
    srun uv run --offline python "$food_vision_dir/evaluate_nutrition5k.py" \
        --methods food_r1 --device cuda \
        --output-dir "$food_vision_dir/results/nutrition5k_50_food_r1"
fi
