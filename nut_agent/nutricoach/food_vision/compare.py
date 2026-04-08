"""
Comparison framework for food image analysis methods.

Run all methods on the same images and produce a side-by-side comparison
of detected items, calorie estimates, latency, and cost.

Usage:
    python -m nutricoach.food_vision.compare --image plate.jpg
    python -m nutricoach.food_vision.compare --image-dir ./test_images/
    python -m nutricoach.food_vision.compare --image plate.jpg --methods vlm_claude,rag_vlm
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import FoodAnalysisResult, FoodAnalyzer

logger = logging.getLogger(__name__)


def get_available_methods() -> Dict[str, type]:
    """Return all available analyzer classes keyed by method name."""
    methods = {}

    try:
        from .rf_detr_analyzer import RFDETRAnalyzer
        methods["rf_detr"] = RFDETRAnalyzer
    except ImportError:
        logger.debug("RF-DETR not available (install rfdetr)")

    try:
        from .vlm_analyzer import VLMAnalyzer, VLMAnalyzerSingleShot
        methods["vlm_claude"] = VLMAnalyzer
        methods["vlm_claude_single"] = VLMAnalyzerSingleShot
    except ImportError:
        logger.debug("VLM analyzer not available (install openai)")

    try:
        from .clip_analyzer import CLIPFoodAnalyzer
        methods["clip_ensemble"] = CLIPFoodAnalyzer
    except ImportError:
        logger.debug("CLIP analyzer not available (install transformers)")

    try:
        from .rag_vlm_analyzer import RAGVLMAnalyzer
        methods["rag_vlm"] = RAGVLMAnalyzer
    except ImportError:
        logger.debug("RAG VLM not available (install openai)")

    return methods


def run_comparison(
    image_path: str,
    methods: Optional[List[str]] = None,
    rf_detr_weights: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
) -> Dict[str, FoodAnalysisResult]:
    """
    Run specified (or all) methods on a single image.

    Args:
        image_path: Path to the food image.
        methods: List of method names to run. None = all available.
        rf_detr_weights: Path to fine-tuned RF-DETR weights.
        openrouter_api_key: API key for OpenRouter (VLM methods).

    Returns:
        Dict mapping method name to FoodAnalysisResult.
    """
    available = get_available_methods()

    if methods:
        selected = {k: v for k, v in available.items() if k in methods}
    else:
        selected = available

    if not selected:
        logger.error("No methods available. Install required dependencies.")
        return {}

    results = {}
    for name, cls in selected.items():
        logger.info("Running method: %s", name)
        try:
            # Instantiate with appropriate kwargs
            kwargs = {}
            if name == "rf_detr" and rf_detr_weights:
                kwargs["model_path"] = rf_detr_weights
            if name in ("vlm_claude", "vlm_claude_single", "rag_vlm") and openrouter_api_key:
                kwargs["api_key"] = openrouter_api_key
            if name == "clip_ensemble" and openrouter_api_key:
                kwargs["openrouter_api_key"] = openrouter_api_key

            analyzer = cls(**kwargs)
            result = analyzer.analyze(image_path)
            results[name] = result
            logger.info("  → %d items, %.0f kcal, %.1fs",
                        len(result.food_items), result.total_calories, result.elapsed_seconds)

        except Exception as e:
            logger.error("  → Failed: %s", e)
            results[name] = FoodAnalysisResult(method=name, error=str(e))

    return results


def format_comparison(results: Dict[str, FoodAnalysisResult]) -> str:
    """Format comparison results as a readable table."""
    lines = []
    lines.append("=" * 80)
    lines.append("FOOD IMAGE ANALYSIS — METHOD COMPARISON")
    lines.append("=" * 80)

    for method, result in results.items():
        lines.append("")
        lines.append(result.summary())

    # Summary table
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{'Method':<25} {'Items':>5} {'Calories':>8} {'Protein':>8} "
                 f"{'Carbs':>8} {'Fat':>8} {'Time':>7}")
    lines.append("-" * 80)

    for method, result in results.items():
        if result.error:
            lines.append(f"{method:<25} {'ERROR':>5} {result.error[:40]}")
        else:
            lines.append(
                f"{method:<25} {len(result.food_items):>5} "
                f"{result.total_calories:>8.0f} {result.total_protein_g:>8.1f} "
                f"{result.total_carbs_g:>8.1f} {result.total_fat_g:>8.1f} "
                f"{result.elapsed_seconds:>6.1f}s"
            )

    lines.append("-" * 80)

    # Calorie agreement analysis
    calorie_values = [
        r.total_calories for r in results.values()
        if not r.error and r.total_calories > 0
    ]
    if len(calorie_values) >= 2:
        avg = sum(calorie_values) / len(calorie_values)
        spread = max(calorie_values) - min(calorie_values)
        lines.append(f"\nCalorie estimates: avg={avg:.0f}, spread={spread:.0f} "
                     f"(±{spread/2/avg*100:.0f}% from mean)")

    return "\n".join(lines)


def save_results(
    results: Dict[str, FoodAnalysisResult],
    output_path: str,
):
    """Save comparison results to JSON."""
    data = {
        method: result.to_dict()
        for method, result in results.items()
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare food image analysis methods"
    )
    parser.add_argument(
        "--image", type=str, help="Path to a single food image"
    )
    parser.add_argument(
        "--image-dir", type=str, help="Directory of food images to analyze"
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated method names (default: all available)"
    )
    parser.add_argument(
        "--rf-detr-weights", type=str, default=None,
        help="Path to fine-tuned RF-DETR weights"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    method_list = args.methods.split(",") if args.methods else None

    if args.image:
        images = [args.image]
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        images = sorted(
            str(p)
            for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    else:
        parser.error("Provide --image or --image-dir")
        return

    all_results = {}
    for img_path in images:
        print(f"\n{'='*60}")
        print(f"Analyzing: {img_path}")
        print(f"{'='*60}")

        results = run_comparison(
            img_path,
            methods=method_list,
            rf_detr_weights=args.rf_detr_weights,
        )
        all_results[img_path] = results
        print(format_comparison(results))

    if args.output:
        flat = {}
        for img, results in all_results.items():
            flat[img] = {m: r.to_dict() for m, r in results.items()}
        with open(args.output, "w") as f:
            json.dump(flat, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
