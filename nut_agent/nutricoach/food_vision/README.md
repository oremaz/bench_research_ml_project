# Food Vision: Multi-Method Food Image Analysis

Analyze food photos to estimate ingredients, portion sizes, and calories/macros.  
Four methods are implemented for comparison, from traditional CV to pure LLM.

## Methods

### Method 1: RF-DETR Fine-Tuning (`rf_detr_analyzer.py`)
- **Architecture**: RF-DETR (ICLR 2026), DINOv2 backbone + deformable attention
- **Approach**: Object detection with bounding boxes → portion estimation from bbox area → nutrition DB lookup
- **Fine-tuning**: Supports training on FoodSeg103, UEC-FoodPix, or custom COCO-format datasets via `RFDETRFoodTrainer`
- **Strengths**: Real-time inference, runs locally, no API costs
- **Weaknesses**: Needs fine-tuning for good food detection (COCO only has 10 food classes); portion estimation from bbox area is crude
- **Install**: `pip install rfdetr supervision`

### Method 2: Pure vLLM via OpenRouter (`vlm_analyzer.py`)
- **Architecture**: OpenRouter vision model (default is a free-tier model set by
  `OPENROUTER_VISION_MODEL` in `shared/config.py`, with server-side fallbacks;
  point it at e.g. `anthropic/claude-opus-4-6` on a paid tier for best quality)
- **Approach**: 3-step chained prompts: (1) identify foods, (2) estimate portions, (3) compute nutrition
- **Also includes**: `VLMAnalyzerSingleShot`, a single comprehensive prompt variant (faster, cheaper)
- **Strengths**: Best food identification (handles complex dishes, mixed meals, sauces); no model training needed
- **Weaknesses**: API cost (~$0.01-0.05/image), latency (3-10s), no local fallback
- **Install**: `pip install openai` + set `OPENROUTER_API_KEY`

### Method 3: CLIP Zero-Shot + LLM Ensemble (`clip_analyzer.py`)
- **Architecture**: CLIP ViT-B/32 + Claude Opus via OpenRouter
- **Approach**: CLIP zero-shot classifies against 100+ food labels → LLM refines detections and estimates portions → nutrition DB lookup
- **Strengths**: CLIP runs locally (fast, free); LLM only needed for refinement; good for common foods
- **Weaknesses**: CLIP single-label classification misses multiple items; LLM refinement still needs API
- **Install**: `pip install transformers torch openai`
- **Alternative backend**: `CLIPFoodAnalyzer(backend="jina")` swaps CLIP for
  `jinaai/jina-embeddings-v5-omni-small` (multimodal embeddings, 1.74B params).
  Benchmarked below; CLIP remains the default because it is both more accurate
  and faster on food classification.

#### Zero-shot backend benchmark (Food101)

1000 Food101 validation images, 101 class prompts, cuda:1
(`ml_pipeline/bench_food_image_zeroshot.py`, results in
`ml_pipeline/results/bench_food_image_zeroshot.json`, 2026-07):

| Backend | Top-1 | Top-5 | ms / image |
|---------|-------|-------|------------|
| CLIP ViT-B/32 (default) | **0.796** | **0.954** | **237** |
| jina-embeddings-v5-omni-small | 0.743 | 0.918 | 687 |

CLIP's contrastive image-text pretraining is a better fit for prompt-based
zero-shot classification than a general-purpose retrieval embedder, at a
third of the latency and 12x fewer parameters. Note jina-v5 is CC BY-NC 4.0
(non-commercial).

### Method 4: RAG-Enhanced VLM, DietAI24-inspired (`rag_vlm_analyzer.py`)
- **Architecture**: Claude Opus via OpenRouter + local nutrition DB retrieval
- **Approach**: VLM identifies foods → retrieve per-100g nutrition data from DB → VLM reasons over image + DB data → cross-validate portions against standard serving sizes
- **Strengths**: Grounds calorie estimates in real nutrition data (reduces hallucination); cross-validates portions; most accurate method
- **Weaknesses**: 2 API calls per image; slightly slower than single-shot VLM
- **Install**: `pip install openai`

## Comparison Framework (`compare.py`)

Run all methods on the same image and compare results:

```bash
# From nut_agent/ directory
python -m nutricoach.food_vision.compare --image path/to/plate.jpg

# Specific methods only
python -m nutricoach.food_vision.compare --image plate.jpg --methods vlm_claude,rag_vlm

# Directory of images, save results
python -m nutricoach.food_vision.compare --image-dir ./test_images/ --output results.json
```

Output includes per-method breakdown, calorie agreement analysis, and timing.

## Nutrition Database (`nutrition_db.py`)

Local database with 130+ common foods (per-100g values from USDA/CIQUAL):
- Proteins, carbs/grains, vegetables, fruits, dairy, legumes
- Common dishes (pizza, burrito, sushi, curry, etc.)
- Snacks, desserts, sauces, drinks
- Fuzzy matching for flexible food name lookup

## Integration with NutriCoach

The `analyze_food_image` tool in `nutricoach/tools.py` integrates food vision into the LangGraph agent:

```
User: "Analyze this photo of my lunch" → Agent calls analyze_food_image tool
  → RAG VLM identifies: grilled chicken (180g), rice (200g), broccoli (120g)
  → Returns: 650 kcal, P:52g, C:58g, F:12g
  → Auto-logs to daily intake
```

## RF-DETR Fine-Tuning Guide

No public food-specific RF-DETR checkpoint exists. This repo fine-tunes on
FoodSeg103 (103 ingredient classes) converted from segmentation masks to COCO
bounding boxes:

```bash
# One-shot: download + convert FoodSeg103, then fine-tune on GPU 1
PYTHONPATH=. uv run python ml_pipeline/prepare_foodseg103_coco.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python ml_pipeline/train_rf_detr_food.py
# Weights land in ml_pipeline/results/rf_detr_food/
```

Trained 2026-07: 8 epochs on 4983 images reached EMA mAP@50:95 = 0.381
(mAP@50 = 0.433) on the 500-image validation split. Load with the fine-tuned
class count so the detection head is not re-initialized:

```python
analyzer = RFDETRAnalyzer(
    model_path="ml_pipeline/results/rf_detr_food/checkpoint_best_total.pth",
    num_classes=103,
)
```

Manual alternative:

```python
from nutricoach.food_vision.rf_detr_analyzer import RFDETRFoodTrainer

trainer = RFDETRFoodTrainer(model_size="base")

# Option 1: Download from Roboflow
trainer.download_food_dataset(
    dataset_name="food-detection",
    api_key="your_roboflow_key",
    output_dir="./food_coco/",
)

# Option 2: Use FoodSeg103 (convert to COCO format first)

# Train
best_checkpoint = trainer.train(
    dataset_dir="./food_coco/",
    epochs=50,
    batch_size=4,
    lr=1e-4,
    output_dir="./rf_detr_food_weights/",
)
```

## Architecture Decision: Why 4 Methods?

| Criterion | RF-DETR | VLM Chain | CLIP+LLM | RAG VLM |
|-----------|---------|-----------|----------|---------|
| Accuracy (food ID) | Low (COCO) / High (fine-tuned) | High | Medium | High |
| Accuracy (portions) | Low (bbox area) | Medium | Medium | High (DB-grounded) |
| Latency | <1s | 5-10s | 2-5s | 4-8s |
| Cost/image | $0 | ~$0.03 | ~$0.01 | ~$0.02 |
| Needs training | Yes | No | No | No |
| Offline capable | Yes | No | Partial | No |

**Recommendation**: Use **RAG VLM** for best accuracy, **VLM single-shot** for speed, **RF-DETR** for offline/free usage after fine-tuning.
