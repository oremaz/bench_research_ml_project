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
- **Architecture**: Claude Opus 4.6 via OpenRouter API
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

No public food-specific RF-DETR checkpoint exists. To fine-tune:

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
