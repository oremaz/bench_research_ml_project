# Data Augmentation Modules

This folder contains augmentation modules for text, image, and tabular/numerical data, plus research-grade wrappers that live in `ml_pipeline/pipelines_torch/augmentations.py`:

- **`text_aug.py`**: Text augmentation using LLM prompting (Google Gemini) and classical techniques (synonym replacement, random insertion, deletion, swap).
- **`image_aug.py`**: Image augmentation using classical computer vision (OpenCV, PIL) and Albumentations library.
- **`augmentations.py`**: Tabular/numerical data augmentation using SMOTE variants, Mixup, hybrid samplers, and research-inspired methods (Simplicial SMOTE, MEB-SMOTE).
- **`../pipelines_torch/augmentations.py`**: Official research integrations (MGS-GRF oversampler, TabEBM generator) accessible through the shared registry pattern.

## Features

### Text Augmentation (`text_aug.py`)
- LLM-based: Paraphrase, synonym substitution, style variation (Google Gemini API required)
- Classical: Synonym replacement, random insertion, random deletion, random swap

#### LLM-based Augmentation
- **Paraphrasing**: Generate semantic variations using Google's Gemini (`llm_paraphrase`)
- **Synonym Substitution**: Replace words with contextually appropriate synonyms (`llm_synonym`)
- **Style Variation**: Rewrite text in different writing styles (`llm_style`)

#### Classical Augmentation
- **Synonym Replacement**: Replace words with predefined synonyms (`classical_synonym`)
- **Random Insertion**: Insert random words at random positions (`classical_insertion`)
- **Random Deletion**: Remove random words from text (`classical_deletion`)
- **Random Swap**: Swap adjacent words randomly (`classical_swap`)
- **Mixed**: Apply a combination of classical techniques (`classical_mixed`)

### Image Augmentation (`image_aug.py`)
- Classical: Rotation, translation, scaling, flipping, brightness/contrast/saturation, noise, blur, elastic transform
- Albumentations: Basic, geometric, noise/blur, color, weather, perspective

#### Classical Techniques (OpenCV + PIL)
- **Geometric Transformations**: Rotation, translation, scaling, flipping (`classical_rotation`, `classical_geometric`)
- **Color Adjustments**: Brightness, contrast, saturation modifications (`classical_color`)
- **Noise and Blur**: Gaussian noise, various blur effects (`classical_noise_blur`)
- **Elastic Transformations**: Non-linear deformations (`classical_elastic`)
- **Mixed**: Apply a combination of classical techniques (`classical_mixed`)

#### Albumentations Library
- **Basic**: Standard augmentations (`albumentations_basic`)
- **Geometric**: Shift-scale-rotate, elastic transform, grid distortion (`albumentations_geometric`)
- **Noise and Blur**: Multiple blur types, noise variations (`albumentations_noise_blur`)
- **Color Manipulation**: RGB shifts, channel shuffling, hue/saturation (`albumentations_color`)
- **Weather Effects**: Rain, fog, sun flare effects (`albumentations_weather`)
- **Perspective Transformations**: Perspective and piecewise affine transforms (`albumentations_perspective`)

### Tabular/Numerical Data Augmentation (`augmentations.py`)
- SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN
- Mixup, MixupSMOTE
- SimplicialSMOTE (Kachan et al., 2025) for simplex-based interpolation
- MEBSMOTE (Shangguan et al., 2024) for minimum enclosing ball oversampling
- SMOTEENN, SMOTETomek
- All oversampling methods (except mixup/none) apply TomekLinks cleaning
- The `max_factor` parameter controls the minority/majority ratio for most samplers.

### Research Integrations (`../pipelines_torch/augmentations.py`)
- **MGS_GRF_Augmentor**: Wraps the official MGS-GRF repository for guided minority sample generation
- **TabEBMGenerator**: Wraps the official TabEBM (NeurIPS 2024) energy-based model for conditional sampling
- **SimplicialSMOTE & MEBSMOTE**: Re-exported for registry-style usage alongside the official wrappers
- Uses the shared `third_party/` loader to dynamically import upstream GitHub repositories without vendoring code
- Configurable via explicit paths or environment variables (`MGS_GRF_REPO`, `TABEBM_REPO`)

## Usage Example

```python
from text_aug import TEXT_AUGMENTATION_REGISTRY
from image_aug import IMAGE_AUGMENTATION_REGISTRY
from augmentations import AUGMENTATION_REGISTRY
from pipelines_torch.augmentations import MGS_GRF_Augmentor

# Text augmentation
augmented_texts = TEXT_AUGMENTATION_REGISTRY["classical_synonym"](["Hello world"])

# Image augmentation
augmented_images = IMAGE_AUGMENTATION_REGISTRY["classical_geometric"]([your_image_array])

# Tabular/numerical augmentation from the built-in registry
X_aug, y_aug = AUGMENTATION_REGISTRY["smote"](X, y)

# Official research augmentor (make sure the upstream repo is available)
mgs = MGS_GRF_Augmentor()
X_bal, y_bal = mgs.fit_resample(X, y, target_class=1, n_samples=128)
```

## Pointing to Official Repositories

The `third_party/` loader automatically searches for cloned research repositories inside `third_party/`, custom paths, or environment variables. For example:

```bash
export MGS_GRF_REPO=/path/to/mgs-grf
export TABEBM_REPO=/path/to/TabEBM
```

## Notes
- For LLM-based text augmentation, set the `GOOGLE_API_KEY` environment variable.
- All registry keys and features listed above are present in the code. See each file for full API and configuration options.
- Research integrations require the official repositories to be available locally; see the example environment variables above.
