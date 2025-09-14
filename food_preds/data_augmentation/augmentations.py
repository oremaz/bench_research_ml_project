from typing import Tuple, Optional, Callable, Dict
import numpy as np

# imbalanced-learn samplers/cleaners
from imblearn.over_sampling import (
    SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN
)
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek

# --- Custom Augmentations ---
class MixupSMOTE:
    """
    Custom MixupSMOTE for binary classification (minority class = 1).
    """
    def __init__(self, n_samples, alpha=0.2, random_state=42, max_factor=2.0):
        self.n_samples = n_samples
        self.alpha = alpha
        self.random_state = random_state
        self.max_factor = max_factor
    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        minor = 1
        major = 0
        X_min = X[y==minor]
        X_maj = X[y==major]
        
        # Calculate target minority size based on max_factor
        # max_factor = majority_count / final_minority_count
        majority_count = len(X_maj)
        current_minor_count = len(X_min)
        target_minor_count = int(majority_count / self.max_factor)
        
        # Ensure we don't go below current count
        target_minor_count = max(target_minor_count, current_minor_count)
        actual_n_samples = target_minor_count - current_minor_count
        
        if actual_n_samples <= 0:
            return X, y
        
        synth = []
        for _ in range(actual_n_samples):
            i,j = np.random.choice(len(X_min),2,replace=True)
            lam = np.random.beta(self.alpha,self.alpha)
            synth.append(lam*X_min[i] + (1-lam)*X_min[j])
        X_new = np.vstack([X, synth])
        y_new = np.hstack([y, [minor]*len(synth)])
        return X_new, y_new

# --- Augmentation Functions ---
def none_augmentation(X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """No augmentation, returns X, y unchanged."""
    return X, y

def _apply_tomek_cleaning(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply TomekLinks cleaning to remove borderline samples."""
    cleaner = TomekLinks()
    return cleaner.fit_resample(X, y)

def _calculate_sampling_strategy(y: np.ndarray, max_factor: float = 2.0) -> Dict:
    """
    Calculate sampling strategy based on max_factor.
    max_factor = majority_count / final_minority_count
    max_factor=1.0 means complete balancing (minority = majority)
    max_factor=2.0 means minority class reaches 50% of majority class size
    max_factor=4.0 means minority class reaches 25% of majority class size
    """
    from collections import Counter
    class_counts = Counter(y)
    majority_count = max(class_counts.values())
    
    sampling_strategy = {}
    for class_label, count in class_counts.items():
        if count < majority_count:  # This is a minority class
            target_samples = int(majority_count / max_factor)
            # Ensure we don't go below current count
            target_samples = max(target_samples, count)
            sampling_strategy[class_label] = target_samples
    
    return sampling_strategy

def random_under_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    random_state: Optional[int] = 42,
    max_factor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly undersample majority classes based on ``max_factor``.

    The ``max_factor`` parameter follows the same definition as in the
    oversampling methods: ``majority_count / final_minority_count``.  When
    ``max_factor`` equals ``1.0`` the classes are perfectly balanced; higher
    values keep a portion of the majority class.
    """
    if max_factor == 1.0:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
        # RandomUnderSampler expects desired majority counts, so invert
        # the mapping produced by ``_calculate_sampling_strategy``
        from collections import Counter
        class_counts = Counter(y)
        for cls in class_counts:
            if cls not in sampling_strategy:
                target = int(max(class_counts.values()) / max_factor)
                sampling_strategy[cls] = max(target, sampling_strategy.get(cls, target))

    sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
    return sampler.fit_resample(X, y)


def nearmiss_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    random_state: Optional[int] = 42,
    max_factor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Under-sample using the NearMiss strategy.

    NearMiss selects majority samples closest to minority ones.  We mirror the
    ``max_factor`` logic used for oversampling to control the final ratio.
    """
    if max_factor == 1.0:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
        from collections import Counter
        class_counts = Counter(y)
        for cls in class_counts:
            if cls not in sampling_strategy:
                target = int(max(class_counts.values()) / max_factor)
                sampling_strategy[cls] = max(target, sampling_strategy.get(cls, target))

    sampler = NearMiss(version=1, sampling_strategy=sampling_strategy)
    return sampler.fit_resample(X, y)


def mgs_grf_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    categorical_features: Optional[np.ndarray] = None,
    random_state: int = 42,
    max_factor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper around `MGSGRFOverSampler` for mixed feature oversampling.

    This augmentation leverages the `mgs-grf` library which blends Gaussian
    resampling for continuous features with generative rules for categorical
    ones.  It gracefully degrades when the package is unavailable.
    """
    try:
        from mgs_grf import MGSGRFOverSampler
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "mgs_grf library is required for mgs_grf_augmentation."
        ) from exc

    sampler = MGSGRFOverSampler(
        categorical_features=categorical_features,
        random_state=random_state,
        ratio=max_factor,
    )
    return sampler.fit_resample(X, y)

def smote_augmentation(X: np.ndarray, y: np.ndarray, random_state: Optional[int] = 42, max_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if max_factor == 1.0:
        # Default behavior: complete balancing
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # Apply TomekLinks cleaning
    return _apply_tomek_cleaning(X_resampled, y_resampled)

def borderline_smote_augmentation(X: np.ndarray, y: np.ndarray, random_state: Optional[int] = 42, max_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if max_factor == 1.0:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
    sampler = BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    # Apply TomekLinks cleaning
    return _apply_tomek_cleaning(X_resampled, y_resampled)

def svm_smote_augmentation(X: np.ndarray, y: np.ndarray, random_state: Optional[int] = 42, max_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if max_factor == 1.0:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
    sampler = SVMSMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    # Apply TomekLinks cleaning
    return _apply_tomek_cleaning(X_resampled, y_resampled)

def kmeans_smote_augmentation(X: np.ndarray, y: np.ndarray, random_state: Optional[int] = 42, max_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if max_factor == 1.0:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
    sampler = KMeansSMOTE(random_state=random_state, cluster_balance_threshold=0.05, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    # Apply TomekLinks cleaning
    return _apply_tomek_cleaning(X_resampled, y_resampled)

def adasyn_augmentation(X: np.ndarray, y: np.ndarray, random_state: Optional[int] = 42, max_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if max_factor == 1.0:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
    sampler = ADASYN(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    # Apply TomekLinks cleaning
    return _apply_tomek_cleaning(X_resampled, y_resampled)

def mixup_augmentation(X: np.ndarray, y: np.ndarray, alpha: float = 0.2, random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    X2 = X[indices]
    y2 = y[indices]
    lam = np.random.beta(alpha, alpha, n_samples)
    lam_X = lam.reshape(-1, 1)
    X_mix = lam_X * X + (1 - lam_X) * X2
    if y.ndim == 1:
        y_mix = lam * y + (1 - lam) * y2
    else:
        y_mix = lam.reshape(-1, 1) * y + (1 - lam).reshape(-1, 1) * y2
    return X_mix, y_mix

def mixup_smote_augmentation(X: np.ndarray, y: np.ndarray, n_samples: int, alpha: float = 0.2, random_state: Optional[int] = 42, max_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    sampler = MixupSMOTE(n_samples=n_samples, alpha=alpha, random_state=random_state, max_factor=max_factor)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    # Apply TomekLinks cleaning
    return _apply_tomek_cleaning(X_resampled, y_resampled)

def smoteenn_augmentation(X: np.ndarray, y: np.ndarray, random_state: Optional[int] = 42, max_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if max_factor == 1.0:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
    sampler = SMOTEENN(random_state=random_state, sampling_strategy=sampling_strategy)
    return sampler.fit_resample(X, y)

def smotetomek_augmentation(X: np.ndarray, y: np.ndarray, random_state: Optional[int] = 42, max_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if max_factor == 1.0:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = _calculate_sampling_strategy(y, max_factor)
    sampler = SMOTETomek(random_state=random_state, sampling_strategy=sampling_strategy)
    return sampler.fit_resample(X, y)

# --- Registry ---
AUGMENTATION_REGISTRY: Dict[str, Callable] = {
    # No augmentation
    "none": none_augmentation,
    # Single samplers (all include TomekLinks cleaning)
    "smote": smote_augmentation,
    "borderline_smote": borderline_smote_augmentation,
    "svm_smote": svm_smote_augmentation,
    "kmeans_smote": kmeans_smote_augmentation,
    "adasyn": adasyn_augmentation,
    # Custom (include TomekLinks cleaning)
    "mixup": mixup_augmentation,
    "mixup_smote": lambda X, y, alpha=0.2, random_state=42, max_factor=2.0: mixup_smote_augmentation(X, y, n_samples=X.shape[0], alpha=alpha, random_state=random_state, max_factor=max_factor),
    # Hybrid samplers (already include cleaning)
    "smoteenn": smoteenn_augmentation,
    "smotetomek": smotetomek_augmentation,
    # Pure undersampling methods
    "random_under": random_under_augmentation,
    "nearmiss": nearmiss_augmentation,
    # Mixed-type oversampler from MGS-GRF
    "mgs_grf": mgs_grf_augmentation,
}

# --- Documentation ---
"""
AUGMENTATION_REGISTRY keys (selection):
- 'smote', 'borderline_smote', 'svm_smote', 'kmeans_smote', 'adasyn'
- 'mixup', 'mixup_smote', 'smoteenn', 'smotetomek'
- Undersampling: 'random_under', 'nearmiss'
- Mixed-type oversampling: 'mgs_grf'
- 'none'

Notes:
- All SMOTE variants and hybrid samplers are for classification only.
- Mixup and none can be used for regression or classification.
- MGS-GRF supports continuous and categorical features.
- All oversampling methods (except mixup and none) automatically apply TomekLinks
  cleaning after generation.

AUTOMATIC CLEANING:
- TomekLinks cleaning is automatically applied after oversampling operations
- This removes borderline samples and improves sample quality
- No need for manual composition - cleaning is built-in

max_factor parameter:
- Most augmentation functions accept a 'max_factor' parameter (default=2.0)
- max_factor = majority_count / final_minority_count
- max_factor=1.0: Complete balancing (minority class equals majority)
- max_factor=2.0: Minority reaches 50% of majority size
- max_factor=4.0: Minority reaches 25% of majority size
- Supported by: smote, borderline_smote, svm_smote, kmeans_smote, adasyn,
  smoteenn, smotetomek, mixup_smote, random_under, nearmiss, mgs_grf
- Not applicable to: mixup, none
"""
