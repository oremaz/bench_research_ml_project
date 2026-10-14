"""Cached local DPF food-photo inference using the existing research adapter."""

import sys
import time
import threading
from functools import lru_cache

from .base import FoodAnalyzer
from .supervised_analyzers import total_result


class DPFAnalyzer(FoodAnalyzer):
    method_name = "dpf_reproduction"

    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def analyze(self, image_path):
        import torch
        from PIL import Image, ImageOps
        from .evaluate_paper_system import DPFReproduction, prepare_release, select_device

        started = time.perf_counter()
        with self.lock:
            if self.model is None:
                if sys.platform == "darwin":
                    # Avoid the LightGBM/PyTorch OpenMP conflict on macOS.
                    torch.set_num_threads(1)
                release = prepare_release(self.method_name)
                if str(release) not in sys.path:
                    sys.path.insert(0, str(release))
                self.model = DPFReproduction(release, select_device("auto"))
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((1024, 1024))
            values, raw_response = self.model.predict(image)
        return total_result(self.method_name, values, started, raw_response)


@lru_cache(maxsize=1)
def get_dpf_analyzer():
    return DPFAnalyzer()
