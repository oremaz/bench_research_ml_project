"""Cached FoodLMM inference in the isolated author environment."""

import atexit
import json
import os
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path

from .base import FoodAnalyzer
from .supervised_analyzers import total_result


class FoodLMMAnalyzer(FoodAnalyzer):
    method_name = "foodlmm_4bit"

    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        atexit.register(self.close)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def analyze(self, image_path):
        started = time.perf_counter()
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.close()
                directory = Path(__file__).parent
                python = Path(os.getenv("FOODLMM_PYTHON") or directory / "data/foodlmm_env/bin/python")
                if not python.is_file():
                    raise RuntimeError("FoodLMM requires its isolated environment. Install requirements_foodlmm.in "
                                       "in food_vision/data/foodlmm_env or set FOODLMM_PYTHON.")
                environment = os.environ.copy()
                environment["PATH"] = str(python.absolute().parent) + os.pathsep + environment.get("PATH", "")
                self.process = subprocess.Popen(
                    [str(python.absolute()), "-u", str(directory / "foodlmm_worker.py")],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=environment,
                )
            try:
                self.process.stdin.write(json.dumps({"image_path": str(Path(image_path).resolve())}) + "\n")
                self.process.stdin.flush()
                response = self.process.stdout.readline()
                if not response:
                    raise RuntimeError("FoodLMM worker stopped. Check its dependencies and available memory.")
                payload = json.loads(response)
            except (OSError, ValueError, RuntimeError):
                self.close()
                raise
            if payload.get("error"):
                raise RuntimeError(payload["error"])
            return total_result(self.method_name, payload["values"], started, payload["raw_response"])


@lru_cache(maxsize=1)
def get_foodlmm_analyzer():
    return FoodLMMAnalyzer()
