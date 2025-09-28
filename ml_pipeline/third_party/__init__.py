"""Utilities for working with locally cloned third-party research repositories."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional

THIRD_PARTY_ROOT = Path(__file__).resolve().parent


def _candidate_paths(repo_name: str, repo_path: Optional[str] = None, env_var: Optional[str] = None) -> Iterable[Path]:
    if repo_path:
        yield Path(repo_path).expanduser()
    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            yield Path(env_value).expanduser()
    yield THIRD_PARTY_ROOT / repo_name


def resolve_repo_path(repo_name: str, repo_path: Optional[str] = None, env_var: Optional[str] = None) -> Path:
    """Return the first existing path for a given repository name."""
    for candidate in _candidate_paths(repo_name, repo_path, env_var):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository '{repo_name}'. Checked: {[str(p) for p in _candidate_paths(repo_name, repo_path, env_var)]}."
    )


def ensure_sys_path(path: Path) -> None:
    """Add *path* to sys.path if not already present."""
    absolute = str(path.resolve())
    if absolute not in sys.path:
        sys.path.insert(0, absolute)


@lru_cache(maxsize=128)
def _load_module_from_file(py_file: str) -> ModuleType:
    spec = spec_from_file_location(f"third_party_{hash(py_file)}", py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for module at {py_file}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _search_symbol_in_repo(
    repo_dir: Path,
    symbol_name: str,
    is_class: bool = True,
) -> Optional[object]:
    token = f"class {symbol_name}" if is_class else f"def {symbol_name}"
    for py_file in repo_dir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if token not in text:
            continue
        module = _load_module_from_file(str(py_file))
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)
    return None


def load_class(
    repo_name: str,
    class_name: str,
    *,
    repo_path: Optional[str] = None,
    env_var: Optional[str] = None,
    module_candidates: Optional[Iterable[str]] = None,
) -> type:
    """Load a class from a third-party repository by name."""
    repo_dir = resolve_repo_path(repo_name, repo_path, env_var)
    ensure_sys_path(repo_dir)

    if module_candidates:
        for module_path in module_candidates:
            try:
                module = import_module(module_path)
            except ImportError:
                continue
            if hasattr(module, class_name):
                return getattr(module, class_name)

    symbol = _search_symbol_in_repo(repo_dir, class_name, is_class=True)
    if symbol is None:
        raise ImportError(
            f"Unable to locate class '{class_name}' inside repository '{repo_name}'. "
            "Consider specifying 'module_candidates' or setting the repo path explicitly."
        )
    return symbol  # type: ignore[return-value]


def load_function(
    repo_name: str,
    function_name: str,
    *,
    repo_path: Optional[str] = None,
    env_var: Optional[str] = None,
    module_candidates: Optional[Iterable[str]] = None,
):
    """Load a function from a third-party repository by name."""
    repo_dir = resolve_repo_path(repo_name, repo_path, env_var)
    ensure_sys_path(repo_dir)

    if module_candidates:
        for module_path in module_candidates:
            try:
                module = import_module(module_path)
            except ImportError:
                continue
            if hasattr(module, function_name):
                return getattr(module, function_name)

    symbol = _search_symbol_in_repo(repo_dir, function_name, is_class=False)
    if symbol is None:
        raise ImportError(
            f"Unable to locate function '{function_name}' inside repository '{repo_name}'."
        )
    return symbol

if __name__ == "__main__":
    # Example usage
    FatFormer = load_class("FatFormer", "CLIPModel")
    print(FatFormer)