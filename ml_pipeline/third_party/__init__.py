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
    verbose: bool = False,
) -> Optional[object]:
    token = f"class {symbol_name}" if is_class else f"def {symbol_name}"
    # Walk the repo and try importing modules by their dotted path relative to repo_dir.
    # This preserves package context so relative imports work.
    for py_file in repo_dir.rglob("*.py"):
        # Skip hidden, cache, and tests directories quickly
        parts = set(p.name for p in py_file.parents)
        if any(skip in parts for skip in {".git", "__pycache__", ".venv", "env", "venv"}):
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if token not in text:
            continue

        # Derive module name relative to repo_dir, e.g., repo_dir/foo/bar.py -> foo.bar
        try:
            rel = py_file.relative_to(repo_dir).with_suffix("")
        except ValueError:
            # Not under repo_dir for some reason
            continue
        # Ignore top-level __init__.py checks; import package itself if matched
        if rel.name == "__init__":
            module_name = ".".join(rel.parts[:-1]) if len(rel.parts) > 1 else ""
        else:
            module_name = ".".join(rel.parts)
        if not module_name:
            # Nothing to import
            continue

        # Prefer fully-qualified import to avoid collisions with generic names like 'models'
        fq_module_name = f"{repo_dir.name}.{module_name}"
        module = None
        for name in (fq_module_name, module_name):
            try:
                module = import_module(name)
                break
            except Exception:
                if verbose:
                    print(f"[third_party] Import failed for {name} while searching for {symbol_name} in {py_file}")
                module = None
        if module is None:
            # If importing by name fails (e.g., missing deps), try best-effort file load as fallback
            try:
                module = _load_module_from_file(str(py_file))
            except Exception:
                if verbose:
                    print(f"[third_party] Fallback file-load failed for {py_file} while searching for {symbol_name}")
                continue

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
    verbose: bool = False,
) -> type:
    """Load a class from a third-party repository by name."""
    repo_dir = resolve_repo_path(repo_name, repo_path, env_var)
    # Add both the repository directory and its parent to sys.path.
    # The parent enables imports like '<repo_name>.<subpkg>.<module>' which helps
    # avoid conflicts with common top-level names (e.g., 'models').
    ensure_sys_path(repo_dir.parent)
    ensure_sys_path(repo_dir)

    if module_candidates:
        for module_path in module_candidates:
            try:
                module = import_module(module_path)
            except ImportError:
                if verbose:
                    print(f"[third_party] ImportError for candidate module '{module_path}'")
                continue
            if hasattr(module, class_name):
                return getattr(module, class_name)
            else:
                if verbose:
                    print(f"[third_party] Candidate module '{module_path}' imported but has no attribute '{class_name}'")

    symbol = _search_symbol_in_repo(repo_dir, class_name, is_class=True, verbose=verbose)
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
    # See rationale in load_class about adding the parent path first
    ensure_sys_path(repo_dir.parent)
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