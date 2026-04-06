"""
Authentication and user management module.
Uses argon2 for password hashing with transparent SHA-256 migration.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

logger = logging.getLogger(__name__)

ph = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a password using argon2."""
    return ph.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """
    Verify a password against a stored hash.
    Supports both argon2 and legacy SHA-256 hashes.
    """
    try:
        return ph.verify(stored_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:
        # Not an argon2 hash — try legacy SHA-256
        return _verify_legacy_sha256(stored_hash, password)


def _is_legacy_sha256_hash(stored_hash: str) -> bool:
    """Check if a hash is a legacy SHA-256 hex string (64 chars)."""
    return len(stored_hash) == 64 and all(c in '0123456789abcdef' for c in stored_hash)


def _verify_legacy_sha256(stored_hash: str, password: str) -> bool:
    """Verify a password against a legacy SHA-256 hash."""
    if not _is_legacy_sha256_hash(stored_hash):
        return False
    return hashlib.sha256(password.encode()).hexdigest() == stored_hash


def load_all_users(users_file: Path) -> Dict[str, Any]:
    """Load all users from users.json."""
    try:
        if users_file.exists() and users_file.stat().st_size > 0:
            with open(users_file, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("users.json was corrupted")
    except Exception as e:
        logger.warning(f"Could not load users: {e}")
    return {}


def save_all_users(users_file: Path, all_users: Dict[str, Any]) -> None:
    """Save all users to users.json."""
    users_file.parent.mkdir(parents=True, exist_ok=True)
    with open(users_file, "w") as f:
        json.dump(all_users, f, indent=2)
        f.flush()


def load_user_info(username: str, users_file: Path) -> Optional[Dict[str, Any]]:
    """Load a single user's info from disk."""
    all_users = load_all_users(users_file)
    return all_users.get(username)


def save_user_info(username: str, user_info: Dict[str, Any], users_file: Path) -> None:
    """Save a single user's info to disk."""
    all_users = load_all_users(users_file)
    all_users[username] = user_info
    save_all_users(users_file, all_users)


def authenticate_user(username: str, password: str, users_file: Path) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user. Returns user_info on success, None on failure.
    Transparently migrates SHA-256 hashes to argon2.
    """
    user_info = load_user_info(username, users_file)
    if user_info is None:
        return None

    stored_hash = user_info.get('password_hash', '')

    if not verify_password(stored_hash, password):
        return None

    # Migrate legacy SHA-256 hash to argon2
    if _is_legacy_sha256_hash(stored_hash):
        user_info['password_hash'] = hash_password(password)
        save_user_info(username, user_info, users_file)
        logger.info(f"Migrated password hash for user '{username}' from SHA-256 to argon2")

    return user_info


def register_user(username: str, password: str, profile: Dict[str, Any], users_file: Path) -> bool:
    """
    Register a new user. Returns True on success, False if username exists.
    """
    all_users = load_all_users(users_file)
    if username in all_users:
        return False

    profile['password_hash'] = hash_password(password)
    all_users[username] = profile
    save_all_users(users_file, all_users)
    return True
