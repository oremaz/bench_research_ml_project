"""Tests for shared.auth module."""

import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.auth import (
    hash_password,
    verify_password,
    _is_legacy_sha256_hash,
    _verify_legacy_sha256,
    load_user_info,
    save_user_info,
    load_all_users,
    authenticate_user,
    register_user,
)


class TestPasswordHashing:
    def test_hash_returns_argon2_format(self):
        hashed = hash_password("mypassword")
        assert hashed.startswith("$argon2")

    def test_verify_correct_password(self):
        hashed = hash_password("mypassword")
        assert verify_password(hashed, "mypassword") is True

    def test_verify_wrong_password(self):
        hashed = hash_password("mypassword")
        assert verify_password(hashed, "wrongpassword") is False

    def test_different_passwords_produce_different_hashes(self):
        h1 = hash_password("password1")
        h2 = hash_password("password2")
        assert h1 != h2

    def test_same_password_produces_different_hashes(self):
        """Argon2 uses random salt, so same password -> different hashes."""
        h1 = hash_password("samepassword")
        h2 = hash_password("samepassword")
        assert h1 != h2


class TestLegacySHA256Migration:
    def test_is_legacy_sha256_hash(self):
        sha256_hash = hashlib.sha256("test".encode()).hexdigest()
        assert _is_legacy_sha256_hash(sha256_hash) is True

    def test_argon2_is_not_legacy(self):
        argon2_hash = hash_password("test")
        assert _is_legacy_sha256_hash(argon2_hash) is False

    def test_verify_legacy_sha256(self):
        sha256_hash = hashlib.sha256("test".encode()).hexdigest()
        assert _verify_legacy_sha256(sha256_hash, "test") is True
        assert _verify_legacy_sha256(sha256_hash, "wrong") is False

    def test_verify_password_with_legacy_hash(self):
        sha256_hash = hashlib.sha256("legacypass".encode()).hexdigest()
        assert verify_password(sha256_hash, "legacypass") is True
        assert verify_password(sha256_hash, "wrongpass") is False


class TestUserCRUD:
    def test_save_and_load_user(self, users_file):
        save_user_info("alice", {"age": 25, "password_hash": "x"}, users_file)
        loaded = load_user_info("alice", users_file)
        assert loaded is not None
        assert loaded["age"] == 25

    def test_load_nonexistent_user(self, users_file):
        loaded = load_user_info("nobody", users_file)
        assert loaded is None

    def test_load_from_empty_file(self, users_file):
        users_file.touch()
        loaded = load_user_info("alice", users_file)
        assert loaded is None

    def test_load_all_users(self, sample_users_json):
        users = load_all_users(sample_users_json)
        assert "testuser" in users

    def test_save_preserves_existing_users(self, users_file):
        save_user_info("alice", {"age": 25}, users_file)
        save_user_info("bob", {"age": 30}, users_file)
        users = load_all_users(users_file)
        assert "alice" in users
        assert "bob" in users


class TestAuthentication:
    def test_authenticate_with_argon2(self, users_file):
        profile = {"password_hash": hash_password("secure123"), "age": 30}
        save_user_info("newuser", profile, users_file)

        result = authenticate_user("newuser", "secure123", users_file)
        assert result is not None
        assert result["age"] == 30

    def test_authenticate_wrong_password(self, users_file):
        profile = {"password_hash": hash_password("secure123"), "age": 30}
        save_user_info("newuser", profile, users_file)

        result = authenticate_user("newuser", "wrong", users_file)
        assert result is None

    def test_authenticate_nonexistent_user(self, users_file):
        result = authenticate_user("ghost", "pass", users_file)
        assert result is None

    def test_authenticate_migrates_sha256_to_argon2(self, sample_users_json):
        """Logging in with a legacy SHA-256 hash should auto-migrate to argon2."""
        result = authenticate_user("testuser", "testpass123", sample_users_json)
        assert result is not None

        # Verify the hash was migrated
        updated = load_user_info("testuser", sample_users_json)
        assert updated["password_hash"].startswith("$argon2")

    def test_register_new_user(self, users_file):
        success = register_user("newuser", "pass123", {"age": 25}, users_file)
        assert success is True

        loaded = load_user_info("newuser", users_file)
        assert loaded is not None
        assert loaded["password_hash"].startswith("$argon2")

    def test_register_duplicate_user(self, users_file):
        register_user("alice", "pass1", {"age": 25}, users_file)
        success = register_user("alice", "pass2", {"age": 30}, users_file)
        assert success is False
