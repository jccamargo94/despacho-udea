import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from services.api.auth import decode_bearer_token


class _FakeSigningKey:
    def __init__(self, key):
        self.key = key


class _FakeJWKClient:
    def __init__(self, public_key):
        self._public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._public_key)


@pytest.fixture
def rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _token(private_pem, **claims):
    payload = {"sub": "user-1", "aud": "authenticated", "exp": int(time.time()) + 3600, **claims}
    return jwt.encode(payload, private_pem, algorithm="RS256")


def test_decode_bearer_token_returns_payload_for_valid_token(rsa_keypair):
    private_pem, public_pem = rsa_keypair
    token = _token(private_pem)
    payload = decode_bearer_token(f"Bearer {token}", _FakeJWKClient(public_pem))
    assert payload["sub"] == "user-1"


def test_decode_bearer_token_rejects_missing_bearer_prefix(rsa_keypair):
    _, public_pem = rsa_keypair
    with pytest.raises(HTTPException) as exc_info:
        decode_bearer_token("not-a-bearer-token", _FakeJWKClient(public_pem))
    assert exc_info.value.status_code == 401


def test_decode_bearer_token_rejects_expired_token(rsa_keypair):
    private_pem, public_pem = rsa_keypair
    token = _token(private_pem, exp=int(time.time()) - 10)
    with pytest.raises(HTTPException) as exc_info:
        decode_bearer_token(f"Bearer {token}", _FakeJWKClient(public_pem))
    assert exc_info.value.status_code == 401


def test_decode_bearer_token_rejects_wrong_audience(rsa_keypair):
    private_pem, public_pem = rsa_keypair
    token = _token(private_pem, aud="something-else")
    with pytest.raises(HTTPException) as exc_info:
        decode_bearer_token(f"Bearer {token}", _FakeJWKClient(public_pem))
    assert exc_info.value.status_code == 401


def test_decode_bearer_token_rejects_signature_from_wrong_key(rsa_keypair):
    private_pem, _ = rsa_keypair
    other_public_pem = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    token = _token(private_pem)
    with pytest.raises(HTTPException) as exc_info:
        decode_bearer_token(f"Bearer {token}", _FakeJWKClient(other_public_pem))
    assert exc_info.value.status_code == 401
