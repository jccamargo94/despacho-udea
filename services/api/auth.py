import os

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

_jwk_client: PyJWKClient | None = None


def _get_jwk_client() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = PyJWKClient(os.environ["SUPABASE_JWKS_URL"])
    return _jwk_client


def decode_bearer_token(authorization: str, jwk_client: PyJWKClient) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ")
    try:
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        return jwt.decode(token, signing_key.key, algorithms=["RS256"], audience="authenticated")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}") from e


def get_current_user_id(authorization: str = Header(...)) -> str:
    payload = decode_bearer_token(authorization, _get_jwk_client())
    return payload["sub"]
