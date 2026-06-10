"""
API Key authentication middleware for production.

Usage:
    Set API_KEY in .env.
    All protected routes require header:  X-API-Key: <your-key>

    Public routes (no auth needed):
        GET /          — root ping
        GET /health    — health check (for uptime monitors)
"""

import os
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Public paths that skip auth ──────────────────────
_PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


def get_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency: validate the X-API-Key header."""
    expected = settings.API_KEY
    if not expected:
        # No key configured → auth is disabled (dev mode)
        return "dev-mode"
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not secrets.compare_digest(api_key, expected):
        logger.warning("Invalid API key attempt.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key
