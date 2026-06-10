"""
Centralized configuration management using Pydantic Settings.
Validates environment variables at startup.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "GST Reconciliation"
    DEBUG: bool = False

    # Backend
    PORT: int = 8001
    HOST: str = "0.0.0.0"

    # ALLOWED_ORIGINS stored as a single string in .env (comma-separated)
    # We parse it manually via property
    ALLOWED_ORIGINS_RAW: str = "http://localhost:8501,http://localhost:8502"

    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        """Parse comma-separated origins string into a list."""
        raw = self.ALLOWED_ORIGINS_RAW.strip()
        return [o.strip().strip("\"'") for o in raw.split(",") if o.strip()]

    # Database - MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "gst_reconciliation"

    # Database - Neo4j (optional)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Security (API Key authentication)
    API_KEY: str | None = None

    # Frontend backend URL
    BACKEND_URL: str = "http://localhost:8001"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
