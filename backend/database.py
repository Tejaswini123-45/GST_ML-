"""
Database connection managers for MongoDB (local or Atlas) and Neo4j (local or Aura).

Both use singleton patterns — one connection per process, reused everywhere.
All credentials are loaded from environment variables.

Neo4j is optional — if unavailable, graph features degrade gracefully (return empty results).
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import GraphDatabase

from backend.utils.logger import get_logger
from backend.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

# ────────────────────────────────────────────
# Singleton holders
# ────────────────────────────────────────────
_mongo_client: AsyncIOMotorClient | None = None
_neo4j_driver = None
_neo4j_available: bool | None = None  # cached availability flag


# ════════════════════════════════════════════
# MongoDB (async via Motor)
# ════════════════════════════════════════════

def get_mongo_client() -> AsyncIOMotorClient:
    """Return the singleton Motor client, creating it on first call."""
    global _mongo_client
    if _mongo_client is None:
        uri = settings.MONGO_URI
        # Build client kwargs
        kwargs: dict = {
            "serverSelectionTimeoutMS": 5000,
            "connectTimeoutMS": 5000,
            "socketTimeoutMS": 30000,
        }
        # Only add TLS CA for Atlas (mongodb+srv or with ssl)
        if "mongodb+srv" in uri or "ssl=true" in uri.lower():
            import certifi
            kwargs["tlsCAFile"] = certifi.where()

        _mongo_client = AsyncIOMotorClient(uri, **kwargs)
        logger.info("MongoDB client created for: %s", uri.split("@")[-1] if "@" in uri else uri)
    return _mongo_client


def get_mongo_db():
    """Return the default MongoDB database handle."""
    return get_mongo_client()[settings.MONGO_DB_NAME]


async def ping_mongo() -> bool:
    """Return True if MongoDB responds to a ping."""
    try:
        result = await get_mongo_client().admin.command("ping")
        return result.get("ok") == 1.0
    except Exception as exc:
        logger.error("MongoDB ping failed: %s", exc)
        return False


# ════════════════════════════════════════════
# Neo4j (sync driver — optional)
# ════════════════════════════════════════════

def get_neo4j_driver():
    """Return the singleton Neo4j driver, creating it on first call.
    Returns None if Neo4j is not configured or unavailable.
    """
    global _neo4j_driver, _neo4j_available
    if _neo4j_driver is not None:
        return _neo4j_driver
    if _neo4j_available is False:
        return None  # already tried and failed

    try:
        uri = settings.NEO4J_URI
        user = settings.NEO4J_USER
        password = settings.NEO4J_PASSWORD

        _neo4j_driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=5,
            connection_acquisition_timeout=5,
            max_transaction_retry_time=10,
        )
        # Test connectivity immediately
        _neo4j_driver.verify_connectivity()
        _neo4j_available = True
        logger.info("Neo4j driver connected to: %s", uri)
        return _neo4j_driver
    except Exception as exc:
        logger.warning("Neo4j unavailable: %s — graph features disabled.", exc)
        _neo4j_driver = None
        _neo4j_available = False
        return None


def ping_neo4j() -> bool:
    """Return True if Neo4j responds to a connectivity check."""
    try:
        driver = get_neo4j_driver()
        if driver is None:
            return False
        driver.verify_connectivity()
        return True
    except Exception as exc:
        logger.error("Neo4j ping failed: %s", exc)
        return False


def is_neo4j_available() -> bool:
    """Return cached Neo4j availability flag."""
    global _neo4j_available
    if _neo4j_available is None:
        # Trigger an availability check
        ping_neo4j()
    return _neo4j_available is True


# ════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════

async def close_connections():
    """Gracefully close both database connections."""
    global _mongo_client, _neo4j_driver, _neo4j_available

    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        logger.info("MongoDB client closed.")

    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None
        logger.info("Neo4j driver closed.")

    _neo4j_available = None
