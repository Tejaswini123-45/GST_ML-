"""
Health check endpoint.
Returns UP / DOWN status for MongoDB and Neo4j.
Neo4j check is run in a thread to avoid blocking the event loop.
"""

import asyncio
from fastapi import APIRouter
from backend.database import ping_mongo, ping_neo4j
from backend.models import HealthStatus

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Ping both databases and report their status."""
    mongo_ok, neo4j_ok = await asyncio.gather(
        ping_mongo(),
        asyncio.to_thread(ping_neo4j),
    )
    return HealthStatus(
        mongodb="UP" if mongo_ok else "DOWN",
        neo4j="UP" if neo4j_ok else "DOWN",
    )
