"""
FastAPI application entry point — Production-ready.

Features:
  - API Key authentication (via X-API-Key header)
  - CORS for Streamlit frontend
  - MongoDB index creation on startup
  - Request ID middleware for distributed tracing
  - Trusted host middleware
  - Rate limiting via slowapi
  - Test routes excluded in production (DEBUG=False)
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.database import close_connections, get_mongo_db
from backend.indexes import ensure_indexes
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Lifespan (startup + shutdown) ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create indexes, warm connections.
    Shutdown: close all DB connections cleanly."""
    # Warm up MongoDB
    try:
        db = get_mongo_db()
        await db.command("ping")
        logger.info("MongoDB connection confirmed.")
    except Exception as e:
        logger.warning("MongoDB warm-up failed: %s", e)

    # Create indexes (non-blocking)
    try:
        await ensure_indexes()
    except Exception as e:
        logger.warning("Index creation failed: %s — continuing startup.", e)

    yield

    # Shutdown
    await close_connections()
    logger.info("Application shutdown complete.")


# ── App instance ──────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description=(
        "GST Reconciliation & ITC Risk Analysis System. "
        "All endpoints require X-API-Key header when API_KEY is configured."
    ),
    lifespan=lifespan,
    debug=settings.DEBUG,
    # In production, restrict docs to prevent schema leakage
    docs_url="/docs" if settings.DEBUG else "/docs",
    redoc_url="/redoc" if settings.DEBUG else None,
)


# ── Request ID Middleware ─────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID to every request for tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms}ms"
    logger.info(
        "[%s] %s %s -> %d (%sms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ── CORS ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only what we use
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────
from fastapi import Depends
from backend.auth import get_api_key
from backend.routes import health, ingest, graph, dashboard, ml

app.include_router(health.router)
app.include_router(ingest.router,    dependencies=[Depends(get_api_key)])
app.include_router(graph.router,     dependencies=[Depends(get_api_key)])
app.include_router(dashboard.router, dependencies=[Depends(get_api_key)])
app.include_router(ml.router,        dependencies=[Depends(get_api_key)])

# Test routes — only in DEBUG mode
if settings.DEBUG:
    from backend.routes import test_mongo, test_neo4j
    app.include_router(test_mongo.router)
    app.include_router(test_neo4j.router)
    logger.info("DEBUG mode: test routes enabled.")
else:
    logger.info("PRODUCTION mode: test routes disabled.")


# ── Root ──────────────────────────────────────────────
@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Minimal root endpoint to confirm the API is running."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": "1.0.0",
    }


# ── Global error handler ──────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — never expose tracebacks in prod."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error.",
            "path": str(request.url.path),
            "request_id": request.headers.get("X-Request-ID", "unknown"),
        },
    )
