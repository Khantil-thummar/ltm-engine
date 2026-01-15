"""
LTM Engine - FastAPI Application Entry Point.

Long-Term Memory Engine for LLM-based applications.
"""

import sys
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ltm_engine import __version__
from ltm_engine.api import create_router
from ltm_engine.config import get_settings
from ltm_engine.dependencies import init_dependencies, close_dependencies

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()

    # Startup
    logger.info(
        "Starting LTM Engine",
        version=__version__,
        debug=settings.debug,
    )

    try:
        await init_dependencies(settings)
        logger.info("LTM Engine started successfully")
        yield
    except Exception as e:
        logger.error("Failed to start LTM Engine", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down LTM Engine")
        await close_dependencies()
        logger.info("LTM Engine shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="LTM Engine",
        description="""
## Long-Term Memory Engine for LLM-based Applications

A production-ready memory subsystem supporting:

### Memory Types
- **Episodic**: Time-stamped events, conversations, interactions (append-only)
- **Semantic**: Facts, summaries, distilled knowledge (versioned)
- **Procedural**: User preferences and patterns (key-value)

### Features
- Hybrid retrieval with semantic similarity + recency + frequency + confidence scoring
- Temporal reasoning with versioned facts and validity intervals
- Conflict detection and resolution using embeddings + LLM
- Memory consolidation (episodic → semantic)
- Decay and forget policies
- Confidence calibration
- Deterministic replay via event sourcing
- Multi-agent support with isolation

### API Operations
- `store`: Store memories by type
- `retrieve`: Hybrid search with scoring
- `consolidate`: Summarize episodic → semantic
- `decay`: Reduce importance over time
- `forget`: Remove/archive memories
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routes
    router = create_router()
    app.include_router(router, prefix="/api/v1")

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "LTM Engine",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


def run():
    """Run the application with uvicorn."""
    settings = get_settings()

    uvicorn.run(
        "ltm_engine.main:create_app",
        factory=True,
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    run()
