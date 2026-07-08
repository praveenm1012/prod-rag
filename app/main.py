"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.router import api_router
from app.api.ui import router as ui_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.observability import setup_observability, shutdown_observability, span
from app.observability.middleware import ObservabilityMiddleware

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks."""
    settings = get_settings()
    setup_logging(settings)
    logger = get_logger(__name__)

    tracer = setup_observability(settings)

    with span(
        tracer,
        "application.startup",
        attributes={
            "service": settings.observability_service_name,
            "environment": settings.app_env,
            "version": __version__,
            "backend": tracer.backend_name,
        },
    ) as startup_span:
        startup_span.set_input(
            {
                "app_name": settings.app_name,
                "environment": settings.app_env,
            }
        )
        logger.info(
            "application_starting",
            app_name=settings.app_name,
            version=__version__,
            environment=settings.app_env,
            observability_backend=tracer.backend_name,
        )
        startup_span.set_output({"status": "ready"})

    yield

    with span(tracer, "application.shutdown"):
        logger.info("application_stopping", app_name=settings.app_name)

    if settings.observability_flush_on_shutdown:
        shutdown_observability()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.app_debug,
        lifespan=lifespan,
    )
    if settings.observability_enabled:
        application.add_middleware(ObservabilityMiddleware)
    application.include_router(ui_router)
    application.include_router(api_router)
    application.mount(
        "/static",
        StaticFiles(directory=_STATIC_DIR),
        name="static",
    )
    return application


app = create_app()


def run() -> None:
    """Run the application with uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_config=None,
    )


if __name__ == "__main__":
    run()
