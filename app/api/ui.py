"""UI routes for the API test console."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@router.get("/", include_in_schema=False)
async def test_console() -> FileResponse:
    """Serve the API test console."""
    return FileResponse(_STATIC_DIR / "index.html")


@router.get("/ui", include_in_schema=False)
async def test_console_alias() -> FileResponse:
    """Serve the API test console at /ui."""
    return FileResponse(_STATIC_DIR / "index.html")
