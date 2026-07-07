"""API router aggregation."""

from fastapi import APIRouter

from app.api import chat, delete, documents, health, upload

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(upload.router)
api_router.include_router(documents.router)
api_router.include_router(delete.router)
