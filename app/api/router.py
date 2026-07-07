"""API router aggregation."""

from fastapi import APIRouter

from app.api import chat, health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(chat.router)
