"""
Master API v1 Router.
Aggregates all modular endpoint routers into a unified prefix hierarchy.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    analysis,
    detections,
    health,
    incidents,
    media,
    priority,
    recommendations,
    resources,
)

api_router = APIRouter()

# Register modular sub-routers
api_router.include_router(health.router)
api_router.include_router(incidents.router)
api_router.include_router(media.router)
api_router.include_router(analysis.router)
api_router.include_router(detections.router)
api_router.include_router(priority.router)
api_router.include_router(recommendations.router)
api_router.include_router(resources.router)
