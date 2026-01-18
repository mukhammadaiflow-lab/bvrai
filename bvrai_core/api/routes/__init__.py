"""
API Routes Module

This module provides all REST API endpoints for the platform.
"""

from .agents import router as agents_router
from .calls import router as calls_router
from .knowledge import router as knowledge_router
from .phone_numbers import router as phone_numbers_router
from .campaigns import router as campaigns_router
from .webhooks import router as webhooks_router
from .analytics import router as analytics_router
from .auth_routes import router as auth_router
from .organization import router as organization_router
from .voice_config import router as voice_config_router
from .api_keys import router as api_keys_router
from .billing import router as billing_router


__all__ = [
    "agents_router",
    "calls_router",
    "knowledge_router",
    "phone_numbers_router",
    "campaigns_router",
    "webhooks_router",
    "analytics_router",
    "auth_router",
    "organization_router",
    "voice_config_router",
    "api_keys_router",
    "billing_router",
]
