"""
API v1 Routes

All versioned API routes for Builder Engine Platform.
"""

from fastapi import APIRouter

from app.api.v1.phone_numbers import router as phone_numbers_router
from app.api.v1.recordings import router as recordings_router
from app.api.v1.workflows import router as workflows_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.voice import router as voice_router
from app.api.v1.llm import router as llm_router
from app.api.v1.billing import router as billing_router
from app.api.v1.auth import router as auth_router
from app.api.v1.tenants import router as tenants_router
from app.api.v1.integrations import router as integrations_router

# Create main v1 router
router = APIRouter(prefix="/v1")

# Include all sub-routers
router.include_router(auth_router, tags=["Authentication"])
router.include_router(tenants_router, tags=["Tenants"])
router.include_router(phone_numbers_router, tags=["Phone Numbers"])
router.include_router(recordings_router, tags=["Recordings"])
router.include_router(workflows_router, tags=["Workflows"])
router.include_router(conversations_router, tags=["Conversations"])
router.include_router(voice_router, tags=["Voice"])
router.include_router(llm_router, tags=["LLM"])
router.include_router(billing_router, tags=["Billing"])
router.include_router(integrations_router, tags=["Integrations"])

__all__ = ["router"]
