"""
Pydantic schemas for Dialog Manager API.

Defines request/response models for the dialog turn endpoint.
"""
from typing import Any, Literal
from pydantic import BaseModel, Field


class DialogTurnRequest(BaseModel):
    """Request model for dialog turn endpoint."""

    tenant_id: str = Field(..., min_length=1, max_length=128, description="Tenant identifier")
    session_id: str = Field(..., min_length=1, max_length=128, description="Session identifier")
    transcript: str = Field(..., min_length=1, max_length=10000, description="User transcript")
    is_final: bool = Field(True, description="Whether this is a final transcript")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenant_id": "tenant-123",
                    "session_id": "session-abc",
                    "transcript": "I would like to book an appointment",
                    "is_final": True,
                }
            ]
        }
    }


class ActionObject(BaseModel):
    """
    Structured action extracted from dialog.

    Actions represent automations or integrations to execute.
    """

    action_type: str = Field(..., description="Type of action (e.g., 'create_booking', 'lookup_faq')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in action extraction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action_type": "create_booking",
                    "parameters": {
                        "service": "haircut",
                        "date": "2024-01-15",
                        "time": "14:00",
                    },
                    "confidence": 0.92,
                }
            ]
        }
    }


class DialogTurnResponse(BaseModel):
    """Response model for dialog turn endpoint."""

    speak_text: str = Field(..., description="Text for TTS synthesis")
    action_object: ActionObject | None = Field(None, description="Extracted action if any")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall response confidence")
    session_id: str = Field(..., description="Session identifier")
    context_used: list[str] = Field(
        default_factory=list,
        description="Document IDs used for context"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "speak_text": "I'd be happy to help you book an appointment. What service would you like?",
                    "action_object": None,
                    "confidence": 0.95,
                    "session_id": "session-abc",
                    "context_used": ["doc-1", "doc-2"],
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = "healthy"
    service: str = "dialog-manager"
    version: str = "1.0.0"
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    code: str
    details: dict[str, Any] | None = None
