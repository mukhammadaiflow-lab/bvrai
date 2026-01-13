"""
Telephony Integration Module

This module provides telephony capabilities including Twilio integration,
WebRTC support, SIP handling, and call routing for voice agents.
"""

from .base import (
    # Call types
    Call,
    CallDirection,
    CallState,
    CallEvent,
    CallEventType,
    CallLeg,
    CallMetadata,
    # Session types
    CallSession,
    SessionConfig,
    # Recording
    Recording,
    RecordingFormat,
    RecordingState,
    # DTMF
    DTMFEvent,
    DTMFMode,
    # Configuration
    TelephonyConfig,
    ProviderConfig,
)

from .twilio_provider import (
    TwilioProvider,
    TwilioConfig,
    TwilioWebhookHandler,
    TwilioStreamHandler,
    create_twilio_response,
)

from .webrtc import (
    WebRTCConnection,
    WebRTCConfig,
    WebRTCSignaling,
    ICECandidate,
    SDPOffer,
    SDPAnswer,
    WebRTCManager,
)

from .sip import (
    SIPProvider,
    SIPConfig,
    SIPMessage,
    SIPDialog,
    SIPSession,
)

from .routing import (
    CallRouter,
    RoutingRule,
    RoutingStrategy,
    RoutingConfig,
    RouteDestination,
)

from .manager import (
    TelephonyManager,
    CallHandler,
    create_telephony_manager,
)


__all__ = [
    # Base types
    "Call",
    "CallDirection",
    "CallState",
    "CallEvent",
    "CallEventType",
    "CallLeg",
    "CallMetadata",
    "CallSession",
    "SessionConfig",
    "Recording",
    "RecordingFormat",
    "RecordingState",
    "DTMFEvent",
    "DTMFMode",
    "TelephonyConfig",
    "ProviderConfig",
    # Twilio
    "TwilioProvider",
    "TwilioConfig",
    "TwilioWebhookHandler",
    "TwilioStreamHandler",
    "create_twilio_response",
    # WebRTC
    "WebRTCConnection",
    "WebRTCConfig",
    "WebRTCSignaling",
    "ICECandidate",
    "SDPOffer",
    "SDPAnswer",
    "WebRTCManager",
    # SIP
    "SIPProvider",
    "SIPConfig",
    "SIPMessage",
    "SIPDialog",
    "SIPSession",
    # Routing
    "CallRouter",
    "RoutingRule",
    "RoutingStrategy",
    "RoutingConfig",
    "RouteDestination",
    # Manager
    "TelephonyManager",
    "CallHandler",
    "create_telephony_manager",
]
