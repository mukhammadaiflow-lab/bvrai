"""
Telephony Manager Module

This module provides a unified interface for managing telephony
operations across multiple providers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from .base import (
    Call,
    CallDirection,
    CallEvent,
    CallEventType,
    CallSession,
    CallState,
    Recording,
    SessionConfig,
    TelephonyConfig,
    ProviderConfig,
)
from .twilio_provider import TwilioProvider, TwilioConfig
from .webrtc import WebRTCManager, WebRTCConnection, WebRTCConfig
from .sip import SIPProvider, SIPConfig
from .routing import CallRouter, RouteDestination, RoutingConfig


logger = logging.getLogger(__name__)


class CallHandler(ABC):
    """Abstract handler for call events."""

    @abstractmethod
    async def on_call_started(self, call: Call, session: CallSession) -> None:
        """Handle call start."""
        pass

    @abstractmethod
    async def on_audio_received(
        self,
        call: Call,
        audio_data: bytes,
    ) -> Optional[bytes]:
        """
        Handle received audio.

        Args:
            call: Call object
            audio_data: Received audio bytes

        Returns:
            Optional audio response
        """
        pass

    @abstractmethod
    async def on_call_ended(self, call: Call) -> None:
        """Handle call end."""
        pass


@dataclass
class ManagerConfig:
    """Configuration for telephony manager."""

    # Provider selection
    default_provider: str = "twilio"
    enabled_providers: List[str] = field(default_factory=lambda: ["twilio", "webrtc"])

    # Provider configs
    twilio_config: Optional[TwilioConfig] = None
    webrtc_config: Optional[WebRTCConfig] = None
    sip_config: Optional[SIPConfig] = None

    # Routing
    routing_config: Optional[RoutingConfig] = None

    # Session defaults
    session_config: SessionConfig = field(default_factory=SessionConfig)

    # Concurrent call limits
    max_concurrent_calls: int = 100
    max_calls_per_second: float = 10.0

    # Events
    on_call_started: Optional[Callable[[Call], None]] = None
    on_call_ended: Optional[Callable[[Call], None]] = None
    on_call_error: Optional[Callable[[Call, str], None]] = None


class TelephonyManager:
    """
    Central manager for all telephony operations.

    Coordinates between multiple providers, handles routing,
    and manages call sessions.
    """

    def __init__(self, config: Optional[ManagerConfig] = None):
        """
        Initialize telephony manager.

        Args:
            config: Manager configuration
        """
        self.config = config or ManagerConfig()

        # Initialize providers
        self._providers: Dict[str, Any] = {}
        self._initialize_providers()

        # Initialize routing
        self._router = CallRouter(self.config.routing_config)

        # Active calls and sessions
        self._calls: Dict[str, Call] = {}
        self._sessions: Dict[str, CallSession] = {}
        self._call_handlers: Dict[str, CallHandler] = {}

        # Rate limiting
        self._call_semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        self._call_count = 0

        # Statistics
        self._stats = {
            "total_calls": 0,
            "active_calls": 0,
            "completed_calls": 0,
            "failed_calls": 0,
            "total_duration_seconds": 0.0,
        }

    def _initialize_providers(self) -> None:
        """Initialize enabled providers."""
        if "twilio" in self.config.enabled_providers:
            twilio_config = self.config.twilio_config or TwilioConfig.from_env()
            self._providers["twilio"] = TwilioProvider(twilio_config)

        if "webrtc" in self.config.enabled_providers:
            webrtc_config = self.config.webrtc_config or WebRTCConfig()
            self._providers["webrtc"] = WebRTCManager(webrtc_config)

        if "sip" in self.config.enabled_providers and self.config.sip_config:
            self._providers["sip"] = SIPProvider(self.config.sip_config)

    async def start(self) -> None:
        """Start the telephony manager."""
        # Start SIP provider if enabled
        if "sip" in self._providers:
            await self._providers["sip"].start()

        logger.info("Telephony manager started")

    async def stop(self) -> None:
        """Stop the telephony manager."""
        # End all active calls
        for call_id in list(self._calls.keys()):
            try:
                await self.end_call(call_id)
            except Exception as e:
                logger.error(f"Error ending call {call_id}: {e}")

        # Stop SIP provider
        if "sip" in self._providers:
            await self._providers["sip"].stop()

        logger.info("Telephony manager stopped")

    async def make_call(
        self,
        to_number: str,
        from_number: Optional[str] = None,
        provider: Optional[str] = None,
        handler: Optional[CallHandler] = None,
        **kwargs,
    ) -> Call:
        """
        Make an outbound call.

        Args:
            to_number: Number to call
            from_number: Caller ID
            provider: Provider to use (default: config default)
            handler: Call handler for events
            **kwargs: Additional provider-specific options

        Returns:
            Call object
        """
        provider_name = provider or self.config.default_provider

        if provider_name not in self._providers:
            raise ValueError(f"Provider not available: {provider_name}")

        async with self._call_semaphore:
            provider_instance = self._providers[provider_name]

            # Make call through provider
            if provider_name == "twilio":
                call = await provider_instance.make_call(
                    to_number=to_number,
                    from_number=from_number,
                    **kwargs,
                )
            elif provider_name == "sip":
                call = await provider_instance.make_call(
                    to_number=to_number,
                    from_number=from_number,
                )
            else:
                raise ValueError(f"Provider {provider_name} doesn't support outbound calls")

            # Track call
            self._calls[call.id] = call
            self._stats["total_calls"] += 1
            self._stats["active_calls"] += 1

            # Create session
            session = CallSession(
                call=call,
                config=self.config.session_config,
            )
            session.is_active = True
            self._sessions[call.id] = session

            # Register handler
            if handler:
                self._call_handlers[call.id] = handler
                await handler.on_call_started(call, session)

            # Emit event
            if self.config.on_call_started:
                self.config.on_call_started(call)

            logger.info(f"Outbound call started: {call.id} -> {to_number}")

            return call

    async def handle_inbound_call(
        self,
        call: Call,
        provider: str = "twilio",
        handler: Optional[CallHandler] = None,
    ) -> CallSession:
        """
        Handle an inbound call.

        Args:
            call: Inbound call from provider
            provider: Provider name
            handler: Call handler

        Returns:
            Call session
        """
        async with self._call_semaphore:
            # Track call
            self._calls[call.id] = call
            self._stats["total_calls"] += 1
            self._stats["active_calls"] += 1

            # Route call
            destination = self._router.route(call)

            # Create session
            session = CallSession(
                call=call,
                config=self.config.session_config,
            )
            session.is_active = True
            self._sessions[call.id] = session

            # Register handler
            if handler:
                self._call_handlers[call.id] = handler
                await handler.on_call_started(call, session)

            # Emit event
            if self.config.on_call_started:
                self.config.on_call_started(call)

            logger.info(f"Inbound call handled: {call.id} from {call.from_number}")

            return session

    async def end_call(self, call_id: str) -> Optional[Call]:
        """
        End a call.

        Args:
            call_id: Call ID

        Returns:
            Ended call or None
        """
        call = self._calls.get(call_id)
        if not call:
            return None

        provider_name = call.provider
        provider = self._providers.get(provider_name)

        try:
            if provider:
                if provider_name == "twilio":
                    await provider.end_call(call.provider_call_id or call_id)
                elif provider_name == "sip":
                    await provider.end_call(call.provider_call_id or call_id)

            # Update call state
            call.state = CallState.COMPLETED
            call.ended_at = datetime.utcnow()

            # Calculate duration
            if call.answered_at:
                call.talk_duration_seconds = (
                    call.ended_at - call.answered_at
                ).total_seconds()

            # Update session
            session = self._sessions.get(call_id)
            if session:
                session.is_active = False

            # Call handler
            handler = self._call_handlers.get(call_id)
            if handler:
                await handler.on_call_ended(call)

            # Update stats
            self._stats["active_calls"] -= 1
            self._stats["completed_calls"] += 1
            self._stats["total_duration_seconds"] += call.talk_duration_seconds

            # Emit event
            if self.config.on_call_ended:
                self.config.on_call_ended(call)

            logger.info(
                f"Call ended: {call_id} "
                f"(duration: {call.talk_duration_seconds:.1f}s)"
            )

        except Exception as e:
            logger.error(f"Error ending call {call_id}: {e}")
            self._stats["failed_calls"] += 1

            if self.config.on_call_error:
                self.config.on_call_error(call, str(e))

        return call

    async def transfer_call(
        self,
        call_id: str,
        to_number: str,
        announce: Optional[str] = None,
    ) -> Call:
        """
        Transfer a call.

        Args:
            call_id: Call ID
            to_number: Transfer destination
            announce: Announcement message

        Returns:
            Updated call
        """
        call = self._calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        provider_name = call.provider
        provider = self._providers.get(provider_name)

        if provider_name == "twilio" and provider:
            await provider.transfer_call(
                call.provider_call_id or call_id,
                to_number,
                announce,
            )

        call.add_event(
            CallEventType.TRANSFER_STARTED,
            {"to": to_number}
        )

        logger.info(f"Call {call_id} transferred to {to_number}")

        return call

    async def hold_call(self, call_id: str) -> Call:
        """
        Put a call on hold.

        Args:
            call_id: Call ID

        Returns:
            Updated call
        """
        call = self._calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        provider_name = call.provider
        provider = self._providers.get(provider_name)

        if provider_name == "twilio" and provider:
            await provider.hold_call(call.provider_call_id or call_id)

        call.is_on_hold = True
        call.state = CallState.ON_HOLD
        call.add_event(CallEventType.HOLD_STARTED)

        return call

    async def resume_call(
        self,
        call_id: str,
        webhook_url: str,
    ) -> Call:
        """
        Resume a held call.

        Args:
            call_id: Call ID
            webhook_url: URL for next instructions

        Returns:
            Updated call
        """
        call = self._calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        provider_name = call.provider
        provider = self._providers.get(provider_name)

        if provider_name == "twilio" and provider:
            await provider.resume_call(
                call.provider_call_id or call_id,
                webhook_url,
            )

        call.is_on_hold = False
        call.state = CallState.IN_PROGRESS
        call.add_event(CallEventType.HOLD_ENDED)

        return call

    async def start_recording(self, call_id: str) -> Optional[Recording]:
        """
        Start recording a call.

        Args:
            call_id: Call ID

        Returns:
            Recording object
        """
        call = self._calls.get(call_id)
        if not call:
            return None

        provider_name = call.provider
        provider = self._providers.get(provider_name)

        if provider_name == "twilio" and provider:
            return await provider.start_recording(
                call.provider_call_id or call_id
            )

        return None

    async def stop_recording(self, call_id: str) -> Optional[Recording]:
        """
        Stop recording a call.

        Args:
            call_id: Call ID

        Returns:
            Recording object
        """
        call = self._calls.get(call_id)
        if not call:
            return None

        provider_name = call.provider
        provider = self._providers.get(provider_name)

        if provider_name == "twilio" and provider:
            return await provider.stop_recording(
                call.provider_call_id or call_id
            )

        return None

    async def send_audio(
        self,
        call_id: str,
        audio_data: bytes,
    ) -> None:
        """
        Send audio to a call.

        Args:
            call_id: Call ID
            audio_data: Audio bytes
        """
        session = self._sessions.get(call_id)
        if not session or not session.is_active:
            return

        # Route to appropriate provider
        call = session.call
        provider_name = call.provider

        if provider_name == "webrtc":
            webrtc = self._providers.get("webrtc")
            if webrtc:
                # Find associated WebRTC connection
                # (In practice, session would track this)
                pass

    async def process_audio(
        self,
        call_id: str,
        audio_data: bytes,
    ) -> Optional[bytes]:
        """
        Process received audio through handler.

        Args:
            call_id: Call ID
            audio_data: Received audio

        Returns:
            Response audio if any
        """
        call = self._calls.get(call_id)
        session = self._sessions.get(call_id)
        handler = self._call_handlers.get(call_id)

        if not call or not session or not handler:
            return None

        session.last_activity_at = datetime.utcnow()

        return await handler.on_audio_received(call, audio_data)

    def get_call(self, call_id: str) -> Optional[Call]:
        """Get a call by ID."""
        return self._calls.get(call_id)

    def get_session(self, call_id: str) -> Optional[CallSession]:
        """Get a session by call ID."""
        return self._sessions.get(call_id)

    def get_active_calls(self) -> List[Call]:
        """Get all active calls."""
        return [
            call for call in self._calls.values()
            if call.state == CallState.IN_PROGRESS
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get telephony statistics."""
        return dict(self._stats)

    def get_provider(self, name: str) -> Optional[Any]:
        """Get a provider by name."""
        return self._providers.get(name)

    # WebRTC-specific methods

    async def create_webrtc_connection(
        self,
        call_id: Optional[str] = None,
    ) -> WebRTCConnection:
        """
        Create a WebRTC connection.

        Args:
            call_id: Associated call ID

        Returns:
            WebRTC connection
        """
        webrtc = self._providers.get("webrtc")
        if not webrtc:
            raise ValueError("WebRTC provider not enabled")

        return await webrtc.create_connection(call_id)

    async def handle_webrtc_offer(
        self,
        connection_id: str,
        offer: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle WebRTC offer.

        Args:
            connection_id: Connection ID
            offer: SDP offer

        Returns:
            SDP answer
        """
        webrtc = self._providers.get("webrtc")
        if not webrtc:
            raise ValueError("WebRTC provider not enabled")

        from .webrtc import SDPOffer

        sdp_offer = SDPOffer.from_dict(offer)
        answer = await webrtc.handle_offer(connection_id, sdp_offer)

        return answer.to_dict() if answer else {}


def create_telephony_manager(
    twilio_account_sid: Optional[str] = None,
    twilio_auth_token: Optional[str] = None,
    default_provider: str = "twilio",
    **kwargs,
) -> TelephonyManager:
    """
    Create a telephony manager with common configuration.

    Args:
        twilio_account_sid: Twilio account SID
        twilio_auth_token: Twilio auth token
        default_provider: Default provider to use
        **kwargs: Additional configuration

    Returns:
        Configured telephony manager
    """
    twilio_config = None
    if twilio_account_sid and twilio_auth_token:
        twilio_config = TwilioConfig(
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
            default_from_number=kwargs.get("default_from_number"),
        )

    config = ManagerConfig(
        default_provider=default_provider,
        twilio_config=twilio_config,
        **{k: v for k, v in kwargs.items() if k not in ["default_from_number"]},
    )

    return TelephonyManager(config)


__all__ = [
    "TelephonyManager",
    "CallHandler",
    "ManagerConfig",
    "create_telephony_manager",
]
