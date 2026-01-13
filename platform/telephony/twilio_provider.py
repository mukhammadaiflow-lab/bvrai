"""
Twilio Provider Module

This module provides Twilio integration for making and receiving
phone calls with media streaming support.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from .base import (
    Call,
    CallDirection,
    CallEvent,
    CallEventType,
    CallSession,
    CallState,
    ProviderConfig,
    Recording,
    RecordingFormat,
    RecordingState,
    SessionConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class TwilioConfig:
    """Twilio-specific configuration."""

    # Credentials
    account_sid: str = ""
    auth_token: str = ""

    # API Key (alternative auth)
    api_key_sid: Optional[str] = None
    api_key_secret: Optional[str] = None

    # Numbers
    default_from_number: Optional[str] = None
    phone_numbers: List[str] = field(default_factory=list)

    # Webhook URLs
    voice_webhook_url: Optional[str] = None
    status_callback_url: Optional[str] = None
    stream_url: Optional[str] = None

    # Recording settings
    record_calls: bool = True
    recording_channels: str = "dual"  # mono, dual
    recording_trim: str = "trim-silence"

    # Stream settings
    stream_track: str = "both"  # inbound, outbound, both
    stream_encoding: str = "mulaw"

    # AMD settings
    enable_amd: bool = False
    amd_timeout: int = 30
    amd_speech_threshold: int = 2400
    amd_speech_end_threshold: int = 1200

    # Timeouts
    timeout: int = 30
    machine_detection_timeout: int = 30

    @classmethod
    def from_env(cls) -> "TwilioConfig":
        """Create config from environment variables."""
        return cls(
            account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
            api_key_sid=os.environ.get("TWILIO_API_KEY_SID"),
            api_key_secret=os.environ.get("TWILIO_API_KEY_SECRET"),
            default_from_number=os.environ.get("TWILIO_PHONE_NUMBER"),
        )


class TwilioProvider:
    """
    Twilio telephony provider.

    Handles call creation, management, and media streaming
    using the Twilio API.
    """

    def __init__(self, config: Optional[TwilioConfig] = None):
        """
        Initialize Twilio provider.

        Args:
            config: Twilio configuration
        """
        self.config = config or TwilioConfig.from_env()
        self._client = None
        self._active_calls: Dict[str, Call] = {}
        self._active_sessions: Dict[str, CallSession] = {}

    async def _get_client(self):
        """Get or create Twilio client."""
        if self._client is None:
            try:
                from twilio.rest import Client
                self._client = Client(
                    self.config.account_sid,
                    self.config.auth_token,
                )
            except ImportError:
                raise ImportError("twilio package required. Install with: pip install twilio")
        return self._client

    async def make_call(
        self,
        to_number: str,
        from_number: Optional[str] = None,
        webhook_url: Optional[str] = None,
        status_callback: Optional[str] = None,
        record: bool = True,
        amd: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Call:
        """
        Make an outbound call.

        Args:
            to_number: Number to call
            from_number: Caller ID (default: config number)
            webhook_url: URL for TwiML instructions
            status_callback: URL for status updates
            record: Whether to record the call
            amd: Enable answering machine detection
            custom_params: Additional Twilio API params

        Returns:
            Call object
        """
        client = await self._get_client()
        from_number = from_number or self.config.default_from_number

        if not from_number:
            raise ValueError("No from_number provided and no default configured")

        # Create call object
        call = Call(
            direction=CallDirection.OUTBOUND,
            state=CallState.INITIATED,
            from_number=from_number,
            to_number=to_number,
            provider="twilio",
            provider_account_id=self.config.account_sid,
        )

        # Build API params
        params = {
            "to": to_number,
            "from_": from_number,
        }

        if webhook_url:
            params["url"] = webhook_url
        elif self.config.voice_webhook_url:
            params["url"] = self.config.voice_webhook_url

        if status_callback:
            params["status_callback"] = status_callback
        elif self.config.status_callback_url:
            params["status_callback"] = self.config.status_callback_url
            params["status_callback_event"] = ["initiated", "ringing", "answered", "completed"]

        if record or self.config.record_calls:
            params["record"] = True
            params["recording_channels"] = self.config.recording_channels
            params["recording_status_callback"] = status_callback or self.config.status_callback_url

        if amd or self.config.enable_amd:
            params["machine_detection"] = "Enable"
            params["machine_detection_timeout"] = self.config.amd_timeout
            params["machine_detection_speech_threshold"] = self.config.amd_speech_threshold
            params["machine_detection_speech_end_threshold"] = self.config.amd_speech_end_threshold

        if custom_params:
            params.update(custom_params)

        try:
            # Make API call
            twilio_call = await asyncio.to_thread(
                client.calls.create,
                **params,
            )

            call.provider_call_id = twilio_call.sid
            call.state = CallState.INITIATED
            call.add_event(CallEventType.INITIATED)

            self._active_calls[call.id] = call
            self._active_calls[twilio_call.sid] = call  # Index by Twilio SID too

            logger.info(f"Outbound call initiated: {call.id} -> {to_number}")

        except Exception as e:
            call.state = CallState.FAILED
            call.add_event(CallEventType.FAILED, {"error": str(e)})
            logger.error(f"Failed to initiate call: {e}")
            raise

        return call

    async def end_call(
        self,
        call_id: str,
        status: str = "completed",
    ) -> Call:
        """
        End an active call.

        Args:
            call_id: Call ID or Twilio SID
            status: End status (completed, canceled)

        Returns:
            Updated call object
        """
        call = self._active_calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        client = await self._get_client()

        try:
            await asyncio.to_thread(
                client.calls(call.provider_call_id).update,
                status=status,
            )

            call.state = CallState.COMPLETED
            call.ended_at = datetime.utcnow()
            call.add_event(CallEventType.COMPLETED)

            logger.info(f"Call ended: {call.id}")

        except Exception as e:
            logger.error(f"Failed to end call: {e}")
            raise

        return call

    async def transfer_call(
        self,
        call_id: str,
        to_number: str,
        announce: Optional[str] = None,
    ) -> Call:
        """
        Transfer a call to another number.

        Args:
            call_id: Call ID
            to_number: Transfer destination
            announce: Optional announcement TwiML

        Returns:
            Updated call object
        """
        call = self._active_calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        client = await self._get_client()

        # Build TwiML for transfer
        twiml = f'<Response><Dial>{to_number}</Dial></Response>'
        if announce:
            twiml = f'<Response><Say>{announce}</Say><Dial>{to_number}</Dial></Response>'

        try:
            call.state = CallState.TRANSFERRING
            call.add_event(CallEventType.TRANSFER_STARTED, {"to": to_number})

            await asyncio.to_thread(
                client.calls(call.provider_call_id).update,
                twiml=twiml,
            )

            logger.info(f"Call transfer initiated: {call.id} -> {to_number}")

        except Exception as e:
            call.add_event(CallEventType.TRANSFER_FAILED, {"error": str(e)})
            logger.error(f"Failed to transfer call: {e}")
            raise

        return call

    async def hold_call(
        self,
        call_id: str,
        hold_music_url: Optional[str] = None,
    ) -> Call:
        """
        Put a call on hold.

        Args:
            call_id: Call ID
            hold_music_url: Optional hold music URL

        Returns:
            Updated call object
        """
        call = self._active_calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        client = await self._get_client()

        # Build TwiML for hold
        if hold_music_url:
            twiml = f'<Response><Play loop="0">{hold_music_url}</Play></Response>'
        else:
            twiml = '<Response><Play loop="0">http://com.twilio.sounds.music.s3.amazonaws.com/ClockworkWaltz.mp3</Play></Response>'

        try:
            await asyncio.to_thread(
                client.calls(call.provider_call_id).update,
                twiml=twiml,
            )

            call.is_on_hold = True
            call.state = CallState.ON_HOLD
            call.add_event(CallEventType.HOLD_STARTED)

            logger.info(f"Call placed on hold: {call.id}")

        except Exception as e:
            logger.error(f"Failed to hold call: {e}")
            raise

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
            Updated call object
        """
        call = self._active_calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        client = await self._get_client()

        try:
            await asyncio.to_thread(
                client.calls(call.provider_call_id).update,
                url=webhook_url,
            )

            call.is_on_hold = False
            call.state = CallState.IN_PROGRESS
            call.add_event(CallEventType.HOLD_ENDED)

            logger.info(f"Call resumed: {call.id}")

        except Exception as e:
            logger.error(f"Failed to resume call: {e}")
            raise

        return call

    async def send_dtmf(
        self,
        call_id: str,
        digits: str,
    ) -> None:
        """
        Send DTMF digits on a call.

        Args:
            call_id: Call ID
            digits: DTMF digits to send
        """
        call = self._active_calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        client = await self._get_client()

        twiml = f'<Response><Play digits="{digits}"/></Response>'

        try:
            await asyncio.to_thread(
                client.calls(call.provider_call_id).update,
                twiml=twiml,
            )

            logger.info(f"DTMF sent on call {call.id}: {digits}")

        except Exception as e:
            logger.error(f"Failed to send DTMF: {e}")
            raise

    async def start_recording(
        self,
        call_id: str,
        format: RecordingFormat = RecordingFormat.WAV,
    ) -> Recording:
        """
        Start recording a call.

        Args:
            call_id: Call ID
            format: Recording format

        Returns:
            Recording object
        """
        call = self._active_calls.get(call_id)
        if not call:
            raise ValueError(f"Call not found: {call_id}")

        client = await self._get_client()

        recording = Recording(
            call_id=call.id,
            state=RecordingState.PENDING,
            format=format,
        )

        try:
            twilio_recording = await asyncio.to_thread(
                client.calls(call.provider_call_id).recordings.create,
                recording_channels=self.config.recording_channels,
            )

            recording.id = twilio_recording.sid
            recording.state = RecordingState.RECORDING
            recording.started_at = datetime.utcnow()

            call.is_recording = True
            call.recording = recording
            call.add_event(CallEventType.RECORDING_STARTED)

            logger.info(f"Recording started for call {call.id}")

        except Exception as e:
            recording.state = RecordingState.FAILED
            logger.error(f"Failed to start recording: {e}")
            raise

        return recording

    async def stop_recording(self, call_id: str) -> Optional[Recording]:
        """
        Stop recording a call.

        Args:
            call_id: Call ID

        Returns:
            Recording object or None
        """
        call = self._active_calls.get(call_id)
        if not call or not call.recording:
            return None

        client = await self._get_client()

        try:
            await asyncio.to_thread(
                client.calls(call.provider_call_id)
                .recordings(call.recording.id)
                .update,
                status="stopped",
            )

            call.recording.state = RecordingState.COMPLETED
            call.recording.ended_at = datetime.utcnow()
            call.is_recording = False
            call.add_event(CallEventType.RECORDING_STOPPED)

            logger.info(f"Recording stopped for call {call.id}")

            return call.recording

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            raise

    async def get_call(self, call_id: str) -> Optional[Call]:
        """Get a call by ID."""
        return self._active_calls.get(call_id)

    async def get_call_by_sid(self, sid: str) -> Optional[Call]:
        """Get a call by Twilio SID."""
        return self._active_calls.get(sid)

    def register_call(self, call: Call) -> None:
        """Register a call from webhook."""
        self._active_calls[call.id] = call
        if call.provider_call_id:
            self._active_calls[call.provider_call_id] = call


class TwilioWebhookHandler:
    """
    Handles incoming Twilio webhooks.

    Validates signatures, parses events, and routes to appropriate handlers.
    """

    def __init__(
        self,
        config: TwilioConfig,
        provider: TwilioProvider,
    ):
        """
        Initialize webhook handler.

        Args:
            config: Twilio configuration
            provider: Twilio provider instance
        """
        self.config = config
        self.provider = provider
        self._handlers: Dict[str, Callable] = {}

    def validate_signature(
        self,
        url: str,
        params: Dict[str, str],
        signature: str,
    ) -> bool:
        """
        Validate Twilio webhook signature.

        Args:
            url: Full webhook URL
            params: Request parameters
            signature: X-Twilio-Signature header

        Returns:
            True if valid
        """
        # Build validation string
        validation_string = url
        for key in sorted(params.keys()):
            validation_string += key + params[key]

        # Calculate expected signature
        expected_signature = base64.b64encode(
            hmac.new(
                self.config.auth_token.encode('utf-8'),
                validation_string.encode('utf-8'),
                hashlib.sha1,
            ).digest()
        ).decode('utf-8')

        return hmac.compare_digest(signature, expected_signature)

    async def handle_voice_webhook(
        self,
        params: Dict[str, str],
    ) -> str:
        """
        Handle incoming voice webhook.

        Args:
            params: Webhook parameters

        Returns:
            TwiML response
        """
        call_sid = params.get("CallSid")
        call_status = params.get("CallStatus")
        direction = params.get("Direction", "inbound")

        # Get or create call
        call = await self.provider.get_call_by_sid(call_sid)

        if not call:
            # New inbound call
            call = Call(
                direction=CallDirection.INBOUND if direction == "inbound" else CallDirection.OUTBOUND,
                state=CallState.RINGING,
                from_number=params.get("From", ""),
                to_number=params.get("To", ""),
                provider="twilio",
                provider_call_id=call_sid,
                provider_account_id=params.get("AccountSid"),
            )
            call.forwarded_from = params.get("ForwardedFrom")
            self.provider.register_call(call)

        # Update state
        if call_status == "ringing":
            call.state = CallState.RINGING
            call.add_event(CallEventType.RINGING)
        elif call_status == "in-progress":
            call.state = CallState.IN_PROGRESS
            call.answered_at = datetime.utcnow()
            call.add_event(CallEventType.ANSWERED)
        elif call_status == "completed":
            call.state = CallState.COMPLETED
            call.ended_at = datetime.utcnow()
            call.hangup_cause = params.get("SipResponseCode")
            call.add_event(CallEventType.COMPLETED)
        elif call_status == "failed":
            call.state = CallState.FAILED
            call.add_event(CallEventType.FAILED, {"reason": params.get("ErrorMessage")})

        # Handle AMD result
        if "AnsweredBy" in params:
            call.answer_state = params["AnsweredBy"]

        # Route to handler
        handler = self._handlers.get("voice", self._default_voice_handler)
        return await handler(call, params)

    async def handle_status_webhook(
        self,
        params: Dict[str, str],
    ) -> str:
        """
        Handle status callback webhook.

        Args:
            params: Webhook parameters

        Returns:
            Response (usually empty)
        """
        call_sid = params.get("CallSid")
        call = await self.provider.get_call_by_sid(call_sid)

        if not call:
            logger.warning(f"Status webhook for unknown call: {call_sid}")
            return ""

        # Update call status
        call_status = params.get("CallStatus")
        if call_status == "completed":
            call.state = CallState.COMPLETED
            call.ended_at = datetime.utcnow()
            call.talk_duration_seconds = float(params.get("CallDuration", 0))
            call.add_event(CallEventType.COMPLETED)
        elif call_status == "busy":
            call.state = CallState.BUSY
            call.add_event(CallEventType.FAILED, {"reason": "busy"})
        elif call_status == "no-answer":
            call.state = CallState.NO_ANSWER
            call.add_event(CallEventType.FAILED, {"reason": "no_answer"})
        elif call_status == "failed":
            call.state = CallState.FAILED
            call.add_event(CallEventType.FAILED, {"reason": params.get("ErrorMessage")})

        # Route to handler
        handler = self._handlers.get("status", lambda c, p: "")
        return await handler(call, params)

    async def handle_recording_webhook(
        self,
        params: Dict[str, str],
    ) -> str:
        """
        Handle recording status webhook.

        Args:
            params: Webhook parameters

        Returns:
            Response
        """
        call_sid = params.get("CallSid")
        call = await self.provider.get_call_by_sid(call_sid)

        if call and call.recording:
            recording_status = params.get("RecordingStatus")
            if recording_status == "completed":
                call.recording.state = RecordingState.COMPLETED
                call.recording.storage_url = params.get("RecordingUrl")
                call.recording.duration_seconds = float(params.get("RecordingDuration", 0))

        return ""

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a webhook handler."""
        self._handlers[event_type] = handler

    async def _default_voice_handler(
        self,
        call: Call,
        params: Dict[str, str],
    ) -> str:
        """Default voice webhook handler."""
        # Return TwiML that starts a media stream
        return create_twilio_response(
            stream_url=self.config.stream_url,
            record=self.config.record_calls,
        )


class TwilioStreamHandler:
    """
    Handles Twilio Media Streams over WebSocket.

    Receives and processes bidirectional audio streams.
    """

    def __init__(self, config: TwilioConfig):
        """
        Initialize stream handler.

        Args:
            config: Twilio configuration
        """
        self.config = config
        self._streams: Dict[str, "TwilioStream"] = {}

    async def handle_websocket(
        self,
        websocket: Any,
        on_audio: Callable[[str, bytes], None],
    ) -> None:
        """
        Handle WebSocket connection for media stream.

        Args:
            websocket: WebSocket connection
            on_audio: Callback for received audio
        """
        stream_sid = None

        try:
            async for message in websocket:
                data = json.loads(message)
                event_type = data.get("event")

                if event_type == "connected":
                    logger.info("Twilio stream connected")

                elif event_type == "start":
                    stream_sid = data.get("streamSid")
                    call_sid = data["start"].get("callSid")

                    stream = TwilioStream(
                        stream_sid=stream_sid,
                        call_sid=call_sid,
                        websocket=websocket,
                    )
                    self._streams[stream_sid] = stream

                    logger.info(f"Twilio stream started: {stream_sid}")

                elif event_type == "media":
                    if stream_sid:
                        payload = data["media"].get("payload")
                        if payload:
                            audio_data = base64.b64decode(payload)
                            await on_audio(stream_sid, audio_data)

                elif event_type == "stop":
                    logger.info(f"Twilio stream stopped: {stream_sid}")
                    if stream_sid in self._streams:
                        del self._streams[stream_sid]
                    break

        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            if stream_sid and stream_sid in self._streams:
                del self._streams[stream_sid]

    async def send_audio(
        self,
        stream_sid: str,
        audio_data: bytes,
    ) -> None:
        """
        Send audio to a stream.

        Args:
            stream_sid: Stream ID
            audio_data: Audio bytes (mulaw)
        """
        stream = self._streams.get(stream_sid)
        if not stream:
            return

        payload = base64.b64encode(audio_data).decode('utf-8')

        message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": payload,
            },
        }

        await stream.websocket.send(json.dumps(message))

    async def send_mark(
        self,
        stream_sid: str,
        name: str,
    ) -> None:
        """
        Send a mark event for synchronization.

        Args:
            stream_sid: Stream ID
            name: Mark name
        """
        stream = self._streams.get(stream_sid)
        if not stream:
            return

        message = {
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {
                "name": name,
            },
        }

        await stream.websocket.send(json.dumps(message))

    async def clear_audio(self, stream_sid: str) -> None:
        """
        Clear pending audio in the buffer.

        Args:
            stream_sid: Stream ID
        """
        stream = self._streams.get(stream_sid)
        if not stream:
            return

        message = {
            "event": "clear",
            "streamSid": stream_sid,
        }

        await stream.websocket.send(json.dumps(message))


@dataclass
class TwilioStream:
    """Active Twilio media stream."""

    stream_sid: str
    call_sid: str
    websocket: Any
    started_at: datetime = field(default_factory=datetime.utcnow)


def create_twilio_response(
    message: Optional[str] = None,
    stream_url: Optional[str] = None,
    record: bool = False,
    gather_dtmf: bool = False,
    dtmf_timeout: int = 5,
    dial_number: Optional[str] = None,
    pause_seconds: int = 0,
) -> str:
    """
    Create TwiML response.

    Args:
        message: Text to speak
        stream_url: WebSocket URL for media stream
        record: Whether to record
        gather_dtmf: Whether to gather DTMF
        dtmf_timeout: DTMF gather timeout
        dial_number: Number to dial/transfer
        pause_seconds: Pause duration

    Returns:
        TwiML string
    """
    parts = ['<Response>']

    if pause_seconds > 0:
        parts.append(f'<Pause length="{pause_seconds}"/>')

    if message:
        parts.append(f'<Say>{message}</Say>')

    if gather_dtmf:
        parts.append(f'<Gather timeout="{dtmf_timeout}" input="dtmf speech"/>')

    if stream_url:
        parts.append(f'<Connect><Stream url="{stream_url}"/></Connect>')

    if dial_number:
        parts.append(f'<Dial>{dial_number}</Dial>')

    if record:
        parts.append('<Record/>')

    parts.append('</Response>')

    return ''.join(parts)


__all__ = [
    "TwilioProvider",
    "TwilioConfig",
    "TwilioWebhookHandler",
    "TwilioStreamHandler",
    "TwilioStream",
    "create_twilio_response",
]
