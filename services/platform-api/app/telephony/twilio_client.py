"""
Twilio Telephony Client

Comprehensive Twilio integration for:
- Outbound call initiation
- Call control (hangup, hold, mute)
- Call transfers (blind, attended, warm)
- Conference management
- Recording control
- Real-time call updates
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
import asyncio
import uuid

import httpx
import structlog
from twilio.rest import Client as TwilioSyncClient
from twilio.base.exceptions import TwilioRestException
from twilio.twiml.voice_response import VoiceResponse, Dial, Conference, Stream

from app.config import get_settings

logger = structlog.get_logger()


class TransferType(str, Enum):
    """Types of call transfer."""
    BLIND = "blind"           # Cold transfer - immediate
    ATTENDED = "attended"     # Consultative transfer
    WARM = "warm"             # Warm handoff with introduction
    CONFERENCE = "conference" # Add party to call


class CallControlAction(str, Enum):
    """Call control actions."""
    HANGUP = "hangup"
    HOLD = "hold"
    UNHOLD = "unhold"
    MUTE = "mute"
    UNMUTE = "unmute"
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    PAUSE_RECORDING = "pause_recording"
    RESUME_RECORDING = "resume_recording"


class TwilioCallStatus(str, Enum):
    """Twilio call statuses."""
    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    BUSY = "busy"
    FAILED = "failed"
    NO_ANSWER = "no-answer"
    CANCELED = "canceled"


@dataclass
class OutboundCallParams:
    """Parameters for initiating an outbound call."""
    to_number: str
    from_number: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    # Webhook URLs
    status_callback_url: Optional[str] = None
    webhook_url: Optional[str] = None

    # Call settings
    timeout: int = 30  # Ring timeout in seconds
    record: bool = False
    recording_channels: str = "dual"  # "mono" or "dual"
    recording_status_callback: Optional[str] = None

    # Caller ID settings
    caller_id_name: Optional[str] = None

    # Machine detection
    machine_detection: Optional[str] = None  # "Enable" or "DetectMessageEnd"
    machine_detection_timeout: int = 30
    machine_detection_speech_threshold: int = 2400

    # Media settings
    media_stream_url: Optional[str] = None

    # Custom parameters
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferParams:
    """Parameters for call transfer."""
    call_sid: str
    target_number: str
    transfer_type: TransferType = TransferType.BLIND

    # Announcement settings
    announce: bool = True
    announce_message: Optional[str] = None

    # Hold music
    hold_music_url: Optional[str] = None

    # Timeout
    timeout: int = 30

    # Caller ID for the transfer leg
    caller_id: Optional[str] = None

    # Context to pass
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallResult:
    """Result of a call operation."""
    success: bool
    call_sid: Optional[str] = None
    status: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferResult:
    """Result of a transfer operation."""
    success: bool
    transfer_id: str = ""
    original_call_sid: Optional[str] = None
    new_call_sid: Optional[str] = None
    conference_sid: Optional[str] = None
    status: str = ""
    message: str = ""
    error: Optional[str] = None


class TwilioClient:
    """
    Async Twilio client for telephony operations.

    Provides:
    - Outbound call initiation
    - Call control (hangup, hold, mute)
    - Call transfers (blind, attended, warm)
    - Conference management
    - Recording control
    """

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        api_key_sid: Optional[str] = None,
        api_key_secret: Optional[str] = None,
    ) -> None:
        """Initialize Twilio client."""
        settings = get_settings()

        # Use provided credentials or fall back to settings
        self.account_sid = account_sid or settings.twilio_account_sid
        self.auth_token = auth_token or settings.twilio_auth_token
        self.api_key_sid = api_key_sid or settings.twilio_api_key_sid
        self.api_key_secret = api_key_secret or settings.twilio_api_key_secret

        # Initialize sync client for operations that don't have async support
        if self.api_key_sid and self.api_key_secret:
            self._sync_client = TwilioSyncClient(
                self.api_key_sid,
                self.api_key_secret,
                self.account_sid,
            )
        elif self.account_sid and self.auth_token:
            self._sync_client = TwilioSyncClient(
                self.account_sid,
                self.auth_token,
            )
        else:
            self._sync_client = None

        # Settings
        self.webhook_base_url = settings.twilio_webhook_base_url
        self.default_caller_id = settings.twilio_default_caller_id
        self.status_callback_url = settings.twilio_status_callback_url
        self.recording_enabled = settings.twilio_recording_enabled

        # HTTP client for async operations
        self._http_client: Optional[httpx.AsyncClient] = None

        # Active transfers tracking
        self._active_transfers: Dict[str, TransferParams] = {}

        self.logger = logger.bind(component="twilio_client")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._http_client is None:
            auth = (self.account_sid, self.auth_token)
            if self.api_key_sid and self.api_key_secret:
                auth = (self.api_key_sid, self.api_key_secret)

            self._http_client = httpx.AsyncClient(
                base_url=f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}",
                auth=auth,
                timeout=30.0,
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ===== Outbound Calls =====

    async def initiate_outbound_call(
        self,
        params: OutboundCallParams,
    ) -> CallResult:
        """
        Initiate an outbound call via Twilio.

        Args:
            params: Outbound call parameters

        Returns:
            CallResult with call SID and status
        """
        if not self._sync_client:
            return CallResult(
                success=False,
                error="Twilio client not configured",
                message="Missing Twilio credentials",
            )

        try:
            # Determine from number
            from_number = params.from_number or self.default_caller_id
            if not from_number:
                return CallResult(
                    success=False,
                    error="No caller ID",
                    message="No from_number provided and no default caller ID configured",
                )

            # Build webhook URL for handling the call
            webhook_url = params.webhook_url
            if not webhook_url and self.webhook_base_url:
                webhook_url = f"{self.webhook_base_url}/api/v1/webhooks/twilio/voice"

            # Build status callback URL
            status_callback = params.status_callback_url or self.status_callback_url
            if not status_callback and self.webhook_base_url:
                status_callback = f"{self.webhook_base_url}/api/v1/webhooks/twilio/status"

            # Build call parameters
            call_params: Dict[str, Any] = {
                "to": params.to_number,
                "from_": from_number,
                "timeout": params.timeout,
            }

            # Add webhook URL
            if webhook_url:
                call_params["url"] = webhook_url

            # Add status callback
            if status_callback:
                call_params["status_callback"] = status_callback
                call_params["status_callback_event"] = [
                    "initiated", "ringing", "answered", "completed"
                ]

            # Add recording if enabled
            if params.record or self.recording_enabled:
                call_params["record"] = True
                call_params["recording_channels"] = params.recording_channels
                if params.recording_status_callback:
                    call_params["recording_status_callback"] = params.recording_status_callback

            # Add machine detection if configured
            if params.machine_detection:
                call_params["machine_detection"] = params.machine_detection
                call_params["machine_detection_timeout"] = params.machine_detection_timeout
                call_params["machine_detection_speech_threshold"] = params.machine_detection_speech_threshold

            # Add caller ID name
            if params.caller_id_name:
                call_params["caller_id"] = params.caller_id_name

            # Run sync call creation in thread pool
            loop = asyncio.get_event_loop()
            call = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls.create(**call_params)
            )

            self.logger.info(
                "Outbound call initiated",
                call_sid=call.sid,
                to=params.to_number,
                from_number=from_number,
                status=call.status,
            )

            return CallResult(
                success=True,
                call_sid=call.sid,
                status=call.status,
                message="Call initiated successfully",
                data={
                    "to": params.to_number,
                    "from": from_number,
                    "direction": "outbound-api",
                    "date_created": call.date_created.isoformat() if call.date_created else None,
                },
            )

        except TwilioRestException as e:
            self.logger.error(
                "Twilio API error initiating call",
                error=str(e),
                code=e.code,
                to=params.to_number,
            )
            return CallResult(
                success=False,
                error=str(e),
                message=f"Twilio error: {e.msg}",
                data={"code": e.code},
            )
        except Exception as e:
            self.logger.error(
                "Error initiating outbound call",
                error=str(e),
                to=params.to_number,
            )
            return CallResult(
                success=False,
                error=str(e),
                message="Failed to initiate call",
            )

    async def initiate_call_with_twiml(
        self,
        to_number: str,
        twiml: str,
        from_number: Optional[str] = None,
        status_callback_url: Optional[str] = None,
    ) -> CallResult:
        """
        Initiate a call with custom TwiML.

        Args:
            to_number: Number to call
            twiml: TwiML instructions
            from_number: Caller ID
            status_callback_url: Status callback URL

        Returns:
            CallResult
        """
        if not self._sync_client:
            return CallResult(
                success=False,
                error="Twilio client not configured",
            )

        try:
            from_num = from_number or self.default_caller_id

            call_params = {
                "to": to_number,
                "from_": from_num,
                "twiml": twiml,
            }

            if status_callback_url:
                call_params["status_callback"] = status_callback_url
                call_params["status_callback_event"] = [
                    "initiated", "ringing", "answered", "completed"
                ]

            loop = asyncio.get_event_loop()
            call = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls.create(**call_params)
            )

            return CallResult(
                success=True,
                call_sid=call.sid,
                status=call.status,
                message="Call initiated with TwiML",
            )

        except TwilioRestException as e:
            return CallResult(
                success=False,
                error=str(e),
                message=f"Twilio error: {e.msg}",
            )

    # ===== Call Control =====

    async def hangup_call(self, call_sid: str) -> CallResult:
        """
        Hang up an active call.

        Args:
            call_sid: Twilio Call SID

        Returns:
            CallResult
        """
        if not self._sync_client:
            return CallResult(success=False, error="Twilio client not configured")

        try:
            loop = asyncio.get_event_loop()
            call = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(call_sid).update(status="completed")
            )

            self.logger.info("Call hung up", call_sid=call_sid)

            return CallResult(
                success=True,
                call_sid=call_sid,
                status="completed",
                message="Call ended",
            )

        except TwilioRestException as e:
            self.logger.error("Error hanging up call", call_sid=call_sid, error=str(e))
            return CallResult(
                success=False,
                call_sid=call_sid,
                error=str(e),
                message=f"Failed to hang up: {e.msg}",
            )

    async def hold_call(
        self,
        call_sid: str,
        hold_music_url: Optional[str] = None,
    ) -> CallResult:
        """
        Put a call on hold.

        Args:
            call_sid: Twilio Call SID
            hold_music_url: URL for hold music

        Returns:
            CallResult
        """
        if not self._sync_client:
            return CallResult(success=False, error="Twilio client not configured")

        try:
            # Create TwiML with hold music
            response = VoiceResponse()

            if hold_music_url:
                response.play(hold_music_url, loop=0)
            else:
                # Use default hold message
                response.say(
                    "Please hold while we connect you.",
                    voice="Polly.Joanna",
                )
                response.play(
                    "http://com.twilio.sounds.music.s3.amazonaws.com/ClockworkWaltz.mp3",
                    loop=0,
                )

            loop = asyncio.get_event_loop()
            call = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(call_sid).update(
                    twiml=str(response)
                )
            )

            self.logger.info("Call placed on hold", call_sid=call_sid)

            return CallResult(
                success=True,
                call_sid=call_sid,
                status="on_hold",
                message="Call placed on hold",
            )

        except TwilioRestException as e:
            self.logger.error("Error holding call", call_sid=call_sid, error=str(e))
            return CallResult(
                success=False,
                call_sid=call_sid,
                error=str(e),
                message=f"Failed to hold: {e.msg}",
            )

    async def resume_call(
        self,
        call_sid: str,
        redirect_url: str,
    ) -> CallResult:
        """
        Resume a call from hold.

        Args:
            call_sid: Twilio Call SID
            redirect_url: URL to redirect the call to

        Returns:
            CallResult
        """
        if not self._sync_client:
            return CallResult(success=False, error="Twilio client not configured")

        try:
            loop = asyncio.get_event_loop()
            call = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(call_sid).update(url=redirect_url)
            )

            self.logger.info("Call resumed", call_sid=call_sid)

            return CallResult(
                success=True,
                call_sid=call_sid,
                status="resumed",
                message="Call resumed",
            )

        except TwilioRestException as e:
            self.logger.error("Error resuming call", call_sid=call_sid, error=str(e))
            return CallResult(
                success=False,
                call_sid=call_sid,
                error=str(e),
                message=f"Failed to resume: {e.msg}",
            )

    async def get_call_status(self, call_sid: str) -> CallResult:
        """
        Get the current status of a call.

        Args:
            call_sid: Twilio Call SID

        Returns:
            CallResult with call details
        """
        if not self._sync_client:
            return CallResult(success=False, error="Twilio client not configured")

        try:
            loop = asyncio.get_event_loop()
            call = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(call_sid).fetch()
            )

            return CallResult(
                success=True,
                call_sid=call_sid,
                status=call.status,
                message="Call status retrieved",
                data={
                    "to": call.to,
                    "from": call.from_,
                    "direction": call.direction,
                    "duration": call.duration,
                    "start_time": call.start_time.isoformat() if call.start_time else None,
                    "end_time": call.end_time.isoformat() if call.end_time else None,
                    "answered_by": call.answered_by,
                },
            )

        except TwilioRestException as e:
            return CallResult(
                success=False,
                call_sid=call_sid,
                error=str(e),
                message=f"Failed to get status: {e.msg}",
            )

    # ===== Call Transfers =====

    async def transfer_call_blind(
        self,
        params: TransferParams,
    ) -> TransferResult:
        """
        Perform a blind (cold) transfer.

        Immediately transfers the call without consultation.

        Args:
            params: Transfer parameters

        Returns:
            TransferResult
        """
        if not self._sync_client:
            return TransferResult(
                success=False,
                error="Twilio client not configured",
            )

        transfer_id = str(uuid.uuid4())

        try:
            # Build transfer TwiML
            response = VoiceResponse()

            # Optional announcement
            if params.announce and params.announce_message:
                response.say(params.announce_message, voice="Polly.Joanna")
            elif params.announce:
                response.say(
                    "Please hold while we transfer your call.",
                    voice="Polly.Joanna",
                )

            # Dial the transfer target
            dial = Dial(
                caller_id=params.caller_id or self.default_caller_id,
                timeout=params.timeout,
            )
            dial.number(params.target_number)
            response.append(dial)

            # Update the call with transfer TwiML
            loop = asyncio.get_event_loop()
            call = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(params.call_sid).update(
                    twiml=str(response)
                )
            )

            self.logger.info(
                "Blind transfer initiated",
                transfer_id=transfer_id,
                call_sid=params.call_sid,
                target=params.target_number,
            )

            return TransferResult(
                success=True,
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
                status="initiated",
                message="Blind transfer initiated",
            )

        except TwilioRestException as e:
            self.logger.error(
                "Blind transfer failed",
                call_sid=params.call_sid,
                error=str(e),
            )
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
                status="failed",
                error=str(e),
                message=f"Transfer failed: {e.msg}",
            )

    async def transfer_call_attended(
        self,
        params: TransferParams,
    ) -> TransferResult:
        """
        Initiate an attended (consultative) transfer.

        This puts the caller on hold and dials the target.
        The agent can then complete or cancel the transfer.

        Args:
            params: Transfer parameters

        Returns:
            TransferResult
        """
        if not self._sync_client:
            return TransferResult(
                success=False,
                error="Twilio client not configured",
            )

        transfer_id = str(uuid.uuid4())

        try:
            # Create conference for the transfer
            conference_name = f"transfer-{transfer_id}"

            # Put original caller on hold in conference
            response = VoiceResponse()
            if params.announce and params.announce_message:
                response.say(params.announce_message, voice="Polly.Joanna")
            else:
                response.say(
                    "Please hold while we connect you.",
                    voice="Polly.Joanna",
                )

            # Add to conference with hold
            dial = Dial()
            conference = Conference(
                conference_name,
                start_conference_on_enter=True,
                end_conference_on_exit=False,
                wait_url=params.hold_music_url or "http://twimlets.com/holdmusic?Bucket=com.twilio.music.classical",
            )
            dial.append(conference)
            response.append(dial)

            # Update original call
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(params.call_sid).update(
                    twiml=str(response)
                )
            )

            # Store transfer state
            self._active_transfers[transfer_id] = params

            self.logger.info(
                "Attended transfer initiated - caller on hold",
                transfer_id=transfer_id,
                call_sid=params.call_sid,
                conference=conference_name,
            )

            return TransferResult(
                success=True,
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
                conference_sid=conference_name,
                status="holding",
                message="Caller on hold - dial consultation target",
            )

        except TwilioRestException as e:
            self.logger.error(
                "Attended transfer initiation failed",
                call_sid=params.call_sid,
                error=str(e),
            )
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
                status="failed",
                error=str(e),
            )

    async def dial_consultation(
        self,
        transfer_id: str,
    ) -> TransferResult:
        """
        Dial the consultation target for attended transfer.

        Args:
            transfer_id: Transfer ID from attended transfer initiation

        Returns:
            TransferResult with new call SID
        """
        params = self._active_transfers.get(transfer_id)
        if not params:
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                status="failed",
                error="Transfer not found",
            )

        if not self._sync_client:
            return TransferResult(
                success=False,
                error="Twilio client not configured",
            )

        try:
            conference_name = f"transfer-{transfer_id}"

            # Build TwiML to join conference
            response = VoiceResponse()
            dial = Dial()
            conference = Conference(
                conference_name,
                start_conference_on_enter=True,
                end_conference_on_exit=False,
                beep=False,
            )
            dial.append(conference)
            response.append(dial)

            # Call the consultation target
            loop = asyncio.get_event_loop()
            call = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls.create(
                    to=params.target_number,
                    from_=params.caller_id or self.default_caller_id,
                    twiml=str(response),
                    timeout=params.timeout,
                )
            )

            self.logger.info(
                "Consultation call initiated",
                transfer_id=transfer_id,
                consultation_call_sid=call.sid,
                target=params.target_number,
            )

            return TransferResult(
                success=True,
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
                new_call_sid=call.sid,
                conference_sid=conference_name,
                status="consulting",
                message="Consultation call in progress",
            )

        except TwilioRestException as e:
            self.logger.error(
                "Consultation call failed",
                transfer_id=transfer_id,
                error=str(e),
            )
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
                status="failed",
                error=str(e),
            )

    async def complete_attended_transfer(
        self,
        transfer_id: str,
        agent_call_sid: str,
    ) -> TransferResult:
        """
        Complete an attended transfer (disconnect agent, keep parties connected).

        Args:
            transfer_id: Transfer ID
            agent_call_sid: Agent's call SID to disconnect

        Returns:
            TransferResult
        """
        params = self._active_transfers.get(transfer_id)
        if not params:
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                status="failed",
                error="Transfer not found",
            )

        if not self._sync_client:
            return TransferResult(
                success=False,
                error="Twilio client not configured",
            )

        try:
            # Hang up the agent's call leg
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(agent_call_sid).update(
                    status="completed"
                )
            )

            # Remove from active transfers
            self._active_transfers.pop(transfer_id, None)

            self.logger.info(
                "Attended transfer completed",
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
            )

            return TransferResult(
                success=True,
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
                status="completed",
                message="Transfer completed - parties connected",
            )

        except TwilioRestException as e:
            self.logger.error(
                "Failed to complete transfer",
                transfer_id=transfer_id,
                error=str(e),
            )
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                status="failed",
                error=str(e),
            )

    async def cancel_attended_transfer(
        self,
        transfer_id: str,
        consultation_call_sid: str,
        resume_url: str,
    ) -> TransferResult:
        """
        Cancel an attended transfer and return to original caller.

        Args:
            transfer_id: Transfer ID
            consultation_call_sid: Consultation call to hang up
            resume_url: URL to redirect original call to

        Returns:
            TransferResult
        """
        params = self._active_transfers.get(transfer_id)
        if not params:
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                status="failed",
                error="Transfer not found",
            )

        if not self._sync_client:
            return TransferResult(
                success=False,
                error="Twilio client not configured",
            )

        try:
            loop = asyncio.get_event_loop()

            # Hang up consultation call
            await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(consultation_call_sid).update(
                    status="completed"
                )
            )

            # Redirect original call back to agent
            await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(params.call_sid).update(
                    url=resume_url
                )
            )

            # Remove from active transfers
            self._active_transfers.pop(transfer_id, None)

            self.logger.info(
                "Attended transfer cancelled",
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
            )

            return TransferResult(
                success=True,
                transfer_id=transfer_id,
                original_call_sid=params.call_sid,
                status="cancelled",
                message="Transfer cancelled - returning to caller",
            )

        except TwilioRestException as e:
            self.logger.error(
                "Failed to cancel transfer",
                transfer_id=transfer_id,
                error=str(e),
            )
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                status="failed",
                error=str(e),
            )

    async def transfer_to_conference(
        self,
        call_sid: str,
        conference_name: str,
        muted: bool = False,
        start_on_enter: bool = True,
        end_on_exit: bool = False,
        announce_url: Optional[str] = None,
    ) -> TransferResult:
        """
        Transfer a call to a conference.

        Args:
            call_sid: Call SID to transfer
            conference_name: Conference room name
            muted: Whether participant joins muted
            start_on_enter: Start conference when participant enters
            end_on_exit: End conference when participant exits
            announce_url: URL for announcement when joining

        Returns:
            TransferResult
        """
        if not self._sync_client:
            return TransferResult(
                success=False,
                error="Twilio client not configured",
            )

        transfer_id = str(uuid.uuid4())

        try:
            response = VoiceResponse()
            dial = Dial()

            conference_params = {
                "muted": muted,
                "start_conference_on_enter": start_on_enter,
                "end_conference_on_exit": end_on_exit,
            }

            conference = Conference(conference_name, **conference_params)
            dial.append(conference)
            response.append(dial)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(call_sid).update(
                    twiml=str(response)
                )
            )

            self.logger.info(
                "Call transferred to conference",
                call_sid=call_sid,
                conference=conference_name,
            )

            return TransferResult(
                success=True,
                transfer_id=transfer_id,
                original_call_sid=call_sid,
                conference_sid=conference_name,
                status="transferred",
                message=f"Transferred to conference: {conference_name}",
            )

        except TwilioRestException as e:
            return TransferResult(
                success=False,
                transfer_id=transfer_id,
                original_call_sid=call_sid,
                status="failed",
                error=str(e),
            )

    # ===== Recording =====

    async def start_recording(
        self,
        call_sid: str,
        recording_channels: str = "dual",
        recording_status_callback: Optional[str] = None,
    ) -> CallResult:
        """
        Start recording a call.

        Args:
            call_sid: Call SID
            recording_channels: "mono" or "dual"
            recording_status_callback: Callback URL

        Returns:
            CallResult with recording SID
        """
        if not self._sync_client:
            return CallResult(success=False, error="Twilio client not configured")

        try:
            params = {
                "recording_channels": recording_channels,
            }
            if recording_status_callback:
                params["recording_status_callback"] = recording_status_callback

            loop = asyncio.get_event_loop()
            recording = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(call_sid).recordings.create(**params)
            )

            self.logger.info(
                "Recording started",
                call_sid=call_sid,
                recording_sid=recording.sid,
            )

            return CallResult(
                success=True,
                call_sid=call_sid,
                status="recording",
                message="Recording started",
                data={"recording_sid": recording.sid},
            )

        except TwilioRestException as e:
            return CallResult(
                success=False,
                call_sid=call_sid,
                error=str(e),
                message=f"Failed to start recording: {e.msg}",
            )

    async def stop_recording(
        self,
        call_sid: str,
        recording_sid: str,
    ) -> CallResult:
        """
        Stop a recording.

        Args:
            call_sid: Call SID
            recording_sid: Recording SID

        Returns:
            CallResult
        """
        if not self._sync_client:
            return CallResult(success=False, error="Twilio client not configured")

        try:
            loop = asyncio.get_event_loop()
            recording = await loop.run_in_executor(
                None,
                lambda: self._sync_client.calls(call_sid).recordings(recording_sid).update(
                    status="stopped"
                )
            )

            self.logger.info(
                "Recording stopped",
                call_sid=call_sid,
                recording_sid=recording_sid,
            )

            return CallResult(
                success=True,
                call_sid=call_sid,
                status="recording_stopped",
                message="Recording stopped",
                data={"recording_sid": recording_sid},
            )

        except TwilioRestException as e:
            return CallResult(
                success=False,
                call_sid=call_sid,
                error=str(e),
                message=f"Failed to stop recording: {e.msg}",
            )

    async def get_recording_url(
        self,
        recording_sid: str,
    ) -> Optional[str]:
        """
        Get the URL for a recording.

        Args:
            recording_sid: Recording SID

        Returns:
            Recording URL or None
        """
        if not self._sync_client:
            return None

        try:
            loop = asyncio.get_event_loop()
            recording = await loop.run_in_executor(
                None,
                lambda: self._sync_client.recordings(recording_sid).fetch()
            )

            # Build the recording URL
            return f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Recordings/{recording_sid}.mp3"

        except TwilioRestException:
            return None

    # ===== TwiML Generation =====

    def generate_stream_twiml(
        self,
        stream_url: str,
        track: str = "both_tracks",
        status_callback: Optional[str] = None,
    ) -> str:
        """
        Generate TwiML for media streaming.

        Args:
            stream_url: WebSocket URL for media streaming
            track: "inbound_track", "outbound_track", or "both_tracks"
            status_callback: Status callback URL

        Returns:
            TwiML string
        """
        response = VoiceResponse()

        start = response.start()
        stream = Stream(url=stream_url, track=track)
        start.append(stream)

        # Keep the call alive
        response.pause(length=3600)

        return str(response)

    def generate_say_twiml(
        self,
        message: str,
        voice: str = "Polly.Joanna",
        language: str = "en-US",
    ) -> str:
        """
        Generate TwiML for text-to-speech.

        Args:
            message: Message to say
            voice: Voice to use
            language: Language code

        Returns:
            TwiML string
        """
        response = VoiceResponse()
        response.say(message, voice=voice, language=language)
        return str(response)

    def generate_gather_twiml(
        self,
        prompt: str,
        action_url: str,
        num_digits: int = 1,
        timeout: int = 5,
        voice: str = "Polly.Joanna",
    ) -> str:
        """
        Generate TwiML for DTMF gathering.

        Args:
            prompt: Prompt message
            action_url: URL to post gathered digits to
            num_digits: Number of digits to gather
            timeout: Timeout in seconds
            voice: Voice for prompt

        Returns:
            TwiML string
        """
        response = VoiceResponse()

        gather = response.gather(
            num_digits=num_digits,
            action=action_url,
            timeout=timeout,
        )
        gather.say(prompt, voice=voice)

        return str(response)


# Singleton instance
_twilio_client: Optional[TwilioClient] = None


def get_twilio_client() -> TwilioClient:
    """Get or create the Twilio client singleton."""
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = TwilioClient()
    return _twilio_client


async def shutdown_twilio_client() -> None:
    """Shutdown the Twilio client."""
    global _twilio_client
    if _twilio_client is not None:
        await _twilio_client.close()
        _twilio_client = None
