"""
Builder Engine Python SDK - Calls Resource

This module provides methods for managing voice calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import Call, CallStatus, CallDirection, Conversation, Message
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class CallsResource(BaseResource):
    """
    Resource for managing voice calls.

    Calls represent individual voice interactions between an agent
    and a phone number. This resource provides methods for initiating
    outbound calls, managing ongoing calls, and retrieving call history.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> # Make an outbound call
        >>> call = client.calls.create(
        ...     agent_id="agent_abc123",
        ...     to_number="+1234567890"
        ... )
        >>> # Check call status
        >>> call = client.calls.get(call.id)
        >>> print(call.status)
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        agent_id: Optional[str] = None,
        status: Optional[CallStatus] = None,
        direction: Optional[CallDirection] = None,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_duration: Optional[int] = None,
        max_duration: Optional[int] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> PaginatedResponse[Call]:
        """
        List all calls.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page (max 100)
            agent_id: Filter by agent ID
            status: Filter by call status
            direction: Filter by call direction
            from_number: Filter by originating number
            to_number: Filter by destination number
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            sort_by: Field to sort by
            sort_order: Sort order (asc, desc)

        Returns:
            PaginatedResponse containing Call objects

        Example:
            >>> calls = client.calls.list(
            ...     status=CallStatus.COMPLETED,
            ...     start_date="2024-01-01",
            ...     min_duration=60
            ... )
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            agent_id=agent_id,
            status=status.value if status else None,
            direction=direction.value if direction else None,
            from_number=from_number,
            to_number=to_number,
            start_date=start_date,
            end_date=end_date,
            min_duration=min_duration,
            max_duration=max_duration,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        response = self._get(Endpoints.CALLS, params=params)
        return self._parse_paginated_response(response, Call)

    def get(self, call_id: str) -> Call:
        """
        Get a call by ID.

        Args:
            call_id: The call's unique identifier

        Returns:
            Call object

        Raises:
            NotFoundError: If the call doesn't exist

        Example:
            >>> call = client.calls.get("call_abc123")
            >>> print(f"Duration: {call.duration_seconds}s")
        """
        path = Endpoints.CALL.format(call_id=call_id)
        response = self._get(path)
        return Call.from_dict(response)

    def create(
        self,
        agent_id: str,
        to_number: str,
        from_number: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        first_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        record: bool = True,
        max_duration: Optional[int] = None,
        webhook_url: Optional[str] = None,
        status_callback_url: Optional[str] = None,
        answering_machine_detection: bool = True,
        voicemail_message: Optional[str] = None,
    ) -> Call:
        """
        Create a new outbound call.

        Args:
            agent_id: The agent to use for this call
            to_number: Destination phone number (E.164 format)
            from_number: Originating phone number (E.164 format)
            phone_number_id: ID of phone number to use
            first_message: Custom first message (overrides agent config)
            context: Additional context for the conversation
            metadata: Custom metadata to attach to the call
            record: Whether to record the call
            max_duration: Maximum call duration in seconds
            webhook_url: Custom webhook URL for call events
            status_callback_url: URL for status callbacks
            answering_machine_detection: Enable AMD
            voicemail_message: Message to leave on voicemail

        Returns:
            Created Call object

        Example:
            >>> call = client.calls.create(
            ...     agent_id="agent_abc123",
            ...     to_number="+1234567890",
            ...     context={"customer_name": "John Doe"},
            ...     metadata={"campaign_id": "campaign_xyz"}
            ... )
        """
        data: Dict[str, Any] = {
            "agent_id": agent_id,
            "to_number": to_number,
        }

        if from_number is not None:
            data["from_number"] = from_number
        if phone_number_id is not None:
            data["phone_number_id"] = phone_number_id
        if first_message is not None:
            data["first_message"] = first_message
        if context is not None:
            data["context"] = context
        if metadata is not None:
            data["metadata"] = metadata
        if record is not None:
            data["record"] = record
        if max_duration is not None:
            data["max_duration"] = max_duration
        if webhook_url is not None:
            data["webhook_url"] = webhook_url
        if status_callback_url is not None:
            data["status_callback_url"] = status_callback_url
        if answering_machine_detection is not None:
            data["answering_machine_detection"] = answering_machine_detection
        if voicemail_message is not None:
            data["voicemail_message"] = voicemail_message

        response = self._post(Endpoints.CALLS, json=data)
        return Call.from_dict(response)

    def hangup(self, call_id: str, reason: Optional[str] = None) -> Call:
        """
        Hang up an active call.

        Args:
            call_id: The call's unique identifier
            reason: Optional reason for hanging up

        Returns:
            Updated Call object

        Example:
            >>> call = client.calls.hangup("call_abc123", reason="User requested")
        """
        path = Endpoints.CALL_HANGUP.format(call_id=call_id)
        data = {}
        if reason:
            data["reason"] = reason
        response = self._post(path, json=data)
        return Call.from_dict(response)

    def transfer(
        self,
        call_id: str,
        to_number: str,
        announce: Optional[str] = None,
        warm_transfer: bool = False,
    ) -> Call:
        """
        Transfer an active call.

        Args:
            call_id: The call's unique identifier
            to_number: Number to transfer to
            announce: Message to announce before transfer
            warm_transfer: Whether to do a warm transfer

        Returns:
            Updated Call object

        Example:
            >>> call = client.calls.transfer(
            ...     call_id="call_abc123",
            ...     to_number="+1987654321",
            ...     announce="Transferring you to a human agent",
            ...     warm_transfer=True
            ... )
        """
        path = Endpoints.CALL_TRANSFER.format(call_id=call_id)
        data = {
            "to_number": to_number,
            "warm_transfer": warm_transfer,
        }
        if announce:
            data["announce"] = announce
        response = self._post(path, json=data)
        return Call.from_dict(response)

    def mute(self, call_id: str, muted: bool = True) -> Call:
        """
        Mute or unmute a call.

        Args:
            call_id: The call's unique identifier
            muted: Whether to mute (True) or unmute (False)

        Returns:
            Updated Call object

        Example:
            >>> call = client.calls.mute("call_abc123", muted=True)
        """
        path = Endpoints.CALL_MUTE.format(call_id=call_id)
        response = self._post(path, json={"muted": muted})
        return Call.from_dict(response)

    def hold(self, call_id: str, on_hold: bool = True, music_url: Optional[str] = None) -> Call:
        """
        Put a call on hold or resume.

        Args:
            call_id: The call's unique identifier
            on_hold: Whether to put on hold (True) or resume (False)
            music_url: Custom hold music URL

        Returns:
            Updated Call object

        Example:
            >>> call = client.calls.hold("call_abc123", on_hold=True)
        """
        path = Endpoints.CALL_HOLD.format(call_id=call_id)
        data = {"on_hold": on_hold}
        if music_url:
            data["music_url"] = music_url
        response = self._post(path, json=data)
        return Call.from_dict(response)

    def send_dtmf(self, call_id: str, digits: str) -> Call:
        """
        Send DTMF tones to a call.

        Args:
            call_id: The call's unique identifier
            digits: DTMF digits to send (0-9, *, #, w for pause)

        Returns:
            Updated Call object

        Example:
            >>> call = client.calls.send_dtmf("call_abc123", "1234#")
        """
        path = Endpoints.CALL_SEND_DTMF.format(call_id=call_id)
        response = self._post(path, json={"digits": digits})
        return Call.from_dict(response)

    def inject_message(
        self,
        call_id: str,
        message: str,
        role: str = "assistant",
        interrupt: bool = False,
    ) -> Call:
        """
        Inject a message into an active call.

        Args:
            call_id: The call's unique identifier
            message: Message to inject
            role: Role of the message (assistant, system)
            interrupt: Whether to interrupt current speech

        Returns:
            Updated Call object

        Example:
            >>> call = client.calls.inject_message(
            ...     call_id="call_abc123",
            ...     message="One moment please, let me check on that.",
            ...     interrupt=True
            ... )
        """
        path = Endpoints.CALL_INJECT_MESSAGE.format(call_id=call_id)
        response = self._post(path, json={
            "message": message,
            "role": role,
            "interrupt": interrupt,
        })
        return Call.from_dict(response)

    def get_recording(self, call_id: str) -> Dict[str, Any]:
        """
        Get the recording for a completed call.

        Args:
            call_id: The call's unique identifier

        Returns:
            Dictionary with recording URL and metadata

        Example:
            >>> recording = client.calls.get_recording("call_abc123")
            >>> print(recording["url"])
        """
        path = Endpoints.CALL_RECORDING.format(call_id=call_id)
        return self._get(path)

    def get_transcript(
        self,
        call_id: str,
        format: str = "json",
        include_timestamps: bool = True,
        include_speaker_labels: bool = True,
    ) -> Dict[str, Any]:
        """
        Get the transcript for a completed call.

        Args:
            call_id: The call's unique identifier
            format: Output format (json, text, srt, vtt)
            include_timestamps: Include word-level timestamps
            include_speaker_labels: Include speaker labels

        Returns:
            Transcript data

        Example:
            >>> transcript = client.calls.get_transcript(
            ...     call_id="call_abc123",
            ...     format="json"
            ... )
            >>> for message in transcript["messages"]:
            ...     print(f"{message['role']}: {message['content']}")
        """
        path = Endpoints.CALL_TRANSCRIPT.format(call_id=call_id)
        params = {
            "format": format,
            "include_timestamps": include_timestamps,
            "include_speaker_labels": include_speaker_labels,
        }
        return self._get(path, params=params)

    def get_conversation(self, call_id: str) -> Conversation:
        """
        Get the conversation for a call.

        Args:
            call_id: The call's unique identifier

        Returns:
            Conversation object with all messages

        Example:
            >>> conversation = client.calls.get_conversation("call_abc123")
            >>> for msg in conversation.messages:
            ...     print(f"{msg.role}: {msg.content}")
        """
        call = self.get(call_id)
        path = Endpoints.CONVERSATION.format(conversation_id=call_id)
        response = self._get(path)
        return Conversation.from_dict(response)

    def bulk_create(
        self,
        agent_id: str,
        calls: List[Dict[str, Any]],
        delay_between_calls_ms: int = 1000,
    ) -> List[Call]:
        """
        Create multiple outbound calls.

        Args:
            agent_id: The agent to use for calls
            calls: List of call configurations with to_number, metadata, etc.
            delay_between_calls_ms: Delay between initiating calls

        Returns:
            List of created Call objects

        Example:
            >>> calls = client.calls.bulk_create(
            ...     agent_id="agent_abc123",
            ...     calls=[
            ...         {"to_number": "+1234567890", "context": {"name": "John"}},
            ...         {"to_number": "+0987654321", "context": {"name": "Jane"}},
            ...     ]
            ... )
        """
        response = self._post(
            f"{Endpoints.CALLS}/bulk",
            json={
                "agent_id": agent_id,
                "calls": calls,
                "delay_between_calls_ms": delay_between_calls_ms,
            },
        )
        return [Call.from_dict(c) for c in response.get("calls", [])]

    def get_active(self, agent_id: Optional[str] = None) -> List[Call]:
        """
        Get all currently active calls.

        Args:
            agent_id: Filter by agent ID

        Returns:
            List of active Call objects

        Example:
            >>> active_calls = client.calls.get_active()
            >>> print(f"Currently {len(active_calls)} active calls")
        """
        params = {"status": "in_progress"}
        if agent_id:
            params["agent_id"] = agent_id
        response = self._get(Endpoints.CALLS, params=params)
        return [Call.from_dict(c) for c in response.get("items", [])]

    def wait_for_completion(
        self,
        call_id: str,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 2,
    ) -> Call:
        """
        Wait for a call to complete.

        Args:
            call_id: The call's unique identifier
            timeout_seconds: Maximum time to wait
            poll_interval_seconds: Polling interval

        Returns:
            Completed Call object

        Raises:
            TimeoutError: If call doesn't complete within timeout

        Example:
            >>> call = client.calls.create(...)
            >>> completed_call = client.calls.wait_for_completion(call.id)
            >>> print(f"Call completed with status: {completed_call.status}")
        """
        import time
        from builderengine.exceptions import TimeoutError

        start_time = time.time()
        terminal_statuses = {
            CallStatus.COMPLETED,
            CallStatus.FAILED,
            CallStatus.BUSY,
            CallStatus.NO_ANSWER,
            CallStatus.CANCELED,
        }

        while True:
            call = self.get(call_id)
            if call.status in terminal_statuses:
                return call

            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"Call {call_id} did not complete within {timeout_seconds}s",
                    timeout_seconds=timeout_seconds,
                )

            time.sleep(poll_interval_seconds)
