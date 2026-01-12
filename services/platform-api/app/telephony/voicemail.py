"""Voicemail detection, recording, and management."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import struct

logger = logging.getLogger(__name__)


class VoicemailStatus(str, Enum):
    """Voicemail message status."""
    NEW = "new"
    HEARD = "heard"
    SAVED = "saved"
    DELETED = "deleted"
    TRANSCRIBING = "transcribing"


class DetectionResult(str, Enum):
    """Voicemail detection result."""
    HUMAN = "human"
    MACHINE = "machine"
    UNKNOWN = "unknown"
    BEEP_DETECTED = "beep_detected"


@dataclass
class VoicemailMessage:
    """A voicemail message."""
    message_id: str
    mailbox_id: str
    call_id: str
    from_number: str
    to_number: str
    duration_seconds: float
    status: VoicemailStatus = VoicemailStatus.NEW
    created_at: datetime = field(default_factory=datetime.utcnow)
    heard_at: Optional[datetime] = None
    audio_url: Optional[str] = None
    audio_path: Optional[str] = None
    transcription: Optional[str] = None
    transcription_confidence: float = 0.0
    sentiment: Optional[str] = None
    is_urgent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "mailbox_id": self.mailbox_id,
            "call_id": self.call_id,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "heard_at": self.heard_at.isoformat() if self.heard_at else None,
            "audio_url": self.audio_url,
            "transcription": self.transcription,
            "transcription_confidence": self.transcription_confidence,
            "sentiment": self.sentiment,
            "is_urgent": self.is_urgent,
            "metadata": self.metadata,
        }


@dataclass
class VoicemailBox:
    """A voicemail box for a user or agent."""
    mailbox_id: str
    owner_id: str
    owner_type: str  # "user", "agent", "department"
    name: str
    greeting_audio_url: Optional[str] = None
    greeting_text: Optional[str] = None
    max_duration_seconds: int = 180
    max_messages: int = 100
    pin: Optional[str] = None
    email_notification: Optional[str] = None
    sms_notification: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mailbox_id": self.mailbox_id,
            "owner_id": self.owner_id,
            "owner_type": self.owner_type,
            "name": self.name,
            "greeting_audio_url": self.greeting_audio_url,
            "greeting_text": self.greeting_text,
            "max_duration_seconds": self.max_duration_seconds,
            "max_messages": self.max_messages,
            "email_notification": self.email_notification,
            "sms_notification": self.sms_notification,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "settings": self.settings,
        }


class VoicemailDetector:
    """
    Detects when a call goes to voicemail/answering machine.

    Uses audio analysis to detect:
    - Voicemail greeting patterns
    - Beep tones indicating recording start
    - Silence patterns typical of machine answers

    Usage:
        detector = VoicemailDetector()

        # Process audio chunks
        result = await detector.process_audio(audio_chunk)

        if result == DetectionResult.MACHINE:
            # Wait for beep
            await detector.wait_for_beep()
            # Start message playback
    """

    def __init__(
        self,
        detection_threshold: float = 0.7,
        beep_frequency_low: float = 400,
        beep_frequency_high: float = 2000,
        min_greeting_duration: float = 1.5,
        max_greeting_duration: float = 30.0,
    ):
        self.detection_threshold = detection_threshold
        self.beep_frequency_low = beep_frequency_low
        self.beep_frequency_high = beep_frequency_high
        self.min_greeting_duration = min_greeting_duration
        self.max_greeting_duration = max_greeting_duration

        self._audio_buffer: List[bytes] = []
        self._detection_result: Optional[DetectionResult] = None
        self._beep_detected = False
        self._greeting_started_at: Optional[float] = None
        self._silence_duration: float = 0.0
        self._callbacks: Dict[str, List[Callable]] = {}

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> DetectionResult:
        """
        Process audio chunk for voicemail detection.

        Returns detection result (may be UNKNOWN while gathering data).
        """
        self._audio_buffer.append(audio_data)

        # Analyze audio characteristics
        audio_features = self._extract_features(audio_data, sample_rate)

        # Check for beep tone
        if self._detect_beep(audio_features):
            self._beep_detected = True
            self._detection_result = DetectionResult.BEEP_DETECTED
            await self._notify("beep_detected", {})
            return DetectionResult.BEEP_DETECTED

        # Analyze for machine vs human patterns
        if not self._detection_result or self._detection_result == DetectionResult.UNKNOWN:
            self._detection_result = self._analyze_patterns(audio_features)

        return self._detection_result or DetectionResult.UNKNOWN

    def _extract_features(
        self,
        audio_data: bytes,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Extract audio features for analysis."""
        # Convert bytes to samples
        if len(audio_data) % 2 != 0:
            audio_data = audio_data[:-1]

        samples = struct.unpack(f'{len(audio_data) // 2}h', audio_data)

        if not samples:
            return {
                "energy": 0.0,
                "zero_crossing_rate": 0.0,
                "dominant_frequency": 0.0,
                "is_silence": True,
            }

        # Calculate energy (RMS)
        energy = (sum(s * s for s in samples) / len(samples)) ** 0.5

        # Calculate zero crossing rate
        zero_crossings = sum(
            1 for i in range(1, len(samples))
            if (samples[i] >= 0) != (samples[i-1] >= 0)
        )
        zcr = zero_crossings / len(samples)

        # Simple frequency estimation (for beep detection)
        # In production, use proper FFT
        dominant_frequency = self._estimate_frequency(samples, sample_rate)

        # Determine if silence
        is_silence = energy < 500  # Threshold for 16-bit audio

        return {
            "energy": energy,
            "zero_crossing_rate": zcr,
            "dominant_frequency": dominant_frequency,
            "is_silence": is_silence,
            "sample_count": len(samples),
        }

    def _estimate_frequency(
        self,
        samples: tuple,
        sample_rate: int,
    ) -> float:
        """Estimate dominant frequency using zero-crossing."""
        if len(samples) < 2:
            return 0.0

        zero_crossings = sum(
            1 for i in range(1, len(samples))
            if (samples[i] >= 0) != (samples[i-1] >= 0)
        )

        duration = len(samples) / sample_rate
        if duration <= 0:
            return 0.0

        # Frequency ≈ zero_crossings / (2 * duration)
        frequency = zero_crossings / (2 * duration)
        return frequency

    def _detect_beep(self, features: Dict[str, Any]) -> bool:
        """Detect if audio contains a beep tone."""
        freq = features.get("dominant_frequency", 0)
        energy = features.get("energy", 0)

        # Check if frequency is in beep range and energy is high enough
        if (self.beep_frequency_low <= freq <= self.beep_frequency_high
                and energy > 1000):
            return True

        return False

    def _analyze_patterns(
        self,
        features: Dict[str, Any],
    ) -> DetectionResult:
        """Analyze audio patterns for machine detection."""
        # This is a simplified implementation
        # In production, use ML models trained on voicemail patterns

        is_silence = features.get("is_silence", False)
        energy = features.get("energy", 0)
        zcr = features.get("zero_crossing_rate", 0)

        # Track silence duration (typical of voicemail)
        if is_silence:
            self._silence_duration += 0.02  # Assume 20ms chunks
        else:
            self._silence_duration = 0

        # If we detect characteristic patterns, classify
        # This is very simplified - real detection uses ML
        if self._silence_duration > 0.5:
            # Long silence after speech often indicates machine
            return DetectionResult.MACHINE

        return DetectionResult.UNKNOWN

    async def wait_for_beep(
        self,
        timeout: float = 30.0,
    ) -> bool:
        """Wait for beep detection."""
        start = asyncio.get_event_loop().time()

        while not self._beep_detected:
            if asyncio.get_event_loop().time() - start > timeout:
                return False
            await asyncio.sleep(0.1)

        return True

    def reset(self) -> None:
        """Reset detector state."""
        self._audio_buffer = []
        self._detection_result = None
        self._beep_detected = False
        self._greeting_started_at = None
        self._silence_duration = 0.0

    def on_event(self, event: str, callback: Callable) -> None:
        """Register callback for detection events."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    async def _notify(self, event: str, data: Dict[str, Any]) -> None:
        """Notify callbacks of event."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Detector callback error: {e}")


class VoicemailManager:
    """
    Manages voicemail boxes and messages.

    Usage:
        manager = VoicemailManager()

        # Create mailbox
        mailbox = await manager.create_mailbox(
            owner_id="user_1",
            owner_type="user",
            name="Main Line",
        )

        # Record voicemail
        message = await manager.record_message(
            mailbox_id=mailbox.mailbox_id,
            call_id="call_123",
            from_number="+15551234567",
            to_number="+15559876543",
            audio_data=audio_bytes,
        )

        # Get messages
        messages = await manager.get_messages(mailbox.mailbox_id)
    """

    def __init__(self):
        self._mailboxes: Dict[str, VoicemailBox] = {}
        self._messages: Dict[str, List[VoicemailMessage]] = {}
        self._lock = asyncio.Lock()
        self._transcription_service: Optional[Any] = None
        self._storage_service: Optional[Any] = None
        self._notification_callbacks: List[Callable] = []

    def set_transcription_service(self, service: Any) -> None:
        """Set transcription service for voicemail transcription."""
        self._transcription_service = service

    def set_storage_service(self, service: Any) -> None:
        """Set storage service for audio files."""
        self._storage_service = service

    async def create_mailbox(
        self,
        owner_id: str,
        owner_type: str,
        name: str,
        greeting_text: Optional[str] = None,
        **settings,
    ) -> VoicemailBox:
        """Create a new voicemail box."""
        import uuid

        mailbox = VoicemailBox(
            mailbox_id=f"vmbox_{uuid.uuid4().hex[:12]}",
            owner_id=owner_id,
            owner_type=owner_type,
            name=name,
            greeting_text=greeting_text or f"You have reached {name}. Please leave a message after the beep.",
            settings=settings,
        )

        async with self._lock:
            self._mailboxes[mailbox.mailbox_id] = mailbox
            self._messages[mailbox.mailbox_id] = []

        logger.info(f"Voicemail box created: {mailbox.mailbox_id}")
        return mailbox

    async def get_mailbox(self, mailbox_id: str) -> Optional[VoicemailBox]:
        """Get a voicemail box."""
        async with self._lock:
            return self._mailboxes.get(mailbox_id)

    async def get_mailbox_by_owner(
        self,
        owner_id: str,
    ) -> List[VoicemailBox]:
        """Get all mailboxes for an owner."""
        async with self._lock:
            return [
                mb for mb in self._mailboxes.values()
                if mb.owner_id == owner_id
            ]

    async def update_mailbox(
        self,
        mailbox_id: str,
        **updates,
    ) -> Optional[VoicemailBox]:
        """Update mailbox settings."""
        async with self._lock:
            mailbox = self._mailboxes.get(mailbox_id)
            if not mailbox:
                return None

            for key, value in updates.items():
                if hasattr(mailbox, key):
                    setattr(mailbox, key, value)

        return mailbox

    async def delete_mailbox(self, mailbox_id: str) -> bool:
        """Delete a voicemail box."""
        async with self._lock:
            if mailbox_id in self._mailboxes:
                del self._mailboxes[mailbox_id]
                if mailbox_id in self._messages:
                    del self._messages[mailbox_id]
                return True
        return False

    async def record_message(
        self,
        mailbox_id: str,
        call_id: str,
        from_number: str,
        to_number: str,
        audio_data: bytes,
        duration_seconds: float,
        sample_rate: int = 16000,
        **metadata,
    ) -> Optional[VoicemailMessage]:
        """Record a new voicemail message."""
        mailbox = await self.get_mailbox(mailbox_id)
        if not mailbox or not mailbox.is_active:
            logger.warning(f"Invalid or inactive mailbox: {mailbox_id}")
            return None

        # Check message limit
        current_messages = await self.get_messages(mailbox_id)
        if len(current_messages) >= mailbox.max_messages:
            logger.warning(f"Mailbox {mailbox_id} is full")
            return None

        import uuid
        message_id = f"vm_{uuid.uuid4().hex[:12]}"

        # Store audio
        audio_path = None
        audio_url = None
        if self._storage_service:
            audio_path = f"voicemail/{mailbox_id}/{message_id}.wav"
            audio_url = await self._storage_service.store(audio_path, audio_data)

        message = VoicemailMessage(
            message_id=message_id,
            mailbox_id=mailbox_id,
            call_id=call_id,
            from_number=from_number,
            to_number=to_number,
            duration_seconds=duration_seconds,
            audio_path=audio_path,
            audio_url=audio_url,
            metadata=metadata,
        )

        async with self._lock:
            if mailbox_id not in self._messages:
                self._messages[mailbox_id] = []
            self._messages[mailbox_id].append(message)

        # Start transcription in background
        asyncio.create_task(self._transcribe_message(message, audio_data))

        # Send notifications
        await self._send_notifications(mailbox, message)

        logger.info(f"Voicemail recorded: {message_id} in {mailbox_id}")
        return message

    async def get_messages(
        self,
        mailbox_id: str,
        status: Optional[VoicemailStatus] = None,
        limit: int = 50,
    ) -> List[VoicemailMessage]:
        """Get messages from a mailbox."""
        async with self._lock:
            messages = self._messages.get(mailbox_id, [])

        if status:
            messages = [m for m in messages if m.status == status]

        # Sort by created_at descending
        messages.sort(key=lambda m: m.created_at, reverse=True)

        return messages[:limit]

    async def get_message(
        self,
        message_id: str,
    ) -> Optional[VoicemailMessage]:
        """Get a specific message."""
        async with self._lock:
            for messages in self._messages.values():
                for message in messages:
                    if message.message_id == message_id:
                        return message
        return None

    async def mark_as_heard(
        self,
        message_id: str,
    ) -> Optional[VoicemailMessage]:
        """Mark a message as heard."""
        message = await self.get_message(message_id)
        if message and message.status == VoicemailStatus.NEW:
            message.status = VoicemailStatus.HEARD
            message.heard_at = datetime.utcnow()
        return message

    async def mark_as_saved(
        self,
        message_id: str,
    ) -> Optional[VoicemailMessage]:
        """Mark a message as saved."""
        message = await self.get_message(message_id)
        if message:
            message.status = VoicemailStatus.SAVED
        return message

    async def delete_message(
        self,
        message_id: str,
    ) -> bool:
        """Delete (soft) a message."""
        message = await self.get_message(message_id)
        if message:
            message.status = VoicemailStatus.DELETED
            return True
        return False

    async def permanently_delete_message(
        self,
        message_id: str,
    ) -> bool:
        """Permanently delete a message."""
        async with self._lock:
            for mailbox_id, messages in self._messages.items():
                for i, message in enumerate(messages):
                    if message.message_id == message_id:
                        del self._messages[mailbox_id][i]
                        return True
        return False

    async def get_new_message_count(
        self,
        mailbox_id: str,
    ) -> int:
        """Get count of new messages."""
        messages = await self.get_messages(mailbox_id, VoicemailStatus.NEW)
        return len(messages)

    async def get_greeting_text(
        self,
        mailbox_id: str,
    ) -> Optional[str]:
        """Get greeting text for a mailbox."""
        mailbox = await self.get_mailbox(mailbox_id)
        if mailbox:
            return mailbox.greeting_text
        return None

    async def set_greeting_text(
        self,
        mailbox_id: str,
        greeting_text: str,
    ) -> Optional[VoicemailBox]:
        """Set greeting text for a mailbox."""
        return await self.update_mailbox(mailbox_id, greeting_text=greeting_text)

    async def cleanup_old_messages(
        self,
        max_age_days: int = 30,
    ) -> int:
        """Clean up old deleted messages."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        removed = 0

        async with self._lock:
            for mailbox_id in self._messages:
                original_count = len(self._messages[mailbox_id])
                self._messages[mailbox_id] = [
                    m for m in self._messages[mailbox_id]
                    if not (m.status == VoicemailStatus.DELETED and m.created_at < cutoff)
                ]
                removed += original_count - len(self._messages[mailbox_id])

        if removed > 0:
            logger.info(f"Cleaned up {removed} old voicemail messages")

        return removed

    def on_new_message(
        self,
        callback: Callable[[VoicemailMessage], Awaitable[None]],
    ) -> None:
        """Register callback for new messages."""
        self._notification_callbacks.append(callback)

    async def _transcribe_message(
        self,
        message: VoicemailMessage,
        audio_data: bytes,
    ) -> None:
        """Transcribe a voicemail message."""
        if not self._transcription_service:
            return

        try:
            message.status = VoicemailStatus.TRANSCRIBING

            result = await self._transcription_service.transcribe(audio_data)

            message.transcription = result.get("text", "")
            message.transcription_confidence = result.get("confidence", 0.0)

            # Detect urgency from transcription
            urgency_keywords = [
                "urgent", "emergency", "asap", "immediately",
                "call back", "important", "critical",
            ]
            if message.transcription:
                lower_text = message.transcription.lower()
                message.is_urgent = any(kw in lower_text for kw in urgency_keywords)

            # Reset status to heard or new
            if message.status == VoicemailStatus.TRANSCRIBING:
                message.status = VoicemailStatus.NEW

            logger.info(f"Voicemail transcribed: {message.message_id}")

        except Exception as e:
            logger.error(f"Transcription failed for {message.message_id}: {e}")
            message.status = VoicemailStatus.NEW

    async def _send_notifications(
        self,
        mailbox: VoicemailBox,
        message: VoicemailMessage,
    ) -> None:
        """Send notifications for new voicemail."""
        # Notify registered callbacks
        for callback in self._notification_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

        # Email notification
        if mailbox.email_notification:
            await self._send_email_notification(mailbox, message)

        # SMS notification
        if mailbox.sms_notification:
            await self._send_sms_notification(mailbox, message)

    async def _send_email_notification(
        self,
        mailbox: VoicemailBox,
        message: VoicemailMessage,
    ) -> None:
        """Send email notification for new voicemail."""
        # In a real implementation, this would send an email
        logger.info(
            f"Email notification would be sent to {mailbox.email_notification} "
            f"for voicemail from {message.from_number}"
        )

    async def _send_sms_notification(
        self,
        mailbox: VoicemailBox,
        message: VoicemailMessage,
    ) -> None:
        """Send SMS notification for new voicemail."""
        # In a real implementation, this would send an SMS
        logger.info(
            f"SMS notification would be sent to {mailbox.sms_notification} "
            f"for voicemail from {message.from_number}"
        )
