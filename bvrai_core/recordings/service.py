"""
Recording Service Module

This module provides a unified service layer for managing call recordings,
transcriptions, retention policies, and redaction operations.
"""

import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import (
    Any,
    AsyncIterator,
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .base import (
    Recording,
    RecordingMetadata,
    AudioProperties,
    RecordingStatus,
    RecordingFormat,
    Transcription,
    TranscriptSegment,
    TranscriptionStatus,
    TranscriptionProvider,
    RetentionPolicy,
    RetentionAction,
    RedactionRule,
    RedactionResult,
    RedactionType,
    StorageConfig,
    StorageBackend,
    AccessLevel,
    RecordingError,
    StorageError,
    TranscriptionError,
    RetentionError,
    RedactionError,
    AccessDeniedError,
)
from .storage import StorageManager, StorageProvider
from .transcription import TranscriptionManager, TranscriptionPostProcessor


logger = logging.getLogger(__name__)


# =============================================================================
# Recording Storage
# =============================================================================


class RecordingStorage:
    """In-memory recording storage for development."""

    def __init__(self):
        self._recordings: Dict[str, Recording] = {}
        self._org_index: Dict[str, Set[str]] = {}
        self._call_index: Dict[str, str] = {}

    async def save(self, recording: Recording) -> None:
        """Save a recording record."""
        recording.updated_at = datetime.utcnow()
        self._recordings[recording.id] = recording

        if recording.organization_id not in self._org_index:
            self._org_index[recording.organization_id] = set()
        self._org_index[recording.organization_id].add(recording.id)
        self._call_index[recording.call_id] = recording.id

    async def get(self, recording_id: str) -> Optional[Recording]:
        """Get a recording by ID."""
        return self._recordings.get(recording_id)

    async def get_by_call_id(self, call_id: str) -> Optional[Recording]:
        """Get recording by call ID."""
        recording_id = self._call_index.get(call_id)
        if recording_id:
            return self._recordings.get(recording_id)
        return None

    async def list(
        self,
        organization_id: str,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[Recording], int]:
        """List recordings with filters."""
        if organization_id not in self._org_index:
            return [], 0

        recordings = []
        for rec_id in self._org_index[organization_id]:
            recording = self._recordings.get(rec_id)
            if not recording:
                continue

            # Apply filters
            if filters:
                if "status" in filters and recording.status != filters["status"]:
                    continue
                if "agent_id" in filters and recording.metadata.agent_id != filters["agent_id"]:
                    continue
                if "campaign_id" in filters and recording.metadata.campaign_id != filters["campaign_id"]:
                    continue
                if "start_date" in filters and recording.started_at < filters["start_date"]:
                    continue
                if "end_date" in filters and recording.started_at > filters["end_date"]:
                    continue
                if "min_duration" in filters and recording.audio.duration_seconds < filters["min_duration"]:
                    continue
                if "tags" in filters:
                    if not any(tag in recording.metadata.tags for tag in filters["tags"]):
                        continue

            recordings.append(recording)

        # Sort by created_at descending
        recordings.sort(key=lambda r: r.created_at, reverse=True)

        total = len(recordings)
        recordings = recordings[offset:offset + limit]

        return recordings, total

    async def delete(self, recording_id: str) -> bool:
        """Delete a recording record."""
        recording = self._recordings.get(recording_id)
        if not recording:
            return False

        if recording.organization_id in self._org_index:
            self._org_index[recording.organization_id].discard(recording_id)
        if recording.call_id in self._call_index:
            del self._call_index[recording.call_id]

        del self._recordings[recording_id]
        return True


# =============================================================================
# Retention Manager
# =============================================================================


class RetentionManager:
    """
    Manages retention policies and lifecycle operations.
    """

    def __init__(self):
        self._policies: Dict[str, RetentionPolicy] = {}
        self._org_index: Dict[str, Set[str]] = {}

    async def create_policy(
        self,
        name: str,
        organization_id: str,
        retention_days: int = 90,
        archive_after_days: Optional[int] = None,
        action_on_expiry: RetentionAction = RetentionAction.DELETE,
        **kwargs,
    ) -> RetentionPolicy:
        """Create a retention policy."""
        policy = RetentionPolicy(
            id=f"rp_{uuid.uuid4().hex[:24]}",
            name=name,
            organization_id=organization_id,
            retention_days=retention_days,
            archive_after_days=archive_after_days,
            action_on_expiry=action_on_expiry,
            applies_to_campaigns=kwargs.get("applies_to_campaigns", []),
            applies_to_agents=kwargs.get("applies_to_agents", []),
            applies_to_tags=kwargs.get("applies_to_tags", []),
            compliance_required=kwargs.get("compliance_required", False),
            legal_hold=kwargs.get("legal_hold", False),
            auto_redact_pii=kwargs.get("auto_redact_pii", False),
            redaction_types=[RedactionType(r) for r in kwargs.get("redaction_types", [])],
            is_default=kwargs.get("is_default", False),
        )

        self._policies[policy.id] = policy
        if organization_id not in self._org_index:
            self._org_index[organization_id] = set()
        self._org_index[organization_id].add(policy.id)

        logger.info(f"Created retention policy: {policy.id} ({name})")
        return policy

    async def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    async def get_default_policy(
        self,
        organization_id: str,
    ) -> Optional[RetentionPolicy]:
        """Get default policy for organization."""
        if organization_id not in self._org_index:
            return None

        for policy_id in self._org_index[organization_id]:
            policy = self._policies.get(policy_id)
            if policy and policy.is_default:
                return policy

        # Return first active policy if no default
        for policy_id in self._org_index[organization_id]:
            policy = self._policies.get(policy_id)
            if policy and policy.is_active:
                return policy

        return None

    async def list_policies(
        self,
        organization_id: str,
    ) -> List[RetentionPolicy]:
        """List policies for organization."""
        if organization_id not in self._org_index:
            return []

        policies = []
        for policy_id in self._org_index[organization_id]:
            policy = self._policies.get(policy_id)
            if policy:
                policies.append(policy)

        return policies

    async def apply_policy(
        self,
        recording: Recording,
        policy: Optional[RetentionPolicy] = None,
    ) -> Recording:
        """Apply retention policy to a recording."""
        if not policy:
            policy = await self.get_default_policy(recording.organization_id)

        if not policy:
            return recording

        # Calculate expiry date
        recording.retention_policy_id = policy.id
        recording.expires_at = policy.calculate_expiry_date(recording.created_at)

        return recording

    async def get_expired_recordings(
        self,
        organization_id: str,
        recording_storage: RecordingStorage,
        as_of: Optional[datetime] = None,
    ) -> List[Recording]:
        """Get recordings that have expired."""
        if as_of is None:
            as_of = datetime.utcnow()

        recordings, _ = await recording_storage.list(
            organization_id=organization_id,
            limit=10000,
        )

        expired = []
        for recording in recordings:
            if recording.expires_at and recording.expires_at <= as_of:
                # Check for legal hold
                if recording.retention_policy_id:
                    policy = await self.get_policy(recording.retention_policy_id)
                    if policy and policy.legal_hold:
                        continue
                expired.append(recording)

        return expired

    async def get_archivable_recordings(
        self,
        organization_id: str,
        recording_storage: RecordingStorage,
        as_of: Optional[datetime] = None,
    ) -> List[Recording]:
        """Get recordings ready for archival."""
        if as_of is None:
            as_of = datetime.utcnow()

        recordings, _ = await recording_storage.list(
            organization_id=organization_id,
            limit=10000,
        )

        archivable = []
        for recording in recordings:
            if recording.archived_at:
                continue

            if recording.retention_policy_id:
                policy = await self.get_policy(recording.retention_policy_id)
                if policy and policy.archive_after_days:
                    archive_date = policy.calculate_archive_date(recording.created_at)
                    if archive_date and archive_date <= as_of:
                        archivable.append(recording)

        return archivable


# =============================================================================
# Redaction Engine
# =============================================================================


class RedactionEngine:
    """
    Handles redaction of sensitive content in recordings and transcripts.
    """

    # Built-in patterns
    PATTERNS = {
        RedactionType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        RedactionType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        RedactionType.PHONE_NUMBER: r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        RedactionType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    }

    def __init__(self):
        self._rules: Dict[str, RedactionRule] = {}
        self._org_rules: Dict[str, Set[str]] = {}

    async def add_rule(
        self,
        name: str,
        redaction_type: RedactionType,
        organization_id: str,
        pattern: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        **kwargs,
    ) -> RedactionRule:
        """Add a redaction rule."""
        rule = RedactionRule(
            id=f"rr_{uuid.uuid4().hex[:16]}",
            name=name,
            redaction_type=redaction_type,
            pattern=pattern or self.PATTERNS.get(redaction_type),
            keywords=keywords or [],
            replacement_text=kwargs.get("replacement_text", "[REDACTED]"),
            replacement_audio=kwargs.get("replacement_audio", "beep"),
            apply_to_transcript=kwargs.get("apply_to_transcript", True),
            apply_to_audio=kwargs.get("apply_to_audio", False),
        )

        self._rules[rule.id] = rule
        if organization_id not in self._org_rules:
            self._org_rules[organization_id] = set()
        self._org_rules[organization_id].add(rule.id)

        return rule

    async def get_rules(
        self,
        organization_id: str,
    ) -> List[RedactionRule]:
        """Get rules for organization."""
        if organization_id not in self._org_rules:
            return []

        rules = []
        for rule_id in self._org_rules[organization_id]:
            rule = self._rules.get(rule_id)
            if rule and rule.is_active:
                rules.append(rule)

        return rules

    async def redact_transcription(
        self,
        transcription: Transcription,
        rules: Optional[List[RedactionRule]] = None,
    ) -> Tuple[Transcription, RedactionResult]:
        """
        Redact sensitive content from transcription.

        Args:
            transcription: Transcription to redact
            rules: Rules to apply (uses org rules if not provided)

        Returns:
            Tuple of (redacted transcription, redaction result)
        """
        if not rules:
            rules = await self.get_rules(transcription.organization_id)

        result = RedactionResult(
            recording_id=transcription.recording_id,
            redaction_rule_ids=[r.id for r in rules],
        )

        # Redact full text
        redacted_text = transcription.full_text
        for rule in rules:
            if not rule.apply_to_transcript:
                continue

            if rule.pattern:
                matches = list(re.finditer(rule.pattern, redacted_text, re.IGNORECASE))
                for match in reversed(matches):
                    redacted_text = (
                        redacted_text[:match.start()] +
                        rule.replacement_text +
                        redacted_text[match.end():]
                    )
                    result.text_redactions_count += 1
                    result.redacted_segments.append({
                        "type": rule.redaction_type.value,
                        "start": match.start(),
                        "end": match.end(),
                        "original_length": match.end() - match.start(),
                    })

            for keyword in rule.keywords:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                matches = list(pattern.finditer(redacted_text))
                for match in reversed(matches):
                    redacted_text = (
                        redacted_text[:match.start()] +
                        rule.replacement_text +
                        redacted_text[match.end():]
                    )
                    result.text_redactions_count += 1

        transcription.full_text = redacted_text

        # Redact segments
        for segment in transcription.segments:
            redacted_segment_text = segment.text
            for rule in rules:
                if not rule.apply_to_transcript:
                    continue

                if rule.pattern:
                    redacted_segment_text = re.sub(
                        rule.pattern,
                        rule.replacement_text,
                        redacted_segment_text,
                        flags=re.IGNORECASE,
                    )

                for keyword in rule.keywords:
                    redacted_segment_text = re.sub(
                        re.escape(keyword),
                        rule.replacement_text,
                        redacted_segment_text,
                        flags=re.IGNORECASE,
                    )

            segment.text = redacted_segment_text

        transcription.status = TranscriptionStatus.REDACTED
        transcription.updated_at = datetime.utcnow()
        result.transcript_redacted = True

        logger.info(f"Redacted transcription {transcription.id}: {result.text_redactions_count} redactions")
        return transcription, result

    def detect_pii(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect PII in text without redacting.

        Args:
            text: Text to analyze

        Returns:
            List of detected PII items
        """
        detected = []

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detected.append({
                    "type": pii_type.value,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                })

        return detected


# =============================================================================
# Recording Service
# =============================================================================


class RecordingService:
    """
    Unified service for managing recordings, transcriptions, and related operations.
    """

    def __init__(
        self,
        storage_manager: Optional[StorageManager] = None,
        transcription_manager: Optional[TranscriptionManager] = None,
        retention_manager: Optional[RetentionManager] = None,
        redaction_engine: Optional[RedactionEngine] = None,
    ):
        """
        Initialize recording service.

        Args:
            storage_manager: Storage manager
            transcription_manager: Transcription manager
            retention_manager: Retention manager
            redaction_engine: Redaction engine
        """
        self.storage = storage_manager or StorageManager()
        self.transcription = transcription_manager or TranscriptionManager()
        self.retention = retention_manager or RetentionManager()
        self.redaction = redaction_engine or RedactionEngine()

        # Recording storage
        self._recording_storage = RecordingStorage()

        # Transcription storage
        self._transcriptions: Dict[str, Transcription] = {}

    async def start_recording(
        self,
        call_id: str,
        organization_id: str,
        agent_id: Optional[str] = None,
        caller_phone: Optional[str] = None,
        agent_phone: Optional[str] = None,
        campaign_id: Optional[str] = None,
        **kwargs,
    ) -> Recording:
        """
        Start a new recording.

        Args:
            call_id: Call ID
            organization_id: Organization ID
            agent_id: Agent ID
            caller_phone: Caller phone number
            agent_phone: Agent phone number
            campaign_id: Campaign ID
            **kwargs: Additional metadata

        Returns:
            New recording object
        """
        metadata = RecordingMetadata(
            call_id=call_id,
            organization_id=organization_id,
            agent_id=agent_id,
            caller_phone=caller_phone,
            agent_phone=agent_phone,
            campaign_id=campaign_id,
            direction=kwargs.get("direction", "inbound"),
            tags=kwargs.get("tags", []),
            custom_fields=kwargs.get("custom_fields", {}),
        )

        recording = Recording(
            id=f"rec_{uuid.uuid4().hex[:24]}",
            call_id=call_id,
            organization_id=organization_id,
            status=RecordingStatus.RECORDING,
            metadata=metadata,
            started_at=datetime.utcnow(),
        )

        # Apply retention policy
        recording = await self.retention.apply_policy(recording)

        # Save
        await self._recording_storage.save(recording)

        logger.info(f"Started recording: {recording.id} for call {call_id}")
        return recording

    async def finish_recording(
        self,
        recording_id: str,
        audio_data: bytes,
        audio_format: RecordingFormat = RecordingFormat.WAV,
        sample_rate: int = 16000,
        duration_seconds: Optional[float] = None,
    ) -> Recording:
        """
        Finish recording and upload audio.

        Args:
            recording_id: Recording ID
            audio_data: Audio data
            audio_format: Audio format
            sample_rate: Sample rate
            duration_seconds: Recording duration

        Returns:
            Updated recording
        """
        recording = await self._recording_storage.get(recording_id)
        if not recording:
            raise RecordingError(f"Recording not found: {recording_id}")

        # Update audio properties
        recording.audio.format = audio_format
        recording.audio.sample_rate = sample_rate
        recording.audio.file_size_bytes = len(audio_data)
        if duration_seconds:
            recording.audio.duration_seconds = duration_seconds

        # Upload to storage
        recording = await self.storage.upload_recording(recording, audio_data)

        # Update status
        recording.status = RecordingStatus.PROCESSING
        recording.ended_at = datetime.utcnow()
        await self._recording_storage.save(recording)

        logger.info(f"Finished recording: {recording_id}")
        return recording

    async def get_recording(
        self,
        recording_id: str,
        user_id: Optional[str] = None,
        check_access: bool = True,
    ) -> Optional[Recording]:
        """
        Get a recording by ID.

        Args:
            recording_id: Recording ID
            user_id: User requesting access
            check_access: Whether to check access permissions

        Returns:
            Recording or None
        """
        recording = await self._recording_storage.get(recording_id)
        if not recording:
            return None

        if check_access and user_id:
            if not self._check_access(recording, user_id):
                raise AccessDeniedError("Access denied to recording")

        return recording

    async def get_recording_by_call(
        self,
        call_id: str,
    ) -> Optional[Recording]:
        """Get recording by call ID."""
        return await self._recording_storage.get_by_call_id(call_id)

    async def list_recordings(
        self,
        organization_id: str,
        agent_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        status: Optional[RecordingStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        min_duration: Optional[float] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[Recording], int]:
        """
        List recordings with filters.

        Args:
            organization_id: Organization ID
            agent_id: Filter by agent
            campaign_id: Filter by campaign
            status: Filter by status
            start_date: Filter by start date
            end_date: Filter by end date
            tags: Filter by tags
            min_duration: Minimum duration in seconds
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (recordings, total_count)
        """
        filters = {}
        if agent_id:
            filters["agent_id"] = agent_id
        if campaign_id:
            filters["campaign_id"] = campaign_id
        if status:
            filters["status"] = status
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date
        if tags:
            filters["tags"] = tags
        if min_duration:
            filters["min_duration"] = min_duration

        return await self._recording_storage.list(
            organization_id=organization_id,
            filters=filters if filters else None,
            offset=offset,
            limit=limit,
        )

    async def delete_recording(
        self,
        recording_id: str,
        delete_storage: bool = True,
    ) -> bool:
        """
        Delete a recording.

        Args:
            recording_id: Recording ID
            delete_storage: Also delete from storage

        Returns:
            True if deleted
        """
        recording = await self._recording_storage.get(recording_id)
        if not recording:
            return False

        # Delete from storage
        if delete_storage and recording.storage_path:
            try:
                await self.storage.delete_recording(recording)
            except StorageError as e:
                logger.warning(f"Failed to delete from storage: {e}")

        # Delete transcription
        if recording.transcription_id and recording.transcription_id in self._transcriptions:
            del self._transcriptions[recording.transcription_id]

        # Delete record
        success = await self._recording_storage.delete(recording_id)

        if success:
            logger.info(f"Deleted recording: {recording_id}")

        return success

    async def get_download_url(
        self,
        recording_id: str,
        expiration_seconds: int = 3600,
    ) -> str:
        """
        Get presigned download URL for recording.

        Args:
            recording_id: Recording ID
            expiration_seconds: URL expiration time

        Returns:
            Presigned URL
        """
        recording = await self._recording_storage.get(recording_id)
        if not recording:
            raise RecordingError(f"Recording not found: {recording_id}")

        return await self.storage.get_download_url(recording, expiration_seconds)

    async def transcribe_recording(
        self,
        recording_id: str,
        provider: Optional[TranscriptionProvider] = None,
        language: str = "en-US",
        diarization: bool = True,
        auto_redact: bool = False,
    ) -> Transcription:
        """
        Transcribe a recording.

        Args:
            recording_id: Recording ID
            provider: Transcription provider
            language: Language code
            diarization: Enable speaker diarization
            auto_redact: Automatically redact PII

        Returns:
            Transcription object
        """
        recording = await self._recording_storage.get(recording_id)
        if not recording:
            raise RecordingError(f"Recording not found: {recording_id}")

        # Download audio
        audio_data = await self.storage.download_recording(recording)

        # Transcribe
        transcription = await self.transcription.transcribe_recording(
            recording=recording,
            audio_data=audio_data,
            provider=provider,
            language=language,
            diarization=diarization,
        )

        # Auto-redact if requested
        if auto_redact:
            transcription, _ = await self.redaction.redact_transcription(transcription)

        # Store
        self._transcriptions[transcription.id] = transcription
        recording.transcription_id = transcription.id
        recording.status = RecordingStatus.READY
        await self._recording_storage.save(recording)

        return transcription

    async def get_transcription(
        self,
        transcription_id: str,
    ) -> Optional[Transcription]:
        """Get transcription by ID."""
        return self._transcriptions.get(transcription_id)

    async def get_recording_transcription(
        self,
        recording_id: str,
    ) -> Optional[Transcription]:
        """Get transcription for a recording."""
        recording = await self._recording_storage.get(recording_id)
        if not recording or not recording.transcription_id:
            return None
        return self._transcriptions.get(recording.transcription_id)

    async def search_recordings(
        self,
        organization_id: str,
        query: str,
    ) -> List[Tuple[Recording, Transcription, List[TranscriptSegment]]]:
        """
        Search recordings by transcript content.

        Args:
            organization_id: Organization ID
            query: Search query

        Returns:
            List of (recording, transcription, matching_segments)
        """
        results = []
        query_lower = query.lower()

        for transcription in self._transcriptions.values():
            if transcription.organization_id != organization_id:
                continue

            matching_segments = []
            for segment in transcription.segments:
                if query_lower in segment.text.lower():
                    matching_segments.append(segment)

            if matching_segments:
                recording = await self._recording_storage.get_by_call_id(
                    transcription.recording_id.replace("rec_", "call_")
                ) or await self._recording_storage.get(transcription.recording_id)

                if recording:
                    results.append((recording, transcription, matching_segments))

        return results

    async def apply_retention_policy(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """
        Apply retention policy to expired recordings.

        Args:
            organization_id: Organization ID

        Returns:
            Summary of actions taken
        """
        summary = {
            "deleted": 0,
            "archived": 0,
            "errors": [],
        }

        # Process expired recordings
        expired = await self.retention.get_expired_recordings(
            organization_id,
            self._recording_storage,
        )

        for recording in expired:
            policy = await self.retention.get_policy(recording.retention_policy_id)
            if not policy:
                continue

            try:
                if policy.action_on_expiry == RetentionAction.DELETE:
                    await self.delete_recording(recording.id)
                    summary["deleted"] += 1
                elif policy.action_on_expiry == RetentionAction.ARCHIVE:
                    # Archive would move to cold storage
                    recording.status = RecordingStatus.ARCHIVED
                    await self._recording_storage.save(recording)
                    summary["archived"] += 1
            except Exception as e:
                summary["errors"].append({
                    "recording_id": recording.id,
                    "error": str(e),
                })

        # Process archivable recordings
        archivable = await self.retention.get_archivable_recordings(
            organization_id,
            self._recording_storage,
        )

        for recording in archivable:
            try:
                recording.archived_at = datetime.utcnow()
                recording.status = RecordingStatus.ARCHIVED
                await self._recording_storage.save(recording)
                summary["archived"] += 1
            except Exception as e:
                summary["errors"].append({
                    "recording_id": recording.id,
                    "error": str(e),
                })

        return summary

    async def get_recording_stats(
        self,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get recording statistics.

        Args:
            organization_id: Organization ID
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Statistics dictionary
        """
        recordings, total = await self.list_recordings(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        total_duration = sum(r.audio.duration_seconds for r in recordings)
        total_size = sum(r.audio.file_size_bytes for r in recordings)

        # Status breakdown
        status_counts: Dict[str, int] = {}
        for recording in recordings:
            status = recording.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Agent breakdown
        agent_counts: Dict[str, int] = {}
        for recording in recordings:
            agent = recording.metadata.agent_id or "unknown"
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        return {
            "total_recordings": total,
            "total_duration_seconds": total_duration,
            "total_duration_hours": total_duration / 3600,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_duration_seconds": total_duration / total if total > 0 else 0,
            "status_breakdown": status_counts,
            "agent_breakdown": agent_counts,
            "transcribed_count": sum(1 for r in recordings if r.transcription_id),
        }

    def _check_access(
        self,
        recording: Recording,
        user_id: str,
    ) -> bool:
        """Check if user has access to recording."""
        if recording.access_level == AccessLevel.PUBLIC:
            return True
        if recording.access_level == AccessLevel.ORGANIZATION:
            return True  # Would check org membership
        if user_id in recording.allowed_users:
            return True
        return False


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "RecordingStorage",
    "RetentionManager",
    "RedactionEngine",
    "RecordingService",
]
