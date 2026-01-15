"""
Voice Registry Service.

Central registry for managing voices, samples, and consent records.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import uuid

from ..config import get_settings, VoiceProvider, VoiceQuality, ConsentStatus
from ..models import (
    Voice,
    VoiceStatus,
    AudioSample,
    VoiceConsent,
    VoiceLibrary,
    VoiceCategory,
    VoiceStyleSettings,
)

logger = logging.getLogger(__name__)


class VoiceRegistry:
    """
    Central registry for voice management.

    In production, this would be backed by a database (PostgreSQL).
    This implementation uses in-memory storage for development.
    """

    def __init__(self):
        # In-memory storage (replace with DB in production)
        self._voices: Dict[str, Voice] = {}
        self._samples: Dict[str, AudioSample] = {}
        self._consents: Dict[str, VoiceConsent] = {}
        self._libraries: Dict[str, VoiceLibrary] = {}

        # Indexes
        self._voices_by_tenant: Dict[str, List[str]] = {}
        self._voices_by_provider: Dict[VoiceProvider, List[str]] = {}

        self._lock = asyncio.Lock()

    # =========================================================================
    # Voice Operations
    # =========================================================================

    async def create_voice(
        self,
        tenant_id: str,
        created_by: str,
        name: str,
        provider: VoiceProvider,
        quality: VoiceQuality = VoiceQuality.STANDARD,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        style: Optional[VoiceStyleSettings] = None,
        consent_id: Optional[str] = None,
    ) -> Voice:
        """Create a new voice record."""
        voice = Voice(
            tenant_id=tenant_id,
            created_by=created_by,
            name=name,
            description=description,
            tags=tags or [],
            provider=provider,
            quality=quality,
            style=style or VoiceStyleSettings(),
            status=VoiceStatus.PENDING,
            consent_id=consent_id,
        )

        async with self._lock:
            self._voices[voice.voice_id] = voice

            # Update indexes
            if tenant_id not in self._voices_by_tenant:
                self._voices_by_tenant[tenant_id] = []
            self._voices_by_tenant[tenant_id].append(voice.voice_id)

            if provider not in self._voices_by_provider:
                self._voices_by_provider[provider] = []
            self._voices_by_provider[provider].append(voice.voice_id)

        logger.info(f"Voice created: {voice.voice_id} for tenant {tenant_id}")
        return voice

    async def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get a voice by ID."""
        return self._voices.get(voice_id)

    async def get_voice_by_tenant(
        self,
        tenant_id: str,
        voice_id: str,
    ) -> Optional[Voice]:
        """Get a voice ensuring it belongs to the tenant."""
        voice = self._voices.get(voice_id)
        if voice and voice.tenant_id == tenant_id:
            return voice
        return None

    async def update_voice(
        self,
        voice_id: str,
        **updates,
    ) -> Optional[Voice]:
        """Update a voice record."""
        voice = self._voices.get(voice_id)
        if not voice:
            return None

        async with self._lock:
            for key, value in updates.items():
                if hasattr(voice, key) and value is not None:
                    setattr(voice, key, value)
            voice.updated_at = datetime.utcnow()

        return voice

    async def update_voice_status(
        self,
        voice_id: str,
        status: VoiceStatus,
        provider_voice_id: Optional[str] = None,
        status_message: Optional[str] = None,
    ) -> Optional[Voice]:
        """Update voice status after cloning."""
        return await self.update_voice(
            voice_id,
            status=status,
            provider_voice_id=provider_voice_id,
            status_message=status_message,
        )

    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice (soft delete)."""
        voice = self._voices.get(voice_id)
        if not voice:
            return False

        async with self._lock:
            voice.status = VoiceStatus.ARCHIVED
            voice.archived_at = datetime.utcnow()

        return True

    async def list_voices(
        self,
        tenant_id: str,
        status: Optional[VoiceStatus] = None,
        provider: Optional[VoiceProvider] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Voice], int]:
        """
        List voices for a tenant with filtering.

        Returns:
            Tuple of (voices, total_count)
        """
        voice_ids = self._voices_by_tenant.get(tenant_id, [])
        voices = [
            self._voices[vid] for vid in voice_ids
            if vid in self._voices
        ]

        # Filter by status
        if status:
            voices = [v for v in voices if v.status == status]
        else:
            # Exclude archived by default
            voices = [v for v in voices if v.status != VoiceStatus.ARCHIVED]

        # Filter by provider
        if provider:
            voices = [v for v in voices if v.provider == provider]

        # Filter by tags
        if tags:
            voices = [
                v for v in voices
                if any(t in v.tags for t in tags)
            ]

        # Search
        if search:
            search_lower = search.lower()
            voices = [
                v for v in voices
                if search_lower in v.name.lower()
                or (v.description and search_lower in v.description.lower())
            ]

        # Sort by created_at descending
        voices.sort(key=lambda v: v.created_at, reverse=True)

        total = len(voices)

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        voices = voices[start:end]

        return voices, total

    async def get_public_voices(
        self,
        tags: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Voice], int]:
        """Get publicly shared voices."""
        voices = [
            v for v in self._voices.values()
            if v.is_public and v.status == VoiceStatus.READY
        ]

        if tags:
            voices = [
                v for v in voices
                if any(t in v.tags for t in tags)
            ]

        total = len(voices)

        start = (page - 1) * page_size
        end = start + page_size
        voices = voices[start:end]

        return voices, total

    async def record_voice_usage(self, voice_id: str) -> None:
        """Record that a voice was used."""
        voice = self._voices.get(voice_id)
        if voice:
            voice.usage_count += 1
            voice.last_used_at = datetime.utcnow()

    # =========================================================================
    # Sample Operations
    # =========================================================================

    async def add_sample(
        self,
        voice_id: str,
        sample: AudioSample,
    ) -> AudioSample:
        """Add a sample to a voice."""
        sample.voice_id = voice_id

        async with self._lock:
            self._samples[sample.sample_id] = sample

            # Update voice
            voice = self._voices.get(voice_id)
            if voice:
                voice.sample_ids.append(sample.sample_id)
                if sample.metadata:
                    voice.total_sample_duration_s += sample.metadata.duration_s

        return sample

    async def get_sample(self, sample_id: str) -> Optional[AudioSample]:
        """Get a sample by ID."""
        return self._samples.get(sample_id)

    async def get_voice_samples(self, voice_id: str) -> List[AudioSample]:
        """Get all samples for a voice."""
        voice = self._voices.get(voice_id)
        if not voice:
            return []

        return [
            self._samples[sid]
            for sid in voice.sample_ids
            if sid in self._samples
        ]

    async def delete_sample(self, sample_id: str) -> bool:
        """Delete a sample."""
        sample = self._samples.get(sample_id)
        if not sample:
            return False

        async with self._lock:
            # Remove from voice
            if sample.voice_id:
                voice = self._voices.get(sample.voice_id)
                if voice and sample_id in voice.sample_ids:
                    voice.sample_ids.remove(sample_id)
                    if sample.metadata:
                        voice.total_sample_duration_s -= sample.metadata.duration_s

            del self._samples[sample_id]

        return True

    # =========================================================================
    # Consent Operations
    # =========================================================================

    async def create_consent(
        self,
        voice_owner_name: str,
        voice_owner_email: Optional[str] = None,
        purpose: str = "ai_agent",
        verification_method: str = "email",
    ) -> VoiceConsent:
        """Create a consent record."""
        settings = get_settings()

        consent = VoiceConsent(
            voice_owner_name=voice_owner_name,
            voice_owner_email=voice_owner_email,
            purpose=purpose,
            verification_method=verification_method,
            verification_code=str(uuid.uuid4())[:8].upper(),
            expires_at=datetime.utcnow() + timedelta(days=settings.consent_expiry_days),
        )

        async with self._lock:
            self._consents[consent.consent_id] = consent

        return consent

    async def get_consent(self, consent_id: str) -> Optional[VoiceConsent]:
        """Get a consent record."""
        return self._consents.get(consent_id)

    async def verify_consent(
        self,
        consent_id: str,
        verification_code: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> bool:
        """Verify a consent record."""
        consent = self._consents.get(consent_id)
        if not consent:
            return False

        if consent.verification_code != verification_code:
            return False

        if consent.status != ConsentStatus.PENDING:
            return False

        async with self._lock:
            consent.status = ConsentStatus.VERIFIED
            consent.verified_at = datetime.utcnow()
            consent.ip_address = ip_address
            consent.user_agent = user_agent

        return True

    async def revoke_consent(self, consent_id: str) -> bool:
        """Revoke a consent."""
        consent = self._consents.get(consent_id)
        if not consent:
            return False

        async with self._lock:
            consent.status = ConsentStatus.REVOKED
            consent.revoked_at = datetime.utcnow()

        return True

    # =========================================================================
    # Library Operations
    # =========================================================================

    async def get_library(self, tenant_id: str) -> VoiceLibrary:
        """Get or create voice library for a tenant."""
        if tenant_id not in self._libraries:
            self._libraries[tenant_id] = VoiceLibrary(
                tenant_id=tenant_id,
                categories=[
                    VoiceCategory(name="Custom", description="Your custom voices"),
                    VoiceCategory(name="Templates", description="Voice templates"),
                ],
            )

        library = self._libraries[tenant_id]

        # Update counts
        voices, total = await self.list_voices(tenant_id)
        library.total_voices = total

        return library

    async def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for a tenant."""
        voices, total = await self.list_voices(tenant_id)
        ready_count = sum(1 for v in voices if v.status == VoiceStatus.READY)

        samples = [
            s for s in self._samples.values()
            if s.voice_id and self._voices.get(s.voice_id, Voice(tenant_id="", created_by="", name="", provider=VoiceProvider.ELEVENLABS)).tenant_id == tenant_id
        ]

        return {
            "tenant_id": tenant_id,
            "total_voices": total,
            "ready_voices": ready_count,
            "total_samples": len(samples),
            "total_duration_s": sum(
                s.metadata.duration_s for s in samples
                if s.metadata
            ),
        }


# Singleton instance
_registry: Optional[VoiceRegistry] = None


def get_registry() -> VoiceRegistry:
    """Get the voice registry singleton."""
    global _registry
    if _registry is None:
        _registry = VoiceRegistry()
    return _registry
