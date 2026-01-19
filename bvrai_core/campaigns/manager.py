"""Campaign management service."""

import asyncio
import csv
import io
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import redis.asyncio as redis

from .models import (
    Campaign,
    CampaignContact,
    CampaignSchedule,
    CampaignSettings,
    CampaignStatus,
    ContactStatus,
    CampaignProgress,
    ContactImportResult,
)

logger = logging.getLogger(__name__)


class CampaignManager:
    """
    Manages campaign lifecycle and contact lists.

    Features:
    - Campaign CRUD operations
    - Contact list management
    - Progress tracking
    - Import/export functionality
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
    ):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None

        # In-memory cache
        self._campaigns: Dict[str, Campaign] = {}
        self._lock = asyncio.Lock()

        # Event callbacks
        self._on_campaign_started: List[Callable[[Campaign], Any]] = []
        self._on_campaign_completed: List[Callable[[Campaign], Any]] = []
        self._on_campaign_paused: List[Callable[[Campaign], Any]] = []

    async def start(self) -> None:
        """Start the campaign manager."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        await self._load_campaigns()
        logger.info("Campaign manager started")

    async def stop(self) -> None:
        """Stop the campaign manager."""
        if self.redis:
            await self.redis.close()
        logger.info("Campaign manager stopped")

    # Campaign CRUD

    async def create_campaign(
        self,
        organization_id: str,
        name: str,
        agent_id: str,
        from_number: str,
        description: str = "",
        phone_number_id: Optional[str] = None,
        schedule: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Campaign:
        """Create a new campaign."""
        campaign_id = f"camp_{uuid4().hex[:12]}"

        # Parse schedule
        campaign_schedule = CampaignSchedule()
        if schedule:
            if schedule.get("start_time"):
                campaign_schedule.start_time = datetime.fromisoformat(schedule["start_time"])
            if schedule.get("end_time"):
                campaign_schedule.end_time = datetime.fromisoformat(schedule["end_time"])
            campaign_schedule.timezone = schedule.get("timezone", "UTC")
            if schedule.get("daily_start_time"):
                from datetime import time as time_type
                parts = schedule["daily_start_time"].split(":")
                campaign_schedule.daily_start_time = time_type(int(parts[0]), int(parts[1]))
            if schedule.get("daily_end_time"):
                from datetime import time as time_type
                parts = schedule["daily_end_time"].split(":")
                campaign_schedule.daily_end_time = time_type(int(parts[0]), int(parts[1]))
            if schedule.get("days_of_week"):
                campaign_schedule.days_of_week = schedule["days_of_week"]

        # Parse settings
        campaign_settings = CampaignSettings()
        if settings:
            for key, value in settings.items():
                if hasattr(campaign_settings, key):
                    setattr(campaign_settings, key, value)

        campaign = Campaign(
            id=campaign_id,
            organization_id=organization_id,
            name=name,
            description=description,
            agent_id=agent_id,
            from_number=from_number,
            phone_number_id=phone_number_id,
            schedule=campaign_schedule,
            settings=campaign_settings,
            tags=tags or [],
            created_by=created_by,
            metadata=metadata or {},
        )

        # Store campaign
        async with self._lock:
            self._campaigns[campaign_id] = campaign

        await self._persist_campaign(campaign)

        logger.info(f"Created campaign: {campaign_id} ({name})")
        return campaign

    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get a campaign by ID."""
        async with self._lock:
            return self._campaigns.get(campaign_id)

    async def list_campaigns(
        self,
        organization_id: str,
        status: Optional[CampaignStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Campaign], int]:
        """List campaigns for an organization."""
        async with self._lock:
            campaigns = [
                c for c in self._campaigns.values()
                if c.organization_id == organization_id
            ]

        if status:
            campaigns = [c for c in campaigns if c.status == status]

        # Sort by created_at descending
        campaigns.sort(key=lambda c: c.created_at, reverse=True)

        total = len(campaigns)
        campaigns = campaigns[offset:offset + limit]

        return campaigns, total

    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Campaign]:
        """Update a campaign."""
        async with self._lock:
            if campaign_id not in self._campaigns:
                return None

            campaign = self._campaigns[campaign_id]

            # Only allow updates for draft campaigns
            if campaign.status != CampaignStatus.DRAFT and "status" not in updates:
                raise ValueError("Can only update draft campaigns")

            for key, value in updates.items():
                if key == "schedule" and isinstance(value, dict):
                    # Update schedule fields
                    for sk, sv in value.items():
                        if hasattr(campaign.schedule, sk):
                            setattr(campaign.schedule, sk, sv)
                elif key == "settings" and isinstance(value, dict):
                    # Update settings fields
                    for sk, sv in value.items():
                        if hasattr(campaign.settings, sk):
                            setattr(campaign.settings, sk, sv)
                elif hasattr(campaign, key):
                    setattr(campaign, key, value)

            campaign.updated_at = datetime.utcnow()
            self._campaigns[campaign_id] = campaign

        await self._persist_campaign(campaign)
        return campaign

    async def delete_campaign(self, campaign_id: str) -> bool:
        """Delete a campaign."""
        async with self._lock:
            if campaign_id not in self._campaigns:
                return False

            campaign = self._campaigns[campaign_id]

            # Can't delete running campaigns
            if campaign.status == CampaignStatus.RUNNING:
                raise ValueError("Cannot delete running campaign")

            del self._campaigns[campaign_id]

        # Remove from Redis
        if self.redis:
            await self.redis.hdel(f"campaigns:{campaign.organization_id}", campaign_id)
            await self.redis.delete(f"campaign_contacts:{campaign_id}")

        logger.info(f"Deleted campaign: {campaign_id}")
        return True

    # Campaign lifecycle

    async def start_campaign(self, campaign_id: str) -> Campaign:
        """Start a campaign."""
        async with self._lock:
            if campaign_id not in self._campaigns:
                raise ValueError("Campaign not found")

            campaign = self._campaigns[campaign_id]

            if not campaign.can_start:
                raise ValueError(f"Cannot start campaign in {campaign.status.value} status")

            if campaign.total_contacts == 0:
                raise ValueError("Campaign has no contacts")

            campaign.status = CampaignStatus.RUNNING
            campaign.started_at = datetime.utcnow()
            campaign.updated_at = datetime.utcnow()

            self._campaigns[campaign_id] = campaign

        await self._persist_campaign(campaign)

        # Notify callbacks
        for callback in self._on_campaign_started:
            try:
                await callback(campaign)
            except Exception as e:
                logger.error(f"Campaign started callback error: {e}")

        logger.info(f"Started campaign: {campaign_id}")
        return campaign

    async def pause_campaign(self, campaign_id: str) -> Campaign:
        """Pause a running campaign."""
        async with self._lock:
            if campaign_id not in self._campaigns:
                raise ValueError("Campaign not found")

            campaign = self._campaigns[campaign_id]

            if not campaign.can_pause:
                raise ValueError(f"Cannot pause campaign in {campaign.status.value} status")

            campaign.status = CampaignStatus.PAUSED
            campaign.paused_at = datetime.utcnow()
            campaign.updated_at = datetime.utcnow()

            self._campaigns[campaign_id] = campaign

        await self._persist_campaign(campaign)

        # Notify callbacks
        for callback in self._on_campaign_paused:
            try:
                await callback(campaign)
            except Exception as e:
                logger.error(f"Campaign paused callback error: {e}")

        logger.info(f"Paused campaign: {campaign_id}")
        return campaign

    async def resume_campaign(self, campaign_id: str) -> Campaign:
        """Resume a paused campaign."""
        async with self._lock:
            if campaign_id not in self._campaigns:
                raise ValueError("Campaign not found")

            campaign = self._campaigns[campaign_id]

            if campaign.status != CampaignStatus.PAUSED:
                raise ValueError("Campaign is not paused")

            campaign.status = CampaignStatus.RUNNING
            campaign.updated_at = datetime.utcnow()

            self._campaigns[campaign_id] = campaign

        await self._persist_campaign(campaign)

        logger.info(f"Resumed campaign: {campaign_id}")
        return campaign

    async def cancel_campaign(self, campaign_id: str) -> Campaign:
        """Cancel a campaign."""
        async with self._lock:
            if campaign_id not in self._campaigns:
                raise ValueError("Campaign not found")

            campaign = self._campaigns[campaign_id]

            if not campaign.can_cancel:
                raise ValueError(f"Cannot cancel campaign in {campaign.status.value} status")

            campaign.status = CampaignStatus.CANCELED
            campaign.canceled_at = datetime.utcnow()
            campaign.updated_at = datetime.utcnow()

            self._campaigns[campaign_id] = campaign

        await self._persist_campaign(campaign)

        logger.info(f"Canceled campaign: {campaign_id}")
        return campaign

    async def complete_campaign(self, campaign_id: str) -> Campaign:
        """Mark a campaign as completed."""
        async with self._lock:
            if campaign_id not in self._campaigns:
                raise ValueError("Campaign not found")

            campaign = self._campaigns[campaign_id]
            campaign.status = CampaignStatus.COMPLETED
            campaign.completed_at = datetime.utcnow()
            campaign.updated_at = datetime.utcnow()

            self._campaigns[campaign_id] = campaign

        await self._persist_campaign(campaign)

        # Notify callbacks
        for callback in self._on_campaign_completed:
            try:
                await callback(campaign)
            except Exception as e:
                logger.error(f"Campaign completed callback error: {e}")

        logger.info(f"Completed campaign: {campaign_id}")
        return campaign

    # Contact management

    async def add_contacts(
        self,
        campaign_id: str,
        contacts: List[Dict[str, Any]],
    ) -> ContactImportResult:
        """Add contacts to a campaign."""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        if campaign.status not in (CampaignStatus.DRAFT, CampaignStatus.PAUSED):
            raise ValueError("Can only add contacts to draft or paused campaigns")

        result = ContactImportResult()

        # Get existing phone numbers to check for duplicates
        existing_numbers = await self._get_existing_numbers(campaign_id)

        new_contacts = []
        for contact_data in contacts:
            phone_number = contact_data.get("phone_number", "")

            # Normalize phone number
            normalized = self._normalize_phone_number(phone_number)
            if not normalized:
                result.invalid_numbers += 1
                continue

            # Check for duplicates
            if normalized in existing_numbers:
                result.duplicates_skipped += 1
                continue

            contact = CampaignContact(
                id=f"cc_{uuid4().hex[:12]}",
                campaign_id=campaign_id,
                phone_number=normalized,
                name=contact_data.get("name"),
                email=contact_data.get("email"),
                custom_fields={
                    k: v for k, v in contact_data.items()
                    if k not in ("phone_number", "name", "email")
                },
            )

            new_contacts.append(contact)
            existing_numbers.add(normalized)
            result.total_imported += 1

        # Store contacts
        if new_contacts:
            await self._store_contacts(campaign_id, new_contacts)

            # Update campaign total
            async with self._lock:
                campaign.total_contacts += len(new_contacts)
                campaign.updated_at = datetime.utcnow()
                self._campaigns[campaign_id] = campaign

            await self._persist_campaign(campaign)

        return result

    async def import_contacts_csv(
        self,
        campaign_id: str,
        csv_content: str,
        phone_column: str = "phone_number",
        name_column: Optional[str] = "name",
        email_column: Optional[str] = "email",
    ) -> ContactImportResult:
        """Import contacts from CSV content."""
        contacts = []

        reader = csv.DictReader(io.StringIO(csv_content))
        for row in reader:
            if phone_column not in row:
                continue

            contact = {
                "phone_number": row[phone_column],
            }

            if name_column and name_column in row:
                contact["name"] = row[name_column]

            if email_column and email_column in row:
                contact["email"] = row[email_column]

            # Include all other columns as custom fields
            for key, value in row.items():
                if key not in (phone_column, name_column, email_column):
                    contact[key] = value

            contacts.append(contact)

        return await self.add_contacts(campaign_id, contacts)

    async def get_contacts(
        self,
        campaign_id: str,
        status: Optional[ContactStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[CampaignContact], int]:
        """Get contacts for a campaign."""
        if not self.redis:
            return [], 0

        key = f"campaign_contacts:{campaign_id}"

        # Get all contacts
        contact_data = await self.redis.lrange(key, 0, -1)

        contacts = []
        for data in contact_data:
            try:
                contact_dict = json.loads(data)
                contact_dict["status"] = ContactStatus(contact_dict["status"])
                if contact_dict.get("outcome"):
                    from .models import CallOutcome
                    contact_dict["outcome"] = CallOutcome(contact_dict["outcome"])

                for date_field in ("last_attempt_at", "next_attempt_at", "completed_at"):
                    if contact_dict.get(date_field):
                        contact_dict[date_field] = datetime.fromisoformat(contact_dict[date_field])

                contact = CampaignContact(**contact_dict)

                if status is None or contact.status == status:
                    contacts.append(contact)

            except Exception as e:
                logger.warning(f"Failed to parse contact: {e}")

        total = len(contacts)
        contacts = contacts[offset:offset + limit]

        return contacts, total

    async def get_contact(self, contact_id: str) -> Optional[CampaignContact]:
        """Get a specific contact."""
        if not self.redis:
            return None

        data = await self.redis.hget("campaign_contacts_by_id", contact_id)
        if not data:
            return None

        try:
            contact_dict = json.loads(data)
            contact_dict["status"] = ContactStatus(contact_dict["status"])
            return CampaignContact(**contact_dict)
        except Exception:
            return None

    async def update_contact(
        self,
        contact_id: str,
        updates: Dict[str, Any],
    ) -> Optional[CampaignContact]:
        """Update a contact."""
        contact = await self.get_contact(contact_id)
        if not contact:
            return None

        for key, value in updates.items():
            if hasattr(contact, key):
                setattr(contact, key, value)

        # Store updated contact
        if self.redis:
            await self.redis.hset(
                "campaign_contacts_by_id",
                contact_id,
                json.dumps(contact.to_dict()),
            )

        return contact

    async def remove_contact(self, campaign_id: str, contact_id: str) -> bool:
        """Remove a contact from a campaign."""
        contact = await self.get_contact(contact_id)
        if not contact or contact.campaign_id != campaign_id:
            return False

        if contact.status in (ContactStatus.IN_PROGRESS, ContactStatus.QUEUED):
            raise ValueError("Cannot remove contact that is being processed")

        if self.redis:
            await self.redis.hdel("campaign_contacts_by_id", contact_id)
            # Note: Also need to update the list, but that's more complex

        # Update campaign total
        campaign = await self.get_campaign(campaign_id)
        if campaign:
            async with self._lock:
                campaign.total_contacts = max(0, campaign.total_contacts - 1)
                self._campaigns[campaign_id] = campaign
            await self._persist_campaign(campaign)

        return True

    # Progress tracking

    async def get_progress(self, campaign_id: str) -> CampaignProgress:
        """Get campaign progress."""
        contacts, _ = await self.get_contacts(campaign_id, limit=100000)

        progress = CampaignProgress()
        progress.total = len(contacts)

        total_duration = 0
        for contact in contacts:
            if contact.status == ContactStatus.PENDING:
                progress.pending += 1
            elif contact.status == ContactStatus.QUEUED:
                progress.queued += 1
            elif contact.status == ContactStatus.IN_PROGRESS:
                progress.in_progress += 1
            elif contact.status == ContactStatus.COMPLETED:
                progress.completed += 1
                progress.successful += 1
                total_duration += contact.duration_seconds
            elif contact.status == ContactStatus.FAILED:
                progress.completed += 1
                progress.failed += 1
            elif contact.status == ContactStatus.NO_ANSWER:
                progress.completed += 1
                progress.no_answer += 1
            elif contact.status == ContactStatus.BUSY:
                progress.completed += 1
                progress.busy += 1
            elif contact.status == ContactStatus.VOICEMAIL:
                progress.completed += 1
                progress.voicemail += 1
            elif contact.status == ContactStatus.SKIPPED:
                progress.skipped += 1
            elif contact.status == ContactStatus.RETRY:
                progress.retry += 1

        if progress.total > 0:
            progress.percent_complete = (progress.completed / progress.total) * 100

        if progress.completed > 0:
            progress.success_rate = (progress.successful / progress.completed) * 100
            progress.avg_duration = total_duration / progress.completed

        progress.total_duration = total_duration

        # Estimate remaining time
        remaining_contacts = progress.total - progress.completed
        if progress.avg_duration > 0 and remaining_contacts > 0:
            # Account for concurrent calls
            campaign = await self.get_campaign(campaign_id)
            if campaign:
                concurrent = campaign.settings.max_concurrent_calls
                progress.estimated_remaining_minutes = (
                    (remaining_contacts * progress.avg_duration) / (concurrent * 60)
                )

        return progress

    # Export

    async def export_contacts(
        self,
        campaign_id: str,
        format: str = "csv",
        include_custom_fields: bool = True,
    ) -> str:
        """Export campaign contacts."""
        contacts, _ = await self.get_contacts(campaign_id, limit=100000)

        if format == "json":
            return json.dumps([c.to_dict() for c in contacts], indent=2)

        # CSV export
        output = io.StringIO()

        fieldnames = ["phone_number", "name", "email", "status", "outcome", "attempts", "duration_seconds"]
        if include_custom_fields and contacts:
            # Get all custom field keys
            custom_keys = set()
            for c in contacts:
                custom_keys.update(c.custom_fields.keys())
            fieldnames.extend(sorted(custom_keys))

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for contact in contacts:
            row = {
                "phone_number": contact.phone_number,
                "name": contact.name or "",
                "email": contact.email or "",
                "status": contact.status.value,
                "outcome": contact.outcome.value if contact.outcome else "",
                "attempts": contact.attempts,
                "duration_seconds": contact.duration_seconds,
            }

            if include_custom_fields:
                for key in contact.custom_fields:
                    row[key] = contact.custom_fields.get(key, "")

            writer.writerow(row)

        return output.getvalue()

    # Helper methods

    def _normalize_phone_number(self, phone: str) -> Optional[str]:
        """Normalize a phone number to E.164 format."""
        # Remove all non-digit characters except +
        cleaned = re.sub(r"[^\d+]", "", phone)

        if not cleaned:
            return None

        # Add + prefix if not present and starts with country code
        if not cleaned.startswith("+"):
            if cleaned.startswith("1") and len(cleaned) == 11:
                cleaned = "+" + cleaned
            elif len(cleaned) == 10:
                cleaned = "+1" + cleaned
            else:
                cleaned = "+" + cleaned

        # Validate length
        if len(cleaned) < 10 or len(cleaned) > 16:
            return None

        return cleaned

    async def _get_existing_numbers(self, campaign_id: str) -> set:
        """Get set of existing phone numbers for a campaign."""
        contacts, _ = await self.get_contacts(campaign_id, limit=100000)
        return {c.phone_number for c in contacts}

    async def _store_contacts(
        self,
        campaign_id: str,
        contacts: List[CampaignContact],
    ) -> None:
        """Store contacts in Redis."""
        if not self.redis:
            return

        pipeline = self.redis.pipeline()

        key = f"campaign_contacts:{campaign_id}"
        for contact in contacts:
            contact_json = json.dumps(contact.to_dict())
            pipeline.rpush(key, contact_json)
            pipeline.hset("campaign_contacts_by_id", contact.id, contact_json)

        await pipeline.execute()

    async def _persist_campaign(self, campaign: Campaign) -> None:
        """Persist campaign to Redis."""
        if not self.redis:
            return

        await self.redis.hset(
            f"campaigns:{campaign.organization_id}",
            campaign.id,
            json.dumps(campaign.to_dict()),
        )

    async def _load_campaigns(self) -> None:
        """Load campaigns from Redis."""
        if not self.redis:
            return

        async for key in self.redis.scan_iter(match="campaigns:*"):
            campaign_data = await self.redis.hgetall(key)

            for campaign_id, data in campaign_data.items():
                try:
                    campaign_dict = json.loads(data)
                    campaign_dict["status"] = CampaignStatus(campaign_dict["status"])

                    # Parse dates
                    for date_field in ("created_at", "updated_at", "started_at", "completed_at", "paused_at", "canceled_at"):
                        if campaign_dict.get(date_field):
                            campaign_dict[date_field] = datetime.fromisoformat(campaign_dict[date_field])

                    # Parse schedule
                    schedule_dict = campaign_dict.pop("schedule", {})
                    campaign_dict["schedule"] = CampaignSchedule(**schedule_dict) if schedule_dict else CampaignSchedule()

                    # Parse settings
                    settings_dict = campaign_dict.pop("settings", {})
                    campaign_dict["settings"] = CampaignSettings(**settings_dict) if settings_dict else CampaignSettings()

                    campaign = Campaign(**campaign_dict)

                    async with self._lock:
                        self._campaigns[campaign.id] = campaign

                except Exception as e:
                    logger.warning(f"Failed to load campaign {campaign_id}: {e}")

        logger.info(f"Loaded {len(self._campaigns)} campaigns")

    # Callbacks

    def on_campaign_started(self, callback: Callable[[Campaign], Any]) -> None:
        """Register callback for campaign started events."""
        self._on_campaign_started.append(callback)

    def on_campaign_completed(self, callback: Callable[[Campaign], Any]) -> None:
        """Register callback for campaign completed events."""
        self._on_campaign_completed.append(callback)

    def on_campaign_paused(self, callback: Callable[[Campaign], Any]) -> None:
        """Register callback for campaign paused events."""
        self._on_campaign_paused.append(callback)
