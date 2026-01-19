"""Campaign execution engine."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

from .models import (
    Campaign,
    CampaignContact,
    CampaignStatus,
    ContactStatus,
    CampaignProgress,
    CampaignCallResult,
    CallOutcome,
)
from .manager import CampaignManager

logger = logging.getLogger(__name__)


@dataclass
class CallRequest:
    """Request to make a call."""
    contact: CampaignContact
    campaign: Campaign
    attempt_number: int


class CampaignExecutor:
    """
    Executes campaign calls with rate limiting and concurrency control.

    Features:
    - Rate limiting (calls per minute)
    - Concurrency control (max concurrent calls)
    - Retry logic with backoff
    - Schedule adherence
    - Progress tracking
    """

    def __init__(
        self,
        manager: CampaignManager,
        max_global_concurrent: int = 1000,
    ):
        self.manager = manager
        self.max_global_concurrent = max_global_concurrent

        # Per-campaign state
        self._campaign_tasks: Dict[str, asyncio.Task] = {}
        self._campaign_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._campaign_queues: Dict[str, asyncio.Queue] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}

        # Global rate limiting
        self._global_semaphore = asyncio.Semaphore(max_global_concurrent)

        # Call provider callback
        self._make_call: Optional[Callable[[CallRequest], CampaignCallResult]] = None

        # Event callbacks
        self._on_call_completed: List[Callable[[CampaignCallResult], Any]] = []

    def set_call_provider(
        self,
        provider: Callable[[CallRequest], CampaignCallResult],
    ) -> None:
        """Set the callback for making calls."""
        self._make_call = provider

    def on_call_completed(
        self,
        callback: Callable[[CampaignCallResult], Any],
    ) -> None:
        """Register callback for call completion."""
        self._on_call_completed.append(callback)

    async def start_campaign_execution(self, campaign_id: str) -> None:
        """Start executing a campaign."""
        if campaign_id in self._campaign_tasks:
            logger.warning(f"Campaign {campaign_id} already executing")
            return

        campaign = await self.manager.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        if campaign.status != CampaignStatus.RUNNING:
            raise ValueError(f"Campaign is not running (status: {campaign.status.value})")

        # Initialize campaign state
        self._campaign_semaphores[campaign_id] = asyncio.Semaphore(
            campaign.settings.max_concurrent_calls
        )
        self._campaign_queues[campaign_id] = asyncio.Queue()
        self._stop_events[campaign_id] = asyncio.Event()

        # Start execution task
        task = asyncio.create_task(self._execute_campaign(campaign_id))
        self._campaign_tasks[campaign_id] = task

        logger.info(f"Started campaign execution: {campaign_id}")

    async def stop_campaign_execution(self, campaign_id: str) -> None:
        """Stop executing a campaign."""
        if campaign_id not in self._campaign_tasks:
            return

        # Signal stop
        self._stop_events[campaign_id].set()

        # Wait for task to complete
        try:
            await asyncio.wait_for(self._campaign_tasks[campaign_id], timeout=30)
        except asyncio.TimeoutError:
            self._campaign_tasks[campaign_id].cancel()

        # Cleanup
        del self._campaign_tasks[campaign_id]
        del self._campaign_semaphores[campaign_id]
        del self._campaign_queues[campaign_id]
        del self._stop_events[campaign_id]

        logger.info(f"Stopped campaign execution: {campaign_id}")

    async def _execute_campaign(self, campaign_id: str) -> None:
        """Main campaign execution loop."""
        stop_event = self._stop_events[campaign_id]

        try:
            # Get campaign
            campaign = await self.manager.get_campaign(campaign_id)
            if not campaign:
                return

            # Get pending contacts
            contacts, _ = await self.manager.get_contacts(
                campaign_id,
                status=ContactStatus.PENDING,
                limit=100000,
            )

            logger.info(f"Campaign {campaign_id}: {len(contacts)} contacts to process")

            # Rate limiter
            rate_limiter = RateLimiter(
                calls_per_minute=campaign.settings.calls_per_minute
            )

            # Worker tasks
            worker_count = min(
                campaign.settings.max_concurrent_calls,
                len(contacts),
            )
            workers = [
                asyncio.create_task(
                    self._call_worker(campaign_id, rate_limiter)
                )
                for _ in range(worker_count)
            ]

            # Enqueue contacts
            queue = self._campaign_queues[campaign_id]
            for contact in contacts:
                if stop_event.is_set():
                    break
                await queue.put(contact)

            # Signal workers to stop
            for _ in range(worker_count):
                await queue.put(None)

            # Wait for workers
            await asyncio.gather(*workers, return_exceptions=True)

            # Check for retries
            retry_contacts, _ = await self.manager.get_contacts(
                campaign_id,
                status=ContactStatus.RETRY,
                limit=100000,
            )

            if retry_contacts and not stop_event.is_set():
                logger.info(f"Campaign {campaign_id}: processing {len(retry_contacts)} retries")
                for contact in retry_contacts:
                    if stop_event.is_set():
                        break
                    await queue.put(contact)

                for _ in range(worker_count):
                    await queue.put(None)

                await asyncio.gather(*workers, return_exceptions=True)

            # Mark campaign complete if all done
            if not stop_event.is_set():
                progress = await self.manager.get_progress(campaign_id)
                if progress.pending == 0 and progress.queued == 0 and progress.in_progress == 0:
                    await self.manager.complete_campaign(campaign_id)

        except asyncio.CancelledError:
            logger.info(f"Campaign {campaign_id} execution cancelled")
        except Exception as e:
            logger.exception(f"Campaign {campaign_id} execution error: {e}")

    async def _call_worker(
        self,
        campaign_id: str,
        rate_limiter: "RateLimiter",
    ) -> None:
        """Worker that processes calls from the queue."""
        queue = self._campaign_queues[campaign_id]
        semaphore = self._campaign_semaphores[campaign_id]
        stop_event = self._stop_events[campaign_id]

        while not stop_event.is_set():
            # Get next contact
            contact = await queue.get()
            if contact is None:
                break

            # Check schedule
            campaign = await self.manager.get_campaign(campaign_id)
            if not campaign or campaign.status != CampaignStatus.RUNNING:
                break

            if not self._is_within_schedule(campaign):
                # Re-queue for later
                await queue.put(contact)
                await asyncio.sleep(60)  # Wait a minute
                continue

            # Rate limit
            await rate_limiter.acquire()

            # Concurrency limit
            async with semaphore:
                async with self._global_semaphore:
                    try:
                        await self._process_contact(campaign, contact)
                    except Exception as e:
                        logger.error(f"Error processing contact {contact.id}: {e}")

    async def _process_contact(
        self,
        campaign: Campaign,
        contact: CampaignContact,
    ) -> None:
        """Process a single contact."""
        if not self._make_call:
            logger.error("No call provider configured")
            return

        # Update contact status
        contact.status = ContactStatus.IN_PROGRESS
        contact.attempts += 1
        contact.last_attempt_at = datetime.utcnow()
        await self.manager.update_contact(contact.id, {
            "status": contact.status,
            "attempts": contact.attempts,
            "last_attempt_at": contact.last_attempt_at,
        })

        # Make the call
        request = CallRequest(
            contact=contact,
            campaign=campaign,
            attempt_number=contact.attempts,
        )

        try:
            result = await self._make_call(request)
        except Exception as e:
            logger.error(f"Call failed for contact {contact.id}: {e}")
            result = CampaignCallResult(
                contact_id=contact.id,
                call_id="",
                status=ContactStatus.FAILED,
                outcome=CallOutcome.FAILED,
                duration_seconds=0,
                error_message=str(e),
            )

        # Update contact with result
        updates = {
            "status": result.status,
            "outcome": result.outcome,
            "call_id": result.call_id,
            "duration_seconds": result.duration_seconds,
        }

        if result.status == ContactStatus.COMPLETED:
            updates["completed_at"] = datetime.utcnow()
        elif contact.can_retry(campaign.settings.max_attempts_per_contact):
            updates["status"] = ContactStatus.RETRY
            updates["next_attempt_at"] = datetime.utcnow() + timedelta(
                minutes=campaign.settings.retry_delay_minutes
            )

        if result.error_message:
            updates["error_message"] = result.error_message

        await self.manager.update_contact(contact.id, updates)

        # Update campaign stats
        campaign_updates = {
            "contacts_processed": campaign.contacts_processed + 1,
            "total_duration_seconds": campaign.total_duration_seconds + result.duration_seconds,
        }

        if result.status == ContactStatus.COMPLETED:
            campaign_updates["successful_calls"] = campaign.successful_calls + 1
        elif result.status == ContactStatus.FAILED:
            campaign_updates["failed_calls"] = campaign.failed_calls + 1

        await self.manager.update_campaign(campaign.id, campaign_updates)

        # Notify callbacks
        for callback in self._on_call_completed:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Call completed callback error: {e}")

    def _is_within_schedule(self, campaign: Campaign) -> bool:
        """Check if current time is within campaign schedule."""
        schedule = campaign.schedule
        now = datetime.utcnow()

        # Check overall campaign dates
        if schedule.start_time and now < schedule.start_time:
            return False
        if schedule.end_time and now > schedule.end_time:
            return False

        # Check day of week
        if schedule.days_of_week:
            if now.weekday() not in schedule.days_of_week:
                return False

        # Check daily time window
        if schedule.daily_start_time and schedule.daily_end_time:
            current_time = now.time()
            if current_time < schedule.daily_start_time:
                return False
            if current_time > schedule.daily_end_time:
                return False

        return True

    async def get_executing_campaigns(self) -> List[str]:
        """Get list of currently executing campaign IDs."""
        return list(self._campaign_tasks.keys())


class RateLimiter:
    """Simple rate limiter for calls per minute."""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until we can make another call."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait_time = max(0, self.last_call + self.interval - now)

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self.last_call = asyncio.get_event_loop().time()
