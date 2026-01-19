"""Campaign scheduling service."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz

from .models import Campaign, CampaignStatus, CampaignSchedule
from .manager import CampaignManager
from .executor import CampaignExecutor

logger = logging.getLogger(__name__)


class CampaignScheduler:
    """
    Schedules campaign execution based on configured schedules.

    Features:
    - Automatic start/stop based on schedule
    - Timezone-aware scheduling
    - Holiday calendar support
    - Daily time window enforcement
    """

    CHECK_INTERVAL = 60  # Check every minute

    def __init__(
        self,
        manager: CampaignManager,
        executor: CampaignExecutor,
    ):
        self.manager = manager
        self.executor = executor

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._scheduled_campaigns: Dict[str, datetime] = {}

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Campaign scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Campaign scheduler stopped")

    async def schedule_campaign(
        self,
        campaign_id: str,
        start_time: datetime,
    ) -> None:
        """Schedule a campaign to start at a specific time."""
        campaign = await self.manager.get_campaign(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        if campaign.status != CampaignStatus.DRAFT:
            raise ValueError("Only draft campaigns can be scheduled")

        # Update campaign
        await self.manager.update_campaign(campaign_id, {
            "status": CampaignStatus.SCHEDULED,
            "schedule": {"start_time": start_time.isoformat()},
        })

        self._scheduled_campaigns[campaign_id] = start_time
        logger.info(f"Scheduled campaign {campaign_id} for {start_time}")

    async def unschedule_campaign(self, campaign_id: str) -> None:
        """Remove a campaign from the schedule."""
        if campaign_id in self._scheduled_campaigns:
            del self._scheduled_campaigns[campaign_id]

        campaign = await self.manager.get_campaign(campaign_id)
        if campaign and campaign.status == CampaignStatus.SCHEDULED:
            await self.manager.update_campaign(campaign_id, {
                "status": CampaignStatus.DRAFT,
            })

        logger.info(f"Unscheduled campaign {campaign_id}")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_schedules()
                await asyncio.sleep(self.CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(self.CHECK_INTERVAL)

    async def _check_schedules(self) -> None:
        """Check all scheduled campaigns and start/stop as needed."""
        now = datetime.utcnow()

        # Get all scheduled campaigns
        for org_id in await self._get_organization_ids():
            campaigns, _ = await self.manager.list_campaigns(org_id)

            for campaign in campaigns:
                await self._check_campaign_schedule(campaign, now)

    async def _check_campaign_schedule(
        self,
        campaign: Campaign,
        now: datetime,
    ) -> None:
        """Check and handle a single campaign's schedule."""
        schedule = campaign.schedule

        # Convert now to campaign timezone
        tz = pytz.timezone(schedule.timezone)
        local_now = datetime.now(tz)

        if campaign.status == CampaignStatus.SCHEDULED:
            # Check if it's time to start
            if self._should_start(campaign, local_now):
                try:
                    await self.manager.start_campaign(campaign.id)
                    await self.executor.start_campaign_execution(campaign.id)
                    logger.info(f"Auto-started campaign: {campaign.id}")
                except Exception as e:
                    logger.error(f"Failed to auto-start campaign {campaign.id}: {e}")

        elif campaign.status == CampaignStatus.RUNNING:
            # Check if it's time to pause (outside daily window)
            if not self._is_within_daily_window(schedule, local_now):
                logger.info(f"Pausing campaign {campaign.id}: outside daily window")
                await self.executor.stop_campaign_execution(campaign.id)
                await self.manager.pause_campaign(campaign.id)

            # Check if campaign has ended
            elif schedule.end_time and now > schedule.end_time:
                logger.info(f"Ending campaign {campaign.id}: past end time")
                await self.executor.stop_campaign_execution(campaign.id)
                await self.manager.complete_campaign(campaign.id)

        elif campaign.status == CampaignStatus.PAUSED:
            # Check if it's time to resume (back in daily window)
            if self._should_resume(campaign, local_now):
                try:
                    await self.manager.resume_campaign(campaign.id)
                    await self.executor.start_campaign_execution(campaign.id)
                    logger.info(f"Auto-resumed campaign: {campaign.id}")
                except Exception as e:
                    logger.error(f"Failed to auto-resume campaign {campaign.id}: {e}")

    def _should_start(self, campaign: Campaign, local_now: datetime) -> bool:
        """Check if a scheduled campaign should start."""
        schedule = campaign.schedule

        # Check start time
        if schedule.start_time:
            if local_now.replace(tzinfo=None) < schedule.start_time:
                return False

        # Check within daily window
        if not self._is_within_daily_window(schedule, local_now):
            return False

        # Check day of week
        if schedule.days_of_week:
            if local_now.weekday() not in schedule.days_of_week:
                return False

        # Check holidays
        if schedule.respect_holidays:
            if self._is_holiday(local_now, schedule.holiday_calendar):
                return False

        return True

    def _should_resume(self, campaign: Campaign, local_now: datetime) -> bool:
        """Check if a paused campaign should resume."""
        schedule = campaign.schedule

        # Only resume if it was auto-paused (not manually paused)
        # This is a simplified check - in production you'd track the pause reason

        # Check still within overall schedule
        if schedule.end_time and datetime.utcnow() > schedule.end_time:
            return False

        # Check within daily window
        if not self._is_within_daily_window(schedule, local_now):
            return False

        # Check day of week
        if schedule.days_of_week:
            if local_now.weekday() not in schedule.days_of_week:
                return False

        return True

    def _is_within_daily_window(
        self,
        schedule: CampaignSchedule,
        local_now: datetime,
    ) -> bool:
        """Check if current time is within the daily calling window."""
        if not schedule.daily_start_time or not schedule.daily_end_time:
            return True

        current_time = local_now.time()
        return schedule.daily_start_time <= current_time <= schedule.daily_end_time

    def _is_holiday(self, date: datetime, calendar: str) -> bool:
        """Check if a date is a holiday."""
        # Simplified holiday check - in production use a proper holiday library
        # like python-holidays

        us_holidays = {
            (1, 1),    # New Year's Day
            (7, 4),    # Independence Day
            (12, 25),  # Christmas
            # Add more as needed
        }

        if calendar == "US":
            return (date.month, date.day) in us_holidays

        return False

    async def _get_organization_ids(self) -> List[str]:
        """Get list of organization IDs with campaigns."""
        # This is a simplified implementation
        # In production, you'd query from database
        org_ids = set()
        for campaign in self.manager._campaigns.values():
            org_ids.add(campaign.organization_id)
        return list(org_ids)

    async def get_scheduled_campaigns(self) -> List[Dict]:
        """Get list of scheduled campaigns with their start times."""
        result = []
        for campaign_id, start_time in self._scheduled_campaigns.items():
            campaign = await self.manager.get_campaign(campaign_id)
            if campaign:
                result.append({
                    "campaign_id": campaign_id,
                    "name": campaign.name,
                    "start_time": start_time.isoformat(),
                    "status": campaign.status.value,
                })
        return result

    async def get_next_scheduled_time(self, campaign_id: str) -> Optional[datetime]:
        """Get the next scheduled execution time for a campaign."""
        campaign = await self.manager.get_campaign(campaign_id)
        if not campaign:
            return None

        schedule = campaign.schedule

        if campaign.status == CampaignStatus.SCHEDULED:
            return schedule.start_time

        if campaign.status == CampaignStatus.PAUSED:
            # Calculate next available time slot
            tz = pytz.timezone(schedule.timezone)
            local_now = datetime.now(tz)

            # Find next valid day
            for days_ahead in range(7):
                check_date = local_now + timedelta(days=days_ahead)

                if schedule.days_of_week and check_date.weekday() not in schedule.days_of_week:
                    continue

                if schedule.respect_holidays and self._is_holiday(check_date, schedule.holiday_calendar):
                    continue

                # Found a valid day
                if schedule.daily_start_time:
                    start = check_date.replace(
                        hour=schedule.daily_start_time.hour,
                        minute=schedule.daily_start_time.minute,
                        second=0,
                        microsecond=0,
                    )
                    if start > local_now:
                        return start.astimezone(pytz.UTC).replace(tzinfo=None)

        return None
