"""
Notifications Service Module

This module provides comprehensive notification delivery services including
email, SMS, push notifications, and in-app messaging.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .base import (
    # Enums
    NotificationChannel,
    NotificationPriority,
    NotificationStatus,
    NotificationType,
    TemplateType,
    # Types
    EmailProviderConfig,
    Notification,
    NotificationContent,
    NotificationDelivery,
    NotificationPreference,
    NotificationRecipient,
    NotificationSubscription,
    NotificationTemplate,
    PushProviderConfig,
    SMSProviderConfig,
    # Exceptions
    DeliveryError,
    NotificationError,
    ProviderError,
    RecipientNotFoundError,
    TemplateNotFoundError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Template Manager
# =============================================================================


class TemplateManager:
    """Manages notification templates."""

    def __init__(self):
        self._templates: Dict[str, NotificationTemplate] = {}
        self._templates_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._default_templates: Dict[str, str] = {}  # type -> template_id

    async def create_template(
        self,
        organization_id: str,
        name: str,
        notification_type: NotificationType,
        template_type: TemplateType = TemplateType.EMAIL_HTML,
        subject: str = "",
        body: str = "",
        html_body: Optional[str] = None,
        variables: Optional[List[str]] = None,
        is_default: bool = False,
        description: str = "",
    ) -> NotificationTemplate:
        """Create a notification template."""
        template = NotificationTemplate(
            id=f"tmpl_{uuid.uuid4().hex[:20]}",
            organization_id=organization_id,
            name=name,
            notification_type=notification_type,
            template_type=template_type,
            subject=subject,
            body=body,
            html_body=html_body,
            variables=variables or [],
            is_default=is_default,
            description=description,
        )

        self._templates[template.id] = template
        self._templates_by_org[organization_id].add(template.id)

        if is_default:
            key = f"{organization_id}:{notification_type.value}:{template_type.value}"
            self._default_templates[key] = template.id

        logger.info(f"Created template {template.id}: {name}")

        return template

    async def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """Get template by ID."""
        return self._templates.get(template_id)

    async def get_default_template(
        self,
        organization_id: str,
        notification_type: NotificationType,
        template_type: TemplateType = TemplateType.EMAIL_HTML,
    ) -> Optional[NotificationTemplate]:
        """Get default template for type."""
        key = f"{organization_id}:{notification_type.value}:{template_type.value}"
        template_id = self._default_templates.get(key)
        if template_id:
            return self._templates.get(template_id)
        return None

    async def list_templates(
        self,
        organization_id: str,
        notification_type: Optional[NotificationType] = None,
        template_type: Optional[TemplateType] = None,
        active_only: bool = True,
    ) -> List[NotificationTemplate]:
        """List templates for organization."""
        template_ids = self._templates_by_org.get(organization_id, set())
        templates = []

        for template_id in template_ids:
            template = self._templates.get(template_id)
            if not template:
                continue
            if active_only and not template.is_active:
                continue
            if notification_type and template.notification_type != notification_type:
                continue
            if template_type and template.template_type != template_type:
                continue
            templates.append(template)

        return sorted(templates, key=lambda t: t.name)

    async def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        html_body: Optional[str] = None,
        variables: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        is_default: Optional[bool] = None,
    ) -> NotificationTemplate:
        """Update a template."""
        template = await self.get_template(template_id)
        if not template:
            raise TemplateNotFoundError(f"Template {template_id} not found")

        if name:
            template.name = name
        if subject is not None:
            template.subject = subject
        if body is not None:
            template.body = body
        if html_body is not None:
            template.html_body = html_body
        if variables is not None:
            template.variables = variables
        if is_active is not None:
            template.is_active = is_active
        if is_default is not None:
            template.is_default = is_default
            if is_default:
                key = f"{template.organization_id}:{template.notification_type.value}:{template.template_type.value}"
                self._default_templates[key] = template.id

        template.updated_at = datetime.utcnow()

        return template

    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        template = self._templates.get(template_id)
        if not template:
            return False

        del self._templates[template_id]
        self._templates_by_org[template.organization_id].discard(template_id)

        return True

    def create_default_templates(self, organization_id: str) -> None:
        """Create default system templates."""
        # Will create templates asynchronously
        asyncio.create_task(self._create_default_templates_async(organization_id))

    async def _create_default_templates_async(self, organization_id: str) -> None:
        """Create default templates asynchronously."""
        default_templates = [
            {
                "name": "Welcome Email",
                "notification_type": NotificationType.WELCOME,
                "template_type": TemplateType.EMAIL_HTML,
                "subject": "Welcome to Voice Agent Platform!",
                "body": "Hello {{name}},\n\nWelcome to our platform! Your account has been created successfully.\n\nBest regards,\nThe Team",
                "html_body": "<h1>Welcome, {{name}}!</h1><p>Your account has been created successfully.</p>",
                "variables": ["name"],
            },
            {
                "name": "Password Reset",
                "notification_type": NotificationType.PASSWORD_RESET,
                "template_type": TemplateType.EMAIL_HTML,
                "subject": "Reset Your Password",
                "body": "Hello {{name}},\n\nClick here to reset your password: {{reset_link}}\n\nThis link expires in 24 hours.",
                "variables": ["name", "reset_link"],
            },
            {
                "name": "Call Missed Alert",
                "notification_type": NotificationType.CALL_MISSED,
                "template_type": TemplateType.EMAIL_HTML,
                "subject": "Missed Call from {{caller_phone}}",
                "body": "You missed a call from {{caller_phone}} at {{call_time}}.",
                "variables": ["caller_phone", "call_time"],
            },
            {
                "name": "Usage Threshold Alert",
                "notification_type": NotificationType.USAGE_THRESHOLD,
                "template_type": TemplateType.EMAIL_HTML,
                "subject": "Usage Alert: {{percent}}% of quota used",
                "body": "You have used {{percent}}% of your monthly quota. Current usage: {{current}} of {{limit}}.",
                "variables": ["percent", "current", "limit"],
            },
        ]

        for tmpl in default_templates:
            await self.create_template(
                organization_id=organization_id,
                is_default=True,
                **tmpl,
            )


# =============================================================================
# Recipient Manager
# =============================================================================


class RecipientManager:
    """Manages notification recipients."""

    def __init__(self):
        self._recipients: Dict[str, NotificationRecipient] = {}
        self._recipients_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._recipients_by_user: Dict[str, str] = {}  # user_id -> recipient_id

    async def create_recipient(
        self,
        organization_id: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        user_id: Optional[str] = None,
        name: str = "",
        preferred_channel: NotificationChannel = NotificationChannel.EMAIL,
        enabled_channels: Optional[List[NotificationChannel]] = None,
        timezone: str = "UTC",
    ) -> NotificationRecipient:
        """Create a notification recipient."""
        recipient = NotificationRecipient(
            id=f"rcpt_{uuid.uuid4().hex[:20]}",
            organization_id=organization_id,
            email=email,
            phone=phone,
            user_id=user_id,
            name=name,
            preferred_channel=preferred_channel,
            enabled_channels=enabled_channels or [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            timezone=timezone,
        )

        self._recipients[recipient.id] = recipient
        self._recipients_by_org[organization_id].add(recipient.id)
        if user_id:
            self._recipients_by_user[user_id] = recipient.id

        logger.info(f"Created recipient {recipient.id}")

        return recipient

    async def get_recipient(self, recipient_id: str) -> Optional[NotificationRecipient]:
        """Get recipient by ID."""
        return self._recipients.get(recipient_id)

    async def get_recipient_by_user(self, user_id: str) -> Optional[NotificationRecipient]:
        """Get recipient by user ID."""
        recipient_id = self._recipients_by_user.get(user_id)
        if recipient_id:
            return self._recipients.get(recipient_id)
        return None

    async def get_recipient_by_email(
        self,
        organization_id: str,
        email: str,
    ) -> Optional[NotificationRecipient]:
        """Get recipient by email."""
        for recipient_id in self._recipients_by_org.get(organization_id, set()):
            recipient = self._recipients.get(recipient_id)
            if recipient and recipient.email == email:
                return recipient
        return None

    async def list_recipients(
        self,
        organization_id: str,
        channel: Optional[NotificationChannel] = None,
    ) -> List[NotificationRecipient]:
        """List recipients for organization."""
        recipient_ids = self._recipients_by_org.get(organization_id, set())
        recipients = []

        for recipient_id in recipient_ids:
            recipient = self._recipients.get(recipient_id)
            if not recipient:
                continue
            if channel and not recipient.can_receive(channel):
                continue
            recipients.append(recipient)

        return recipients

    async def update_recipient(
        self,
        recipient_id: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        push_token: Optional[str] = None,
        name: Optional[str] = None,
        enabled_channels: Optional[List[NotificationChannel]] = None,
        timezone: Optional[str] = None,
    ) -> NotificationRecipient:
        """Update a recipient."""
        recipient = await self.get_recipient(recipient_id)
        if not recipient:
            raise RecipientNotFoundError(f"Recipient {recipient_id} not found")

        if email is not None:
            recipient.email = email
        if phone is not None:
            recipient.phone = phone
        if push_token is not None:
            recipient.push_token = push_token
        if name is not None:
            recipient.name = name
        if enabled_channels is not None:
            recipient.enabled_channels = enabled_channels
        if timezone is not None:
            recipient.timezone = timezone

        return recipient

    async def delete_recipient(self, recipient_id: str) -> bool:
        """Delete a recipient."""
        recipient = self._recipients.get(recipient_id)
        if not recipient:
            return False

        del self._recipients[recipient_id]
        self._recipients_by_org[recipient.organization_id].discard(recipient_id)
        if recipient.user_id and recipient.user_id in self._recipients_by_user:
            del self._recipients_by_user[recipient.user_id]

        return True


# =============================================================================
# Notification Providers
# =============================================================================


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""

    @property
    @abstractmethod
    def channel(self) -> NotificationChannel:
        """Get provider channel."""
        pass

    @abstractmethod
    async def send(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
        notification: Notification,
    ) -> NotificationDelivery:
        """Send notification."""
        pass


class EmailProvider(NotificationProvider):
    """Email notification provider."""

    def __init__(self, config: EmailProviderConfig):
        self.config = config

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.EMAIL

    async def send(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
        notification: Notification,
    ) -> NotificationDelivery:
        """Send email notification."""
        delivery = NotificationDelivery(
            id=f"dlvr_{uuid.uuid4().hex[:20]}",
            notification_id=notification.id,
            recipient_id=recipient.id,
            channel=NotificationChannel.EMAIL,
            destination=recipient.email or "",
            provider=self.config.provider,
        )

        if not recipient.email:
            delivery.status = NotificationStatus.FAILED
            delivery.error_message = "No email address"
            return delivery

        try:
            # Simulate sending (in real implementation, use actual email service)
            if self.config.provider == "smtp":
                await self._send_smtp(recipient, content)
            elif self.config.provider == "sendgrid":
                await self._send_sendgrid(recipient, content)
            elif self.config.provider == "ses":
                await self._send_ses(recipient, content)

            delivery.status = NotificationStatus.SENT
            delivery.sent_at = datetime.utcnow()
            delivery.provider_message_id = f"msg_{uuid.uuid4().hex[:16]}"

            logger.info(f"Email sent to {recipient.email}")

        except Exception as e:
            delivery.status = NotificationStatus.FAILED
            delivery.error_message = str(e)
            logger.error(f"Failed to send email to {recipient.email}: {e}")

        delivery.attempts += 1
        delivery.last_attempt_at = datetime.utcnow()

        return delivery

    async def _send_smtp(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
    ) -> None:
        """Send via SMTP."""
        # In real implementation:
        # import smtplib
        # from email.mime.multipart import MIMEMultipart
        # from email.mime.text import MIMEText
        pass

    async def _send_sendgrid(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
    ) -> None:
        """Send via SendGrid."""
        # In real implementation, use SendGrid API
        pass

    async def _send_ses(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
    ) -> None:
        """Send via AWS SES."""
        # In real implementation, use boto3
        pass


class SMSProvider(NotificationProvider):
    """SMS notification provider."""

    def __init__(self, config: SMSProviderConfig):
        self.config = config

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.SMS

    async def send(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
        notification: Notification,
    ) -> NotificationDelivery:
        """Send SMS notification."""
        delivery = NotificationDelivery(
            id=f"dlvr_{uuid.uuid4().hex[:20]}",
            notification_id=notification.id,
            recipient_id=recipient.id,
            channel=NotificationChannel.SMS,
            destination=recipient.phone or "",
            provider=self.config.provider,
        )

        if not recipient.phone:
            delivery.status = NotificationStatus.FAILED
            delivery.error_message = "No phone number"
            return delivery

        try:
            if self.config.provider == "twilio":
                await self._send_twilio(recipient, content)
            elif self.config.provider == "nexmo":
                await self._send_nexmo(recipient, content)

            delivery.status = NotificationStatus.SENT
            delivery.sent_at = datetime.utcnow()
            delivery.provider_message_id = f"sms_{uuid.uuid4().hex[:16]}"

            logger.info(f"SMS sent to {recipient.phone}")

        except Exception as e:
            delivery.status = NotificationStatus.FAILED
            delivery.error_message = str(e)
            logger.error(f"Failed to send SMS to {recipient.phone}: {e}")

        delivery.attempts += 1
        delivery.last_attempt_at = datetime.utcnow()

        return delivery

    async def _send_twilio(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
    ) -> None:
        """Send via Twilio."""
        # In real implementation, use Twilio API
        pass

    async def _send_nexmo(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
    ) -> None:
        """Send via Nexmo/Vonage."""
        # In real implementation, use Vonage API
        pass


class PushProvider(NotificationProvider):
    """Push notification provider."""

    def __init__(self, config: PushProviderConfig):
        self.config = config

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.PUSH

    async def send(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
        notification: Notification,
    ) -> NotificationDelivery:
        """Send push notification."""
        delivery = NotificationDelivery(
            id=f"dlvr_{uuid.uuid4().hex[:20]}",
            notification_id=notification.id,
            recipient_id=recipient.id,
            channel=NotificationChannel.PUSH,
            destination=recipient.push_token or "",
            provider=self.config.provider,
        )

        if not recipient.push_token:
            delivery.status = NotificationStatus.FAILED
            delivery.error_message = "No push token"
            return delivery

        try:
            if self.config.provider == "firebase":
                await self._send_firebase(recipient, content)

            delivery.status = NotificationStatus.SENT
            delivery.sent_at = datetime.utcnow()
            delivery.provider_message_id = f"push_{uuid.uuid4().hex[:16]}"

            logger.info(f"Push sent to {recipient.push_token[:20]}...")

        except Exception as e:
            delivery.status = NotificationStatus.FAILED
            delivery.error_message = str(e)
            logger.error(f"Failed to send push notification: {e}")

        delivery.attempts += 1
        delivery.last_attempt_at = datetime.utcnow()

        return delivery

    async def _send_firebase(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
    ) -> None:
        """Send via Firebase Cloud Messaging."""
        # In real implementation, use Firebase Admin SDK
        pass


class InAppProvider(NotificationProvider):
    """In-app notification provider."""

    def __init__(self):
        self._notifications: Dict[str, List[NotificationDelivery]] = defaultdict(list)

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.IN_APP

    async def send(
        self,
        recipient: NotificationRecipient,
        content: NotificationContent,
        notification: Notification,
    ) -> NotificationDelivery:
        """Send in-app notification."""
        delivery = NotificationDelivery(
            id=f"dlvr_{uuid.uuid4().hex[:20]}",
            notification_id=notification.id,
            recipient_id=recipient.id,
            channel=NotificationChannel.IN_APP,
            destination=recipient.user_id or "",
            provider="in_app",
        )

        if not recipient.user_id:
            delivery.status = NotificationStatus.FAILED
            delivery.error_message = "No user ID"
            return delivery

        # Store for retrieval
        delivery.status = NotificationStatus.DELIVERED
        delivery.sent_at = datetime.utcnow()
        delivery.delivered_at = datetime.utcnow()

        self._notifications[recipient.user_id].append(delivery)

        logger.info(f"In-app notification delivered to user {recipient.user_id}")

        return delivery

    async def get_unread(self, user_id: str) -> List[NotificationDelivery]:
        """Get unread notifications for user."""
        deliveries = self._notifications.get(user_id, [])
        return [d for d in deliveries if not d.opened_at]

    async def mark_read(self, delivery_id: str, user_id: str) -> bool:
        """Mark notification as read."""
        deliveries = self._notifications.get(user_id, [])
        for delivery in deliveries:
            if delivery.id == delivery_id:
                delivery.opened_at = datetime.utcnow()
                return True
        return False


# =============================================================================
# Notification Queue
# =============================================================================


class NotificationQueue:
    """Queue for processing notifications."""

    def __init__(self):
        self._queue: List[Notification] = []
        self._processing: Set[str] = set()
        self._retry_queue: Dict[str, Tuple[Notification, int]] = {}
        self._max_retries = 3
        self._retry_delay_seconds = 60

    async def enqueue(
        self,
        notification: Notification,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> None:
        """Add notification to queue."""
        notification.priority = priority
        notification.status = NotificationStatus.QUEUED

        # Insert based on priority
        if priority == NotificationPriority.CRITICAL:
            self._queue.insert(0, notification)
        else:
            self._queue.append(notification)

        logger.debug(f"Queued notification {notification.id}")

    async def dequeue(self) -> Optional[Notification]:
        """Get next notification from queue."""
        if not self._queue:
            return None

        # Sort by priority before dequeuing
        self._queue.sort(
            key=lambda n: {
                NotificationPriority.CRITICAL: 0,
                NotificationPriority.HIGH: 1,
                NotificationPriority.NORMAL: 2,
                NotificationPriority.LOW: 3,
            }.get(n.priority, 2)
        )

        notification = self._queue.pop(0)
        self._processing.add(notification.id)
        notification.status = NotificationStatus.SENDING

        return notification

    async def complete(self, notification_id: str, success: bool) -> None:
        """Mark notification as complete."""
        self._processing.discard(notification_id)

        if not success:
            # Check if should retry
            retry_info = self._retry_queue.get(notification_id)
            if retry_info:
                notification, attempts = retry_info
                if attempts < self._max_retries:
                    self._retry_queue[notification_id] = (notification, attempts + 1)
                    # Schedule retry
                    asyncio.create_task(self._schedule_retry(notification, attempts + 1))

    async def _schedule_retry(
        self,
        notification: Notification,
        attempt: int,
    ) -> None:
        """Schedule retry with exponential backoff."""
        delay = self._retry_delay_seconds * (2 ** (attempt - 1))
        await asyncio.sleep(delay)
        await self.enqueue(notification, notification.priority)

    @property
    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)

    @property
    def processing_count(self) -> int:
        """Get count of processing notifications."""
        return len(self._processing)


# =============================================================================
# Preference Manager
# =============================================================================


class PreferenceManager:
    """Manages user notification preferences."""

    def __init__(self):
        self._preferences: Dict[str, NotificationPreference] = {}
        self._subscriptions: Dict[str, NotificationSubscription] = {}
        self._subscriptions_by_user: Dict[str, Set[str]] = defaultdict(set)

    async def get_preference(self, user_id: str) -> Optional[NotificationPreference]:
        """Get user preferences."""
        return self._preferences.get(user_id)

    async def create_preference(
        self,
        user_id: str,
        organization_id: str,
        **kwargs,
    ) -> NotificationPreference:
        """Create user preferences."""
        preference = NotificationPreference(
            id=f"pref_{uuid.uuid4().hex[:20]}",
            user_id=user_id,
            organization_id=organization_id,
            **kwargs,
        )

        self._preferences[user_id] = preference

        return preference

    async def update_preference(
        self,
        user_id: str,
        email_enabled: Optional[bool] = None,
        sms_enabled: Optional[bool] = None,
        push_enabled: Optional[bool] = None,
        in_app_enabled: Optional[bool] = None,
        disabled_types: Optional[List[NotificationType]] = None,
        digest_enabled: Optional[bool] = None,
    ) -> NotificationPreference:
        """Update user preferences."""
        preference = self._preferences.get(user_id)
        if not preference:
            raise NotificationError(f"Preferences not found for user {user_id}")

        if email_enabled is not None:
            preference.email_enabled = email_enabled
        if sms_enabled is not None:
            preference.sms_enabled = sms_enabled
        if push_enabled is not None:
            preference.push_enabled = push_enabled
        if in_app_enabled is not None:
            preference.in_app_enabled = in_app_enabled
        if disabled_types is not None:
            preference.disabled_types = disabled_types
        if digest_enabled is not None:
            preference.digest_enabled = digest_enabled

        preference.updated_at = datetime.utcnow()

        return preference

    async def create_subscription(
        self,
        organization_id: str,
        user_id: str,
        notification_types: List[NotificationType],
        channels: List[NotificationChannel],
        filters: Optional[Dict[str, Any]] = None,
    ) -> NotificationSubscription:
        """Create a notification subscription."""
        subscription = NotificationSubscription(
            id=f"sub_{uuid.uuid4().hex[:20]}",
            organization_id=organization_id,
            user_id=user_id,
            notification_types=notification_types,
            channels=channels,
            filters=filters or {},
        )

        self._subscriptions[subscription.id] = subscription
        self._subscriptions_by_user[user_id].add(subscription.id)

        return subscription

    async def get_matching_subscriptions(
        self,
        organization_id: str,
        notification_type: NotificationType,
        context: Dict[str, Any],
    ) -> List[NotificationSubscription]:
        """Get subscriptions matching criteria."""
        matching = []
        for sub in self._subscriptions.values():
            if sub.organization_id != organization_id:
                continue
            if sub.matches(notification_type, context):
                matching.append(sub)
        return matching


# =============================================================================
# Notification Service
# =============================================================================


class NotificationService:
    """
    Unified notification service.

    Provides:
    - Multi-channel delivery (email, SMS, push, in-app)
    - Template management
    - Recipient management
    - Preference handling
    - Delivery tracking
    """

    def __init__(self):
        self.templates = TemplateManager()
        self.recipients = RecipientManager()
        self.preferences = PreferenceManager()
        self.queue = NotificationQueue()

        # Providers
        self._providers: Dict[NotificationChannel, NotificationProvider] = {}
        self._in_app_provider = InAppProvider()
        self._providers[NotificationChannel.IN_APP] = self._in_app_provider

        # Delivery tracking
        self._deliveries: Dict[str, List[NotificationDelivery]] = defaultdict(list)
        self._notifications: Dict[str, Notification] = {}

    def configure_email(self, config: EmailProviderConfig) -> None:
        """Configure email provider."""
        self._providers[NotificationChannel.EMAIL] = EmailProvider(config)
        logger.info(f"Email provider configured: {config.provider}")

    def configure_sms(self, config: SMSProviderConfig) -> None:
        """Configure SMS provider."""
        self._providers[NotificationChannel.SMS] = SMSProvider(config)
        logger.info(f"SMS provider configured: {config.provider}")

    def configure_push(self, config: PushProviderConfig) -> None:
        """Configure push provider."""
        self._providers[NotificationChannel.PUSH] = PushProvider(config)
        logger.info(f"Push provider configured: {config.provider}")

    async def send(
        self,
        organization_id: str,
        notification_type: NotificationType,
        recipient_ids: Optional[List[str]] = None,
        recipient_emails: Optional[List[str]] = None,
        channels: Optional[List[NotificationChannel]] = None,
        template_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        content: Optional[NotificationContent] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
    ) -> Notification:
        """
        Send a notification.

        Either provide template_id + context or content directly.
        """
        # Create notification
        notification = Notification(
            id=f"notif_{uuid.uuid4().hex[:18]}",
            organization_id=organization_id,
            notification_type=notification_type,
            recipient_ids=recipient_ids or [],
            recipient_emails=recipient_emails or [],
            channels=channels or [NotificationChannel.EMAIL],
            template_id=template_id,
            context=context or {},
            priority=priority,
            scheduled_at=scheduled_at,
        )

        # Resolve content
        if template_id:
            template = await self.templates.get_template(template_id)
            if template:
                notification.content = NotificationContent(
                    subject=template.render_subject(context or {}),
                    body=template.render(context or {}),
                    html_body=template.render_html(context or {}),
                )
        elif content:
            notification.content = content

        self._notifications[notification.id] = notification

        # Handle scheduled notifications
        if scheduled_at and scheduled_at > datetime.utcnow():
            notification.status = NotificationStatus.PENDING
            asyncio.create_task(self._schedule_send(notification))
            return notification

        # Send immediately
        await self._process_notification(notification)

        return notification

    async def _schedule_send(self, notification: Notification) -> None:
        """Schedule notification for future delivery."""
        if notification.scheduled_at:
            delay = (notification.scheduled_at - datetime.utcnow()).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
            await self._process_notification(notification)

    async def _process_notification(self, notification: Notification) -> None:
        """Process and deliver notification."""
        notification.status = NotificationStatus.SENDING

        # Get recipients
        recipients = []
        for recipient_id in notification.recipient_ids:
            recipient = await self.recipients.get_recipient(recipient_id)
            if recipient:
                recipients.append(recipient)

        for email in notification.recipient_emails:
            recipient = await self.recipients.get_recipient_by_email(
                notification.organization_id, email
            )
            if recipient:
                recipients.append(recipient)
            else:
                # Create temporary recipient
                recipients.append(NotificationRecipient(
                    id=f"tmp_{uuid.uuid4().hex[:16]}",
                    organization_id=notification.organization_id,
                    email=email,
                ))

        # Send to each recipient via each channel
        all_success = True
        for recipient in recipients:
            for channel in notification.channels:
                # Check preferences
                if recipient.user_id:
                    preference = await self.preferences.get_preference(recipient.user_id)
                    if preference and not preference.should_notify(channel, notification.notification_type):
                        continue

                # Check if recipient can receive
                if not recipient.can_receive(channel):
                    continue

                # Get provider
                provider = self._providers.get(channel)
                if not provider:
                    logger.warning(f"No provider for channel {channel}")
                    continue

                # Send
                try:
                    delivery = await provider.send(
                        recipient,
                        notification.content,
                        notification,
                    )
                    self._deliveries[notification.id].append(delivery)

                    if delivery.status == NotificationStatus.FAILED:
                        all_success = False

                except Exception as e:
                    logger.error(f"Error sending via {channel}: {e}")
                    all_success = False

        # Update notification status
        notification.status = NotificationStatus.SENT if all_success else NotificationStatus.FAILED
        notification.sent_at = datetime.utcnow()
        notification.updated_at = datetime.utcnow()

    async def send_system_alert(
        self,
        organization_id: str,
        subject: str,
        message: str,
        recipient_ids: List[str],
        priority: NotificationPriority = NotificationPriority.HIGH,
    ) -> Notification:
        """Send a system alert."""
        return await self.send(
            organization_id=organization_id,
            notification_type=NotificationType.SYSTEM_ALERT,
            recipient_ids=recipient_ids,
            channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            content=NotificationContent(
                subject=subject,
                body=message,
                title="System Alert",
            ),
            priority=priority,
        )

    async def send_call_notification(
        self,
        organization_id: str,
        notification_type: NotificationType,
        recipient_ids: List[str],
        call_id: str,
        caller_phone: str,
        caller_name: Optional[str] = None,
    ) -> Notification:
        """Send a call-related notification."""
        context = {
            "call_id": call_id,
            "caller_phone": caller_phone,
            "caller_name": caller_name or "Unknown",
            "call_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        }

        return await self.send(
            organization_id=organization_id,
            notification_type=notification_type,
            recipient_ids=recipient_ids,
            channels=[NotificationChannel.EMAIL, NotificationChannel.PUSH],
            context=context,
        )

    async def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get notification by ID."""
        return self._notifications.get(notification_id)

    async def get_deliveries(
        self,
        notification_id: str,
    ) -> List[NotificationDelivery]:
        """Get delivery records for notification."""
        return self._deliveries.get(notification_id, [])

    async def get_unread_count(self, user_id: str) -> int:
        """Get unread in-app notification count."""
        unread = await self._in_app_provider.get_unread(user_id)
        return len(unread)

    async def get_statistics(
        self,
        organization_id: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Get notification statistics."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        notifications = [
            n for n in self._notifications.values()
            if n.organization_id == organization_id
            and n.created_at >= cutoff
        ]

        sent = [n for n in notifications if n.is_sent]
        failed = [n for n in notifications if n.is_failed]

        # Channel breakdown
        channel_counts: Dict[str, int] = defaultdict(int)
        for n in notifications:
            for channel in n.channels:
                channel_counts[channel.value] += 1

        # Type breakdown
        type_counts: Dict[str, int] = defaultdict(int)
        for n in notifications:
            type_counts[n.notification_type.value] += 1

        return {
            "total": len(notifications),
            "sent": len(sent),
            "failed": len(failed),
            "success_rate": len(sent) / len(notifications) if notifications else 0,
            "by_channel": dict(channel_counts),
            "by_type": dict(type_counts),
        }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Managers
    "TemplateManager",
    "RecipientManager",
    "PreferenceManager",
    # Providers
    "NotificationProvider",
    "EmailProvider",
    "SMSProvider",
    "PushProvider",
    "InAppProvider",
    # Queue
    "NotificationQueue",
    # Main Service
    "NotificationService",
]
