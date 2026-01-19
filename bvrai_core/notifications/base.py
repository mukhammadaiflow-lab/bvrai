"""
Notifications Base Types Module

This module defines core types for multi-channel notification delivery
including email, SMS, push notifications, and in-app messages.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
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


# =============================================================================
# Enums
# =============================================================================


class NotificationChannel(str, Enum):
    """Notification delivery channels."""

    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    SLACK = "slack"
    VOICE = "voice"  # Voice call notification


class NotificationType(str, Enum):
    """Types of notifications."""

    # System notifications
    SYSTEM_ALERT = "system_alert"
    SYSTEM_UPDATE = "system_update"
    MAINTENANCE = "maintenance"

    # Call notifications
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"
    CALL_MISSED = "call_missed"
    CALL_FAILED = "call_failed"
    VOICEMAIL_RECEIVED = "voicemail_received"

    # Agent notifications
    AGENT_ASSIGNED = "agent_assigned"
    AGENT_OFFLINE = "agent_offline"
    QUEUE_THRESHOLD = "queue_threshold"

    # Campaign notifications
    CAMPAIGN_STARTED = "campaign_started"
    CAMPAIGN_COMPLETED = "campaign_completed"
    CAMPAIGN_PAUSED = "campaign_paused"
    CAMPAIGN_FAILED = "campaign_failed"

    # Billing notifications
    PAYMENT_RECEIVED = "payment_received"
    PAYMENT_FAILED = "payment_failed"
    INVOICE_GENERATED = "invoice_generated"
    USAGE_THRESHOLD = "usage_threshold"

    # Account notifications
    ACCOUNT_CREATED = "account_created"
    PASSWORD_RESET = "password_reset"
    API_KEY_CREATED = "api_key_created"
    WELCOME = "welcome"

    # Custom
    CUSTOM = "custom"


class NotificationStatus(str, Enum):
    """Status of notification delivery."""

    PENDING = "pending"
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPENED = "opened"
    CLICKED = "clicked"


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""

    CRITICAL = "critical"  # Immediate delivery
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TemplateType(str, Enum):
    """Types of notification templates."""

    EMAIL_HTML = "email_html"
    EMAIL_TEXT = "email_text"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


# =============================================================================
# Recipient Types
# =============================================================================


@dataclass
class NotificationRecipient:
    """A notification recipient."""

    id: str
    organization_id: str

    # Contact information
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    user_id: Optional[str] = None

    # Preferences
    name: str = ""
    preferred_channel: NotificationChannel = NotificationChannel.EMAIL
    enabled_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.EMAIL, NotificationChannel.IN_APP]
    )

    # Timezone for delivery scheduling
    timezone: str = "UTC"

    # Quiet hours
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None  # "08:00"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"rcpt_{uuid.uuid4().hex[:20]}"

    def can_receive(self, channel: NotificationChannel) -> bool:
        """Check if recipient can receive on channel."""
        if channel not in self.enabled_channels:
            return False

        if channel == NotificationChannel.EMAIL:
            return bool(self.email)
        if channel == NotificationChannel.SMS:
            return bool(self.phone)
        if channel == NotificationChannel.PUSH:
            return bool(self.push_token)
        if channel == NotificationChannel.IN_APP:
            return bool(self.user_id)

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "email": self.email,
            "phone": self.phone,
            "push_token": self.push_token,
            "user_id": self.user_id,
            "name": self.name,
            "preferred_channel": self.preferred_channel.value,
            "enabled_channels": [c.value for c in self.enabled_channels],
            "timezone": self.timezone,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationRecipient":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            organization_id=data["organization_id"],
            email=data.get("email"),
            phone=data.get("phone"),
            push_token=data.get("push_token"),
            user_id=data.get("user_id"),
            name=data.get("name", ""),
            preferred_channel=NotificationChannel(data.get("preferred_channel", "email")),
            enabled_channels=[
                NotificationChannel(c) for c in data.get("enabled_channels", ["email"])
            ],
            timezone=data.get("timezone", "UTC"),
            quiet_hours_start=data.get("quiet_hours_start"),
            quiet_hours_end=data.get("quiet_hours_end"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )


# =============================================================================
# Template Types
# =============================================================================


@dataclass
class NotificationTemplate:
    """A notification template."""

    id: str
    organization_id: str
    name: str
    notification_type: NotificationType

    # Template content
    template_type: TemplateType = TemplateType.EMAIL_HTML
    subject: str = ""  # For email
    body: str = ""
    html_body: Optional[str] = None  # For HTML email

    # Variables
    variables: List[str] = field(default_factory=list)  # Expected variables
    default_values: Dict[str, str] = field(default_factory=dict)

    # Localization
    locale: str = "en"
    translations: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Status
    is_active: bool = True
    is_default: bool = False

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"tmpl_{uuid.uuid4().hex[:20]}"

    def render(self, context: Dict[str, Any]) -> str:
        """Render template with context."""
        result = self.body
        for key, value in context.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))

        # Apply defaults for missing variables
        for key, default in self.default_values.items():
            placeholder = "{{" + key + "}}"
            if placeholder in result:
                result = result.replace(placeholder, default)

        return result

    def render_html(self, context: Dict[str, Any]) -> Optional[str]:
        """Render HTML template with context."""
        if not self.html_body:
            return None

        result = self.html_body
        for key, value in context.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))

        return result

    def render_subject(self, context: Dict[str, Any]) -> str:
        """Render subject with context."""
        result = self.subject
        for key, value in context.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "notification_type": self.notification_type.value,
            "template_type": self.template_type.value,
            "subject": self.subject,
            "body": self.body,
            "html_body": self.html_body,
            "variables": self.variables,
            "default_values": self.default_values,
            "locale": self.locale,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Notification Types
# =============================================================================


@dataclass
class NotificationContent:
    """Content of a notification."""

    subject: str = ""
    body: str = ""
    html_body: Optional[str] = None

    # For push notifications
    title: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    # Actions
    action_url: Optional[str] = None
    action_label: Optional[str] = None

    # Rich content
    image_url: Optional[str] = None
    icon_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "body": self.body,
            "html_body": self.html_body,
            "title": self.title,
            "data": self.data,
            "action_url": self.action_url,
            "action_label": self.action_label,
            "image_url": self.image_url,
            "icon_url": self.icon_url,
        }


@dataclass
class Notification:
    """A notification to be sent."""

    id: str
    organization_id: str
    notification_type: NotificationType

    # Recipients
    recipient_ids: List[str] = field(default_factory=list)
    recipient_emails: List[str] = field(default_factory=list)
    recipient_phones: List[str] = field(default_factory=list)

    # Channels
    channels: List[NotificationChannel] = field(default_factory=list)

    # Content
    content: NotificationContent = field(default_factory=NotificationContent)
    template_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Priority and scheduling
    priority: NotificationPriority = NotificationPriority.NORMAL
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Status
    status: NotificationStatus = NotificationStatus.PENDING
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Tracking
    external_id: Optional[str] = None  # ID from external provider
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"notif_{uuid.uuid4().hex[:18]}"

    @property
    def is_sent(self) -> bool:
        """Check if notification has been sent."""
        return self.status in [
            NotificationStatus.SENT,
            NotificationStatus.DELIVERED,
            NotificationStatus.OPENED,
            NotificationStatus.CLICKED,
        ]

    @property
    def is_failed(self) -> bool:
        """Check if notification failed."""
        return self.status in [
            NotificationStatus.FAILED,
            NotificationStatus.BOUNCED,
        ]

    @property
    def is_scheduled(self) -> bool:
        """Check if notification is scheduled for future."""
        if not self.scheduled_at:
            return False
        return self.scheduled_at > datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "notification_type": self.notification_type.value,
            "recipient_ids": self.recipient_ids,
            "recipient_emails": self.recipient_emails,
            "recipient_phones": self.recipient_phones,
            "channels": [c.value for c in self.channels],
            "content": self.content.to_dict(),
            "template_id": self.template_id,
            "context": self.context,
            "priority": self.priority.value,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "error_message": self.error_message,
            "external_id": self.external_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class NotificationDelivery:
    """Record of a notification delivery attempt."""

    id: str
    notification_id: str
    recipient_id: Optional[str] = None

    # Delivery details
    channel: NotificationChannel = NotificationChannel.EMAIL
    destination: str = ""  # Email, phone, token

    # Status
    status: NotificationStatus = NotificationStatus.PENDING
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None

    # Provider info
    provider: str = ""
    provider_message_id: Optional[str] = None
    provider_response: Optional[str] = None

    # Timing
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None

    # Error info
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"dlvr_{uuid.uuid4().hex[:20]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "notification_id": self.notification_id,
            "recipient_id": self.recipient_id,
            "channel": self.channel.value,
            "destination": self.destination,
            "status": self.status.value,
            "attempts": self.attempts,
            "provider": self.provider,
            "provider_message_id": self.provider_message_id,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "clicked_at": self.clicked_at.isoformat() if self.clicked_at else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Preference Types
# =============================================================================


@dataclass
class NotificationPreference:
    """User notification preferences."""

    id: str
    user_id: str
    organization_id: str

    # Channel preferences
    email_enabled: bool = True
    sms_enabled: bool = True
    push_enabled: bool = True
    in_app_enabled: bool = True

    # Type preferences (which notifications to receive)
    disabled_types: List[NotificationType] = field(default_factory=list)

    # Frequency
    digest_enabled: bool = False
    digest_frequency: str = "daily"  # daily, weekly

    # Quiet hours
    quiet_hours_enabled: bool = False
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "08:00"
    quiet_hours_timezone: str = "UTC"

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"pref_{uuid.uuid4().hex[:20]}"

    def should_notify(
        self,
        channel: NotificationChannel,
        notification_type: NotificationType,
    ) -> bool:
        """Check if user should be notified."""
        # Check channel preference
        if channel == NotificationChannel.EMAIL and not self.email_enabled:
            return False
        if channel == NotificationChannel.SMS and not self.sms_enabled:
            return False
        if channel == NotificationChannel.PUSH and not self.push_enabled:
            return False
        if channel == NotificationChannel.IN_APP and not self.in_app_enabled:
            return False

        # Check type preference
        if notification_type in self.disabled_types:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "email_enabled": self.email_enabled,
            "sms_enabled": self.sms_enabled,
            "push_enabled": self.push_enabled,
            "in_app_enabled": self.in_app_enabled,
            "disabled_types": [t.value for t in self.disabled_types],
            "digest_enabled": self.digest_enabled,
            "digest_frequency": self.digest_frequency,
            "quiet_hours_enabled": self.quiet_hours_enabled,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "quiet_hours_timezone": self.quiet_hours_timezone,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Subscription Types
# =============================================================================


@dataclass
class NotificationSubscription:
    """A subscription to notification events."""

    id: str
    organization_id: str
    user_id: Optional[str] = None

    # What to subscribe to
    notification_types: List[NotificationType] = field(default_factory=list)
    channels: List[NotificationChannel] = field(default_factory=list)

    # Filters
    filters: Dict[str, Any] = field(default_factory=dict)  # e.g., {"queue_id": "xxx"}

    # Status
    is_active: bool = True

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"sub_{uuid.uuid4().hex[:20]}"

    def matches(
        self,
        notification_type: NotificationType,
        context: Dict[str, Any],
    ) -> bool:
        """Check if subscription matches notification."""
        if not self.is_active:
            return False

        # Check type
        if self.notification_types and notification_type not in self.notification_types:
            return False

        # Check filters
        for key, value in self.filters.items():
            if key in context and context[key] != value:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "notification_types": [t.value for t in self.notification_types],
            "channels": [c.value for c in self.channels],
            "filters": self.filters,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Provider Configuration
# =============================================================================


@dataclass
class EmailProviderConfig:
    """Configuration for email provider."""

    provider: str = "smtp"  # smtp, sendgrid, ses, mailgun
    from_email: str = ""
    from_name: str = ""

    # SMTP settings
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True

    # API settings
    api_key: str = ""
    api_endpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes sensitive data)."""
        return {
            "provider": self.provider,
            "from_email": self.from_email,
            "from_name": self.from_name,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
        }


@dataclass
class SMSProviderConfig:
    """Configuration for SMS provider."""

    provider: str = "twilio"  # twilio, nexmo, plivo
    from_number: str = ""

    # API settings
    account_sid: str = ""
    auth_token: str = ""
    api_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes sensitive data)."""
        return {
            "provider": self.provider,
            "from_number": self.from_number,
        }


@dataclass
class PushProviderConfig:
    """Configuration for push notification provider."""

    provider: str = "firebase"  # firebase, apns, onesignal

    # Firebase settings
    fcm_server_key: str = ""
    fcm_project_id: str = ""

    # APNS settings
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_bundle_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes sensitive data)."""
        return {
            "provider": self.provider,
            "fcm_project_id": self.fcm_project_id,
            "apns_bundle_id": self.apns_bundle_id,
        }


# =============================================================================
# Exceptions
# =============================================================================


class NotificationError(Exception):
    """Base exception for notification errors."""
    pass


class TemplateNotFoundError(NotificationError):
    """Template not found."""
    pass


class RecipientNotFoundError(NotificationError):
    """Recipient not found."""
    pass


class DeliveryError(NotificationError):
    """Error delivering notification."""

    def __init__(self, message: str, channel: NotificationChannel, error_code: Optional[str] = None):
        super().__init__(message)
        self.channel = channel
        self.error_code = error_code


class ProviderError(NotificationError):
    """Error from notification provider."""

    def __init__(self, message: str, provider: str):
        super().__init__(message)
        self.provider = provider


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "NotificationChannel",
    "NotificationType",
    "NotificationStatus",
    "NotificationPriority",
    "TemplateType",
    # Recipient types
    "NotificationRecipient",
    # Template types
    "NotificationTemplate",
    # Notification types
    "NotificationContent",
    "Notification",
    "NotificationDelivery",
    # Preference types
    "NotificationPreference",
    "NotificationSubscription",
    # Provider configs
    "EmailProviderConfig",
    "SMSProviderConfig",
    "PushProviderConfig",
    # Exceptions
    "NotificationError",
    "TemplateNotFoundError",
    "RecipientNotFoundError",
    "DeliveryError",
    "ProviderError",
]
