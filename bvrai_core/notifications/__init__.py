"""
Notifications Module

This module provides comprehensive multi-channel notification delivery
for voice agent platforms, including email, SMS, push notifications,
and in-app messaging.

Features:
- Multi-Channel Delivery: Email, SMS, push, in-app, webhook
- Template Management: Customizable templates with variable substitution
- Recipient Management: Contact info, preferences, quiet hours
- Provider Flexibility: Support for multiple providers per channel
- Preference System: User-controlled notification preferences
- Delivery Tracking: Status tracking, retry logic, analytics

Example usage:

    from bvrai_core.notifications import (
        NotificationService,
        NotificationType,
        NotificationChannel,
        NotificationPriority,
        NotificationContent,
        EmailProviderConfig,
        SMSProviderConfig,
    )

    # Initialize service
    service = NotificationService()

    # Configure providers
    service.configure_email(EmailProviderConfig(
        provider="sendgrid",
        from_email="noreply@example.com",
        from_name="Voice Agent Platform",
        api_key="SG.xxxx",
    ))

    service.configure_sms(SMSProviderConfig(
        provider="twilio",
        from_number="+15551234567",
        account_sid="AC...",
        auth_token="...",
    ))

    # Create recipient
    recipient = await service.recipients.create_recipient(
        organization_id="org_123",
        email="user@example.com",
        phone="+15559876543",
        name="John Doe",
        enabled_channels=[
            NotificationChannel.EMAIL,
            NotificationChannel.SMS,
            NotificationChannel.IN_APP,
        ],
    )

    # Create template
    template = await service.templates.create_template(
        organization_id="org_123",
        name="Missed Call Alert",
        notification_type=NotificationType.CALL_MISSED,
        subject="Missed call from {{caller_phone}}",
        body="You missed a call from {{caller_name}} ({{caller_phone}}) at {{call_time}}.",
        html_body="<p>You missed a call from <b>{{caller_name}}</b> at {{call_time}}.</p>",
        variables=["caller_phone", "caller_name", "call_time"],
        is_default=True,
    )

    # Send notification using template
    notification = await service.send(
        organization_id="org_123",
        notification_type=NotificationType.CALL_MISSED,
        recipient_ids=[recipient.id],
        channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
        template_id=template.id,
        context={
            "caller_phone": "+15551234567",
            "caller_name": "Jane Smith",
            "call_time": "2024-01-15 10:30 AM",
        },
        priority=NotificationPriority.HIGH,
    )

    # Send notification with direct content
    notification = await service.send(
        organization_id="org_123",
        notification_type=NotificationType.SYSTEM_ALERT,
        recipient_emails=["admin@example.com"],
        channels=[NotificationChannel.EMAIL],
        content=NotificationContent(
            subject="High Queue Volume Alert",
            body="Queue 'Support' has exceeded threshold with 50 waiting calls.",
        ),
        priority=NotificationPriority.CRITICAL,
    )

    # Send system alert helper
    await service.send_system_alert(
        organization_id="org_123",
        subject="Usage Threshold Alert",
        message="You have used 80% of your monthly API quota.",
        recipient_ids=[recipient.id],
    )

    # Get unread count for user
    unread = await service.get_unread_count(user_id="user_456")

    # Set user preferences
    await service.preferences.create_preference(
        user_id="user_456",
        organization_id="org_123",
        email_enabled=True,
        sms_enabled=False,
        push_enabled=True,
    )

    # Get statistics
    stats = await service.get_statistics("org_123", hours=24)
    print(f"Sent: {stats['sent']}, Failed: {stats['failed']}")
"""

# Base types and enums
from .base import (
    # Enums
    NotificationChannel,
    NotificationType,
    NotificationStatus,
    NotificationPriority,
    TemplateType,
    # Recipient types
    NotificationRecipient,
    # Template types
    NotificationTemplate,
    # Notification types
    NotificationContent,
    Notification,
    NotificationDelivery,
    # Preference types
    NotificationPreference,
    NotificationSubscription,
    # Provider configs
    EmailProviderConfig,
    SMSProviderConfig,
    PushProviderConfig,
    # Exceptions
    NotificationError,
    TemplateNotFoundError,
    RecipientNotFoundError,
    DeliveryError,
    ProviderError,
)

# Services
from .service import (
    # Managers
    TemplateManager,
    RecipientManager,
    PreferenceManager,
    # Providers
    NotificationProvider,
    EmailProvider,
    SMSProvider,
    PushProvider,
    InAppProvider,
    # Queue
    NotificationQueue,
    # Main Service
    NotificationService,
)


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
    # Exceptions
    "NotificationError",
    "TemplateNotFoundError",
    "RecipientNotFoundError",
    "DeliveryError",
    "ProviderError",
]
