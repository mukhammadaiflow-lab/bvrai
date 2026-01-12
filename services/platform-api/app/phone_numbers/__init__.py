"""
Phone Number Management Module

Complete phone number lifecycle:
- Multi-provider support (Twilio, Telnyx, Vonage)
- Number provisioning and release
- Agent binding
- Configuration management
"""

from app.phone_numbers.service import (
    PhoneNumber,
    PhoneNumberService,
    NumberType,
    NumberStatus,
    NumberCapability,
    Provider,
    NumberSearchCriteria,
    AvailableNumber,
    TelephonyProvider,
    TwilioProvider,
    TelnyxProvider,
    format_e164,
    validate_e164,
    get_phone_number_service,
)

__all__ = [
    "PhoneNumber",
    "PhoneNumberService",
    "NumberType",
    "NumberStatus",
    "NumberCapability",
    "Provider",
    "NumberSearchCriteria",
    "AvailableNumber",
    "TelephonyProvider",
    "TwilioProvider",
    "TelnyxProvider",
    "format_e164",
    "validate_e164",
    "get_phone_number_service",
]
