"""
Builder Engine Python SDK - Configuration

This module contains configuration classes and defaults for the SDK.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClientConfig:
    """
    Configuration for the Builder Engine client.

    Attributes:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        organization_id: Organization ID for multi-tenant requests
        debug: Enable debug logging
    """
    base_url: str = "https://api.builderengine.io"
    timeout: float = 30.0
    max_retries: int = 3
    organization_id: Optional[str] = None
    debug: bool = False


# Default configuration
DEFAULT_CONFIG = ClientConfig()


# API version
API_VERSION = "v1"

# Endpoints
class Endpoints:
    """API endpoint paths."""

    # Agents
    AGENTS = "/api/v1/agents"
    AGENT = "/api/v1/agents/{agent_id}"
    AGENT_CALLS = "/api/v1/agents/{agent_id}/calls"
    AGENT_ANALYTICS = "/api/v1/agents/{agent_id}/analytics"
    AGENT_DUPLICATE = "/api/v1/agents/{agent_id}/duplicate"
    AGENT_TEST = "/api/v1/agents/{agent_id}/test"

    # Calls
    CALLS = "/api/v1/calls"
    CALL = "/api/v1/calls/{call_id}"
    CALL_HANGUP = "/api/v1/calls/{call_id}/hangup"
    CALL_TRANSFER = "/api/v1/calls/{call_id}/transfer"
    CALL_MUTE = "/api/v1/calls/{call_id}/mute"
    CALL_HOLD = "/api/v1/calls/{call_id}/hold"
    CALL_RECORDING = "/api/v1/calls/{call_id}/recording"
    CALL_TRANSCRIPT = "/api/v1/calls/{call_id}/transcript"
    CALL_SEND_DTMF = "/api/v1/calls/{call_id}/dtmf"
    CALL_INJECT_MESSAGE = "/api/v1/calls/{call_id}/inject"

    # Conversations
    CONVERSATIONS = "/api/v1/conversations"
    CONVERSATION = "/api/v1/conversations/{conversation_id}"
    CONVERSATION_MESSAGES = "/api/v1/conversations/{conversation_id}/messages"

    # Phone Numbers
    PHONE_NUMBERS = "/api/v1/phone-numbers"
    PHONE_NUMBER = "/api/v1/phone-numbers/{phone_number_id}"
    PHONE_NUMBERS_AVAILABLE = "/api/v1/phone-numbers/available"
    PHONE_NUMBER_PURCHASE = "/api/v1/phone-numbers/purchase"
    PHONE_NUMBER_RELEASE = "/api/v1/phone-numbers/{phone_number_id}/release"

    # Voices
    VOICES = "/api/v1/voices"
    VOICE = "/api/v1/voices/{voice_id}"
    VOICES_LIBRARY = "/api/v1/voices/library"
    VOICE_PREVIEW = "/api/v1/voices/{voice_id}/preview"
    VOICE_CLONE = "/api/v1/voices/clone"

    # Webhooks
    WEBHOOKS = "/api/v1/webhooks"
    WEBHOOK = "/api/v1/webhooks/{webhook_id}"
    WEBHOOK_TEST = "/api/v1/webhooks/{webhook_id}/test"
    WEBHOOK_DELIVERIES = "/api/v1/webhooks/{webhook_id}/deliveries"
    WEBHOOK_ROTATE_SECRET = "/api/v1/webhooks/{webhook_id}/rotate-secret"

    # Knowledge Base
    KNOWLEDGE_BASES = "/api/v1/knowledge-bases"
    KNOWLEDGE_BASE = "/api/v1/knowledge-bases/{knowledge_base_id}"
    KNOWLEDGE_BASE_DOCUMENTS = "/api/v1/knowledge-bases/{knowledge_base_id}/documents"
    KNOWLEDGE_BASE_DOCUMENT = "/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}"
    KNOWLEDGE_BASE_QUERY = "/api/v1/knowledge-bases/{knowledge_base_id}/query"
    KNOWLEDGE_BASE_SYNC = "/api/v1/knowledge-bases/{knowledge_base_id}/sync"

    # Workflows
    WORKFLOWS = "/api/v1/workflows"
    WORKFLOW = "/api/v1/workflows/{workflow_id}"
    WORKFLOW_ENABLE = "/api/v1/workflows/{workflow_id}/enable"
    WORKFLOW_DISABLE = "/api/v1/workflows/{workflow_id}/disable"
    WORKFLOW_EXECUTE = "/api/v1/workflows/{workflow_id}/execute"
    WORKFLOW_EXECUTIONS = "/api/v1/workflows/{workflow_id}/executions"

    # Campaigns
    CAMPAIGNS = "/api/v1/campaigns"
    CAMPAIGN = "/api/v1/campaigns/{campaign_id}"
    CAMPAIGN_START = "/api/v1/campaigns/{campaign_id}/start"
    CAMPAIGN_PAUSE = "/api/v1/campaigns/{campaign_id}/pause"
    CAMPAIGN_RESUME = "/api/v1/campaigns/{campaign_id}/resume"
    CAMPAIGN_CANCEL = "/api/v1/campaigns/{campaign_id}/cancel"
    CAMPAIGN_CONTACTS = "/api/v1/campaigns/{campaign_id}/contacts"
    CAMPAIGN_CONTACT = "/api/v1/campaigns/{campaign_id}/contacts/{contact_id}"
    CAMPAIGN_IMPORT_CONTACTS = "/api/v1/campaigns/{campaign_id}/contacts/import"
    CAMPAIGN_ANALYTICS = "/api/v1/campaigns/{campaign_id}/analytics"

    # Analytics
    ANALYTICS_OVERVIEW = "/api/v1/analytics/overview"
    ANALYTICS_CALLS = "/api/v1/analytics/calls"
    ANALYTICS_AGENTS = "/api/v1/analytics/agents"
    ANALYTICS_USAGE = "/api/v1/analytics/usage"
    ANALYTICS_COSTS = "/api/v1/analytics/costs"
    ANALYTICS_EXPORT = "/api/v1/analytics/export"

    # Organizations
    ORGANIZATIONS = "/api/v1/organizations"
    ORGANIZATION = "/api/v1/organizations/{organization_id}"
    ORGANIZATION_MEMBERS = "/api/v1/organizations/{organization_id}/members"
    ORGANIZATION_MEMBER = "/api/v1/organizations/{organization_id}/members/{user_id}"
    ORGANIZATION_INVITE = "/api/v1/organizations/{organization_id}/invite"
    ORGANIZATION_SETTINGS = "/api/v1/organizations/{organization_id}/settings"

    # Users
    USERS = "/api/v1/users"
    USER = "/api/v1/users/{user_id}"
    USER_ME = "/api/v1/users/me"
    USER_PROFILE = "/api/v1/users/me/profile"
    USER_PASSWORD = "/api/v1/users/me/password"
    USER_NOTIFICATIONS = "/api/v1/users/me/notifications"

    # API Keys
    API_KEYS = "/api/v1/api-keys"
    API_KEY = "/api/v1/api-keys/{api_key_id}"
    API_KEY_REGENERATE = "/api/v1/api-keys/{api_key_id}/regenerate"

    # Billing
    BILLING_SUBSCRIPTION = "/api/v1/billing/subscription"
    BILLING_USAGE = "/api/v1/billing/usage"
    BILLING_INVOICES = "/api/v1/billing/invoices"
    BILLING_INVOICE = "/api/v1/billing/invoices/{invoice_id}"
    BILLING_PAYMENT_METHODS = "/api/v1/billing/payment-methods"
    BILLING_PAYMENT_METHOD = "/api/v1/billing/payment-methods/{payment_method_id}"
    BILLING_CHECKOUT = "/api/v1/billing/checkout"
    BILLING_PORTAL = "/api/v1/billing/portal"


# Request limits
class Limits:
    """API limits and constraints."""

    # Pagination
    MAX_PAGE_SIZE = 100
    DEFAULT_PAGE_SIZE = 20

    # Uploads
    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
    MAX_DOCUMENTS_PER_KB = 1000
    MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

    # Calls
    MAX_CALL_DURATION_SECONDS = 7200  # 2 hours
    MAX_CONCURRENT_CALLS = 100

    # Campaigns
    MAX_CONTACTS_PER_CAMPAIGN = 100000
    MAX_CONCURRENT_CAMPAIGN_CALLS = 50

    # Webhooks
    MAX_WEBHOOKS_PER_ORG = 50
    MAX_WEBHOOK_TIMEOUT_SECONDS = 60

    # Rate limits (requests per minute)
    RATE_LIMIT_STANDARD = 1000
    RATE_LIMIT_CALLS = 100
    RATE_LIMIT_BULK = 10
