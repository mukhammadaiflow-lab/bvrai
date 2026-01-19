"""
Agent Factory Base Types

This module defines the core data structures for business information
and agent configuration used throughout the agent factory.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)


class BusinessCategory(str, Enum):
    """Business category/industry classification."""

    # Healthcare
    HEALTHCARE = "healthcare"
    DENTAL = "dental"
    MEDICAL_PRACTICE = "medical_practice"
    PHARMACY = "pharmacy"
    MENTAL_HEALTH = "mental_health"
    VETERINARY = "veterinary"

    # Professional Services
    LEGAL = "legal"
    ACCOUNTING = "accounting"
    CONSULTING = "consulting"
    REAL_ESTATE = "real_estate"
    INSURANCE = "insurance"
    FINANCIAL_SERVICES = "financial_services"

    # Home Services
    PLUMBING = "plumbing"
    HVAC = "hvac"
    ELECTRICAL = "electrical"
    ROOFING = "roofing"
    LANDSCAPING = "landscaping"
    CLEANING = "cleaning"
    PEST_CONTROL = "pest_control"

    # Automotive
    AUTO_DEALERSHIP = "auto_dealership"
    AUTO_REPAIR = "auto_repair"
    AUTO_BODY = "auto_body"
    TOWING = "towing"

    # Hospitality
    RESTAURANT = "restaurant"
    HOTEL = "hotel"
    CATERING = "catering"
    EVENT_VENUE = "event_venue"

    # Retail
    RETAIL_STORE = "retail_store"
    ECOMMERCE = "ecommerce"

    # Personal Services
    SALON = "salon"
    SPA = "spa"
    FITNESS = "fitness"

    # Education
    EDUCATION = "education"
    TUTORING = "tutoring"
    TRAINING = "training"

    # Technology
    IT_SERVICES = "it_services"
    SOFTWARE = "software"

    # Other
    OTHER = "other"


class GenerationStatus(str, Enum):
    """Status of agent generation."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    GENERATING_PERSONA = "generating_persona"
    BUILDING_KNOWLEDGE = "building_knowledge"
    CREATING_FLOWS = "creating_flows"
    GENERATING_PROMPTS = "generating_prompts"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ContactInfo:
    """Business contact information."""

    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: str = "US"

    # Social media
    facebook: Optional[str] = None
    twitter: Optional[str] = None
    instagram: Optional[str] = None
    linkedin: Optional[str] = None

    # Additional contact methods
    fax: Optional[str] = None
    toll_free: Optional[str] = None
    text_number: Optional[str] = None


@dataclass
class BusinessHours:
    """Hours for a single day."""

    open_time: Optional[time] = None
    close_time: Optional[time] = None
    is_closed: bool = False

    # Break periods
    break_start: Optional[time] = None
    break_end: Optional[time] = None

    def is_open(self, check_time: time) -> bool:
        """Check if business is open at given time."""
        if self.is_closed or not self.open_time or not self.close_time:
            return False

        if self.open_time <= check_time <= self.close_time:
            # Check break
            if self.break_start and self.break_end:
                if self.break_start <= check_time <= self.break_end:
                    return False
            return True
        return False


@dataclass
class HoursOfOperation:
    """Weekly hours of operation."""

    monday: Optional[BusinessHours] = None
    tuesday: Optional[BusinessHours] = None
    wednesday: Optional[BusinessHours] = None
    thursday: Optional[BusinessHours] = None
    friday: Optional[BusinessHours] = None
    saturday: Optional[BusinessHours] = None
    sunday: Optional[BusinessHours] = None

    # Special hours/holidays
    holiday_hours: Dict[str, BusinessHours] = field(default_factory=dict)
    timezone: str = "America/New_York"

    def get_day_hours(self, day: int) -> Optional[BusinessHours]:
        """Get hours for day (0=Monday)."""
        days = [
            self.monday, self.tuesday, self.wednesday,
            self.thursday, self.friday, self.saturday, self.sunday
        ]
        return days[day] if 0 <= day < 7 else None


@dataclass
class ProductInfo:
    """Product information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    price: Optional[float] = None
    price_range: Optional[str] = None  # e.g., "$50-$100"
    category: str = ""

    # Features
    features: List[str] = field(default_factory=list)
    specifications: Dict[str, str] = field(default_factory=dict)

    # Availability
    in_stock: bool = True
    lead_time: Optional[str] = None

    # Media
    image_url: Optional[str] = None
    video_url: Optional[str] = None

    # Relationships
    related_products: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price": self.price,
            "price_range": self.price_range,
            "category": self.category,
            "features": self.features,
            "in_stock": self.in_stock,
        }


@dataclass
class ServiceInfo:
    """Service information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Pricing
    price: Optional[float] = None
    price_type: str = "fixed"  # fixed, hourly, estimate, quote
    price_range: Optional[str] = None

    # Duration
    duration_minutes: Optional[int] = None
    duration_range: Optional[str] = None  # e.g., "1-2 hours"

    # Requirements
    requirements: List[str] = field(default_factory=list)
    preparation: Optional[str] = None

    # Category
    category: str = ""
    subcategory: str = ""

    # Availability
    available: bool = True
    booking_required: bool = False
    advance_notice_hours: int = 0

    # Service area (for field services)
    service_area: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price": self.price,
            "price_type": self.price_type,
            "duration_minutes": self.duration_minutes,
            "category": self.category,
            "available": self.available,
            "booking_required": self.booking_required,
        }


@dataclass
class FAQEntry:
    """Frequently asked question entry."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    category: str = "general"

    # Variations
    question_variations: List[str] = field(default_factory=list)

    # Metadata
    keywords: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = more important

    # Follow-up
    follow_up_questions: List[str] = field(default_factory=list)
    related_faqs: List[str] = field(default_factory=list)


@dataclass
class PolicyInfo:
    """Business policy information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ""  # cancellation, refund, privacy, terms, warranty
    summary: str = ""
    full_text: str = ""

    # Conditions
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)

    # Time frames
    time_limit_days: Optional[int] = None
    notice_required_hours: Optional[int] = None

    # Fees
    fee_amount: Optional[float] = None
    fee_percentage: Optional[float] = None


@dataclass
class TeamMember:
    """Team/staff member information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: str = ""
    title: Optional[str] = None

    # Contact
    email: Optional[str] = None
    phone: Optional[str] = None
    extension: Optional[str] = None

    # Availability
    available_for_calls: bool = True
    schedule: Optional[HoursOfOperation] = None

    # Expertise
    specialties: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)

    # Bio
    bio: Optional[str] = None

    # Transfer settings
    can_receive_transfers: bool = True
    transfer_priority: int = 0


@dataclass
class LocationInfo:
    """Business location information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    is_primary: bool = False

    # Address
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    country: str = "US"

    # Contact
    phone: Optional[str] = None
    email: Optional[str] = None

    # Hours
    hours: Optional[HoursOfOperation] = None

    # Services at this location
    services_available: List[str] = field(default_factory=list)

    # Parking/access
    parking_info: Optional[str] = None
    accessibility_info: Optional[str] = None

    # Geographic
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class BusinessInfo:
    """Complete business information for agent generation."""

    # Basic info
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tagline: Optional[str] = None

    # Classification
    category: BusinessCategory = BusinessCategory.OTHER
    subcategories: List[str] = field(default_factory=list)

    # Contact
    contact: ContactInfo = field(default_factory=ContactInfo)

    # Hours
    hours: Optional[HoursOfOperation] = None

    # Products and services
    products: List[ProductInfo] = field(default_factory=list)
    services: List[ServiceInfo] = field(default_factory=list)

    # FAQs
    faqs: List[FAQEntry] = field(default_factory=list)

    # Policies
    policies: List[PolicyInfo] = field(default_factory=list)

    # Team
    team_members: List[TeamMember] = field(default_factory=list)

    # Locations
    locations: List[LocationInfo] = field(default_factory=list)

    # Additional content
    about_us: Optional[str] = None
    mission_statement: Optional[str] = None
    unique_value_proposition: Optional[str] = None

    # Target audience
    target_audience: List[str] = field(default_factory=list)

    # Differentiators
    differentiators: List[str] = field(default_factory=list)
    awards_certifications: List[str] = field(default_factory=list)

    # Common caller intents
    common_call_reasons: List[str] = field(default_factory=list)

    # Brand voice
    brand_tone: str = "professional"  # professional, friendly, casual, formal
    brand_personality: List[str] = field(default_factory=list)

    # Documents (paths or URLs)
    documents: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "products": [p.to_dict() for p in self.products],
            "services": [s.to_dict() for s in self.services],
            "faqs": len(self.faqs),
            "policies": len(self.policies),
            "team_members": len(self.team_members),
        }


# Agent Configuration Types

@dataclass
class VoiceConfig:
    """Voice/TTS configuration for the agent."""

    provider: str = "elevenlabs"
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None

    # Voice characteristics
    gender: str = "neutral"  # male, female, neutral
    age_range: str = "adult"  # young, adult, mature
    accent: str = "neutral"  # neutral, american, british, etc.

    # Voice settings
    speed: float = 1.0
    pitch: float = 1.0
    stability: float = 0.75
    similarity_boost: float = 0.75

    # Style
    style: str = "conversational"  # conversational, narrative, news
    emotion: str = "friendly"  # friendly, professional, empathetic


@dataclass
class GreetingConfig:
    """Greeting configuration."""

    # Main greeting
    greeting_template: str = "Hello, thank you for calling {business_name}. How can I help you today?"

    # Time-based greetings
    morning_greeting: Optional[str] = None
    afternoon_greeting: Optional[str] = None
    evening_greeting: Optional[str] = None

    # After-hours greeting
    after_hours_greeting: Optional[str] = None

    # Holiday greetings
    holiday_greetings: Dict[str, str] = field(default_factory=dict)

    # Returning caller greeting
    returning_caller_greeting: Optional[str] = None

    # Name capture
    ask_for_name: bool = False
    name_prompt: str = "May I ask who I'm speaking with?"


@dataclass
class TransferConfig:
    """Call transfer configuration."""

    enabled: bool = True

    # Transfer destinations
    default_transfer_number: Optional[str] = None
    department_numbers: Dict[str, str] = field(default_factory=dict)

    # Transfer behavior
    warm_transfer: bool = True  # Announce caller before transferring
    max_hold_seconds: int = 180
    transfer_announcement: str = "I'm transferring you now. Please hold."

    # Fallback
    voicemail_enabled: bool = True
    voicemail_prompt: str = "Please leave a message after the beep."


@dataclass
class EscalationConfig:
    """Escalation configuration."""

    # Triggers
    escalation_keywords: List[str] = field(default_factory=lambda: [
        "speak to a human", "real person", "supervisor", "manager",
        "escalate", "complaint", "frustrated"
    ])

    # Sentiment threshold
    negative_sentiment_threshold: float = -0.5
    max_failed_intents: int = 3

    # Action
    escalation_message: str = "I understand you'd like to speak with someone. Let me connect you."
    escalation_number: Optional[str] = None


@dataclass
class ComplianceConfig:
    """Compliance and safety configuration."""

    # Recording
    recording_disclosure: bool = True
    recording_message: str = "This call may be recorded for quality assurance."

    # Privacy
    collect_sensitive_data: bool = False
    data_retention_days: int = 90

    # Industry-specific
    hipaa_compliant: bool = False
    pci_compliant: bool = False

    # Prohibited topics
    prohibited_topics: List[str] = field(default_factory=list)

    # Required disclosures
    required_disclosures: List[str] = field(default_factory=list)


@dataclass
class BehaviorConfig:
    """Agent behavior configuration."""

    # Response style
    response_length: str = "concise"  # concise, moderate, detailed
    formality_level: str = "professional"  # casual, professional, formal

    # Conversation control
    max_turn_duration_seconds: int = 30
    silence_timeout_seconds: int = 5
    max_silence_prompts: int = 2

    # Interruption handling
    allow_interruptions: bool = True
    interruption_sensitivity: float = 0.5

    # Clarification
    ask_for_clarification: bool = True
    max_clarification_attempts: int = 2

    # Confirmation
    confirm_important_info: bool = True
    repeat_back_details: bool = True

    # Error handling
    apologize_for_errors: bool = True
    graceful_failure_message: str = "I apologize, but I'm having trouble with that. Let me connect you with someone who can help."

    # Personality traits
    use_fillers: bool = False  # um, uh, etc.
    show_empathy: bool = True
    use_humor: bool = False

    # Language
    primary_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class AgentPersona:
    """Agent persona definition."""

    # Identity
    name: str = ""
    role: str = "AI Assistant"

    # Background
    background_story: Optional[str] = None
    expertise_areas: List[str] = field(default_factory=list)

    # Personality
    personality_traits: List[str] = field(default_factory=list)
    communication_style: str = "professional"

    # Voice
    voice_config: VoiceConfig = field(default_factory=VoiceConfig)

    # Sample phrases
    sample_greetings: List[str] = field(default_factory=list)
    sample_responses: List[str] = field(default_factory=list)
    common_phrases: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Complete agent configuration."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"

    # Business reference
    business_id: str = ""
    business_name: str = ""

    # Persona
    persona: AgentPersona = field(default_factory=AgentPersona)

    # Behavior
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)

    # Features
    greeting: GreetingConfig = field(default_factory=GreetingConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

    # System prompt
    system_prompt: str = ""

    # Dialog flows (IDs)
    dialog_flows: List[str] = field(default_factory=list)

    # Intent handlers
    intent_handlers: Dict[str, str] = field(default_factory=dict)

    # Knowledge base reference
    knowledge_base_id: Optional[str] = None

    # Tools/functions enabled
    enabled_tools: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "business_id": self.business_id,
            "persona_name": self.persona.name,
            "dialog_flows": self.dialog_flows,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class GenerationRequest:
    """Request to generate an agent."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Business info
    business_info: BusinessInfo = field(default_factory=BusinessInfo)

    # Customization options
    persona_preferences: Dict[str, Any] = field(default_factory=dict)
    voice_preferences: Dict[str, Any] = field(default_factory=dict)
    behavior_preferences: Dict[str, Any] = field(default_factory=dict)

    # Features to enable
    enable_appointment_booking: bool = False
    enable_order_taking: bool = False
    enable_lead_qualification: bool = False
    enable_support_tickets: bool = False

    # Integrations
    integrations: List[str] = field(default_factory=list)

    # Priority flows
    priority_call_reasons: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    organization_id: Optional[str] = None


@dataclass
class GenerationResult:
    """Result of agent generation."""

    # Request reference
    request_id: str = ""

    # Status
    status: GenerationStatus = GenerationStatus.PENDING
    error: Optional[str] = None

    # Generated config
    agent_config: Optional[AgentConfig] = None

    # Generated artifacts
    system_prompt: str = ""
    dialog_flows: List[Dict[str, Any]] = field(default_factory=list)
    intent_definitions: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_base_id: Optional[str] = None

    # Metrics
    processing_time_ms: float = 0.0
    tokens_used: int = 0

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "error": self.error,
            "agent_config": self.agent_config.to_dict() if self.agent_config else None,
            "processing_time_ms": self.processing_time_ms,
        }


__all__ = [
    # Enums
    "BusinessCategory",
    "GenerationStatus",
    # Business Info
    "ContactInfo",
    "BusinessHours",
    "HoursOfOperation",
    "ProductInfo",
    "ServiceInfo",
    "FAQEntry",
    "PolicyInfo",
    "TeamMember",
    "LocationInfo",
    "BusinessInfo",
    # Agent Config
    "VoiceConfig",
    "GreetingConfig",
    "TransferConfig",
    "EscalationConfig",
    "ComplianceConfig",
    "BehaviorConfig",
    "AgentPersona",
    "AgentConfig",
    # Generation
    "GenerationRequest",
    "GenerationResult",
]
