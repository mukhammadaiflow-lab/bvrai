"""
Industry Intelligence Base Types Module

This module defines the core data types for industry-specific
intelligence, compliance, and behavioral patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


class IndustryType(str, Enum):
    """Industry classifications."""

    # Healthcare
    HEALTHCARE_GENERAL = "healthcare_general"
    HEALTHCARE_DENTAL = "healthcare_dental"
    HEALTHCARE_MENTAL = "healthcare_mental"
    HEALTHCARE_VETERINARY = "healthcare_veterinary"
    HEALTHCARE_PHARMACY = "healthcare_pharmacy"
    HEALTHCARE_OPTOMETRY = "healthcare_optometry"

    # Professional Services
    LEGAL = "legal"
    ACCOUNTING = "accounting"
    FINANCIAL_SERVICES = "financial_services"
    INSURANCE = "insurance"
    REAL_ESTATE = "real_estate"
    CONSULTING = "consulting"

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
    EVENT_PLANNING = "event_planning"

    # Personal Services
    SALON = "salon"
    SPA = "spa"
    FITNESS = "fitness"
    PHOTOGRAPHY = "photography"

    # Retail
    RETAIL_GENERAL = "retail_general"
    ECOMMERCE = "ecommerce"

    # Education
    EDUCATION_K12 = "education_k12"
    EDUCATION_HIGHER = "education_higher"
    TUTORING = "tutoring"

    # Technology
    IT_SERVICES = "it_services"
    SOFTWARE = "software"

    # Other
    GOVERNMENT = "government"
    NONPROFIT = "nonprofit"
    OTHER = "other"


class ComplianceLevel(str, Enum):
    """Compliance requirement levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ADVISORY = "advisory"


class RegulationType(str, Enum):
    """Types of regulations."""

    # Healthcare
    HIPAA = "hipaa"
    HITECH = "hitech"
    FDA = "fda"

    # Financial
    PCI_DSS = "pci_dss"
    SOX = "sox"
    GLBA = "glba"
    FINRA = "finra"
    SEC = "sec"

    # Privacy
    GDPR = "gdpr"
    CCPA = "ccpa"
    COPPA = "coppa"

    # Telecom
    TCPA = "tcpa"
    DNC = "do_not_call"
    FCC = "fcc"

    # Legal
    ABA_ETHICS = "aba_ethics"
    ATTORNEY_CLIENT = "attorney_client_privilege"

    # Insurance
    STATE_INSURANCE = "state_insurance"
    NAIC = "naic"

    # Real Estate
    RESPA = "respa"
    FAIR_HOUSING = "fair_housing"

    # General
    ADA = "ada"
    EEOC = "eeoc"
    OSHA = "osha"
    FTC = "ftc"


class ConversationPhase(str, Enum):
    """Phases of a customer conversation."""

    GREETING = "greeting"
    IDENTIFICATION = "identification"
    DISCOVERY = "discovery"
    QUALIFICATION = "qualification"
    INFORMATION = "information"
    RECOMMENDATION = "recommendation"
    OBJECTION_HANDLING = "objection_handling"
    COMMITMENT = "commitment"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"


class SentimentCategory(str, Enum):
    """Customer sentiment categories."""

    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    CONFUSED = "confused"
    ANXIOUS = "anxious"
    URGENT = "urgent"
    APPRECIATIVE = "appreciative"


@dataclass
class ComplianceRequirement:
    """A specific compliance requirement."""

    id: str = ""
    regulation: RegulationType = RegulationType.FTC
    title: str = ""
    description: str = ""
    level: ComplianceLevel = ComplianceLevel.MEDIUM

    # Requirements
    required_disclosures: List[str] = field(default_factory=list)
    prohibited_phrases: List[str] = field(default_factory=list)
    required_phrases: List[str] = field(default_factory=list)
    data_handling: Dict[str, str] = field(default_factory=dict)

    # Actions
    on_violation: str = ""  # Action to take on violation
    escalation_required: bool = False

    # Metadata
    effective_date: Optional[datetime] = None
    reference_url: str = ""


@dataclass
class TermDefinition:
    """Definition of an industry term."""

    term: str = ""
    definition: str = ""
    abbreviation: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    customer_friendly: str = ""  # Simple explanation for customers


@dataclass
class ConversationPattern:
    """A common conversation pattern for an industry."""

    id: str = ""
    name: str = ""
    description: str = ""
    phase: ConversationPhase = ConversationPhase.DISCOVERY

    # Pattern details
    trigger_intents: List[str] = field(default_factory=list)
    trigger_keywords: List[str] = field(default_factory=list)
    response_templates: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)

    # Conditions
    sentiment_conditions: List[SentimentCategory] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)

    # Outcomes
    expected_outcomes: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)


@dataclass
class BehaviorGuideline:
    """Behavioral guideline for agent interactions."""

    id: str = ""
    category: str = ""
    guideline: str = ""
    rationale: str = ""
    priority: int = 0

    # Examples
    do_examples: List[str] = field(default_factory=list)
    dont_examples: List[str] = field(default_factory=list)

    # Conditions
    applies_when: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)


@dataclass
class ServiceOffering:
    """A service offering for an industry."""

    id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""

    # Details
    typical_duration: str = ""
    price_range: str = ""
    prerequisites: List[str] = field(default_factory=list)

    # Selling points
    benefits: List[str] = field(default_factory=list)
    common_objections: List[str] = field(default_factory=list)
    objection_responses: Dict[str, str] = field(default_factory=dict)

    # Questions to ask
    qualification_questions: List[str] = field(default_factory=list)


@dataclass
class EmergencyProtocol:
    """Protocol for handling emergencies."""

    id: str = ""
    trigger_keywords: List[str] = field(default_factory=list)
    trigger_phrases: List[str] = field(default_factory=list)
    severity: str = ""  # critical, high, medium

    # Actions
    immediate_response: str = ""
    transfer_required: bool = True
    transfer_to: str = ""  # Role or number
    notification_required: bool = False
    notification_contacts: List[str] = field(default_factory=list)

    # Script
    acknowledgment_script: str = ""
    hold_script: str = ""


@dataclass
class IndustryMetric:
    """Key performance metric for an industry."""

    id: str = ""
    name: str = ""
    description: str = ""
    unit: str = ""  # percentage, count, duration, currency

    # Benchmarks
    industry_average: float = 0.0
    good_threshold: float = 0.0
    excellent_threshold: float = 0.0

    # Calculation
    calculation_method: str = ""
    data_sources: List[str] = field(default_factory=list)


@dataclass
class SeasonalPattern:
    """Seasonal pattern affecting the industry."""

    id: str = ""
    name: str = ""
    description: str = ""

    # Timing
    start_month: int = 1
    end_month: int = 12
    peak_months: List[int] = field(default_factory=list)

    # Impact
    demand_multiplier: float = 1.0
    staffing_recommendation: str = ""
    marketing_focus: str = ""

    # Conversation adjustments
    talking_points: List[str] = field(default_factory=list)
    urgency_messaging: str = ""


@dataclass
class IndustryProfile:
    """Complete profile for an industry."""

    industry_type: IndustryType = IndustryType.OTHER
    name: str = ""
    description: str = ""

    # Core characteristics
    typical_customer_segments: List[str] = field(default_factory=list)
    common_pain_points: List[str] = field(default_factory=list)
    value_propositions: List[str] = field(default_factory=list)

    # Compliance
    applicable_regulations: List[RegulationType] = field(default_factory=list)
    compliance_requirements: List[ComplianceRequirement] = field(default_factory=list)

    # Terminology
    key_terms: List[TermDefinition] = field(default_factory=list)

    # Conversation
    conversation_patterns: List[ConversationPattern] = field(default_factory=list)
    behavior_guidelines: List[BehaviorGuideline] = field(default_factory=list)

    # Services
    common_services: List[ServiceOffering] = field(default_factory=list)

    # Emergency handling
    emergency_protocols: List[EmergencyProtocol] = field(default_factory=list)

    # Performance
    key_metrics: List[IndustryMetric] = field(default_factory=list)

    # Seasonality
    seasonal_patterns: List[SeasonalPattern] = field(default_factory=list)

    # Communication style
    tone: str = "professional"  # professional, friendly, formal, casual
    formality_level: int = 3  # 1-5 scale
    empathy_level: int = 3  # 1-5 scale
    urgency_sensitivity: int = 3  # 1-5 scale

    # Business hours consideration
    typical_business_hours: Dict[str, str] = field(default_factory=dict)
    after_hours_handling: str = ""

    # Metadata
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IndustryContext:
    """Runtime context for industry intelligence."""

    profile: IndustryProfile
    active_compliance: List[ComplianceRequirement] = field(default_factory=list)
    current_season: Optional[SeasonalPattern] = None

    # Conversation state
    current_phase: ConversationPhase = ConversationPhase.GREETING
    detected_sentiment: SentimentCategory = SentimentCategory.NEUTRAL
    identified_pain_points: List[str] = field(default_factory=list)
    discussed_services: List[str] = field(default_factory=list)

    # Flags
    emergency_detected: bool = False
    compliance_warning: bool = False
    escalation_recommended: bool = False


__all__ = [
    "IndustryType",
    "ComplianceLevel",
    "RegulationType",
    "ConversationPhase",
    "SentimentCategory",
    "ComplianceRequirement",
    "TermDefinition",
    "ConversationPattern",
    "BehaviorGuideline",
    "ServiceOffering",
    "EmergencyProtocol",
    "IndustryMetric",
    "SeasonalPattern",
    "IndustryProfile",
    "IndustryContext",
]
