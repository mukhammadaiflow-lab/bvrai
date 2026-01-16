"""
Business Analyzer Module

This module analyzes business information to extract insights
and prepare data for agent generation.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .base import (
    BusinessInfo,
    BusinessCategory,
    FAQEntry,
    ServiceInfo,
    ProductInfo,
)


logger = logging.getLogger(__name__)


@dataclass
class KeyEntities:
    """Key entities extracted from business info."""

    # Business entities
    business_name: str = ""
    business_type: str = ""
    industry: str = ""

    # Products and services
    product_names: List[str] = field(default_factory=list)
    service_names: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # People
    team_names: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

    # Locations
    locations: List[str] = field(default_factory=list)
    service_areas: List[str] = field(default_factory=list)

    # Pricing
    price_points: List[str] = field(default_factory=list)

    # Dates/times
    business_hours: List[str] = field(default_factory=list)

    # Contact info
    phone_numbers: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)

    # Custom terms
    industry_terms: List[str] = field(default_factory=list)
    brand_terms: List[str] = field(default_factory=list)


@dataclass
class ConversationTopics:
    """Topics the agent should handle."""

    # Primary topics
    primary_topics: List[str] = field(default_factory=list)

    # Secondary topics
    secondary_topics: List[str] = field(default_factory=list)

    # FAQ topics
    faq_topics: List[str] = field(default_factory=list)

    # Off-topic handling
    redirect_topics: List[str] = field(default_factory=list)
    prohibited_topics: List[str] = field(default_factory=list)

    # Topic relationships
    topic_hierarchy: Dict[str, List[str]] = field(default_factory=dict)

    def get_all_topics(self) -> List[str]:
        """Get all topics."""
        return list(set(
            self.primary_topics +
            self.secondary_topics +
            self.faq_topics
        ))


@dataclass
class BusinessInsights:
    """Insights extracted from business analysis."""

    # Business profile
    business_size: str = "small"  # small, medium, large
    business_maturity: str = "established"  # startup, growing, established
    service_complexity: str = "moderate"  # simple, moderate, complex

    # Customer profile
    customer_type: str = "b2c"  # b2b, b2c, both
    typical_call_duration: int = 5  # minutes
    call_volume_estimate: str = "moderate"

    # Communication style
    recommended_tone: str = "professional"
    formality_level: str = "professional"
    urgency_level: str = "normal"

    # Key differentiators
    main_differentiators: List[str] = field(default_factory=list)
    competitive_advantages: List[str] = field(default_factory=list)

    # Pain points
    common_customer_problems: List[str] = field(default_factory=list)
    resolution_patterns: List[str] = field(default_factory=list)

    # Upsell opportunities
    upsell_opportunities: List[str] = field(default_factory=list)
    cross_sell_products: List[str] = field(default_factory=list)

    # Compliance requirements
    requires_hipaa: bool = False
    requires_pci: bool = False
    industry_regulations: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result."""

    # Extracted data
    entities: KeyEntities = field(default_factory=KeyEntities)
    topics: ConversationTopics = field(default_factory=ConversationTopics)
    insights: BusinessInsights = field(default_factory=BusinessInsights)

    # Generated intents
    suggested_intents: List[Dict[str, Any]] = field(default_factory=list)

    # Generated slots
    suggested_slots: List[Dict[str, Any]] = field(default_factory=list)

    # Knowledge gaps
    missing_information: List[str] = field(default_factory=list)
    suggested_faqs: List[str] = field(default_factory=list)

    # Processing metadata
    analysis_confidence: float = 0.0
    processing_notes: List[str] = field(default_factory=list)


class BusinessAnalyzer:
    """
    Analyzes business information to extract insights for agent generation.

    Performs entity extraction, topic identification, and generates
    recommendations for agent configuration.
    """

    # Industry-specific patterns
    INDUSTRY_PATTERNS = {
        BusinessCategory.HEALTHCARE: {
            "terms": ["appointment", "patient", "doctor", "medical", "health", "insurance", "prescription"],
            "intents": ["schedule_appointment", "check_insurance", "request_prescription", "ask_symptoms"],
            "compliance": ["hipaa"],
        },
        BusinessCategory.DENTAL: {
            "terms": ["appointment", "cleaning", "dentist", "dental", "teeth", "emergency"],
            "intents": ["schedule_appointment", "check_insurance", "emergency_dental", "ask_procedures"],
            "compliance": ["hipaa"],
        },
        BusinessCategory.LEGAL: {
            "terms": ["consultation", "attorney", "lawyer", "case", "legal", "court"],
            "intents": ["schedule_consultation", "case_status", "ask_services", "get_quote"],
            "compliance": ["confidentiality"],
        },
        BusinessCategory.REAL_ESTATE: {
            "terms": ["listing", "property", "agent", "showing", "mortgage", "buyer", "seller"],
            "intents": ["schedule_showing", "property_inquiry", "market_info", "agent_info"],
            "compliance": [],
        },
        BusinessCategory.PLUMBING: {
            "terms": ["emergency", "leak", "repair", "install", "plumber", "drain", "pipe"],
            "intents": ["schedule_service", "emergency_service", "get_quote", "service_area"],
            "compliance": [],
        },
        BusinessCategory.HVAC: {
            "terms": ["heating", "cooling", "AC", "furnace", "maintenance", "repair", "install"],
            "intents": ["schedule_service", "emergency_service", "get_quote", "maintenance_plan"],
            "compliance": [],
        },
        BusinessCategory.AUTO_DEALERSHIP: {
            "terms": ["vehicle", "car", "trade-in", "financing", "test drive", "inventory"],
            "intents": ["schedule_test_drive", "check_inventory", "get_quote", "financing_options"],
            "compliance": [],
        },
        BusinessCategory.RESTAURANT: {
            "terms": ["reservation", "menu", "hours", "location", "catering", "takeout", "delivery"],
            "intents": ["make_reservation", "check_hours", "ask_menu", "order_takeout"],
            "compliance": [],
        },
        BusinessCategory.SALON: {
            "terms": ["appointment", "stylist", "haircut", "color", "treatment", "availability"],
            "intents": ["schedule_appointment", "check_availability", "ask_services", "get_pricing"],
            "compliance": [],
        },
        BusinessCategory.INSURANCE: {
            "terms": ["policy", "claim", "coverage", "premium", "deductible", "quote"],
            "intents": ["get_quote", "file_claim", "check_coverage", "policy_inquiry"],
            "compliance": ["privacy"],
        },
    }

    # Common intent templates
    COMMON_INTENTS = [
        {
            "name": "greeting",
            "description": "Customer greeting",
            "examples": ["hello", "hi", "good morning"],
        },
        {
            "name": "hours_inquiry",
            "description": "Ask about business hours",
            "examples": ["what are your hours", "are you open", "when do you close"],
        },
        {
            "name": "location_inquiry",
            "description": "Ask about location",
            "examples": ["where are you located", "what's your address", "how do I get there"],
        },
        {
            "name": "contact_inquiry",
            "description": "Ask for contact info",
            "examples": ["what's your email", "phone number", "how can I reach you"],
        },
        {
            "name": "pricing_inquiry",
            "description": "Ask about prices",
            "examples": ["how much does it cost", "what's the price", "do you have pricing"],
        },
        {
            "name": "service_inquiry",
            "description": "Ask about services",
            "examples": ["what services do you offer", "do you do", "can you help with"],
        },
        {
            "name": "human_transfer",
            "description": "Request to speak with human",
            "examples": ["speak to someone", "real person", "talk to a human"],
        },
        {
            "name": "complaint",
            "description": "Customer complaint",
            "examples": ["I have a complaint", "not satisfied", "problem with"],
        },
    ]

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        enable_llm_analysis: bool = False,
    ):
        """
        Initialize analyzer.

        Args:
            llm_provider: Optional LLM provider for advanced analysis
            enable_llm_analysis: Whether to use LLM for deeper analysis
        """
        self._llm_provider = llm_provider
        self._enable_llm = enable_llm_analysis

    async def analyze(self, business_info: BusinessInfo) -> AnalysisResult:
        """
        Analyze business information.

        Args:
            business_info: Business information to analyze

        Returns:
            Analysis result with entities, topics, and insights
        """
        result = AnalysisResult()

        # Extract entities
        result.entities = self._extract_entities(business_info)

        # Identify topics
        result.topics = self._identify_topics(business_info)

        # Generate insights
        result.insights = self._generate_insights(business_info)

        # Suggest intents
        result.suggested_intents = self._suggest_intents(business_info)

        # Suggest slots
        result.suggested_slots = self._suggest_slots(business_info)

        # Identify gaps
        result.missing_information = self._identify_gaps(business_info)
        result.suggested_faqs = self._suggest_faqs(business_info)

        # LLM-enhanced analysis if available
        if self._enable_llm and self._llm_provider:
            result = await self._enhance_with_llm(business_info, result)

        # Calculate confidence
        result.analysis_confidence = self._calculate_confidence(business_info, result)

        return result

    def _extract_entities(self, business_info: BusinessInfo) -> KeyEntities:
        """Extract key entities from business info."""
        entities = KeyEntities()

        # Business info
        entities.business_name = business_info.name
        entities.business_type = business_info.category.value
        entities.industry = self._get_industry_name(business_info.category)

        # Products
        for product in business_info.products:
            entities.product_names.append(product.name)
            if product.category and product.category not in entities.categories:
                entities.categories.append(product.category)

        # Services
        for service in business_info.services:
            entities.service_names.append(service.name)
            if service.category and service.category not in entities.categories:
                entities.categories.append(service.category)
            entities.service_areas.extend(service.service_area)

        # Team
        for member in business_info.team_members:
            entities.team_names.append(member.name)
            if member.role and member.role not in entities.roles:
                entities.roles.append(member.role)

        # Locations
        for location in business_info.locations:
            location_str = f"{location.city}, {location.state}" if location.city else location.address
            if location_str:
                entities.locations.append(location_str)

        # Contact info
        if business_info.contact.phone:
            entities.phone_numbers.append(business_info.contact.phone)
        if business_info.contact.email:
            entities.emails.append(business_info.contact.email)

        # Pricing
        for product in business_info.products:
            if product.price:
                entities.price_points.append(f"${product.price}")
            elif product.price_range:
                entities.price_points.append(product.price_range)

        for service in business_info.services:
            if service.price:
                entities.price_points.append(f"${service.price}")
            elif service.price_range:
                entities.price_points.append(service.price_range)

        # Industry terms
        industry_patterns = self.INDUSTRY_PATTERNS.get(business_info.category, {})
        entities.industry_terms = industry_patterns.get("terms", [])

        # Brand terms from description and about
        entities.brand_terms = self._extract_brand_terms(business_info)

        return entities

    def _identify_topics(self, business_info: BusinessInfo) -> ConversationTopics:
        """Identify conversation topics."""
        topics = ConversationTopics()

        # Primary topics from common call reasons
        topics.primary_topics = list(business_info.common_call_reasons)

        # Add industry-specific primary topics
        industry_patterns = self.INDUSTRY_PATTERNS.get(business_info.category, {})
        industry_intents = industry_patterns.get("intents", [])
        for intent in industry_intents:
            topic = intent.replace("_", " ")
            if topic not in topics.primary_topics:
                topics.primary_topics.append(topic)

        # Secondary topics from services
        for service in business_info.services:
            topic = f"about {service.name}"
            topics.secondary_topics.append(topic)

        # FAQ topics
        for faq in business_info.faqs:
            topics.faq_topics.append(faq.category)

        # Build topic hierarchy
        topics.topic_hierarchy = {
            "general": ["hours", "location", "contact"],
            "services": [s.name for s in business_info.services],
            "products": [p.name for p in business_info.products],
            "support": ["complaint", "feedback", "question"],
        }

        return topics

    def _generate_insights(self, business_info: BusinessInfo) -> BusinessInsights:
        """Generate business insights."""
        insights = BusinessInsights()

        # Business size estimation
        team_size = len(business_info.team_members)
        location_count = len(business_info.locations)

        if team_size > 50 or location_count > 5:
            insights.business_size = "large"
        elif team_size > 10 or location_count > 1:
            insights.business_size = "medium"
        else:
            insights.business_size = "small"

        # Service complexity
        service_count = len(business_info.services)
        product_count = len(business_info.products)

        if service_count + product_count > 20:
            insights.service_complexity = "complex"
        elif service_count + product_count > 5:
            insights.service_complexity = "moderate"
        else:
            insights.service_complexity = "simple"

        # Customer type inference
        b2b_keywords = ["enterprise", "business", "corporate", "commercial", "wholesale"]
        b2c_keywords = ["consumer", "residential", "personal", "individual", "home"]

        desc_lower = business_info.description.lower()
        b2b_matches = sum(1 for k in b2b_keywords if k in desc_lower)
        b2c_matches = sum(1 for k in b2c_keywords if k in desc_lower)

        if b2b_matches > b2c_matches:
            insights.customer_type = "b2b"
        elif b2c_matches > b2b_matches:
            insights.customer_type = "b2c"
        else:
            insights.customer_type = "both"

        # Recommended tone from brand settings
        insights.recommended_tone = business_info.brand_tone
        insights.formality_level = self._infer_formality(business_info)

        # Differentiators
        insights.main_differentiators = list(business_info.differentiators)

        # Compliance requirements
        industry_patterns = self.INDUSTRY_PATTERNS.get(business_info.category, {})
        compliance = industry_patterns.get("compliance", [])

        insights.requires_hipaa = "hipaa" in compliance
        insights.requires_pci = "pci" in compliance
        insights.industry_regulations = compliance

        # Common problems from FAQs
        for faq in business_info.faqs:
            if any(word in faq.question.lower() for word in ["problem", "issue", "not working", "help"]):
                insights.common_customer_problems.append(faq.question)

        return insights

    def _suggest_intents(self, business_info: BusinessInfo) -> List[Dict[str, Any]]:
        """Suggest intents based on business info."""
        intents = []

        # Add common intents
        intents.extend(self.COMMON_INTENTS.copy())

        # Add industry-specific intents
        industry_patterns = self.INDUSTRY_PATTERNS.get(business_info.category, {})
        for intent_name in industry_patterns.get("intents", []):
            intent = {
                "name": intent_name,
                "description": f"Industry-specific: {intent_name.replace('_', ' ')}",
                "examples": [],
                "industry_specific": True,
            }
            intents.append(intent)

        # Add service-specific intents
        for service in business_info.services:
            if service.booking_required:
                intent = {
                    "name": f"book_{self._slugify(service.name)}",
                    "description": f"Book {service.name}",
                    "examples": [
                        f"book {service.name}",
                        f"schedule {service.name}",
                        f"make appointment for {service.name}",
                    ],
                    "service_id": service.id,
                }
                intents.append(intent)

            # Service inquiry intent
            intent = {
                "name": f"inquire_{self._slugify(service.name)}",
                "description": f"Ask about {service.name}",
                "examples": [
                    f"tell me about {service.name}",
                    f"what is {service.name}",
                    f"how much is {service.name}",
                ],
                "service_id": service.id,
            }
            intents.append(intent)

        # Add FAQ-based intents
        for faq in business_info.faqs:
            intent = {
                "name": f"faq_{self._slugify(faq.category)}_{faq.id[:8]}",
                "description": faq.question,
                "examples": [faq.question] + faq.question_variations,
                "faq_id": faq.id,
            }
            intents.append(intent)

        return intents

    def _suggest_slots(self, business_info: BusinessInfo) -> List[Dict[str, Any]]:
        """Suggest slots based on business info."""
        slots = []

        # Common slots
        common_slots = [
            {
                "name": "customer_name",
                "type": "name",
                "description": "Customer's name",
                "required": False,
            },
            {
                "name": "phone_number",
                "type": "phone",
                "description": "Customer's phone number",
                "required": False,
            },
            {
                "name": "email",
                "type": "email",
                "description": "Customer's email",
                "required": False,
            },
        ]
        slots.extend(common_slots)

        # Service-specific slots
        for service in business_info.services:
            if service.booking_required:
                slots.append({
                    "name": f"appointment_date_{self._slugify(service.name)}",
                    "type": "date",
                    "description": f"Preferred date for {service.name}",
                    "service_id": service.id,
                })
                slots.append({
                    "name": f"appointment_time_{self._slugify(service.name)}",
                    "type": "time",
                    "description": f"Preferred time for {service.name}",
                    "service_id": service.id,
                })

        # Location slot if multiple locations
        if len(business_info.locations) > 1:
            slots.append({
                "name": "preferred_location",
                "type": "enum",
                "description": "Preferred location",
                "values": [loc.name for loc in business_info.locations],
            })

        # Team member slot if transfers enabled
        if business_info.team_members:
            slots.append({
                "name": "requested_team_member",
                "type": "enum",
                "description": "Requested team member",
                "values": [m.name for m in business_info.team_members if m.available_for_calls],
            })

        # Industry-specific slots
        if business_info.category in [BusinessCategory.HEALTHCARE, BusinessCategory.DENTAL]:
            slots.extend([
                {"name": "insurance_provider", "type": "string", "description": "Insurance provider"},
                {"name": "patient_dob", "type": "date", "description": "Patient date of birth"},
                {"name": "is_new_patient", "type": "boolean", "description": "New or existing patient"},
            ])
        elif business_info.category == BusinessCategory.AUTO_DEALERSHIP:
            slots.extend([
                {"name": "vehicle_interest", "type": "string", "description": "Vehicle of interest"},
                {"name": "trade_in", "type": "boolean", "description": "Has trade-in vehicle"},
            ])
        elif business_info.category in [BusinessCategory.PLUMBING, BusinessCategory.HVAC, BusinessCategory.ELECTRICAL]:
            slots.extend([
                {"name": "service_address", "type": "address", "description": "Service location"},
                {"name": "is_emergency", "type": "boolean", "description": "Emergency service needed"},
                {"name": "problem_description", "type": "string", "description": "Description of issue"},
            ])

        return slots

    def _identify_gaps(self, business_info: BusinessInfo) -> List[str]:
        """Identify missing information."""
        gaps = []

        # Check required fields
        if not business_info.name:
            gaps.append("Business name is missing")

        if not business_info.description:
            gaps.append("Business description is missing")

        if not business_info.contact.phone:
            gaps.append("Phone number is missing")

        # Check for minimal content
        if not business_info.services and not business_info.products:
            gaps.append("No services or products defined")

        if not business_info.faqs:
            gaps.append("No FAQs provided - recommend adding common questions")

        if not business_info.hours:
            gaps.append("Business hours not specified")

        # Check for industry-specific requirements
        if business_info.category in [BusinessCategory.HEALTHCARE, BusinessCategory.DENTAL]:
            if not any("insurance" in str(faq.question).lower() for faq in business_info.faqs):
                gaps.append("No insurance-related FAQs (recommended for healthcare)")

        # Check content depth
        if len(business_info.services) > 0:
            services_without_description = [s for s in business_info.services if not s.description]
            if services_without_description:
                gaps.append(f"{len(services_without_description)} services lack descriptions")

            services_without_pricing = [s for s in business_info.services if not s.price and not s.price_range]
            if services_without_pricing:
                gaps.append(f"{len(services_without_pricing)} services lack pricing information")

        return gaps

    def _suggest_faqs(self, business_info: BusinessInfo) -> List[str]:
        """Suggest additional FAQs."""
        suggestions = []
        existing_topics = {faq.category.lower() for faq in business_info.faqs}
        existing_questions = {faq.question.lower() for faq in business_info.faqs}

        # Common FAQs to check
        common_faqs = [
            ("What are your hours of operation?", "hours"),
            ("Where are you located?", "location"),
            ("How do I contact you?", "contact"),
            ("Do you offer free estimates?", "pricing"),
            ("What payment methods do you accept?", "payment"),
            ("Do you offer financing?", "financing"),
            ("What is your cancellation policy?", "policy"),
            ("Do you offer warranties?", "warranty"),
            ("What areas do you serve?", "service_area"),
        ]

        for question, topic in common_faqs:
            if topic not in existing_topics and question.lower() not in existing_questions:
                suggestions.append(question)

        # Industry-specific FAQs
        industry_faqs = {
            BusinessCategory.HEALTHCARE: [
                "Do you accept my insurance?",
                "How do I request prescription refills?",
                "What should I bring to my first appointment?",
            ],
            BusinessCategory.DENTAL: [
                "Do you accept my dental insurance?",
                "What is included in a dental cleaning?",
                "Do you offer payment plans?",
            ],
            BusinessCategory.PLUMBING: [
                "Do you offer 24/7 emergency service?",
                "How quickly can you respond to emergencies?",
                "Do you provide free estimates?",
            ],
            BusinessCategory.AUTO_DEALERSHIP: [
                "Can I schedule a test drive?",
                "What financing options do you offer?",
                "Do you accept trade-ins?",
            ],
            BusinessCategory.RESTAURANT: [
                "Do you take reservations?",
                "Do you offer takeout or delivery?",
                "Do you accommodate dietary restrictions?",
            ],
        }

        if business_info.category in industry_faqs:
            for question in industry_faqs[business_info.category]:
                if question.lower() not in existing_questions:
                    suggestions.append(question)

        return suggestions[:10]  # Limit suggestions

    async def _enhance_with_llm(
        self,
        business_info: BusinessInfo,
        result: AnalysisResult,
    ) -> AnalysisResult:
        """Enhance analysis with LLM."""
        if not self._llm_provider:
            return result

        try:
            # Prepare context
            context = self._prepare_llm_context(business_info)

            # Generate enhanced insights
            prompt = f"""Analyze this business information and provide insights:

Business: {business_info.name}
Category: {business_info.category.value}
Description: {business_info.description}

Services: {[s.name for s in business_info.services]}
Products: {[p.name for p in business_info.products]}

Please provide:
1. Key customer pain points this business likely addresses
2. Suggested conversation topics for a phone agent
3. Potential upsell opportunities
4. Recommended communication tone

Format as JSON with keys: pain_points, topics, upsells, tone"""

            response = await self._llm_provider.generate(prompt)

            # Parse and merge insights
            if response:
                # Update result with LLM insights
                result.processing_notes.append("Enhanced with LLM analysis")

        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            result.processing_notes.append(f"LLM enhancement skipped: {e}")

        return result

    def _calculate_confidence(
        self,
        business_info: BusinessInfo,
        result: AnalysisResult,
    ) -> float:
        """Calculate analysis confidence score."""
        score = 0.0
        max_score = 100.0

        # Basic info completeness (30 points)
        if business_info.name:
            score += 10
        if business_info.description:
            score += 10
        if business_info.category != BusinessCategory.OTHER:
            score += 10

        # Contact info (10 points)
        if business_info.contact.phone:
            score += 5
        if business_info.contact.email:
            score += 5

        # Content depth (30 points)
        if business_info.services:
            score += min(len(business_info.services) * 2, 10)
        if business_info.products:
            score += min(len(business_info.products) * 2, 10)
        if business_info.faqs:
            score += min(len(business_info.faqs), 10)

        # Rich content (20 points)
        if business_info.about_us:
            score += 5
        if business_info.policies:
            score += 5
        if business_info.team_members:
            score += 5
        if business_info.hours:
            score += 5

        # Gaps penalty (10 points max penalty)
        gap_penalty = min(len(result.missing_information) * 2, 10)
        score = max(0, score - gap_penalty)

        return score / max_score

    def _get_industry_name(self, category: BusinessCategory) -> str:
        """Get human-readable industry name."""
        names = {
            BusinessCategory.HEALTHCARE: "Healthcare",
            BusinessCategory.DENTAL: "Dental",
            BusinessCategory.MEDICAL_PRACTICE: "Medical Practice",
            BusinessCategory.LEGAL: "Legal Services",
            BusinessCategory.REAL_ESTATE: "Real Estate",
            BusinessCategory.PLUMBING: "Plumbing",
            BusinessCategory.HVAC: "HVAC",
            BusinessCategory.AUTO_DEALERSHIP: "Auto Dealership",
            BusinessCategory.RESTAURANT: "Restaurant",
            BusinessCategory.SALON: "Salon & Spa",
            BusinessCategory.INSURANCE: "Insurance",
        }
        return names.get(category, category.value.replace("_", " ").title())

    def _infer_formality(self, business_info: BusinessInfo) -> str:
        """Infer formality level from business info."""
        formal_categories = {
            BusinessCategory.LEGAL,
            BusinessCategory.FINANCIAL_SERVICES,
            BusinessCategory.INSURANCE,
            BusinessCategory.HEALTHCARE,
        }

        casual_categories = {
            BusinessCategory.RESTAURANT,
            BusinessCategory.SALON,
            BusinessCategory.FITNESS,
            BusinessCategory.RETAIL_STORE,
        }

        if business_info.category in formal_categories:
            return "formal"
        elif business_info.category in casual_categories:
            return "casual"
        else:
            return "professional"

    def _extract_brand_terms(self, business_info: BusinessInfo) -> List[str]:
        """Extract brand-specific terms from business content."""
        terms = set()

        # Extract from name
        name_words = business_info.name.split()
        terms.update(word for word in name_words if len(word) > 2)

        # Extract from tagline
        if business_info.tagline:
            terms.update(
                word for word in business_info.tagline.split()
                if len(word) > 3 and word.isalpha()
            )

        # Extract from unique value proposition
        if business_info.unique_value_proposition:
            terms.update(
                word for word in business_info.unique_value_proposition.split()
                if len(word) > 4 and word.isalpha()
            )

        return list(terms)[:20]

    def _slugify(self, text: str) -> str:
        """Convert text to slug format."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '_', text)
        text = text.strip('_')
        return text

    def _prepare_llm_context(self, business_info: BusinessInfo) -> str:
        """Prepare business info as context for LLM."""
        parts = [
            f"Business Name: {business_info.name}",
            f"Industry: {business_info.category.value}",
            f"Description: {business_info.description}",
        ]

        if business_info.services:
            parts.append(f"Services: {', '.join(s.name for s in business_info.services[:10])}")

        if business_info.products:
            parts.append(f"Products: {', '.join(p.name for p in business_info.products[:10])}")

        if business_info.about_us:
            parts.append(f"About: {business_info.about_us[:500]}")

        return "\n".join(parts)


__all__ = [
    "BusinessAnalyzer",
    "AnalysisResult",
    "BusinessInsights",
    "ConversationTopics",
    "KeyEntities",
]
