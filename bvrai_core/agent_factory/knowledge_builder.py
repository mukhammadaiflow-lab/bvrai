"""
Knowledge Builder Module

This module builds knowledge bases from business information,
preparing content for RAG retrieval and agent responses.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from .base import (
    BusinessInfo,
    BusinessCategory,
    FAQEntry,
    ServiceInfo,
    ProductInfo,
    PolicyInfo,
)


logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    """Types of knowledge content."""
    FAQ = "faq"
    SERVICE = "service"
    PRODUCT = "product"
    POLICY = "policy"
    CONTACT = "contact"
    HOURS = "hours"
    LOCATION = "location"
    ABOUT = "about"
    TEAM = "team"
    DOCUMENT = "document"
    CUSTOM = "custom"


@dataclass
class KnowledgeSource:
    """Source of knowledge content."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: KnowledgeType = KnowledgeType.CUSTOM
    name: str = ""

    # Content
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source reference
    source_id: Optional[str] = None  # ID of original entity (FAQ, service, etc.)
    source_path: Optional[str] = None  # Path to document if applicable

    # Processing status
    processed: bool = False
    chunk_ids: List[str] = field(default_factory=list)


@dataclass
class KnowledgeConfig:
    """Configuration for knowledge building."""

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embedding settings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # Vector store settings
    vector_store_type: str = "memory"
    collection_name: str = "business_knowledge"

    # Processing options
    include_faqs: bool = True
    include_services: bool = True
    include_products: bool = True
    include_policies: bool = True
    include_contact_info: bool = True
    include_team_info: bool = True
    include_documents: bool = True

    # Enhancement options
    generate_qa_pairs: bool = True
    expand_abbreviations: bool = True


@dataclass
class ProcessedKnowledge:
    """Result of knowledge processing."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    business_id: str = ""

    # Sources processed
    sources: List[KnowledgeSource] = field(default_factory=list)
    total_sources: int = 0

    # Chunks created
    total_chunks: int = 0
    chunk_ids: List[str] = field(default_factory=list)

    # Vector store reference
    collection_name: str = ""
    vector_store_type: str = ""

    # Processing metadata
    processed_at: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "business_id": self.business_id,
            "total_sources": self.total_sources,
            "total_chunks": self.total_chunks,
            "collection_name": self.collection_name,
            "processed_at": self.processed_at.isoformat(),
        }


class KnowledgeBuilder:
    """
    Builds knowledge bases from business information.

    Processes business data into searchable knowledge chunks
    for RAG retrieval.
    """

    def __init__(
        self,
        config: Optional[KnowledgeConfig] = None,
        ingestion_pipeline: Optional[Any] = None,
    ):
        """
        Initialize knowledge builder.

        Args:
            config: Knowledge configuration
            ingestion_pipeline: Optional pre-configured ingestion pipeline
        """
        self.config = config or KnowledgeConfig()
        self._ingestion_pipeline = ingestion_pipeline

    async def build(
        self,
        business_info: BusinessInfo,
    ) -> ProcessedKnowledge:
        """
        Build knowledge base from business information.

        Args:
            business_info: Business information

        Returns:
            Processed knowledge result
        """
        import time
        start_time = time.time()

        result = ProcessedKnowledge(
            business_id=business_info.id,
            collection_name=f"{self.config.collection_name}_{business_info.id[:8]}",
            vector_store_type=self.config.vector_store_type,
        )

        # Extract knowledge sources
        sources = self._extract_sources(business_info)
        result.sources = sources
        result.total_sources = len(sources)

        # Initialize ingestion pipeline if not provided
        if not self._ingestion_pipeline:
            self._ingestion_pipeline = await self._create_pipeline(result.collection_name)

        # Process each source
        for source in sources:
            try:
                chunk_ids = await self._process_source(source)
                source.processed = True
                source.chunk_ids = chunk_ids
                result.chunk_ids.extend(chunk_ids)
            except Exception as e:
                logger.error(f"Failed to process source {source.name}: {e}")
                result.errors.append(f"Failed to process {source.name}: {str(e)}")

        result.total_chunks = len(result.chunk_ids)
        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def _extract_sources(self, business_info: BusinessInfo) -> List[KnowledgeSource]:
        """Extract knowledge sources from business info."""
        sources = []

        # Extract FAQs
        if self.config.include_faqs:
            for faq in business_info.faqs:
                source = self._create_faq_source(faq, business_info)
                sources.append(source)

        # Extract services
        if self.config.include_services:
            for service in business_info.services:
                source = self._create_service_source(service, business_info)
                sources.append(source)

        # Extract products
        if self.config.include_products:
            for product in business_info.products:
                source = self._create_product_source(product, business_info)
                sources.append(source)

        # Extract policies
        if self.config.include_policies:
            for policy in business_info.policies:
                source = self._create_policy_source(policy, business_info)
                sources.append(source)

        # Extract contact information
        if self.config.include_contact_info:
            source = self._create_contact_source(business_info)
            sources.append(source)

        # Extract hours
        if business_info.hours:
            source = self._create_hours_source(business_info)
            sources.append(source)

        # Extract about/description
        if business_info.about_us or business_info.description:
            source = self._create_about_source(business_info)
            sources.append(source)

        # Extract team info
        if self.config.include_team_info and business_info.team_members:
            source = self._create_team_source(business_info)
            sources.append(source)

        # Extract locations
        for location in business_info.locations:
            source = self._create_location_source(location, business_info)
            sources.append(source)

        # Generate additional QA pairs
        if self.config.generate_qa_pairs:
            qa_sources = self._generate_qa_pairs(business_info)
            sources.extend(qa_sources)

        return sources

    def _create_faq_source(
        self,
        faq: FAQEntry,
        business_info: BusinessInfo,
    ) -> KnowledgeSource:
        """Create knowledge source from FAQ."""
        content = f"""Question: {faq.question}
Answer: {faq.answer}"""

        if faq.question_variations:
            content += f"\n\nAlternative questions:\n- " + "\n- ".join(faq.question_variations)

        return KnowledgeSource(
            type=KnowledgeType.FAQ,
            name=f"FAQ: {faq.question[:50]}",
            content=content,
            source_id=faq.id,
            metadata={
                "category": faq.category,
                "priority": faq.priority,
                "business_name": business_info.name,
                "keywords": faq.keywords,
            },
        )

    def _create_service_source(
        self,
        service: ServiceInfo,
        business_info: BusinessInfo,
    ) -> KnowledgeSource:
        """Create knowledge source from service."""
        content_parts = [
            f"Service: {service.name}",
            f"Description: {service.description}",
        ]

        if service.price:
            content_parts.append(f"Price: ${service.price}")
        elif service.price_range:
            content_parts.append(f"Price range: {service.price_range}")

        if service.price_type:
            content_parts.append(f"Pricing type: {service.price_type}")

        if service.duration_minutes:
            content_parts.append(f"Duration: approximately {service.duration_minutes} minutes")
        elif service.duration_range:
            content_parts.append(f"Duration: {service.duration_range}")

        if service.requirements:
            content_parts.append(f"Requirements: {', '.join(service.requirements)}")

        if service.booking_required:
            content_parts.append("Note: Appointment booking is required for this service.")

        if service.service_area:
            content_parts.append(f"Service area: {', '.join(service.service_area)}")

        return KnowledgeSource(
            type=KnowledgeType.SERVICE,
            name=f"Service: {service.name}",
            content="\n".join(content_parts),
            source_id=service.id,
            metadata={
                "category": service.category,
                "subcategory": service.subcategory,
                "business_name": business_info.name,
                "booking_required": service.booking_required,
                "available": service.available,
            },
        )

    def _create_product_source(
        self,
        product: ProductInfo,
        business_info: BusinessInfo,
    ) -> KnowledgeSource:
        """Create knowledge source from product."""
        content_parts = [
            f"Product: {product.name}",
            f"Description: {product.description}",
        ]

        if product.price:
            content_parts.append(f"Price: ${product.price}")
        elif product.price_range:
            content_parts.append(f"Price range: {product.price_range}")

        if product.features:
            content_parts.append(f"Features: {', '.join(product.features)}")

        if product.specifications:
            specs = [f"{k}: {v}" for k, v in product.specifications.items()]
            content_parts.append(f"Specifications: {', '.join(specs)}")

        content_parts.append(f"Availability: {'In stock' if product.in_stock else 'Out of stock'}")

        if product.lead_time:
            content_parts.append(f"Lead time: {product.lead_time}")

        return KnowledgeSource(
            type=KnowledgeType.PRODUCT,
            name=f"Product: {product.name}",
            content="\n".join(content_parts),
            source_id=product.id,
            metadata={
                "category": product.category,
                "business_name": business_info.name,
                "in_stock": product.in_stock,
            },
        )

    def _create_policy_source(
        self,
        policy: PolicyInfo,
        business_info: BusinessInfo,
    ) -> KnowledgeSource:
        """Create knowledge source from policy."""
        content_parts = [
            f"Policy: {policy.name}",
            f"Type: {policy.type}",
            f"Summary: {policy.summary}",
        ]

        if policy.full_text:
            content_parts.append(f"\nFull Policy:\n{policy.full_text}")

        if policy.conditions:
            content_parts.append(f"\nConditions:\n- " + "\n- ".join(policy.conditions))

        if policy.exceptions:
            content_parts.append(f"\nExceptions:\n- " + "\n- ".join(policy.exceptions))

        if policy.time_limit_days:
            content_parts.append(f"\nTime limit: {policy.time_limit_days} days")

        if policy.fee_amount:
            content_parts.append(f"Fee: ${policy.fee_amount}")
        elif policy.fee_percentage:
            content_parts.append(f"Fee: {policy.fee_percentage}%")

        return KnowledgeSource(
            type=KnowledgeType.POLICY,
            name=f"Policy: {policy.name}",
            content="\n".join(content_parts),
            source_id=policy.id,
            metadata={
                "policy_type": policy.type,
                "business_name": business_info.name,
            },
        )

    def _create_contact_source(self, business_info: BusinessInfo) -> KnowledgeSource:
        """Create knowledge source from contact information."""
        contact = business_info.contact
        content_parts = [
            f"Contact information for {business_info.name}:",
        ]

        if contact.phone:
            content_parts.append(f"Phone: {contact.phone}")
        if contact.toll_free:
            content_parts.append(f"Toll-free: {contact.toll_free}")
        if contact.email:
            content_parts.append(f"Email: {contact.email}")
        if contact.website:
            content_parts.append(f"Website: {contact.website}")
        if contact.fax:
            content_parts.append(f"Fax: {contact.fax}")

        if contact.address:
            address = contact.address
            if contact.city:
                address += f", {contact.city}"
            if contact.state:
                address += f", {contact.state}"
            if contact.zip_code:
                address += f" {contact.zip_code}"
            content_parts.append(f"Address: {address}")

        # Social media
        social = []
        if contact.facebook:
            social.append(f"Facebook: {contact.facebook}")
        if contact.instagram:
            social.append(f"Instagram: {contact.instagram}")
        if contact.twitter:
            social.append(f"Twitter: {contact.twitter}")
        if contact.linkedin:
            social.append(f"LinkedIn: {contact.linkedin}")

        if social:
            content_parts.append("\nSocial Media:")
            content_parts.extend(social)

        return KnowledgeSource(
            type=KnowledgeType.CONTACT,
            name="Contact Information",
            content="\n".join(content_parts),
            metadata={
                "business_name": business_info.name,
                "has_phone": bool(contact.phone),
                "has_email": bool(contact.email),
            },
        )

    def _create_hours_source(self, business_info: BusinessInfo) -> KnowledgeSource:
        """Create knowledge source from business hours."""
        hours = business_info.hours
        content_parts = [
            f"Business hours for {business_info.name}:",
        ]

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i, day in enumerate(days):
            day_hours = hours.get_day_hours(i)
            if day_hours:
                if day_hours.is_closed:
                    content_parts.append(f"{day}: Closed")
                elif day_hours.open_time and day_hours.close_time:
                    open_str = day_hours.open_time.strftime("%I:%M %p")
                    close_str = day_hours.close_time.strftime("%I:%M %p")
                    content_parts.append(f"{day}: {open_str} - {close_str}")

                    if day_hours.break_start and day_hours.break_end:
                        break_start = day_hours.break_start.strftime("%I:%M %p")
                        break_end = day_hours.break_end.strftime("%I:%M %p")
                        content_parts.append(f"  Break: {break_start} - {break_end}")

        if hours.timezone:
            content_parts.append(f"\nTimezone: {hours.timezone}")

        if hours.holiday_hours:
            content_parts.append("\nHoliday Hours:")
            for holiday, hrs in hours.holiday_hours.items():
                if hrs.is_closed:
                    content_parts.append(f"  {holiday}: Closed")
                else:
                    content_parts.append(f"  {holiday}: Special hours")

        return KnowledgeSource(
            type=KnowledgeType.HOURS,
            name="Business Hours",
            content="\n".join(content_parts),
            metadata={
                "business_name": business_info.name,
                "timezone": hours.timezone,
            },
        )

    def _create_about_source(self, business_info: BusinessInfo) -> KnowledgeSource:
        """Create knowledge source from about information."""
        content_parts = [f"About {business_info.name}:"]

        if business_info.description:
            content_parts.append(f"\n{business_info.description}")

        if business_info.about_us:
            content_parts.append(f"\n{business_info.about_us}")

        if business_info.tagline:
            content_parts.append(f"\nTagline: {business_info.tagline}")

        if business_info.mission_statement:
            content_parts.append(f"\nMission: {business_info.mission_statement}")

        if business_info.unique_value_proposition:
            content_parts.append(f"\nWhat makes us unique: {business_info.unique_value_proposition}")

        if business_info.differentiators:
            content_parts.append(f"\nKey differentiators:")
            for diff in business_info.differentiators:
                content_parts.append(f"- {diff}")

        if business_info.awards_certifications:
            content_parts.append(f"\nAwards & Certifications:")
            for award in business_info.awards_certifications:
                content_parts.append(f"- {award}")

        return KnowledgeSource(
            type=KnowledgeType.ABOUT,
            name="About Us",
            content="\n".join(content_parts),
            metadata={
                "business_name": business_info.name,
                "category": business_info.category.value,
            },
        )

    def _create_team_source(self, business_info: BusinessInfo) -> KnowledgeSource:
        """Create knowledge source from team information."""
        content_parts = [f"Team at {business_info.name}:"]

        for member in business_info.team_members:
            member_info = [f"\n{member.name}"]
            if member.title:
                member_info.append(f"  Title: {member.title}")
            if member.role:
                member_info.append(f"  Role: {member.role}")
            if member.specialties:
                member_info.append(f"  Specialties: {', '.join(member.specialties)}")
            if member.certifications:
                member_info.append(f"  Certifications: {', '.join(member.certifications)}")
            if member.bio:
                member_info.append(f"  Bio: {member.bio}")

            content_parts.extend(member_info)

        return KnowledgeSource(
            type=KnowledgeType.TEAM,
            name="Team Members",
            content="\n".join(content_parts),
            metadata={
                "business_name": business_info.name,
                "team_size": len(business_info.team_members),
            },
        )

    def _create_location_source(
        self,
        location: Any,
        business_info: BusinessInfo,
    ) -> KnowledgeSource:
        """Create knowledge source from location."""
        content_parts = [f"Location: {location.name or 'Main Office'}"]

        address = location.address
        if location.city:
            address += f", {location.city}"
        if location.state:
            address += f", {location.state}"
        if location.zip_code:
            address += f" {location.zip_code}"
        content_parts.append(f"Address: {address}")

        if location.phone:
            content_parts.append(f"Phone: {location.phone}")

        if location.email:
            content_parts.append(f"Email: {location.email}")

        if location.services_available:
            content_parts.append(f"Services at this location: {', '.join(location.services_available)}")

        if location.parking_info:
            content_parts.append(f"Parking: {location.parking_info}")

        if location.accessibility_info:
            content_parts.append(f"Accessibility: {location.accessibility_info}")

        if location.is_primary:
            content_parts.append("This is our primary location.")

        return KnowledgeSource(
            type=KnowledgeType.LOCATION,
            name=f"Location: {location.name or 'Main'}",
            content="\n".join(content_parts),
            source_id=location.id,
            metadata={
                "business_name": business_info.name,
                "is_primary": location.is_primary,
                "city": location.city,
                "state": location.state,
            },
        )

    def _generate_qa_pairs(self, business_info: BusinessInfo) -> List[KnowledgeSource]:
        """Generate additional QA pairs from business info."""
        sources = []

        # Common questions based on business info
        if business_info.contact.phone:
            sources.append(KnowledgeSource(
                type=KnowledgeType.FAQ,
                name="FAQ: Phone number",
                content=f"Question: What is your phone number?\nAnswer: You can reach us at {business_info.contact.phone}.",
                metadata={"generated": True},
            ))

        if business_info.contact.email:
            sources.append(KnowledgeSource(
                type=KnowledgeType.FAQ,
                name="FAQ: Email",
                content=f"Question: What is your email address?\nAnswer: You can email us at {business_info.contact.email}.",
                metadata={"generated": True},
            ))

        if business_info.contact.address:
            address = business_info.contact.address
            if business_info.contact.city:
                address += f", {business_info.contact.city}"
            if business_info.contact.state:
                address += f", {business_info.contact.state}"

            sources.append(KnowledgeSource(
                type=KnowledgeType.FAQ,
                name="FAQ: Location",
                content=f"Question: Where are you located?\nAnswer: We are located at {address}.",
                metadata={"generated": True},
            ))

        if business_info.services:
            service_names = [s.name for s in business_info.services]
            sources.append(KnowledgeSource(
                type=KnowledgeType.FAQ,
                name="FAQ: Services offered",
                content=f"Question: What services do you offer?\nAnswer: We offer {', '.join(service_names)}.",
                metadata={"generated": True},
            ))

        # Industry-specific generated QA
        if business_info.category == BusinessCategory.HEALTHCARE:
            sources.append(KnowledgeSource(
                type=KnowledgeType.FAQ,
                name="FAQ: New patients",
                content="Question: Are you accepting new patients?\nAnswer: Yes, we welcome new patients! Please call to schedule your first appointment.",
                metadata={"generated": True, "industry": "healthcare"},
            ))

        if business_info.category in [BusinessCategory.PLUMBING, BusinessCategory.HVAC]:
            sources.append(KnowledgeSource(
                type=KnowledgeType.FAQ,
                name="FAQ: Emergency service",
                content="Question: Do you offer emergency service?\nAnswer: Yes, we provide emergency service. Please call our main number and select the emergency option.",
                metadata={"generated": True, "industry": "home_service"},
            ))

        return sources

    async def _create_pipeline(self, collection_name: str) -> Any:
        """Create ingestion pipeline."""
        try:
            from ..knowledge import IngestionPipeline, IngestionConfig

            config = IngestionConfig(
                embedding_provider=self.config.embedding_provider,
                embedding_model=self.config.embedding_model,
                vector_store_type=self.config.vector_store_type,
                collection_name=collection_name,
            )

            return IngestionPipeline(config=config)
        except ImportError:
            logger.warning("Knowledge module not available, using mock pipeline")
            return MockPipeline()

    async def _process_source(self, source: KnowledgeSource) -> List[str]:
        """Process a knowledge source."""
        if not self._ingestion_pipeline:
            logger.warning("No ingestion pipeline available")
            return []

        try:
            # Ingest the content
            result = await self._ingestion_pipeline.ingest_text(
                text=source.content,
                source_name=source.name,
            )

            return result.chunk_ids if hasattr(result, 'chunk_ids') else []
        except Exception as e:
            logger.error(f"Failed to process source: {e}")
            raise

    async def process_documents(
        self,
        documents: List[Union[str, Path]],
        business_info: BusinessInfo,
    ) -> ProcessedKnowledge:
        """
        Process additional documents into the knowledge base.

        Args:
            documents: List of document paths
            business_info: Business information for context

        Returns:
            Processing result
        """
        result = ProcessedKnowledge(
            business_id=business_info.id,
        )

        for doc_path in documents:
            try:
                source = KnowledgeSource(
                    type=KnowledgeType.DOCUMENT,
                    name=str(doc_path),
                    source_path=str(doc_path),
                    metadata={
                        "business_name": business_info.name,
                        "document_path": str(doc_path),
                    },
                )

                if self._ingestion_pipeline:
                    ingest_result = await self._ingestion_pipeline.ingest(doc_path)
                    source.chunk_ids = ingest_result.chunk_ids if hasattr(ingest_result, 'chunk_ids') else []
                    source.processed = True

                result.sources.append(source)
                result.chunk_ids.extend(source.chunk_ids)

            except Exception as e:
                logger.error(f"Failed to process document {doc_path}: {e}")
                result.errors.append(f"Failed to process {doc_path}: {str(e)}")

        result.total_sources = len(result.sources)
        result.total_chunks = len(result.chunk_ids)

        return result


class MockPipeline:
    """Mock pipeline for when knowledge module is not available."""

    async def ingest_text(self, text: str, source_name: str) -> Any:
        """Mock ingest."""
        class MockResult:
            chunk_ids = [str(uuid.uuid4())[:8]]
        return MockResult()

    async def ingest(self, path: Any) -> Any:
        """Mock ingest."""
        class MockResult:
            chunk_ids = [str(uuid.uuid4())[:8]]
        return MockResult()


__all__ = [
    "KnowledgeBuilder",
    "KnowledgeConfig",
    "KnowledgeSource",
    "KnowledgeType",
    "ProcessedKnowledge",
]
