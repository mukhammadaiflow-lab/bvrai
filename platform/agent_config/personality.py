"""
Personality Management Module

This module provides comprehensive personality profile management for voice agents,
including creation, storage, industry-specific configurations, and trait analysis.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
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
    PersonalityProfile,
    PersonalityTrait,
    VoiceSettings,
    BehaviorSettings,
    IndustryType,
    ComplianceMode,
    EscalationTrigger,
    ConfigurationError,
    ValidationError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Personality Storage Interface
# =============================================================================


class PersonalityStorage(ABC):
    """Abstract base class for personality storage backends."""

    @abstractmethod
    async def save(self, profile: PersonalityProfile) -> None:
        """Save a personality profile."""
        pass

    @abstractmethod
    async def get(self, profile_id: str) -> Optional[PersonalityProfile]:
        """Get a profile by ID."""
        pass

    @abstractmethod
    async def get_by_name(
        self,
        organization_id: str,
        name: str,
    ) -> Optional[PersonalityProfile]:
        """Get a profile by name within organization."""
        pass

    @abstractmethod
    async def list(
        self,
        organization_id: str,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[PersonalityProfile], int]:
        """List profiles with pagination."""
        pass

    @abstractmethod
    async def delete(self, profile_id: str) -> bool:
        """Delete a profile."""
        pass


class InMemoryPersonalityStorage(PersonalityStorage):
    """In-memory personality storage for testing and development."""

    def __init__(self):
        self._profiles: Dict[str, PersonalityProfile] = {}
        self._org_index: Dict[str, Set[str]] = {}
        self._industry_index: Dict[IndustryType, Set[str]] = {}

    async def save(self, profile: PersonalityProfile) -> None:
        """Save a profile to memory."""
        profile.updated_at = datetime.utcnow()
        self._profiles[profile.id] = profile

        # Update organization index
        if profile.organization_id:
            if profile.organization_id not in self._org_index:
                self._org_index[profile.organization_id] = set()
            self._org_index[profile.organization_id].add(profile.id)

        # Update industry index
        if profile.industry != IndustryType.CUSTOM:
            if profile.industry not in self._industry_index:
                self._industry_index[profile.industry] = set()
            self._industry_index[profile.industry].add(profile.id)

    async def get(self, profile_id: str) -> Optional[PersonalityProfile]:
        """Get a profile by ID."""
        return self._profiles.get(profile_id)

    async def get_by_name(
        self,
        organization_id: str,
        name: str,
    ) -> Optional[PersonalityProfile]:
        """Get a profile by name within organization."""
        if organization_id not in self._org_index:
            return None

        name_lower = name.lower()
        for profile_id in self._org_index[organization_id]:
            profile = self._profiles.get(profile_id)
            if profile and profile.name.lower() == name_lower:
                return profile
        return None

    async def list(
        self,
        organization_id: str,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[PersonalityProfile], int]:
        """List profiles with pagination."""
        if organization_id not in self._org_index:
            return [], 0

        profiles = []
        for profile_id in self._org_index[organization_id]:
            profile = self._profiles.get(profile_id)
            if not profile:
                continue

            # Apply filters
            if filters:
                if "industry" in filters:
                    if profile.industry != filters["industry"]:
                        continue
                if "is_active" in filters:
                    if profile.is_active != filters["is_active"]:
                        continue
                if "traits" in filters:
                    has_trait = any(
                        t in profile.primary_traits or t in profile.secondary_traits
                        for t in filters["traits"]
                    )
                    if not has_trait:
                        continue
                if "search" in filters:
                    search = filters["search"].lower()
                    if search not in profile.name.lower() and search not in profile.description.lower():
                        continue

            profiles.append(profile)

        # Sort by updated_at descending
        profiles.sort(key=lambda p: p.updated_at, reverse=True)

        total = len(profiles)
        profiles = profiles[offset:offset + limit]

        return profiles, total

    async def delete(self, profile_id: str) -> bool:
        """Delete a profile."""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False

        # Remove from indices
        if profile.organization_id in self._org_index:
            self._org_index[profile.organization_id].discard(profile_id)

        if profile.industry in self._industry_index:
            self._industry_index[profile.industry].discard(profile_id)

        del self._profiles[profile_id]
        return True


# =============================================================================
# Trait Analyzer
# =============================================================================


class TraitAnalyzer:
    """
    Analyzes and recommends personality traits based on business context.
    """

    # Industry to trait mappings
    INDUSTRY_TRAITS: Dict[IndustryType, Dict[str, List[PersonalityTrait]]] = {
        IndustryType.HEALTHCARE: {
            "primary": [PersonalityTrait.EMPATHETIC, PersonalityTrait.CALM, PersonalityTrait.PATIENT],
            "secondary": [PersonalityTrait.PROFESSIONAL, PersonalityTrait.THOROUGH, PersonalityTrait.WARM],
        },
        IndustryType.FINANCIAL_SERVICES: {
            "primary": [PersonalityTrait.PROFESSIONAL, PersonalityTrait.THOROUGH, PersonalityTrait.DIRECT],
            "secondary": [PersonalityTrait.CALM, PersonalityTrait.PATIENT, PersonalityTrait.CONCISE],
        },
        IndustryType.INSURANCE: {
            "primary": [PersonalityTrait.PROFESSIONAL, PersonalityTrait.PATIENT, PersonalityTrait.THOROUGH],
            "secondary": [PersonalityTrait.EMPATHETIC, PersonalityTrait.DIPLOMATIC, PersonalityTrait.CALM],
        },
        IndustryType.REAL_ESTATE: {
            "primary": [PersonalityTrait.ENTHUSIASTIC, PersonalityTrait.FRIENDLY, PersonalityTrait.PROACTIVE],
            "secondary": [PersonalityTrait.PROFESSIONAL, PersonalityTrait.DETAILED, PersonalityTrait.PATIENT],
        },
        IndustryType.LEGAL: {
            "primary": [PersonalityTrait.PROFESSIONAL, PersonalityTrait.FORMAL, PersonalityTrait.THOROUGH],
            "secondary": [PersonalityTrait.CALM, PersonalityTrait.DIPLOMATIC, PersonalityTrait.DIRECT],
        },
        IndustryType.RETAIL: {
            "primary": [PersonalityTrait.FRIENDLY, PersonalityTrait.ENTHUSIASTIC, PersonalityTrait.EFFICIENT],
            "secondary": [PersonalityTrait.PATIENT, PersonalityTrait.CONVERSATIONAL, PersonalityTrait.PROACTIVE],
        },
        IndustryType.HOSPITALITY: {
            "primary": [PersonalityTrait.WARM, PersonalityTrait.FRIENDLY, PersonalityTrait.ENTHUSIASTIC],
            "secondary": [PersonalityTrait.PATIENT, PersonalityTrait.EMPATHETIC, PersonalityTrait.EFFICIENT],
        },
        IndustryType.AUTOMOTIVE: {
            "primary": [PersonalityTrait.FRIENDLY, PersonalityTrait.PROFESSIONAL, PersonalityTrait.ENTHUSIASTIC],
            "secondary": [PersonalityTrait.PATIENT, PersonalityTrait.DETAILED, PersonalityTrait.PROACTIVE],
        },
        IndustryType.TELECOMMUNICATIONS: {
            "primary": [PersonalityTrait.PATIENT, PersonalityTrait.EFFICIENT, PersonalityTrait.PROFESSIONAL],
            "secondary": [PersonalityTrait.CALM, PersonalityTrait.THOROUGH, PersonalityTrait.DIRECT],
        },
        IndustryType.TECHNOLOGY: {
            "primary": [PersonalityTrait.PROFESSIONAL, PersonalityTrait.DETAILED, PersonalityTrait.PATIENT],
            "secondary": [PersonalityTrait.EFFICIENT, PersonalityTrait.DIRECT, PersonalityTrait.THOROUGH],
        },
        IndustryType.EDUCATION: {
            "primary": [PersonalityTrait.PATIENT, PersonalityTrait.FRIENDLY, PersonalityTrait.WARM],
            "secondary": [PersonalityTrait.THOROUGH, PersonalityTrait.EMPATHETIC, PersonalityTrait.DETAILED],
        },
        IndustryType.FITNESS: {
            "primary": [PersonalityTrait.ENTHUSIASTIC, PersonalityTrait.FRIENDLY, PersonalityTrait.PROACTIVE],
            "secondary": [PersonalityTrait.WARM, PersonalityTrait.EFFICIENT, PersonalityTrait.DIRECT],
        },
        IndustryType.BEAUTY: {
            "primary": [PersonalityTrait.FRIENDLY, PersonalityTrait.WARM, PersonalityTrait.ENTHUSIASTIC],
            "secondary": [PersonalityTrait.PATIENT, PersonalityTrait.CONVERSATIONAL, PersonalityTrait.EMPATHETIC],
        },
        IndustryType.FOOD_SERVICE: {
            "primary": [PersonalityTrait.FRIENDLY, PersonalityTrait.EFFICIENT, PersonalityTrait.WARM],
            "secondary": [PersonalityTrait.PATIENT, PersonalityTrait.ENTHUSIASTIC, PersonalityTrait.PROACTIVE],
        },
        IndustryType.HOME_SERVICES: {
            "primary": [PersonalityTrait.PROFESSIONAL, PersonalityTrait.FRIENDLY, PersonalityTrait.EFFICIENT],
            "secondary": [PersonalityTrait.PATIENT, PersonalityTrait.THOROUGH, PersonalityTrait.DIRECT],
        },
    }

    # Compliance mode mappings
    COMPLIANCE_TRAITS: Dict[ComplianceMode, List[PersonalityTrait]] = {
        ComplianceMode.HIPAA: [PersonalityTrait.THOROUGH, PersonalityTrait.PROFESSIONAL, PersonalityTrait.CALM],
        ComplianceMode.PCI_DSS: [PersonalityTrait.PROFESSIONAL, PersonalityTrait.THOROUGH, PersonalityTrait.DIRECT],
        ComplianceMode.GDPR: [PersonalityTrait.PROFESSIONAL, PersonalityTrait.THOROUGH, PersonalityTrait.DIPLOMATIC],
    }

    # Trait compatibility matrix
    TRAIT_COMPATIBILITY: Dict[PersonalityTrait, Set[PersonalityTrait]] = {
        PersonalityTrait.PROFESSIONAL: {PersonalityTrait.FRIENDLY, PersonalityTrait.EFFICIENT, PersonalityTrait.DIRECT},
        PersonalityTrait.FRIENDLY: {PersonalityTrait.WARM, PersonalityTrait.ENTHUSIASTIC, PersonalityTrait.PATIENT},
        PersonalityTrait.FORMAL: {PersonalityTrait.PROFESSIONAL, PersonalityTrait.THOROUGH, PersonalityTrait.DIRECT},
        PersonalityTrait.CASUAL: {PersonalityTrait.FRIENDLY, PersonalityTrait.WARM, PersonalityTrait.CONVERSATIONAL},
        PersonalityTrait.WARM: {PersonalityTrait.EMPATHETIC, PersonalityTrait.PATIENT, PersonalityTrait.FRIENDLY},
        PersonalityTrait.EMPATHETIC: {PersonalityTrait.WARM, PersonalityTrait.PATIENT, PersonalityTrait.CALM},
        PersonalityTrait.ENTHUSIASTIC: {PersonalityTrait.FRIENDLY, PersonalityTrait.PROACTIVE, PersonalityTrait.WARM},
        PersonalityTrait.CALM: {PersonalityTrait.PATIENT, PersonalityTrait.PROFESSIONAL, PersonalityTrait.EMPATHETIC},
        PersonalityTrait.ASSERTIVE: {PersonalityTrait.DIRECT, PersonalityTrait.PROFESSIONAL, PersonalityTrait.EFFICIENT},
    }

    # Trait conflicts
    TRAIT_CONFLICTS: Dict[PersonalityTrait, Set[PersonalityTrait]] = {
        PersonalityTrait.FORMAL: {PersonalityTrait.CASUAL},
        PersonalityTrait.CASUAL: {PersonalityTrait.FORMAL},
        PersonalityTrait.ENTHUSIASTIC: {PersonalityTrait.CALM},
        PersonalityTrait.CONCISE: {PersonalityTrait.DETAILED},
        PersonalityTrait.DETAILED: {PersonalityTrait.CONCISE},
        PersonalityTrait.PROACTIVE: {PersonalityTrait.REACTIVE},
        PersonalityTrait.REACTIVE: {PersonalityTrait.PROACTIVE},
    }

    def recommend_traits(
        self,
        industry: IndustryType,
        compliance_mode: Optional[ComplianceMode] = None,
        preferred_traits: Optional[List[PersonalityTrait]] = None,
        excluded_traits: Optional[List[PersonalityTrait]] = None,
    ) -> Tuple[List[PersonalityTrait], List[PersonalityTrait]]:
        """
        Recommend personality traits based on industry and context.

        Args:
            industry: Industry type
            compliance_mode: Compliance requirements
            preferred_traits: User-preferred traits to include
            excluded_traits: Traits to exclude

        Returns:
            Tuple of (primary_traits, secondary_traits)
        """
        excluded = set(excluded_traits or [])
        primary = []
        secondary = []

        # Start with industry defaults
        if industry in self.INDUSTRY_TRAITS:
            industry_traits = self.INDUSTRY_TRAITS[industry]
            primary = [t for t in industry_traits["primary"] if t not in excluded]
            secondary = [t for t in industry_traits["secondary"] if t not in excluded]
        else:
            # Default traits for custom/unknown industries
            primary = [PersonalityTrait.PROFESSIONAL, PersonalityTrait.FRIENDLY]
            secondary = [PersonalityTrait.PATIENT, PersonalityTrait.EFFICIENT]

        # Add compliance-related traits
        if compliance_mode and compliance_mode in self.COMPLIANCE_TRAITS:
            for trait in self.COMPLIANCE_TRAITS[compliance_mode]:
                if trait not in excluded and trait not in primary:
                    if len(primary) < 4:
                        primary.append(trait)
                    elif trait not in secondary:
                        secondary.append(trait)

        # Incorporate preferred traits
        if preferred_traits:
            for trait in preferred_traits:
                if trait in excluded:
                    continue
                # Check for conflicts
                has_conflict = False
                if trait in self.TRAIT_CONFLICTS:
                    for existing in primary + secondary:
                        if existing in self.TRAIT_CONFLICTS[trait]:
                            has_conflict = True
                            break

                if not has_conflict:
                    if trait not in primary:
                        if len(primary) < 4:
                            primary.insert(0, trait)
                        elif trait not in secondary:
                            secondary.insert(0, trait)

        # Ensure no conflicts in final lists
        primary = self._remove_conflicts(primary)
        secondary = self._remove_conflicts(secondary, primary)

        return primary[:4], secondary[:5]

    def _remove_conflicts(
        self,
        traits: List[PersonalityTrait],
        additional_check: Optional[List[PersonalityTrait]] = None,
    ) -> List[PersonalityTrait]:
        """Remove conflicting traits from list."""
        result = []
        check_against = set(additional_check or [])

        for trait in traits:
            has_conflict = False
            if trait in self.TRAIT_CONFLICTS:
                for existing in result:
                    if existing in self.TRAIT_CONFLICTS[trait]:
                        has_conflict = True
                        break
                for existing in check_against:
                    if existing in self.TRAIT_CONFLICTS[trait]:
                        has_conflict = True
                        break

            if not has_conflict:
                result.append(trait)

        return result

    def analyze_compatibility(
        self,
        traits: List[PersonalityTrait],
    ) -> Dict[str, Any]:
        """
        Analyze compatibility of trait combination.

        Args:
            traits: List of traits to analyze

        Returns:
            Analysis result with score and recommendations
        """
        conflicts = []
        enhancements = []
        score = 100.0

        # Check for conflicts
        for i, trait1 in enumerate(traits):
            if trait1 in self.TRAIT_CONFLICTS:
                for trait2 in traits[i+1:]:
                    if trait2 in self.TRAIT_CONFLICTS[trait1]:
                        conflicts.append((trait1, trait2))
                        score -= 20.0

        # Check for compatible combinations
        for i, trait1 in enumerate(traits):
            if trait1 in self.TRAIT_COMPATIBILITY:
                for trait2 in traits[i+1:]:
                    if trait2 in self.TRAIT_COMPATIBILITY[trait1]:
                        enhancements.append((trait1, trait2))
                        score += 5.0

        # Cap score
        score = max(0.0, min(100.0, score))

        return {
            "score": score,
            "conflicts": [
                {"trait1": c[0].value, "trait2": c[1].value}
                for c in conflicts
            ],
            "enhancements": [
                {"trait1": e[0].value, "trait2": e[1].value}
                for e in enhancements
            ],
            "is_compatible": len(conflicts) == 0,
            "recommendations": self._get_recommendations(traits, conflicts),
        }

    def _get_recommendations(
        self,
        traits: List[PersonalityTrait],
        conflicts: List[Tuple[PersonalityTrait, PersonalityTrait]],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        for conflict in conflicts:
            recommendations.append(
                f"Consider removing either '{conflict[0].value}' or '{conflict[1].value}' "
                f"as they create conflicting personality characteristics."
            )

        if len(traits) < 2:
            recommendations.append(
                "Consider adding more traits for a more nuanced personality profile."
            )

        if len(traits) > 5:
            recommendations.append(
                "Consider reducing the number of traits to maintain a focused personality."
            )

        return recommendations


# =============================================================================
# Voice Configuration Helper
# =============================================================================


class VoiceConfigurationHelper:
    """
    Helper for configuring voice settings based on personality and industry.
    """

    # Voice provider presets
    VOICE_PRESETS: Dict[str, Dict[str, VoiceSettings]] = {
        "elevenlabs": {
            "professional_male": VoiceSettings(
                provider="elevenlabs",
                voice_id="pNInz6obpgDQGcFmaJgB",
                voice_name="Adam",
                speed=1.0,
                stability=0.7,
                similarity_boost=0.8,
            ),
            "professional_female": VoiceSettings(
                provider="elevenlabs",
                voice_id="21m00Tcm4TlvDq8ikWAM",
                voice_name="Rachel",
                speed=1.0,
                stability=0.7,
                similarity_boost=0.8,
            ),
            "friendly_male": VoiceSettings(
                provider="elevenlabs",
                voice_id="ErXwobaYiN019PkySvjV",
                voice_name="Antoni",
                speed=1.05,
                stability=0.5,
                similarity_boost=0.75,
            ),
            "friendly_female": VoiceSettings(
                provider="elevenlabs",
                voice_id="EXAVITQu4vr4xnSDxMaL",
                voice_name="Bella",
                speed=1.05,
                stability=0.5,
                similarity_boost=0.75,
            ),
            "calm_male": VoiceSettings(
                provider="elevenlabs",
                voice_id="VR6AewLTigWG4xSOukaG",
                voice_name="Arnold",
                speed=0.95,
                stability=0.8,
                similarity_boost=0.7,
            ),
            "calm_female": VoiceSettings(
                provider="elevenlabs",
                voice_id="MF3mGyEYCl7XYWbV9V6O",
                voice_name="Elli",
                speed=0.95,
                stability=0.8,
                similarity_boost=0.7,
            ),
            "energetic_male": VoiceSettings(
                provider="elevenlabs",
                voice_id="TxGEqnHWrfWFTfGW9XjX",
                voice_name="Josh",
                speed=1.1,
                stability=0.4,
                similarity_boost=0.8,
            ),
            "energetic_female": VoiceSettings(
                provider="elevenlabs",
                voice_id="jBpfuIE2acCO8z3wKNLl",
                voice_name="Gigi",
                speed=1.1,
                stability=0.4,
                similarity_boost=0.8,
            ),
        },
        "openai": {
            "professional_male": VoiceSettings(
                provider="openai",
                voice_id="onyx",
                voice_name="Onyx",
                speed=1.0,
            ),
            "professional_female": VoiceSettings(
                provider="openai",
                voice_id="nova",
                voice_name="Nova",
                speed=1.0,
            ),
            "friendly_male": VoiceSettings(
                provider="openai",
                voice_id="echo",
                voice_name="Echo",
                speed=1.05,
            ),
            "friendly_female": VoiceSettings(
                provider="openai",
                voice_id="shimmer",
                voice_name="Shimmer",
                speed=1.05,
            ),
            "calm_male": VoiceSettings(
                provider="openai",
                voice_id="fable",
                voice_name="Fable",
                speed=0.95,
            ),
            "calm_female": VoiceSettings(
                provider="openai",
                voice_id="alloy",
                voice_name="Alloy",
                speed=0.95,
            ),
        },
    }

    # Trait to voice style mapping
    TRAIT_VOICE_STYLE: Dict[PersonalityTrait, str] = {
        PersonalityTrait.PROFESSIONAL: "professional",
        PersonalityTrait.FORMAL: "professional",
        PersonalityTrait.FRIENDLY: "friendly",
        PersonalityTrait.WARM: "friendly",
        PersonalityTrait.CASUAL: "friendly",
        PersonalityTrait.EMPATHETIC: "calm",
        PersonalityTrait.CALM: "calm",
        PersonalityTrait.PATIENT: "calm",
        PersonalityTrait.ENTHUSIASTIC: "energetic",
        PersonalityTrait.PROACTIVE: "energetic",
    }

    def recommend_voice(
        self,
        traits: List[PersonalityTrait],
        provider: str = "elevenlabs",
        gender: str = "female",
    ) -> VoiceSettings:
        """
        Recommend voice settings based on personality traits.

        Args:
            traits: Personality traits
            provider: Voice provider
            gender: Preferred gender

        Returns:
            Recommended voice settings
        """
        # Determine voice style from traits
        style_counts: Dict[str, int] = {}
        for trait in traits:
            if trait in self.TRAIT_VOICE_STYLE:
                style = self.TRAIT_VOICE_STYLE[trait]
                style_counts[style] = style_counts.get(style, 0) + 1

        # Get dominant style
        dominant_style = "professional"  # default
        if style_counts:
            dominant_style = max(style_counts, key=style_counts.get)

        # Get voice preset
        preset_key = f"{dominant_style}_{gender}"
        provider_presets = self.VOICE_PRESETS.get(provider, self.VOICE_PRESETS["elevenlabs"])

        if preset_key in provider_presets:
            return provider_presets[preset_key]

        # Fallback to professional
        fallback_key = f"professional_{gender}"
        return provider_presets.get(fallback_key, VoiceSettings(provider=provider))

    def adjust_for_industry(
        self,
        voice: VoiceSettings,
        industry: IndustryType,
    ) -> VoiceSettings:
        """
        Adjust voice settings for specific industry requirements.

        Args:
            voice: Base voice settings
            industry: Industry type

        Returns:
            Adjusted voice settings
        """
        # Create a copy
        adjusted = VoiceSettings.from_dict(voice.to_dict())

        # Industry-specific adjustments
        if industry in [IndustryType.HEALTHCARE, IndustryType.LEGAL]:
            # More stable, slower pace
            adjusted.stability = min(0.9, voice.stability + 0.1)
            adjusted.speed = max(0.9, voice.speed - 0.05)

        elif industry in [IndustryType.RETAIL, IndustryType.FITNESS, IndustryType.ENTERTAINMENT]:
            # More energetic
            adjusted.stability = max(0.3, voice.stability - 0.1)
            adjusted.speed = min(1.15, voice.speed + 0.05)

        elif industry in [IndustryType.FINANCIAL_SERVICES, IndustryType.INSURANCE]:
            # Professional, clear
            adjusted.stability = 0.7
            adjusted.speed = 1.0

        return adjusted


# =============================================================================
# Personality Manager
# =============================================================================


class PersonalityManager:
    """
    Manages personality profiles with CRUD operations, recommendations, and validation.
    """

    def __init__(
        self,
        storage: Optional[PersonalityStorage] = None,
        trait_analyzer: Optional[TraitAnalyzer] = None,
        voice_helper: Optional[VoiceConfigurationHelper] = None,
    ):
        """
        Initialize personality manager.

        Args:
            storage: Storage backend
            trait_analyzer: Trait analyzer
            voice_helper: Voice configuration helper
        """
        self.storage = storage or InMemoryPersonalityStorage()
        self.trait_analyzer = trait_analyzer or TraitAnalyzer()
        self.voice_helper = voice_helper or VoiceConfigurationHelper()

        # Cache
        self._cache: Dict[str, Tuple[PersonalityProfile, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def create_profile(
        self,
        name: str,
        organization_id: str,
        agent_name: str,
        agent_role: str,
        company_name: str,
        industry: IndustryType = IndustryType.CUSTOM,
        primary_traits: Optional[List[PersonalityTrait]] = None,
        secondary_traits: Optional[List[PersonalityTrait]] = None,
        voice: Optional[VoiceSettings] = None,
        behavior: Optional[BehaviorSettings] = None,
        description: str = "",
        auto_configure: bool = True,
    ) -> PersonalityProfile:
        """
        Create a new personality profile.

        Args:
            name: Profile name
            organization_id: Organization ID
            agent_name: Name for the agent persona
            agent_role: Role/title
            company_name: Company name
            industry: Industry type
            primary_traits: Primary personality traits
            secondary_traits: Secondary traits
            voice: Voice settings
            behavior: Behavior settings
            description: Profile description
            auto_configure: Auto-configure traits and voice if not provided

        Returns:
            Created profile
        """
        # Auto-configure traits if not provided
        if auto_configure and not primary_traits:
            primary_traits, secondary_traits = self.trait_analyzer.recommend_traits(
                industry=industry,
                compliance_mode=behavior.compliance_mode if behavior else None,
            )

        # Auto-configure voice if not provided
        if auto_configure and not voice:
            voice = self.voice_helper.recommend_voice(
                traits=primary_traits or [],
                provider="elevenlabs",
                gender="female",
            )
            voice = self.voice_helper.adjust_for_industry(voice, industry)

        profile = PersonalityProfile(
            id=f"pers_{uuid.uuid4().hex[:24]}",
            name=name,
            description=description,
            organization_id=organization_id,
            agent_name=agent_name,
            agent_role=agent_role,
            company_name=company_name,
            primary_traits=primary_traits or [],
            secondary_traits=secondary_traits or [],
            industry=industry,
            voice=voice or VoiceSettings(),
            behavior=behavior or BehaviorSettings(),
        )

        # Validate
        self._validate_profile(profile)

        # Save
        await self.storage.save(profile)

        logger.info(f"Created personality profile: {profile.id} ({name})")
        return profile

    async def create_profile_from_industry(
        self,
        name: str,
        organization_id: str,
        agent_name: str,
        company_name: str,
        industry: IndustryType,
        agent_role: Optional[str] = None,
        voice_gender: str = "female",
        voice_provider: str = "elevenlabs",
    ) -> PersonalityProfile:
        """
        Create a fully configured profile based on industry best practices.

        Args:
            name: Profile name
            organization_id: Organization ID
            agent_name: Name for the agent persona
            company_name: Company name
            industry: Industry type
            agent_role: Role (auto-generated if not provided)
            voice_gender: Preferred voice gender
            voice_provider: Voice provider

        Returns:
            Created profile
        """
        # Auto-generate role if not provided
        if not agent_role:
            agent_role = self._get_default_role_for_industry(industry)

        # Get recommended traits
        primary, secondary = self.trait_analyzer.recommend_traits(industry=industry)

        # Get recommended voice
        voice = self.voice_helper.recommend_voice(
            traits=primary,
            provider=voice_provider,
            gender=voice_gender,
        )
        voice = self.voice_helper.adjust_for_industry(voice, industry)

        # Get industry-specific behavior settings
        behavior = self._get_default_behavior_for_industry(industry)

        return await self.create_profile(
            name=name,
            organization_id=organization_id,
            agent_name=agent_name,
            agent_role=agent_role,
            company_name=company_name,
            industry=industry,
            primary_traits=primary,
            secondary_traits=secondary,
            voice=voice,
            behavior=behavior,
            description=f"Auto-configured profile for {industry.value}",
            auto_configure=False,
        )

    async def get_profile(self, profile_id: str) -> Optional[PersonalityProfile]:
        """
        Get a profile by ID.

        Args:
            profile_id: Profile ID

        Returns:
            Profile or None
        """
        # Check cache
        if profile_id in self._cache:
            profile, cached_at = self._cache[profile_id]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return profile

        # Get from storage
        profile = await self.storage.get(profile_id)

        # Update cache
        if profile:
            self._cache[profile_id] = (profile, datetime.utcnow())

        return profile

    async def update_profile(
        self,
        profile_id: str,
        updates: Dict[str, Any],
    ) -> PersonalityProfile:
        """
        Update a profile.

        Args:
            profile_id: Profile ID
            updates: Fields to update

        Returns:
            Updated profile
        """
        profile = await self.storage.get(profile_id)
        if not profile:
            raise ConfigurationError(f"Profile not found: {profile_id}")

        # Apply updates
        if "name" in updates:
            profile.name = updates["name"]
        if "description" in updates:
            profile.description = updates["description"]
        if "agent_name" in updates:
            profile.agent_name = updates["agent_name"]
        if "agent_role" in updates:
            profile.agent_role = updates["agent_role"]
        if "company_name" in updates:
            profile.company_name = updates["company_name"]
        if "primary_traits" in updates:
            profile.primary_traits = [
                PersonalityTrait(t) if isinstance(t, str) else t
                for t in updates["primary_traits"]
            ]
        if "secondary_traits" in updates:
            profile.secondary_traits = [
                PersonalityTrait(t) if isinstance(t, str) else t
                for t in updates["secondary_traits"]
            ]
        if "industry" in updates:
            profile.industry = IndustryType(updates["industry"]) if isinstance(updates["industry"], str) else updates["industry"]
        if "voice" in updates:
            profile.voice = VoiceSettings.from_dict(updates["voice"]) if isinstance(updates["voice"], dict) else updates["voice"]
        if "behavior" in updates:
            profile.behavior = BehaviorSettings.from_dict(updates["behavior"]) if isinstance(updates["behavior"], dict) else updates["behavior"]
        if "greeting_style" in updates:
            profile.greeting_style = updates["greeting_style"]
        if "farewell_style" in updates:
            profile.farewell_style = updates["farewell_style"]
        if "empathy_level" in updates:
            profile.empathy_level = updates["empathy_level"]
        if "assertiveness" in updates:
            profile.assertiveness = updates["assertiveness"]
        if "preferred_phrases" in updates:
            profile.preferred_phrases = updates["preferred_phrases"]
        if "avoided_phrases" in updates:
            profile.avoided_phrases = updates["avoided_phrases"]
        if "is_active" in updates:
            profile.is_active = updates["is_active"]
        if "is_default" in updates:
            profile.is_default = updates["is_default"]

        profile.updated_at = datetime.utcnow()

        # Validate
        self._validate_profile(profile)

        # Save
        await self.storage.save(profile)

        # Invalidate cache
        if profile_id in self._cache:
            del self._cache[profile_id]

        logger.info(f"Updated personality profile: {profile_id}")
        return profile

    async def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a profile.

        Args:
            profile_id: Profile ID

        Returns:
            True if deleted
        """
        success = await self.storage.delete(profile_id)

        # Invalidate cache
        if profile_id in self._cache:
            del self._cache[profile_id]

        if success:
            logger.info(f"Deleted personality profile: {profile_id}")

        return success

    async def list_profiles(
        self,
        organization_id: str,
        industry: Optional[IndustryType] = None,
        traits: Optional[List[PersonalityTrait]] = None,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[PersonalityProfile], int]:
        """
        List profiles with filtering.

        Args:
            organization_id: Organization ID
            industry: Filter by industry
            traits: Filter by traits
            search: Search in name/description
            is_active: Filter by active status
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (profiles, total_count)
        """
        filters = {}
        if industry:
            filters["industry"] = industry
        if traits:
            filters["traits"] = traits
        if search:
            filters["search"] = search
        if is_active is not None:
            filters["is_active"] = is_active

        return await self.storage.list(
            organization_id=organization_id,
            filters=filters if filters else None,
            offset=offset,
            limit=limit,
        )

    async def clone_profile(
        self,
        profile_id: str,
        new_name: Optional[str] = None,
    ) -> PersonalityProfile:
        """
        Clone a profile.

        Args:
            profile_id: Profile ID to clone
            new_name: Name for the clone

        Returns:
            Cloned profile
        """
        original = await self.storage.get(profile_id)
        if not original:
            raise ConfigurationError(f"Profile not found: {profile_id}")

        clone = PersonalityProfile(
            id=f"pers_{uuid.uuid4().hex[:24]}",
            name=new_name or f"{original.name} (Copy)",
            description=original.description,
            organization_id=original.organization_id,
            agent_name=original.agent_name,
            agent_role=original.agent_role,
            company_name=original.company_name,
            primary_traits=original.primary_traits.copy(),
            secondary_traits=original.secondary_traits.copy(),
            industry=original.industry,
            specialization=original.specialization,
            voice=VoiceSettings.from_dict(original.voice.to_dict()),
            behavior=BehaviorSettings.from_dict(original.behavior.to_dict()),
            greeting_style=original.greeting_style,
            farewell_style=original.farewell_style,
            empathy_level=original.empathy_level,
            assertiveness=original.assertiveness,
            preferred_phrases=original.preferred_phrases.copy(),
            avoided_phrases=original.avoided_phrases.copy(),
            custom_vocabulary=original.custom_vocabulary.copy(),
            is_active=False,
            is_default=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        await self.storage.save(clone)

        logger.info(f"Cloned personality profile: {profile_id} -> {clone.id}")
        return clone

    def analyze_traits(
        self,
        traits: List[PersonalityTrait],
    ) -> Dict[str, Any]:
        """
        Analyze a set of traits for compatibility.

        Args:
            traits: Traits to analyze

        Returns:
            Analysis result
        """
        return self.trait_analyzer.analyze_compatibility(traits)

    def recommend_traits_for_industry(
        self,
        industry: IndustryType,
        compliance_mode: Optional[ComplianceMode] = None,
    ) -> Tuple[List[PersonalityTrait], List[PersonalityTrait]]:
        """
        Get recommended traits for an industry.

        Args:
            industry: Industry type
            compliance_mode: Compliance requirements

        Returns:
            Tuple of (primary, secondary) traits
        """
        return self.trait_analyzer.recommend_traits(
            industry=industry,
            compliance_mode=compliance_mode,
        )

    def _validate_profile(self, profile: PersonalityProfile) -> None:
        """Validate a personality profile."""
        errors = []

        if not profile.name:
            errors.append("Profile name is required")

        if not profile.agent_name:
            errors.append("Agent name is required")

        if not profile.company_name:
            errors.append("Company name is required")

        # Validate trait compatibility
        all_traits = profile.primary_traits + profile.secondary_traits
        analysis = self.trait_analyzer.analyze_compatibility(all_traits)
        if not analysis["is_compatible"]:
            for conflict in analysis["conflicts"]:
                errors.append(
                    f"Trait conflict: {conflict['trait1']} conflicts with {conflict['trait2']}"
                )

        if errors:
            raise ValidationError(f"Profile validation failed: {'; '.join(errors)}")

    def _get_default_role_for_industry(self, industry: IndustryType) -> str:
        """Get default role name for industry."""
        roles = {
            IndustryType.HEALTHCARE: "Healthcare Assistant",
            IndustryType.FINANCIAL_SERVICES: "Financial Services Representative",
            IndustryType.INSURANCE: "Insurance Specialist",
            IndustryType.REAL_ESTATE: "Real Estate Assistant",
            IndustryType.LEGAL: "Legal Assistant",
            IndustryType.RETAIL: "Customer Service Representative",
            IndustryType.HOSPITALITY: "Concierge",
            IndustryType.AUTOMOTIVE: "Sales Consultant",
            IndustryType.TELECOMMUNICATIONS: "Support Specialist",
            IndustryType.TECHNOLOGY: "Technical Support Specialist",
            IndustryType.EDUCATION: "Admissions Advisor",
            IndustryType.FITNESS: "Fitness Coordinator",
            IndustryType.BEAUTY: "Booking Specialist",
            IndustryType.FOOD_SERVICE: "Restaurant Host",
            IndustryType.HOME_SERVICES: "Service Coordinator",
        }
        return roles.get(industry, "Assistant")

    def _get_default_behavior_for_industry(
        self,
        industry: IndustryType,
    ) -> BehaviorSettings:
        """Get default behavior settings for industry."""
        behavior = BehaviorSettings()

        # Industry-specific settings
        if industry == IndustryType.HEALTHCARE:
            behavior.compliance_mode = ComplianceMode.HIPAA
            behavior.response_style = "conversational"
            behavior.response_length = "moderate"
            behavior.voicemail_action = "leave_message"
            behavior.required_disclosures = [
                "This call may be recorded for quality assurance.",
            ]

        elif industry == IndustryType.FINANCIAL_SERVICES:
            behavior.compliance_mode = ComplianceMode.GLBA
            behavior.response_style = "formal"
            behavior.response_length = "moderate"

        elif industry in [IndustryType.RETAIL, IndustryType.FITNESS]:
            behavior.response_style = "conversational"
            behavior.response_length = "brief"
            behavior.use_filler_words = True

        elif industry == IndustryType.LEGAL:
            behavior.response_style = "formal"
            behavior.response_length = "detailed"
            behavior.use_filler_words = False
            behavior.required_disclosures = [
                "I am an AI assistant and cannot provide legal advice.",
            ]

        return behavior


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "PersonalityStorage",
    "InMemoryPersonalityStorage",
    "TraitAnalyzer",
    "VoiceConfigurationHelper",
    "PersonalityManager",
]
