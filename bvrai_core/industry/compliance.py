"""
Industry Compliance Module

This module provides compliance requirements, regulations,
and validation for industry-specific voice agent interactions.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .base import (
    IndustryType,
    ComplianceLevel,
    RegulationType,
    ComplianceRequirement,
)


logger = logging.getLogger(__name__)


# HIPAA Compliance Requirements
HIPAA_REQUIREMENTS = [
    ComplianceRequirement(
        id="hipaa_verification",
        regulation=RegulationType.HIPAA,
        title="Patient Identity Verification",
        description="Verify patient identity before discussing PHI",
        level=ComplianceLevel.CRITICAL,
        required_phrases=[
            "verify your identity",
            "confirm your date of birth",
        ],
        data_handling={
            "pii_collection": "Collect only minimum necessary",
            "storage": "Do not store in conversation logs",
            "transmission": "Use secure channels only",
        },
        on_violation="Immediately stop discussing PHI, verify identity",
        escalation_required=False,
    ),
    ComplianceRequirement(
        id="hipaa_minimum_necessary",
        regulation=RegulationType.HIPAA,
        title="Minimum Necessary Standard",
        description="Only discuss PHI necessary for the purpose",
        level=ComplianceLevel.HIGH,
        data_handling={
            "scope": "Limit to information needed for current interaction",
            "disclosure": "Do not volunteer additional health information",
        },
    ),
    ComplianceRequirement(
        id="hipaa_authorization",
        regulation=RegulationType.HIPAA,
        title="Authorization for Third Parties",
        description="Require authorization before discussing with third parties",
        level=ComplianceLevel.CRITICAL,
        required_disclosures=[
            "We cannot discuss patient information without authorization",
        ],
        on_violation="Do not disclose any information",
        escalation_required=True,
    ),
]


# TCPA Compliance Requirements
TCPA_REQUIREMENTS = [
    ComplianceRequirement(
        id="tcpa_consent",
        regulation=RegulationType.TCPA,
        title="Prior Express Consent",
        description="Ensure consent before making automated/recorded calls",
        level=ComplianceLevel.CRITICAL,
        required_disclosures=[
            "This call may be recorded for quality assurance",
        ],
        data_handling={
            "consent_tracking": "Record consent status",
            "opt_out": "Honor all opt-out requests immediately",
        },
    ),
    ComplianceRequirement(
        id="tcpa_calling_hours",
        regulation=RegulationType.TCPA,
        title="Calling Time Restrictions",
        description="Only call during permitted hours (8am-9pm local time)",
        level=ComplianceLevel.HIGH,
        on_violation="Do not place call outside permitted hours",
    ),
    ComplianceRequirement(
        id="tcpa_dnc",
        regulation=RegulationType.DNC,
        title="Do Not Call List Compliance",
        description="Honor Do Not Call requests",
        level=ComplianceLevel.CRITICAL,
        required_phrases=[
            "add you to our do not call list",
            "remove you from our calling list",
        ],
        on_violation="Immediately cease calling and add to internal DNC list",
    ),
]


# PCI-DSS Compliance Requirements
PCI_DSS_REQUIREMENTS = [
    ComplianceRequirement(
        id="pci_no_storage",
        regulation=RegulationType.PCI_DSS,
        title="No Card Data Storage",
        description="Never store full card numbers or CVV",
        level=ComplianceLevel.CRITICAL,
        prohibited_phrases=[
            "read me your card number",
            "what's your CVV",
            "give me your security code",
        ],
        data_handling={
            "card_numbers": "Never store or repeat back full card numbers",
            "cvv": "Never store CVV, do not repeat back",
            "expiration": "Do not store after transaction",
        },
        on_violation="Transfer to secure payment system",
        escalation_required=True,
    ),
    ComplianceRequirement(
        id="pci_secure_transfer",
        regulation=RegulationType.PCI_DSS,
        title="Secure Payment Transfer",
        description="Transfer to secure IVR for payment details",
        level=ComplianceLevel.HIGH,
        required_phrases=[
            "transfer you to our secure payment system",
            "use our secure payment line",
        ],
    ),
]


# Fair Housing Act Requirements
FAIR_HOUSING_REQUIREMENTS = [
    ComplianceRequirement(
        id="fha_no_steering",
        regulation=RegulationType.FAIR_HOUSING,
        title="No Steering",
        description="Do not steer clients toward or away from areas based on protected classes",
        level=ComplianceLevel.CRITICAL,
        prohibited_phrases=[
            "family neighborhood",
            "great for families",
            "mostly",
            "that area has a lot of",
            "you'd fit in there",
            "people like you",
        ],
        on_violation="Refocus on property features and client needs",
    ),
    ComplianceRequirement(
        id="fha_equal_service",
        regulation=RegulationType.FAIR_HOUSING,
        title="Equal Service",
        description="Provide equal service regardless of protected class",
        level=ComplianceLevel.CRITICAL,
        required_phrases=[
            "show you any properties",
            "help you find what you're looking for",
        ],
    ),
]


# Legal Ethics Requirements
LEGAL_ETHICS_REQUIREMENTS = [
    ComplianceRequirement(
        id="aba_no_advice",
        regulation=RegulationType.ABA_ETHICS,
        title="No Unauthorized Legal Advice",
        description="Only attorneys may provide legal advice",
        level=ComplianceLevel.CRITICAL,
        prohibited_phrases=[
            "you should sue",
            "you have a strong case",
            "they're liable",
            "you'll win",
            "my legal advice is",
        ],
        required_phrases=[
            "speak with an attorney",
            "attorney can advise you",
            "schedule a consultation",
        ],
        on_violation="Clarify role and offer to schedule with attorney",
    ),
    ComplianceRequirement(
        id="aba_confidentiality",
        regulation=RegulationType.ATTORNEY_CLIENT,
        title="Confidentiality",
        description="All communications are confidential",
        level=ComplianceLevel.CRITICAL,
        required_disclosures=[
            "confidential",
        ],
        data_handling={
            "storage": "Treat all information as privileged",
            "disclosure": "Never discuss with third parties",
        },
    ),
]


# Insurance Compliance Requirements
INSURANCE_REQUIREMENTS = [
    ComplianceRequirement(
        id="insurance_disclosure",
        regulation=RegulationType.STATE_INSURANCE,
        title="Required Disclosures",
        description="Provide required disclosures when discussing coverage",
        level=ComplianceLevel.HIGH,
        required_disclosures=[
            "Coverage depends on policy terms and conditions",
            "Actual premium may vary based on underwriting",
        ],
    ),
    ComplianceRequirement(
        id="insurance_no_guarantees",
        regulation=RegulationType.STATE_INSURANCE,
        title="No Coverage Guarantees",
        description="Cannot guarantee coverage without underwriting",
        level=ComplianceLevel.HIGH,
        prohibited_phrases=[
            "guaranteed coverage",
            "you're definitely covered",
            "we'll cover that",
        ],
    ),
]


# Financial Services Compliance
FINANCIAL_REQUIREMENTS = [
    ComplianceRequirement(
        id="finra_no_guarantees",
        regulation=RegulationType.FINRA,
        title="No Investment Guarantees",
        description="Cannot guarantee investment returns",
        level=ComplianceLevel.CRITICAL,
        prohibited_phrases=[
            "guaranteed return",
            "you'll make money",
            "can't lose",
            "sure thing",
            "100% safe",
        ],
        required_disclosures=[
            "Past performance does not guarantee future results",
            "Investments involve risk",
        ],
    ),
    ComplianceRequirement(
        id="finra_suitability",
        regulation=RegulationType.FINRA,
        title="Suitability",
        description="Must understand client needs before recommendations",
        level=ComplianceLevel.HIGH,
        required_phrases=[
            "understand your situation",
            "investment objectives",
            "risk tolerance",
        ],
    ),
]


# GDPR/Privacy Requirements
PRIVACY_REQUIREMENTS = [
    ComplianceRequirement(
        id="gdpr_consent",
        regulation=RegulationType.GDPR,
        title="Consent for Data Processing",
        description="Obtain consent before processing personal data",
        level=ComplianceLevel.HIGH,
        required_disclosures=[
            "We collect your information to provide service",
            "You can request deletion of your data",
        ],
    ),
    ComplianceRequirement(
        id="ccpa_disclosure",
        regulation=RegulationType.CCPA,
        title="California Privacy Disclosure",
        description="Disclose data collection to California residents",
        level=ComplianceLevel.MEDIUM,
        required_disclosures=[
            "right to know what personal information we collect",
            "right to delete your personal information",
            "right to opt-out of sale of personal information",
        ],
    ),
]


# Registry of requirements by regulation type
COMPLIANCE_REGISTRY: Dict[RegulationType, List[ComplianceRequirement]] = {
    RegulationType.HIPAA: HIPAA_REQUIREMENTS,
    RegulationType.TCPA: TCPA_REQUIREMENTS,
    RegulationType.DNC: [r for r in TCPA_REQUIREMENTS if r.regulation == RegulationType.DNC],
    RegulationType.PCI_DSS: PCI_DSS_REQUIREMENTS,
    RegulationType.FAIR_HOUSING: FAIR_HOUSING_REQUIREMENTS,
    RegulationType.ABA_ETHICS: LEGAL_ETHICS_REQUIREMENTS,
    RegulationType.ATTORNEY_CLIENT: [
        r for r in LEGAL_ETHICS_REQUIREMENTS
        if r.regulation == RegulationType.ATTORNEY_CLIENT
    ],
    RegulationType.STATE_INSURANCE: INSURANCE_REQUIREMENTS,
    RegulationType.FINRA: FINANCIAL_REQUIREMENTS,
    RegulationType.SEC: FINANCIAL_REQUIREMENTS,
    RegulationType.GDPR: PRIVACY_REQUIREMENTS,
    RegulationType.CCPA: [r for r in PRIVACY_REQUIREMENTS if r.regulation == RegulationType.CCPA],
}


@dataclass
class ComplianceViolation:
    """A detected compliance violation."""

    requirement_id: str
    regulation: RegulationType
    level: ComplianceLevel
    description: str
    matched_phrase: str = ""
    context: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    remediation: str = ""


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check."""

    is_compliant: bool = True
    violations: List[ComplianceViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_disclosures: List[str] = field(default_factory=list)
    escalation_required: bool = False


class ComplianceChecker:
    """
    Checks conversation content for compliance violations.

    Validates text against prohibited phrases, required disclosures,
    and other compliance requirements.
    """

    def __init__(
        self,
        regulations: Optional[List[RegulationType]] = None,
    ):
        """
        Initialize compliance checker.

        Args:
            regulations: List of applicable regulations
        """
        self.regulations = regulations or []
        self._requirements: List[ComplianceRequirement] = []
        self._load_requirements()

        # Compiled patterns for efficiency
        self._prohibited_patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _load_requirements(self) -> None:
        """Load requirements for applicable regulations."""
        for regulation in self.regulations:
            requirements = COMPLIANCE_REGISTRY.get(regulation, [])
            self._requirements.extend(requirements)

    def _compile_patterns(self) -> None:
        """Compile regex patterns for prohibited phrases."""
        for requirement in self._requirements:
            for phrase in requirement.prohibited_phrases:
                # Create case-insensitive pattern with word boundaries
                pattern_key = f"{requirement.id}:{phrase}"
                try:
                    self._prohibited_patterns[pattern_key] = re.compile(
                        rf"\b{re.escape(phrase)}\b",
                        re.IGNORECASE,
                    )
                except re.error:
                    logger.warning(f"Invalid regex pattern: {phrase}")

    def check_text(self, text: str) -> ComplianceCheckResult:
        """
        Check text for compliance violations.

        Args:
            text: Text to check

        Returns:
            Compliance check result
        """
        result = ComplianceCheckResult()
        text_lower = text.lower()

        # Check for prohibited phrases
        for pattern_key, pattern in self._prohibited_patterns.items():
            req_id, phrase = pattern_key.split(":", 1)
            match = pattern.search(text)

            if match:
                requirement = self._get_requirement(req_id)
                if requirement:
                    violation = ComplianceViolation(
                        requirement_id=req_id,
                        regulation=requirement.regulation,
                        level=requirement.level,
                        description=requirement.title,
                        matched_phrase=phrase,
                        context=text[max(0, match.start() - 50):match.end() + 50],
                        remediation=requirement.on_violation,
                    )
                    result.violations.append(violation)
                    result.is_compliant = False

                    if requirement.escalation_required:
                        result.escalation_required = True

        # Collect required disclosures
        for requirement in self._requirements:
            result.required_disclosures.extend(requirement.required_disclosures)

        # Remove duplicates
        result.required_disclosures = list(set(result.required_disclosures))

        return result

    def check_response(
        self,
        proposed_response: str,
        conversation_context: Optional[str] = None,
    ) -> ComplianceCheckResult:
        """
        Check a proposed agent response for compliance.

        Args:
            proposed_response: Response to check
            conversation_context: Full conversation context

        Returns:
            Compliance check result
        """
        # Check the response itself
        result = self.check_text(proposed_response)

        # Add context-specific checks
        if conversation_context:
            context_result = self._check_context_compliance(
                proposed_response,
                conversation_context,
            )
            result.warnings.extend(context_result.warnings)

        return result

    def _check_context_compliance(
        self,
        response: str,
        context: str,
    ) -> ComplianceCheckResult:
        """Check context-specific compliance."""
        result = ComplianceCheckResult()

        # Check for PHI discussion without verification
        if RegulationType.HIPAA in self.regulations:
            if self._mentions_health_info(response):
                if not self._has_verification(context):
                    result.warnings.append(
                        "Discussing health information without visible verification"
                    )

        # Check for payment info discussion
        if RegulationType.PCI_DSS in self.regulations:
            if self._mentions_payment_info(response):
                result.warnings.append(
                    "Discussing payment information - ensure secure channel"
                )

        return result

    def _mentions_health_info(self, text: str) -> bool:
        """Check if text mentions health information."""
        health_terms = [
            "diagnosis", "prescription", "medication", "treatment",
            "symptoms", "condition", "test results", "lab results",
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in health_terms)

    def _mentions_payment_info(self, text: str) -> bool:
        """Check if text mentions payment information."""
        payment_terms = [
            "card number", "credit card", "debit card", "cvv",
            "security code", "expiration", "billing",
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in payment_terms)

    def _has_verification(self, context: str) -> bool:
        """Check if context includes identity verification."""
        verification_terms = [
            "date of birth", "dob", "verified", "confirm your identity",
            "last four of", "social security",
        ]
        context_lower = context.lower()
        return any(term in context_lower for term in verification_terms)

    def _get_requirement(self, requirement_id: str) -> Optional[ComplianceRequirement]:
        """Get requirement by ID."""
        for req in self._requirements:
            if req.id == requirement_id:
                return req
        return None

    def get_required_disclosures(self) -> List[str]:
        """Get all required disclosures."""
        disclosures = []
        for req in self._requirements:
            disclosures.extend(req.required_disclosures)
        return list(set(disclosures))

    def add_regulation(self, regulation: RegulationType) -> None:
        """Add a regulation to check against."""
        if regulation not in self.regulations:
            self.regulations.append(regulation)
            requirements = COMPLIANCE_REGISTRY.get(regulation, [])
            self._requirements.extend(requirements)
            self._compile_patterns()


class ComplianceManager:
    """
    Manages compliance across multiple industries and sessions.

    Provides:
    - Industry-specific compliance checking
    - Compliance reporting
    - Violation tracking
    - Remediation guidance
    """

    def __init__(self):
        """Initialize compliance manager."""
        self._checkers: Dict[str, ComplianceChecker] = {}
        self._violation_log: List[ComplianceViolation] = []

    def get_checker_for_industry(
        self,
        industry_type: IndustryType,
    ) -> ComplianceChecker:
        """
        Get compliance checker for an industry.

        Args:
            industry_type: Industry type

        Returns:
            Configured compliance checker
        """
        from .profiles import get_industry_profile

        profile = get_industry_profile(industry_type)
        if not profile:
            return ComplianceChecker()

        checker_key = str(industry_type)
        if checker_key not in self._checkers:
            self._checkers[checker_key] = ComplianceChecker(
                regulations=profile.applicable_regulations,
            )

        return self._checkers[checker_key]

    def check_and_log(
        self,
        text: str,
        industry_type: IndustryType,
        session_id: Optional[str] = None,
    ) -> ComplianceCheckResult:
        """
        Check text for compliance and log violations.

        Args:
            text: Text to check
            industry_type: Industry type
            session_id: Session ID for logging

        Returns:
            Compliance check result
        """
        checker = self.get_checker_for_industry(industry_type)
        result = checker.check_text(text)

        # Log violations
        for violation in result.violations:
            self._violation_log.append(violation)
            logger.warning(
                f"Compliance violation [{violation.level.value}] "
                f"in {industry_type.value}: {violation.description} "
                f"- '{violation.matched_phrase}'"
            )

        return result

    def get_violations(
        self,
        regulation: Optional[RegulationType] = None,
        level: Optional[ComplianceLevel] = None,
    ) -> List[ComplianceViolation]:
        """
        Get logged violations with optional filtering.

        Args:
            regulation: Filter by regulation
            level: Filter by level

        Returns:
            List of violations
        """
        violations = self._violation_log

        if regulation:
            violations = [v for v in violations if v.regulation == regulation]

        if level:
            violations = [v for v in violations if v.level == level]

        return violations

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all violations."""
        summary = {
            "total_violations": len(self._violation_log),
            "by_level": {},
            "by_regulation": {},
        }

        for level in ComplianceLevel:
            count = len([v for v in self._violation_log if v.level == level])
            if count > 0:
                summary["by_level"][level.value] = count

        for violation in self._violation_log:
            reg = violation.regulation.value
            summary["by_regulation"][reg] = summary["by_regulation"].get(reg, 0) + 1

        return summary

    def clear_violations(self) -> None:
        """Clear violation log."""
        self._violation_log.clear()


def get_requirements_for_regulation(
    regulation: RegulationType,
) -> List[ComplianceRequirement]:
    """Get all requirements for a regulation."""
    return COMPLIANCE_REGISTRY.get(regulation, [])


def get_industry_regulations(industry_type: IndustryType) -> List[RegulationType]:
    """Get applicable regulations for an industry."""
    from .profiles import get_industry_profile

    profile = get_industry_profile(industry_type)
    if profile:
        return profile.applicable_regulations
    return []


def create_compliance_checker(
    industry_type: Optional[IndustryType] = None,
    regulations: Optional[List[RegulationType]] = None,
) -> ComplianceChecker:
    """
    Create a compliance checker.

    Args:
        industry_type: Industry type (uses industry's regulations)
        regulations: Explicit list of regulations

    Returns:
        Configured compliance checker
    """
    if regulations:
        return ComplianceChecker(regulations=regulations)

    if industry_type:
        industry_regulations = get_industry_regulations(industry_type)
        return ComplianceChecker(regulations=industry_regulations)

    return ComplianceChecker()


__all__ = [
    "ComplianceViolation",
    "ComplianceCheckResult",
    "ComplianceChecker",
    "ComplianceManager",
    "COMPLIANCE_REGISTRY",
    "get_requirements_for_regulation",
    "get_industry_regulations",
    "create_compliance_checker",
    "HIPAA_REQUIREMENTS",
    "TCPA_REQUIREMENTS",
    "PCI_DSS_REQUIREMENTS",
    "FAIR_HOUSING_REQUIREMENTS",
    "LEGAL_ETHICS_REQUIREMENTS",
    "INSURANCE_REQUIREMENTS",
    "FINANCIAL_REQUIREMENTS",
    "PRIVACY_REQUIREMENTS",
]
