"""
Industry Intelligence Module

This module provides the central intelligence system for industry-specific
voice agent behavior, combining profiles, compliance, terminology, and behaviors.
"""

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
    Union,
)

from .base import (
    IndustryType,
    IndustryProfile,
    IndustryContext,
    ComplianceLevel,
    RegulationType,
    ConversationPhase,
    SentimentCategory,
    ComplianceRequirement,
    TermDefinition,
    ConversationPattern,
    BehaviorGuideline,
    EmergencyProtocol,
    SeasonalPattern,
)
from .profiles import (
    INDUSTRY_PROFILES,
    get_industry_profile,
    get_all_industry_types,
)
from .compliance import (
    ComplianceChecker,
    ComplianceManager,
    ComplianceCheckResult,
    ComplianceViolation,
)
from .terminology import (
    TerminologyManager,
    get_industry_terminology,
)
from .behaviors import (
    ConversationBehavior,
    BehaviorEngine,
    ResponseStyle,
    SentimentResponse,
    SENTIMENT_RESPONSES,
)


logger = logging.getLogger(__name__)


@dataclass
class IntelligenceConfig:
    """Configuration for industry intelligence."""

    # Feature toggles
    compliance_checking_enabled: bool = True
    terminology_assistance_enabled: bool = True
    sentiment_detection_enabled: bool = True
    emergency_detection_enabled: bool = True

    # Behavior settings
    auto_adjust_tone: bool = True
    provide_follow_up_questions: bool = True
    include_guidelines: bool = True

    # Compliance settings
    block_on_critical_violation: bool = True
    log_all_violations: bool = True

    # Response generation
    max_response_suggestions: int = 3
    include_customer_explanations: bool = True


@dataclass
class ProcessingResult:
    """Result of processing a message through intelligence."""

    # Detection results
    detected_intent: Optional[str] = None
    detected_sentiment: SentimentCategory = SentimentCategory.NEUTRAL
    detected_phase: ConversationPhase = ConversationPhase.DISCOVERY

    # Emergency handling
    is_emergency: bool = False
    emergency_protocol: Optional[EmergencyProtocol] = None

    # Compliance
    compliance_result: Optional[ComplianceCheckResult] = None
    requires_human: bool = False

    # Response suggestions
    suggested_responses: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    sentiment_acknowledgment: Optional[str] = None

    # Guidelines
    applicable_guidelines: List[BehaviorGuideline] = field(default_factory=list)

    # Terminology
    terms_to_explain: List[TermDefinition] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0
    confidence: float = 0.0


@dataclass
class SessionIntelligence:
    """Intelligence context for a conversation session."""

    session_id: str
    industry_type: IndustryType
    context: IndustryContext

    # Components
    behavior: ConversationBehavior
    compliance_checker: ComplianceChecker
    terminology_manager: TerminologyManager

    # State
    messages_processed: int = 0
    current_phase: ConversationPhase = ConversationPhase.GREETING
    dominant_sentiment: SentimentCategory = SentimentCategory.NEUTRAL
    topics_discussed: Set[str] = field(default_factory=set)
    info_collected: Dict[str, Any] = field(default_factory=dict)

    # Flags
    escalation_triggered: bool = False
    compliance_warning_given: bool = False


class IndustryIntelligence:
    """
    Central intelligence system for industry-specific voice agents.

    Combines:
    - Industry profiles with domain knowledge
    - Compliance checking and enforcement
    - Terminology management
    - Behavior patterns and guidelines
    - Sentiment and intent detection
    - Emergency handling
    """

    def __init__(self, config: Optional[IntelligenceConfig] = None):
        """
        Initialize industry intelligence.

        Args:
            config: Intelligence configuration
        """
        self.config = config or IntelligenceConfig()

        # Core managers
        self._compliance_manager = ComplianceManager()
        self._terminology_manager = TerminologyManager()
        self._behavior_engine = BehaviorEngine()

        # Session tracking
        self._sessions: Dict[str, SessionIntelligence] = {}

        # Statistics
        self._stats = {
            "messages_processed": 0,
            "emergencies_detected": 0,
            "compliance_violations": 0,
            "escalations_triggered": 0,
        }

    def create_session(
        self,
        session_id: str,
        industry_type: IndustryType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionIntelligence:
        """
        Create intelligence session for a conversation.

        Args:
            session_id: Session ID
            industry_type: Industry type
            metadata: Additional metadata

        Returns:
            Session intelligence context
        """
        # Get industry profile
        profile = get_industry_profile(industry_type)
        if not profile:
            profile = IndustryProfile(industry_type=industry_type)

        # Create context
        context = IndustryContext(
            profile=profile,
            active_compliance=profile.compliance_requirements,
        )

        # Check for seasonal patterns
        current_month = datetime.utcnow().month
        for pattern in profile.seasonal_patterns:
            if pattern.start_month <= current_month <= pattern.end_month:
                context.current_season = pattern
                break
            # Handle wrap-around (e.g., Oct-Mar)
            if pattern.start_month > pattern.end_month:
                if current_month >= pattern.start_month or current_month <= pattern.end_month:
                    context.current_season = pattern
                    break

        # Create components
        behavior = ConversationBehavior(industry_type)
        compliance_checker = self._compliance_manager.get_checker_for_industry(
            industry_type
        )

        # Create session
        session = SessionIntelligence(
            session_id=session_id,
            industry_type=industry_type,
            context=context,
            behavior=behavior,
            compliance_checker=compliance_checker,
            terminology_manager=self._terminology_manager,
        )

        self._sessions[session_id] = session

        logger.info(f"Intelligence session created: {session_id} for {industry_type.value}")

        return session

    def get_session(self, session_id: str) -> Optional[SessionIntelligence]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        End an intelligence session.

        Args:
            session_id: Session ID

        Returns:
            Session summary
        """
        session = self._sessions.pop(session_id, None)

        if session:
            # End behavior engine session
            behavior_stats = self._behavior_engine.end_session(session_id)

            summary = {
                "session_id": session_id,
                "industry_type": session.industry_type.value,
                "messages_processed": session.messages_processed,
                "topics_discussed": list(session.topics_discussed),
                "info_collected": session.info_collected,
                "final_phase": session.current_phase.value,
                "dominant_sentiment": session.dominant_sentiment.value,
                "escalation_triggered": session.escalation_triggered,
                "behavior_stats": behavior_stats,
            }

            logger.info(f"Intelligence session ended: {session_id}")

            return summary

        return None

    def process_message(
        self,
        session_id: str,
        message: str,
        speaker: str = "customer",
    ) -> ProcessingResult:
        """
        Process a message through intelligence system.

        Args:
            session_id: Session ID
            message: Message text
            speaker: Who said it (customer or agent)

        Returns:
            Processing result with intelligence
        """
        start_time = datetime.utcnow()
        result = ProcessingResult()

        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return result

        session.messages_processed += 1
        self._stats["messages_processed"] += 1

        # Process based on speaker
        if speaker == "customer":
            result = self._process_customer_message(session, message)
        else:
            result = self._process_agent_message(session, message)

        # Calculate processing time
        end_time = datetime.utcnow()
        result.processing_time_ms = (end_time - start_time).total_seconds() * 1000

        return result

    def _process_customer_message(
        self,
        session: SessionIntelligence,
        message: str,
    ) -> ProcessingResult:
        """Process a customer message."""
        result = ProcessingResult()

        # Check for emergency
        if self.config.emergency_detection_enabled:
            emergency = session.behavior.check_emergency(message)
            if emergency:
                result.is_emergency = True
                result.emergency_protocol = emergency
                result.requires_human = emergency.transfer_required
                result.suggested_responses = [emergency.acknowledgment_script]
                self._stats["emergencies_detected"] += 1

                session.context.emergency_detected = True

                return result

        # Detect sentiment
        if self.config.sentiment_detection_enabled:
            sentiment = session.behavior.detect_sentiment(message)
            result.detected_sentiment = sentiment
            session.dominant_sentiment = sentiment
            session.context.detected_sentiment = sentiment

            # Get sentiment acknowledgment
            sentiment_response = SENTIMENT_RESPONSES.get(sentiment)
            if sentiment_response and sentiment_response.acknowledgment_phrases:
                result.sentiment_acknowledgment = sentiment_response.acknowledgment_phrases[0]

        # Detect intent
        intent = session.behavior.detect_intent(message)
        result.detected_intent = intent

        # Get behavior suggestions
        behavior_suggestion = session.behavior.suggest_response(message)

        if behavior_suggestion.get("suggested_responses"):
            result.suggested_responses = behavior_suggestion["suggested_responses"][
                :self.config.max_response_suggestions
            ]

        if self.config.provide_follow_up_questions:
            result.follow_up_questions = behavior_suggestion.get("follow_up_questions", [])

        # Update phase
        if behavior_suggestion.get("phase"):
            phase_str = behavior_suggestion["phase"]
            try:
                result.detected_phase = ConversationPhase(phase_str)
                session.current_phase = result.detected_phase
                session.context.current_phase = result.detected_phase
            except ValueError:
                pass

        # Get applicable guidelines
        if self.config.include_guidelines:
            guidelines = session.behavior.get_guidelines_for_phase(session.current_phase)
            result.applicable_guidelines = guidelines[:3]

        # Check for terms to explain
        if self.config.terminology_assistance_enabled:
            terms = self._find_terms_in_message(session, message)
            result.terms_to_explain = terms

        # Track topics
        if intent:
            session.topics_discussed.add(intent)

        return result

    def _process_agent_message(
        self,
        session: SessionIntelligence,
        message: str,
    ) -> ProcessingResult:
        """Process an agent message (for compliance checking)."""
        result = ProcessingResult()

        # Check compliance
        if self.config.compliance_checking_enabled:
            compliance_result = session.compliance_checker.check_text(message)
            result.compliance_result = compliance_result

            if not compliance_result.is_compliant:
                self._stats["compliance_violations"] += len(compliance_result.violations)

                if self.config.log_all_violations:
                    for violation in compliance_result.violations:
                        logger.warning(
                            f"Compliance violation in session {session.session_id}: "
                            f"{violation.description}"
                        )

                if compliance_result.escalation_required:
                    result.requires_human = True
                    session.escalation_triggered = True
                    self._stats["escalations_triggered"] += 1

        return result

    def _find_terms_in_message(
        self,
        session: SessionIntelligence,
        message: str,
    ) -> List[TermDefinition]:
        """Find industry terms in a message that might need explanation."""
        terms = []
        message_lower = message.lower()

        industry_terms = get_industry_terminology(session.industry_type)

        for term_def in industry_terms:
            # Check main term
            if term_def.term.lower() in message_lower:
                terms.append(term_def)
                continue

            # Check abbreviation
            if term_def.abbreviation and term_def.abbreviation.lower() in message_lower:
                terms.append(term_def)
                continue

            # Check synonyms
            for synonym in term_def.synonyms:
                if synonym.lower() in message_lower:
                    terms.append(term_def)
                    break

        return terms

    def get_response_guidance(
        self,
        session_id: str,
        customer_message: str,
    ) -> Dict[str, Any]:
        """
        Get comprehensive guidance for responding to a customer.

        Args:
            session_id: Session ID
            customer_message: Customer's message

        Returns:
            Complete response guidance
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        # Process the message
        result = self.process_message(session_id, customer_message, speaker="customer")

        # Build comprehensive guidance
        guidance = {
            "session_context": {
                "industry": session.industry_type.value,
                "current_phase": session.current_phase.value,
                "messages_so_far": session.messages_processed,
            },
            "detection": {
                "intent": result.detected_intent,
                "sentiment": result.detected_sentiment.value,
                "is_emergency": result.is_emergency,
            },
            "response": {
                "suggested_responses": result.suggested_responses,
                "sentiment_acknowledgment": result.sentiment_acknowledgment,
                "follow_up_questions": result.follow_up_questions,
            },
            "guidelines": [
                {
                    "guideline": g.guideline,
                    "do": g.do_examples[:2] if g.do_examples else [],
                    "dont": g.dont_examples[:2] if g.dont_examples else [],
                }
                for g in result.applicable_guidelines
            ],
            "terminology": [
                {
                    "term": t.term,
                    "explanation": t.customer_friendly or t.definition,
                }
                for t in result.terms_to_explain
            ],
            "requires_human": result.requires_human,
        }

        # Add emergency info if detected
        if result.is_emergency and result.emergency_protocol:
            guidance["emergency"] = {
                "severity": result.emergency_protocol.severity,
                "immediate_action": result.emergency_protocol.immediate_response,
                "script": result.emergency_protocol.acknowledgment_script,
                "transfer_required": result.emergency_protocol.transfer_required,
            }

        # Add seasonal context
        if session.context.current_season:
            guidance["seasonal"] = {
                "name": session.context.current_season.name,
                "talking_points": session.context.current_season.talking_points,
            }

        # Add required disclosures
        disclosures = session.compliance_checker.get_required_disclosures()
        if disclosures:
            guidance["required_disclosures"] = disclosures

        return guidance

    def validate_response(
        self,
        session_id: str,
        proposed_response: str,
    ) -> Dict[str, Any]:
        """
        Validate a proposed agent response before sending.

        Args:
            session_id: Session ID
            proposed_response: Proposed response text

        Returns:
            Validation result
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"valid": False, "error": "Session not found"}

        # Check compliance
        result = self.process_message(session_id, proposed_response, speaker="agent")

        validation = {
            "valid": True,
            "warnings": [],
            "violations": [],
            "suggestions": [],
        }

        if result.compliance_result:
            if not result.compliance_result.is_compliant:
                validation["valid"] = False

                for violation in result.compliance_result.violations:
                    validation["violations"].append({
                        "level": violation.level.value,
                        "description": violation.description,
                        "matched_phrase": violation.matched_phrase,
                        "remediation": violation.remediation,
                    })

            validation["warnings"].extend(result.compliance_result.warnings)

        if result.requires_human:
            validation["requires_human"] = True
            validation["suggestions"].append(
                "This response should be reviewed by a human before sending."
            )

        return validation

    def explain_term(
        self,
        session_id: str,
        term: str,
    ) -> Optional[str]:
        """
        Get customer-friendly explanation for a term.

        Args:
            session_id: Session ID
            term: Term to explain

        Returns:
            Customer-friendly explanation
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        return session.terminology_manager.get_customer_explanation(
            term,
            session.industry_type,
        )

    def get_industry_info(
        self,
        industry_type: IndustryType,
    ) -> Dict[str, Any]:
        """
        Get comprehensive information about an industry.

        Args:
            industry_type: Industry type

        Returns:
            Industry information
        """
        profile = get_industry_profile(industry_type)
        if not profile:
            return {"error": "Industry not found"}

        return {
            "name": profile.name,
            "description": profile.description,
            "customer_segments": profile.typical_customer_segments,
            "pain_points": profile.common_pain_points,
            "value_propositions": profile.value_propositions,
            "regulations": [r.value for r in profile.applicable_regulations],
            "services": [
                {
                    "name": s.name,
                    "description": s.description,
                    "category": s.category,
                }
                for s in profile.common_services
            ],
            "tone": profile.tone,
            "formality_level": profile.formality_level,
            "empathy_level": profile.empathy_level,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get intelligence system statistics."""
        return {
            **self._stats,
            "active_sessions": len(self._sessions),
            "supported_industries": len(INDUSTRY_PROFILES),
        }


def create_industry_intelligence(
    config: Optional[IntelligenceConfig] = None,
) -> IndustryIntelligence:
    """
    Create an industry intelligence instance.

    Args:
        config: Intelligence configuration

    Returns:
        Configured intelligence instance
    """
    return IndustryIntelligence(config)


__all__ = [
    "IntelligenceConfig",
    "ProcessingResult",
    "SessionIntelligence",
    "IndustryIntelligence",
    "create_industry_intelligence",
]
