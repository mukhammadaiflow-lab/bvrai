"""
Response Advisor.

Generates recommendations for AI response adaptation based on
detected emotional state and conversation context.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..config import EmotionCategory, ArousalLevel, ValenceLevel, get_settings
from ..models import (
    EmotionalState,
    ResponseRecommendation,
    ToneGuidance,
    StyleAdjustment,
)

logger = logging.getLogger(__name__)


class ResponseStrategy(str, Enum):
    """High-level response strategies."""

    EMPATHETIC = "empathetic"
    PROFESSIONAL = "professional"
    ENTHUSIASTIC = "enthusiastic"
    CALMING = "calming"
    CLARIFYING = "clarifying"
    REASSURING = "reassuring"
    SOLUTION_FOCUSED = "solution_focused"
    ENGAGING = "engaging"
    NEUTRAL = "neutral"


@dataclass
class EmotionStrategy:
    """Strategy configuration for an emotion."""

    primary_strategy: ResponseStrategy
    secondary_strategy: Optional[ResponseStrategy]
    tone_warmth: float  # -1 to 1 (cold to warm)
    speaking_rate_adjustment: float  # -1 to 1 (slower to faster)
    energy_level: float  # 0 to 1
    formality: float  # 0 to 1 (casual to formal)
    empathy_level: float  # 0 to 1
    directness: float  # 0 to 1
    suggested_phrases: List[str]
    avoid_phrases: List[str]
    priority_actions: List[str]


# Strategy configurations for each emotion
EMOTION_STRATEGIES: Dict[EmotionCategory, EmotionStrategy] = {
    EmotionCategory.HAPPY: EmotionStrategy(
        primary_strategy=ResponseStrategy.ENTHUSIASTIC,
        secondary_strategy=ResponseStrategy.ENGAGING,
        tone_warmth=0.8,
        speaking_rate_adjustment=0.2,
        energy_level=0.8,
        formality=0.3,
        empathy_level=0.6,
        directness=0.7,
        suggested_phrases=[
            "That's wonderful!",
            "I'm glad to hear that",
            "Great choice!",
        ],
        avoid_phrases=[
            "I'm sorry",
            "Unfortunately",
            "I understand your frustration",
        ],
        priority_actions=["match_enthusiasm", "reinforce_positive", "explore_further"],
    ),
    EmotionCategory.SAD: EmotionStrategy(
        primary_strategy=ResponseStrategy.EMPATHETIC,
        secondary_strategy=ResponseStrategy.REASSURING,
        tone_warmth=0.9,
        speaking_rate_adjustment=-0.3,
        energy_level=0.4,
        formality=0.4,
        empathy_level=0.95,
        directness=0.3,
        suggested_phrases=[
            "I understand how you feel",
            "That sounds difficult",
            "I'm here to help",
        ],
        avoid_phrases=[
            "Cheer up!",
            "It's not that bad",
            "Look on the bright side",
        ],
        priority_actions=["acknowledge_feeling", "offer_support", "gentle_guidance"],
    ),
    EmotionCategory.ANGRY: EmotionStrategy(
        primary_strategy=ResponseStrategy.CALMING,
        secondary_strategy=ResponseStrategy.SOLUTION_FOCUSED,
        tone_warmth=0.5,
        speaking_rate_adjustment=-0.2,
        energy_level=0.5,
        formality=0.6,
        empathy_level=0.8,
        directness=0.4,
        suggested_phrases=[
            "I completely understand your frustration",
            "Let me help resolve this",
            "You're right to be concerned",
        ],
        avoid_phrases=[
            "Calm down",
            "There's no need to be upset",
            "You're overreacting",
        ],
        priority_actions=[
            "validate_frustration",
            "take_ownership",
            "offer_immediate_action",
        ],
    ),
    EmotionCategory.FEARFUL: EmotionStrategy(
        primary_strategy=ResponseStrategy.REASSURING,
        secondary_strategy=ResponseStrategy.CLARIFYING,
        tone_warmth=0.7,
        speaking_rate_adjustment=-0.2,
        energy_level=0.5,
        formality=0.5,
        empathy_level=0.85,
        directness=0.5,
        suggested_phrases=[
            "Don't worry",
            "You're in good hands",
            "Let me explain step by step",
        ],
        avoid_phrases=[
            "You should be worried",
            "This is serious",
            "I can't guarantee",
        ],
        priority_actions=["provide_reassurance", "explain_clearly", "build_confidence"],
    ),
    EmotionCategory.SURPRISED: EmotionStrategy(
        primary_strategy=ResponseStrategy.CLARIFYING,
        secondary_strategy=ResponseStrategy.ENGAGING,
        tone_warmth=0.6,
        speaking_rate_adjustment=0.0,
        energy_level=0.6,
        formality=0.4,
        empathy_level=0.5,
        directness=0.7,
        suggested_phrases=[
            "I can explain further",
            "Here's what happened",
            "Let me clarify",
        ],
        avoid_phrases=[
            "Obviously",
            "As you know",
            "Clearly",
        ],
        priority_actions=["provide_context", "address_surprise", "ensure_understanding"],
    ),
    EmotionCategory.DISGUSTED: EmotionStrategy(
        primary_strategy=ResponseStrategy.PROFESSIONAL,
        secondary_strategy=ResponseStrategy.SOLUTION_FOCUSED,
        tone_warmth=0.4,
        speaking_rate_adjustment=-0.1,
        energy_level=0.5,
        formality=0.7,
        empathy_level=0.7,
        directness=0.6,
        suggested_phrases=[
            "I understand your concern",
            "Let me address that",
            "We can fix this",
        ],
        avoid_phrases=[
            "It's not that bad",
            "You're being picky",
            "That's normal",
        ],
        priority_actions=["acknowledge_issue", "offer_alternative", "maintain_respect"],
    ),
    EmotionCategory.NEUTRAL: EmotionStrategy(
        primary_strategy=ResponseStrategy.NEUTRAL,
        secondary_strategy=ResponseStrategy.PROFESSIONAL,
        tone_warmth=0.5,
        speaking_rate_adjustment=0.0,
        energy_level=0.5,
        formality=0.5,
        empathy_level=0.5,
        directness=0.6,
        suggested_phrases=[],
        avoid_phrases=[],
        priority_actions=["provide_information", "be_helpful", "stay_focused"],
    ),
    EmotionCategory.EXCITED: EmotionStrategy(
        primary_strategy=ResponseStrategy.ENTHUSIASTIC,
        secondary_strategy=ResponseStrategy.ENGAGING,
        tone_warmth=0.9,
        speaking_rate_adjustment=0.3,
        energy_level=0.9,
        formality=0.2,
        empathy_level=0.7,
        directness=0.7,
        suggested_phrases=[
            "Absolutely!",
            "Let's do this!",
            "That's exciting!",
        ],
        avoid_phrases=[
            "Slow down",
            "Let's be realistic",
            "Hold on",
        ],
        priority_actions=["match_energy", "channel_enthusiasm", "build_momentum"],
    ),
    EmotionCategory.FRUSTRATED: EmotionStrategy(
        primary_strategy=ResponseStrategy.SOLUTION_FOCUSED,
        secondary_strategy=ResponseStrategy.EMPATHETIC,
        tone_warmth=0.6,
        speaking_rate_adjustment=-0.1,
        energy_level=0.6,
        formality=0.5,
        empathy_level=0.85,
        directness=0.7,
        suggested_phrases=[
            "I hear you",
            "Let's solve this together",
            "I'll take care of this",
        ],
        avoid_phrases=[
            "That's just how it works",
            "There's nothing I can do",
            "You should have",
        ],
        priority_actions=[
            "acknowledge_difficulty",
            "take_responsibility",
            "provide_solution",
        ],
    ),
    EmotionCategory.CONFUSED: EmotionStrategy(
        primary_strategy=ResponseStrategy.CLARIFYING,
        secondary_strategy=ResponseStrategy.REASSURING,
        tone_warmth=0.6,
        speaking_rate_adjustment=-0.2,
        energy_level=0.5,
        formality=0.4,
        empathy_level=0.6,
        directness=0.8,
        suggested_phrases=[
            "Let me explain more clearly",
            "Here's what that means",
            "To put it simply",
        ],
        avoid_phrases=[
            "As I said before",
            "It's simple",
            "Obviously",
        ],
        priority_actions=["simplify_explanation", "use_examples", "check_understanding"],
    ),
    EmotionCategory.BORED: EmotionStrategy(
        primary_strategy=ResponseStrategy.ENGAGING,
        secondary_strategy=ResponseStrategy.ENTHUSIASTIC,
        tone_warmth=0.6,
        speaking_rate_adjustment=0.1,
        energy_level=0.7,
        formality=0.3,
        empathy_level=0.4,
        directness=0.8,
        suggested_phrases=[
            "Here's something interesting",
            "Let me show you",
            "You might find this helpful",
        ],
        avoid_phrases=[
            "To continue with the process",
            "As I was saying",
            "Moving on",
        ],
        priority_actions=["add_interest", "be_concise", "offer_options"],
    ),
    EmotionCategory.ANXIOUS: EmotionStrategy(
        primary_strategy=ResponseStrategy.CALMING,
        secondary_strategy=ResponseStrategy.REASSURING,
        tone_warmth=0.7,
        speaking_rate_adjustment=-0.3,
        energy_level=0.4,
        formality=0.5,
        empathy_level=0.85,
        directness=0.5,
        suggested_phrases=[
            "Take your time",
            "There's no rush",
            "I'm here to help",
        ],
        avoid_phrases=[
            "Hurry",
            "Time is running out",
            "You need to decide now",
        ],
        priority_actions=["reduce_pressure", "provide_clarity", "offer_support"],
    ),
    EmotionCategory.CONTENT: EmotionStrategy(
        primary_strategy=ResponseStrategy.PROFESSIONAL,
        secondary_strategy=ResponseStrategy.ENGAGING,
        tone_warmth=0.6,
        speaking_rate_adjustment=0.0,
        energy_level=0.5,
        formality=0.4,
        empathy_level=0.5,
        directness=0.6,
        suggested_phrases=[
            "Is there anything else I can help with?",
            "Perfect",
            "Glad to help",
        ],
        avoid_phrases=[],
        priority_actions=["maintain_satisfaction", "offer_additional_help", "be_efficient"],
    ),
    EmotionCategory.EMPATHETIC: EmotionStrategy(
        primary_strategy=ResponseStrategy.EMPATHETIC,
        secondary_strategy=ResponseStrategy.PROFESSIONAL,
        tone_warmth=0.8,
        speaking_rate_adjustment=-0.1,
        energy_level=0.5,
        formality=0.4,
        empathy_level=0.9,
        directness=0.5,
        suggested_phrases=[
            "I appreciate you sharing that",
            "That means a lot",
            "Thank you for understanding",
        ],
        avoid_phrases=[],
        priority_actions=["reciprocate_empathy", "build_connection", "show_appreciation"],
    ),
    EmotionCategory.URGENT: EmotionStrategy(
        primary_strategy=ResponseStrategy.SOLUTION_FOCUSED,
        secondary_strategy=ResponseStrategy.PROFESSIONAL,
        tone_warmth=0.5,
        speaking_rate_adjustment=0.2,
        energy_level=0.7,
        formality=0.6,
        empathy_level=0.6,
        directness=0.9,
        suggested_phrases=[
            "Right away",
            "Let me help you immediately",
            "Here's what we'll do",
        ],
        avoid_phrases=[
            "When you have a moment",
            "Eventually",
            "At some point",
        ],
        priority_actions=["act_quickly", "be_direct", "prioritize_solution"],
    ),
}


class ResponseAdvisor:
    """
    Generates response recommendations based on emotional state.

    Provides:
    - Tone and style guidance
    - Suggested phrases and approaches
    - Things to avoid
    - Speaking rate and energy adjustments
    """

    def __init__(self):
        """Initialize advisor."""
        self.strategies = EMOTION_STRATEGIES
        self.settings = get_settings()

    def get_recommendation(
        self,
        state: EmotionalState,
        context: Optional[Dict] = None,
    ) -> ResponseRecommendation:
        """
        Generate response recommendation based on emotional state.

        Args:
            state: Current emotional state
            context: Optional additional context

        Returns:
            ResponseRecommendation with guidance
        """
        # Get primary strategy
        strategy = self.strategies.get(
            state.current_emotion,
            self.strategies[EmotionCategory.NEUTRAL],
        )

        # Adjust based on trends
        strategy = self._adjust_for_trends(strategy, state)

        # Adjust based on shift
        if state.recent_shift:
            strategy = self._adjust_for_shift(strategy, state)

        # Generate tone guidance
        tone = self._generate_tone_guidance(strategy, state)

        # Generate style adjustments
        style = self._generate_style_adjustments(strategy, state)

        # Get action priorities
        actions = self._prioritize_actions(strategy, state, context)

        # Generate suggested phrases
        suggested = self._select_phrases(strategy, state)

        # Generate avoid list
        avoid = strategy.avoid_phrases.copy()

        # Determine urgency
        urgency = self._calculate_urgency(state)

        return ResponseRecommendation(
            timestamp_ms=state.timestamp_ms,
            primary_strategy=strategy.primary_strategy.value,
            secondary_strategy=(
                strategy.secondary_strategy.value
                if strategy.secondary_strategy
                else None
            ),
            tone_guidance=tone,
            style_adjustments=style,
            priority_actions=actions,
            suggested_phrases=suggested,
            phrases_to_avoid=avoid,
            urgency_level=urgency,
            confidence=state.current_confidence,
            reasoning=self._generate_reasoning(state, strategy),
        )

    def _adjust_for_trends(
        self,
        strategy: EmotionStrategy,
        state: EmotionalState,
    ) -> EmotionStrategy:
        """Adjust strategy based on emotional trends."""
        # Create a modified copy
        import copy
        adjusted = copy.deepcopy(strategy)

        # If arousal is decreasing, slightly increase energy
        if state.arousal_trend < -0.1:
            adjusted.energy_level = min(1.0, adjusted.energy_level + 0.1)

        # If valence is decreasing, increase empathy
        if state.valence_trend < -0.1:
            adjusted.empathy_level = min(1.0, adjusted.empathy_level + 0.15)

        # If stability is low, increase warmth and decrease directness
        if state.stability < 0.5:
            adjusted.tone_warmth = min(1.0, adjusted.tone_warmth + 0.1)
            adjusted.directness = max(0.0, adjusted.directness - 0.1)

        return adjusted

    def _adjust_for_shift(
        self,
        strategy: EmotionStrategy,
        state: EmotionalState,
    ) -> EmotionStrategy:
        """Adjust strategy based on recent emotional shift."""
        import copy
        adjusted = copy.deepcopy(strategy)

        shift = state.recent_shift

        if shift.direction == "negative":
            # More empathetic, slower pace
            adjusted.empathy_level = min(1.0, adjusted.empathy_level + 0.2)
            adjusted.speaking_rate_adjustment -= 0.1
            adjusted.primary_strategy = ResponseStrategy.EMPATHETIC

        elif shift.direction == "escalation":
            # Calming approach
            adjusted.primary_strategy = ResponseStrategy.CALMING
            adjusted.speaking_rate_adjustment -= 0.2
            adjusted.energy_level = max(0.3, adjusted.energy_level - 0.2)

        elif shift.direction == "positive":
            # Match the positive shift
            adjusted.tone_warmth = min(1.0, adjusted.tone_warmth + 0.1)
            adjusted.energy_level = min(1.0, adjusted.energy_level + 0.1)

        return adjusted

    def _generate_tone_guidance(
        self,
        strategy: EmotionStrategy,
        state: EmotionalState,
    ) -> ToneGuidance:
        """Generate tone guidance."""
        # Map warmth to description
        if strategy.tone_warmth > 0.7:
            warmth_desc = "warm_friendly"
        elif strategy.tone_warmth > 0.4:
            warmth_desc = "pleasant_professional"
        elif strategy.tone_warmth > 0.1:
            warmth_desc = "neutral"
        else:
            warmth_desc = "formal"

        # Map energy to description
        if strategy.energy_level > 0.7:
            energy_desc = "high_energy"
        elif strategy.energy_level > 0.4:
            energy_desc = "moderate"
        else:
            energy_desc = "calm"

        # Map formality
        if strategy.formality > 0.7:
            formality_desc = "formal"
        elif strategy.formality > 0.4:
            formality_desc = "semi_formal"
        else:
            formality_desc = "casual"

        return ToneGuidance(
            warmth=warmth_desc,
            warmth_value=strategy.tone_warmth,
            energy=energy_desc,
            energy_value=strategy.energy_level,
            formality=formality_desc,
            formality_value=strategy.formality,
            empathy_level=strategy.empathy_level,
            directness_level=strategy.directness,
        )

    def _generate_style_adjustments(
        self,
        strategy: EmotionStrategy,
        state: EmotionalState,
    ) -> StyleAdjustment:
        """Generate style adjustments for TTS."""
        return StyleAdjustment(
            speaking_rate_modifier=strategy.speaking_rate_adjustment,
            pitch_modifier=0.0,  # No pitch change by default
            volume_modifier=0.0,  # No volume change by default
            pause_frequency=(
                "increased" if strategy.speaking_rate_adjustment < 0 else "normal"
            ),
            emphasis_level=(
                "high" if strategy.energy_level > 0.7 else "normal"
            ),
        )

    def _prioritize_actions(
        self,
        strategy: EmotionStrategy,
        state: EmotionalState,
        context: Optional[Dict],
    ) -> List[str]:
        """Prioritize actions based on state."""
        actions = strategy.priority_actions.copy()

        # Add context-specific actions
        if state.deviation_from_baseline > 0.3:
            actions.insert(0, "acknowledge_change")

        if state.conversation_trajectory == "negative":
            actions.insert(0, "address_concerns")

        if state.stability < 0.5:
            actions.insert(0, "provide_stability")

        return actions[:5]  # Top 5 actions

    def _select_phrases(
        self,
        strategy: EmotionStrategy,
        state: EmotionalState,
    ) -> List[str]:
        """Select appropriate suggested phrases."""
        phrases = strategy.suggested_phrases.copy()

        # Add trajectory-specific phrases
        if state.conversation_trajectory == "positive":
            phrases.extend([
                "Great progress",
                "We're on the right track",
            ])
        elif state.conversation_trajectory == "negative":
            phrases.extend([
                "I want to help make this better",
                "Let's work through this together",
            ])

        return phrases[:5]  # Top 5 phrases

    def _calculate_urgency(self, state: EmotionalState) -> str:
        """Calculate urgency level."""
        if state.current_emotion == EmotionCategory.URGENT:
            return "high"

        if state.current_emotion in [
            EmotionCategory.ANGRY,
            EmotionCategory.FRUSTRATED,
        ]:
            if state.current_confidence > 0.7:
                return "high"
            return "medium"

        if state.recent_shift and state.recent_shift.direction == "escalation":
            return "medium"

        return "normal"

    def _generate_reasoning(
        self,
        state: EmotionalState,
        strategy: EmotionStrategy,
    ) -> str:
        """Generate human-readable reasoning."""
        reasons = []

        reasons.append(
            f"Detected {state.current_emotion.value} emotion "
            f"with {state.current_confidence:.0%} confidence"
        )

        if state.recent_shift:
            reasons.append(
                f"Recent {state.recent_shift.direction} shift "
                f"from {state.recent_shift.from_emotion.value}"
            )

        if state.stability < 0.5:
            reasons.append("Emotional state is unstable")

        if state.deviation_from_baseline > 0.3:
            reasons.append("Significant deviation from baseline")

        reasons.append(
            f"Recommended {strategy.primary_strategy.value} approach"
        )

        return ". ".join(reasons) + "."

    def get_quick_recommendation(
        self,
        emotion: EmotionCategory,
        confidence: float = 0.5,
    ) -> Dict:
        """Get quick recommendation without full state."""
        strategy = self.strategies.get(
            emotion,
            self.strategies[EmotionCategory.NEUTRAL],
        )

        return {
            "strategy": strategy.primary_strategy.value,
            "tone_warmth": strategy.tone_warmth,
            "energy_level": strategy.energy_level,
            "empathy_level": strategy.empathy_level,
            "speaking_rate_adjustment": strategy.speaking_rate_adjustment,
            "suggested_phrases": strategy.suggested_phrases[:3],
            "avoid_phrases": strategy.avoid_phrases[:3],
            "actions": strategy.priority_actions[:3],
        }
