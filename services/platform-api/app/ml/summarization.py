"""
Conversation Summarization System

Summarization for voice conversations:
- Extractive summarization
- Abstractive summarization
- Key point extraction
- Action item detection
- Call disposition
"""

from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import re
import math
import logging

logger = logging.getLogger(__name__)


class SummaryType(str, Enum):
    """Summary types."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    BULLET_POINTS = "bullet_points"
    NARRATIVE = "narrative"


class KeyPointType(str, Enum):
    """Types of key points."""
    MAIN_TOPIC = "main_topic"
    ISSUE = "issue"
    RESOLUTION = "resolution"
    REQUEST = "request"
    INFORMATION = "information"
    DECISION = "decision"
    OUTCOME = "outcome"


class ActionItemPriority(str, Enum):
    """Action item priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CallDisposition(str, Enum):
    """Call outcome dispositions."""
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FOLLOW_UP = "follow_up"
    TRANSFERRED = "transferred"
    ABANDONED = "abandoned"
    PENDING = "pending"
    NO_ACTION = "no_action"


@dataclass
class KeyPoint:
    """Extracted key point."""
    text: str
    type: KeyPointType
    confidence: float
    source_turn: Optional[int] = None
    speaker: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "type": self.type.value,
            "confidence": self.confidence,
            "source_turn": self.source_turn,
            "speaker": self.speaker,
            "metadata": self.metadata,
        }


@dataclass
class ActionItem:
    """Detected action item."""
    description: str
    assignee: Optional[str] = None
    priority: ActionItemPriority = ActionItemPriority.MEDIUM
    due_date: Optional[str] = None
    source_text: Optional[str] = None
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "assignee": self.assignee,
            "priority": self.priority.value,
            "due_date": self.due_date,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class Summary:
    """Conversation summary."""
    text: str
    summary_type: SummaryType
    key_points: List[KeyPoint] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    disposition: Optional[CallDisposition] = None
    topics: List[str] = field(default_factory=list)
    sentiment_summary: Optional[str] = None
    duration_seconds: Optional[float] = None
    turn_count: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "summary_type": self.summary_type.value,
            "key_points": [kp.to_dict() for kp in self.key_points],
            "action_items": [ai.to_dict() for ai in self.action_items],
            "disposition": self.disposition.value if self.disposition else None,
            "topics": self.topics,
            "sentiment_summary": self.sentiment_summary,
            "duration_seconds": self.duration_seconds,
            "turn_count": self.turn_count,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


class ConversationSummarizer(ABC):
    """Abstract base for conversation summarizers."""

    @abstractmethod
    async def summarize(
        self,
        conversation: List[Dict[str, str]],
        max_length: int = 200,
    ) -> Summary:
        """Summarize conversation."""
        pass


class ExtractiveSummarizer(ConversationSummarizer):
    """
    Extractive summarization.

    Selects important sentences from the conversation.
    """

    def __init__(
        self,
        sentence_limit: int = 5,
        include_speakers: bool = True,
    ):
        self.sentence_limit = sentence_limit
        self.include_speakers = include_speakers
        self._important_words: Set[str] = self._load_important_words()

    def _load_important_words(self) -> Set[str]:
        """Load domain-important words."""
        return {
            # Issues and problems
            "problem", "issue", "error", "bug", "broken", "wrong",
            "not working", "doesn't work", "failed", "failure",
            # Resolutions
            "resolved", "fixed", "solution", "solved", "helped",
            "working now", "completed", "done",
            # Actions
            "will", "need to", "have to", "must", "should",
            "please", "request", "want", "require",
            # Important markers
            "important", "urgent", "critical", "priority",
            "deadline", "asap", "immediately",
            # Customer service
            "refund", "cancel", "return", "exchange", "complaint",
            "feedback", "satisfied", "dissatisfied",
            # Follow-up
            "follow up", "call back", "email", "contact",
            "schedule", "appointment", "meeting",
        }

    async def summarize(
        self,
        conversation: List[Dict[str, str]],
        max_length: int = 200,
    ) -> Summary:
        """Extract key sentences as summary."""
        import time
        start_time = time.time()

        # Score each turn
        scored_turns = []
        for i, turn in enumerate(conversation):
            content = turn.get("content", turn.get("text", ""))
            speaker = turn.get("role", turn.get("speaker", "unknown"))

            score = self._score_sentence(content, i, len(conversation))
            scored_turns.append((score, i, speaker, content))

        # Select top sentences
        scored_turns.sort(reverse=True)
        selected = scored_turns[:self.sentence_limit]
        selected.sort(key=lambda x: x[1])  # Sort by original order

        # Build summary text
        if self.include_speakers:
            summary_parts = [
                f"{speaker}: {content}"
                for _, _, speaker, content in selected
            ]
        else:
            summary_parts = [content for _, _, _, content in selected]

        summary_text = " ".join(summary_parts)

        # Truncate if needed
        if len(summary_text) > max_length:
            summary_text = summary_text[:max_length - 3] + "..."

        # Extract key points
        key_points = self._extract_key_points(conversation)

        # Detect action items
        action_items = self._detect_action_items(conversation)

        # Detect topics
        topics = self._detect_topics(conversation)

        # Determine disposition
        disposition = self._determine_disposition(conversation)

        return Summary(
            text=summary_text,
            summary_type=SummaryType.EXTRACTIVE,
            key_points=key_points,
            action_items=action_items,
            disposition=disposition,
            topics=topics,
            turn_count=len(conversation),
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def _score_sentence(
        self,
        text: str,
        position: int,
        total_turns: int,
    ) -> float:
        """Score sentence importance."""
        score = 0.0
        text_lower = text.lower()

        # Important word score
        for word in self._important_words:
            if word in text_lower:
                score += 1.0

        # Position score (first and last are important)
        if position == 0:
            score += 2.0  # First turn
        elif position == total_turns - 1:
            score += 1.5  # Last turn
        elif position < 3:
            score += 1.0  # Early turns

        # Length score (prefer medium-length)
        word_count = len(text.split())
        if 10 <= word_count <= 30:
            score += 0.5

        # Question/answer score
        if "?" in text:
            score += 0.3
        if text_lower.startswith(("yes", "no", "sure", "okay")):
            score += 0.2

        return score

    def _extract_key_points(
        self,
        conversation: List[Dict[str, str]],
    ) -> List[KeyPoint]:
        """Extract key points from conversation."""
        key_points = []

        patterns = {
            KeyPointType.ISSUE: [
                r"(?:problem|issue|error)\s+(?:with|is|was)\s+(.+?)(?:\.|$)",
                r"(?:doesn't|doesn't|can't|cannot)\s+(.+?)(?:\.|$)",
            ],
            KeyPointType.RESOLUTION: [
                r"(?:resolved|fixed|solved)\s+(.+?)(?:\.|$)",
                r"(?:that|this)\s+should\s+(?:fix|resolve|solve)\s+(.+?)(?:\.|$)",
            ],
            KeyPointType.REQUEST: [
                r"(?:please|can you|could you|would you)\s+(.+?)(?:\.|$|\?)",
                r"(?:i need|i want|i would like)\s+(.+?)(?:\.|$)",
            ],
        }

        for i, turn in enumerate(conversation):
            content = turn.get("content", turn.get("text", ""))
            speaker = turn.get("role", turn.get("speaker", ""))

            for point_type, type_patterns in patterns.items():
                for pattern in type_patterns:
                    matches = re.findall(pattern, content.lower())
                    for match in matches:
                        if len(match) > 10:  # Filter short matches
                            key_points.append(KeyPoint(
                                text=match.strip()[:100],
                                type=point_type,
                                confidence=0.7,
                                source_turn=i,
                                speaker=speaker,
                            ))

        return key_points[:10]  # Limit key points

    def _detect_action_items(
        self,
        conversation: List[Dict[str, str]],
    ) -> List[ActionItem]:
        """Detect action items in conversation."""
        action_items = []

        patterns = [
            (r"(?:i will|i'll|we will|we'll)\s+(.+?)(?:\.|$)", ActionItemPriority.MEDIUM),
            (r"(?:please|can you)\s+(.+?)(?:\.|$)", ActionItemPriority.HIGH),
            (r"(?:need to|have to|must|should)\s+(.+?)(?:\.|$)", ActionItemPriority.MEDIUM),
            (r"(?:follow up|call back|email|send)\s+(.+?)(?:\.|$)", ActionItemPriority.MEDIUM),
            (r"(?:schedule|book|arrange)\s+(.+?)(?:\.|$)", ActionItemPriority.LOW),
        ]

        for turn in conversation:
            content = turn.get("content", turn.get("text", ""))
            speaker = turn.get("role", turn.get("speaker", ""))

            for pattern, priority in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) > 10:  # Filter short matches
                        action_items.append(ActionItem(
                            description=match.strip()[:100],
                            assignee=speaker if speaker else None,
                            priority=priority,
                            source_text=content[:100],
                            confidence=0.7,
                        ))

        return action_items[:5]  # Limit action items

    def _detect_topics(
        self,
        conversation: List[Dict[str, str]],
    ) -> List[str]:
        """Detect main topics in conversation."""
        # Simple topic detection based on noun phrase frequency
        topic_patterns = [
            r"\b(account|order|payment|refund|subscription)\b",
            r"\b(product|service|item|delivery|shipping)\b",
            r"\b(support|help|assistance|question)\b",
            r"\b(billing|invoice|charge|fee)\b",
            r"\b(password|login|access|security)\b",
            r"\b(appointment|booking|reservation|schedule)\b",
        ]

        topic_counts: Dict[str, int] = {}
        full_text = " ".join(
            turn.get("content", turn.get("text", "")).lower()
            for turn in conversation
        )

        for pattern in topic_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                topic_counts[match] = topic_counts.get(match, 0) + 1

        # Return top topics
        sorted_topics = sorted(
            topic_counts.items(),
            key=lambda x: -x[1]
        )
        return [topic for topic, _ in sorted_topics[:5]]

    def _determine_disposition(
        self,
        conversation: List[Dict[str, str]],
    ) -> CallDisposition:
        """Determine call disposition."""
        full_text = " ".join(
            turn.get("content", turn.get("text", "")).lower()
            for turn in conversation
        )

        disposition_indicators = {
            CallDisposition.RESOLVED: [
                "resolved", "fixed", "solved", "that's it",
                "thank you", "thanks for your help", "working now",
            ],
            CallDisposition.ESCALATED: [
                "supervisor", "manager", "escalate", "transfer to",
            ],
            CallDisposition.FOLLOW_UP: [
                "follow up", "call back", "call you back",
                "get back to you", "will contact",
            ],
            CallDisposition.TRANSFERRED: [
                "transfer you", "connecting you", "another department",
            ],
        }

        for disposition, indicators in disposition_indicators.items():
            for indicator in indicators:
                if indicator in full_text:
                    return disposition

        # Check if last turn indicates resolution
        if conversation:
            last_content = conversation[-1].get("content", "").lower()
            if any(word in last_content for word in ["bye", "goodbye", "thank you"]):
                return CallDisposition.RESOLVED

        return CallDisposition.PENDING


class AbstractiveSummarizer(ConversationSummarizer):
    """
    Abstractive summarization.

    Generates new summary text (simplified implementation).
    """

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self._templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load summary templates."""
        return {
            "support_resolved": (
                "The customer contacted support regarding {issue}. "
                "The agent {resolution}. The call ended with the issue resolved."
            ),
            "support_pending": (
                "The customer inquired about {topics}. "
                "The agent provided {information}. Follow-up may be required."
            ),
            "inquiry": (
                "The customer had questions about {topics}. "
                "The agent explained {details}."
            ),
            "complaint": (
                "The customer expressed dissatisfaction with {issue}. "
                "The agent {response}. {outcome}"
            ),
        }

    async def summarize(
        self,
        conversation: List[Dict[str, str]],
        max_length: int = 200,
    ) -> Summary:
        """Generate abstractive summary."""
        import time
        start_time = time.time()

        # Analyze conversation
        analysis = self._analyze_conversation(conversation)

        # Select template
        template_type = self._select_template(analysis)
        template = self._templates[template_type]

        # Fill template
        summary_text = self._fill_template(template, analysis)

        # Truncate if needed
        if len(summary_text) > max_length:
            summary_text = summary_text[:max_length - 3] + "..."

        # Use extractive for key points and actions
        extractive = ExtractiveSummarizer()
        extractive_result = await extractive.summarize(conversation, max_length)

        return Summary(
            text=summary_text,
            summary_type=SummaryType.ABSTRACTIVE,
            key_points=extractive_result.key_points,
            action_items=extractive_result.action_items,
            disposition=extractive_result.disposition,
            topics=analysis.get("topics", []),
            sentiment_summary=analysis.get("sentiment", "neutral"),
            turn_count=len(conversation),
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def _analyze_conversation(
        self,
        conversation: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Analyze conversation for summary generation."""
        analysis = {
            "topics": [],
            "issues": [],
            "resolutions": [],
            "sentiment": "neutral",
            "call_type": "inquiry",
        }

        full_text = " ".join(
            turn.get("content", turn.get("text", "")).lower()
            for turn in conversation
        )

        # Detect issues
        issue_patterns = [
            r"(?:problem|issue|error) (?:with|is) (.+?)(?:\.|,|$)",
            r"(?:can't|cannot|doesn't|doesn't work) (.+?)(?:\.|$)",
        ]
        for pattern in issue_patterns:
            matches = re.findall(pattern, full_text)
            analysis["issues"].extend(matches[:3])

        # Detect resolutions
        resolution_patterns = [
            r"(?:fixed|resolved|solved|helped with) (.+?)(?:\.|$)",
            r"(?:should now|will now|can now) (.+?)(?:\.|$)",
        ]
        for pattern in resolution_patterns:
            matches = re.findall(pattern, full_text)
            analysis["resolutions"].extend(matches[:3])

        # Detect topics (simplified)
        topic_words = ["account", "order", "payment", "product", "service"]
        for word in topic_words:
            if word in full_text:
                analysis["topics"].append(word)

        # Detect sentiment
        negative_words = ["angry", "frustrated", "disappointed", "terrible"]
        positive_words = ["thank", "great", "helpful", "appreciate"]

        neg_count = sum(1 for w in negative_words if w in full_text)
        pos_count = sum(1 for w in positive_words if w in full_text)

        if neg_count > pos_count:
            analysis["sentiment"] = "negative"
        elif pos_count > neg_count:
            analysis["sentiment"] = "positive"

        # Determine call type
        if analysis["issues"]:
            if neg_count > 0:
                analysis["call_type"] = "complaint"
            else:
                analysis["call_type"] = "support_resolved" if analysis["resolutions"] else "support_pending"

        return analysis

    def _select_template(self, analysis: Dict[str, Any]) -> str:
        """Select appropriate template."""
        return analysis.get("call_type", "inquiry")

    def _fill_template(
        self,
        template: str,
        analysis: Dict[str, Any],
    ) -> str:
        """Fill template with analysis data."""
        replacements = {
            "{issue}": ", ".join(analysis.get("issues", ["their inquiry"]))[:50] or "their inquiry",
            "{issues}": ", ".join(analysis.get("issues", ["various matters"]))[:50] or "various matters",
            "{resolution}": ", ".join(analysis.get("resolutions", ["assisted the customer"]))[:50] or "assisted the customer",
            "{topics}": ", ".join(analysis.get("topics", ["general matters"]))[:50] or "general matters",
            "{information}": "relevant information",
            "{details}": "the relevant details",
            "{response}": "addressed their concerns",
            "{outcome}": "The matter was noted for follow-up." if not analysis.get("resolutions") else "The issue was addressed.",
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result


class HybridSummarizer(ConversationSummarizer):
    """
    Hybrid summarization combining extractive and abstractive.
    """

    def __init__(
        self,
        extractive_weight: float = 0.5,
    ):
        self.extractive_weight = extractive_weight
        self._extractive = ExtractiveSummarizer()
        self._abstractive = AbstractiveSummarizer()

    async def summarize(
        self,
        conversation: List[Dict[str, str]],
        max_length: int = 200,
    ) -> Summary:
        """Generate hybrid summary."""
        import time
        start_time = time.time()

        # Get both summaries
        extractive, abstractive = await asyncio.gather(
            self._extractive.summarize(conversation, max_length),
            self._abstractive.summarize(conversation, max_length),
        )

        # Combine: use abstractive text with extractive key points
        summary_text = abstractive.text

        # If abstractive is too short, append extractive
        if len(summary_text) < max_length * 0.5:
            remaining = max_length - len(summary_text) - 1
            if remaining > 50:
                summary_text += " " + extractive.text[:remaining]

        # Merge key points
        key_points = abstractive.key_points or extractive.key_points

        return Summary(
            text=summary_text[:max_length],
            summary_type=SummaryType.HYBRID,
            key_points=key_points,
            action_items=extractive.action_items,
            disposition=extractive.disposition or abstractive.disposition,
            topics=abstractive.topics or extractive.topics,
            sentiment_summary=abstractive.sentiment_summary,
            turn_count=len(conversation),
            processing_time_ms=(time.time() - start_time) * 1000,
        )


class BulletPointSummarizer(ConversationSummarizer):
    """
    Bullet point summary generator.

    Creates concise bullet-point summaries.
    """

    def __init__(self, max_bullets: int = 5):
        self.max_bullets = max_bullets
        self._extractive = ExtractiveSummarizer()

    async def summarize(
        self,
        conversation: List[Dict[str, str]],
        max_length: int = 500,
    ) -> Summary:
        """Generate bullet point summary."""
        import time
        start_time = time.time()

        # Get extractive analysis
        base_summary = await self._extractive.summarize(conversation, max_length)

        # Create bullets
        bullets = []

        # Add topic bullet
        if base_summary.topics:
            bullets.append(f"• Topics: {', '.join(base_summary.topics[:3])}")

        # Add key points
        for kp in base_summary.key_points[:3]:
            bullets.append(f"• {kp.type.value.title()}: {kp.text}")

        # Add disposition
        if base_summary.disposition:
            bullets.append(f"• Outcome: {base_summary.disposition.value.replace('_', ' ').title()}")

        # Add action items
        for ai in base_summary.action_items[:2]:
            bullets.append(f"• Action: {ai.description}")

        # Join bullets
        summary_text = "\n".join(bullets[:self.max_bullets])

        return Summary(
            text=summary_text,
            summary_type=SummaryType.BULLET_POINTS,
            key_points=base_summary.key_points,
            action_items=base_summary.action_items,
            disposition=base_summary.disposition,
            topics=base_summary.topics,
            turn_count=len(conversation),
            processing_time_ms=(time.time() - start_time) * 1000,
        )


class ConversationSummarizationPipeline:
    """
    Complete summarization pipeline.

    Combines multiple summarization strategies.
    """

    def __init__(
        self,
        default_type: SummaryType = SummaryType.HYBRID,
    ):
        self.default_type = default_type
        self._summarizers: Dict[SummaryType, ConversationSummarizer] = {
            SummaryType.EXTRACTIVE: ExtractiveSummarizer(),
            SummaryType.ABSTRACTIVE: AbstractiveSummarizer(),
            SummaryType.HYBRID: HybridSummarizer(),
            SummaryType.BULLET_POINTS: BulletPointSummarizer(),
        }

    async def summarize(
        self,
        conversation: List[Dict[str, str]],
        summary_type: Optional[SummaryType] = None,
        max_length: int = 200,
        include_all_types: bool = False,
    ) -> Union[Summary, Dict[SummaryType, Summary]]:
        """Generate summary."""
        if include_all_types:
            results = await asyncio.gather(*[
                summarizer.summarize(conversation, max_length)
                for summarizer in self._summarizers.values()
            ])
            return dict(zip(self._summarizers.keys(), results))

        type_to_use = summary_type or self.default_type
        summarizer = self._summarizers.get(type_to_use, self._summarizers[SummaryType.HYBRID])
        return await summarizer.summarize(conversation, max_length)

    async def quick_summary(
        self,
        conversation: List[Dict[str, str]],
        max_length: int = 100,
    ) -> str:
        """Generate quick text-only summary."""
        summary = await self._summarizers[SummaryType.EXTRACTIVE].summarize(
            conversation, max_length
        )
        return summary.text

    async def detailed_summary(
        self,
        conversation: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Generate detailed summary with all components."""
        summary = await self._summarizers[SummaryType.HYBRID].summarize(
            conversation, max_length=500
        )

        return {
            "summary": summary.text,
            "key_points": [kp.to_dict() for kp in summary.key_points],
            "action_items": [ai.to_dict() for ai in summary.action_items],
            "disposition": summary.disposition.value if summary.disposition else None,
            "topics": summary.topics,
            "sentiment": summary.sentiment_summary,
            "turn_count": summary.turn_count,
        }


class RealTimeSummarizer:
    """
    Real-time conversation summarizer.

    Updates summary incrementally as conversation progresses.
    """

    def __init__(self):
        self._conversation: List[Dict[str, str]] = []
        self._current_summary: Optional[Summary] = None
        self._pipeline = ConversationSummarizationPipeline()

    def add_turn(self, content: str, speaker: str = "unknown") -> None:
        """Add conversation turn."""
        self._conversation.append({
            "content": content,
            "role": speaker,
        })

    async def get_current_summary(self) -> Summary:
        """Get current summary."""
        if not self._conversation:
            return Summary(
                text="No conversation yet.",
                summary_type=SummaryType.EXTRACTIVE,
            )

        self._current_summary = await self._pipeline.summarize(
            self._conversation,
            summary_type=SummaryType.EXTRACTIVE,
            max_length=200,
        )
        return self._current_summary

    async def get_live_insights(self) -> Dict[str, Any]:
        """Get live conversation insights."""
        summary = await self.get_current_summary()

        return {
            "turn_count": len(self._conversation),
            "current_topics": summary.topics,
            "sentiment": summary.sentiment_summary,
            "key_points_count": len(summary.key_points),
            "action_items_count": len(summary.action_items),
            "likely_disposition": summary.disposition.value if summary.disposition else "pending",
        }

    def clear(self) -> None:
        """Clear conversation."""
        self._conversation.clear()
        self._current_summary = None


# Factory function
def create_summarizer(
    summarizer_type: str = "hybrid",
    **kwargs,
) -> ConversationSummarizer:
    """Create summarizer by type."""
    summarizers = {
        "extractive": ExtractiveSummarizer,
        "abstractive": AbstractiveSummarizer,
        "hybrid": HybridSummarizer,
        "bullet_points": BulletPointSummarizer,
    }

    cls = summarizers.get(summarizer_type, HybridSummarizer)
    return cls(**kwargs)
