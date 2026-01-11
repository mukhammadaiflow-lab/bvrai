"""
Dialog Service - RAG orchestration and LLM response generation.

This is the core service that:
1. Queries vector DB for relevant context
2. Builds prompts with system message, few-shots, context, and history
3. Calls LLM to generate response
4. Extracts actions from response
5. Returns speak_text and action_object
"""
import json
import re
from dataclasses import dataclass
from typing import Any

import structlog

from app.adapters.llm_adapter import LLMAdapter, LLMMessage
from app.adapters.vector_adapter import LocalVectorAdapter, VectorDBAdapter
from app.config import get_settings
from app.models.schemas import ActionObject, DialogTurnResponse
from app.services.session_service import SessionService

logger = structlog.get_logger()


@dataclass
class DialogContext:
    """Context for dialog generation."""

    tenant_id: str
    session_id: str
    transcript: str
    is_final: bool
    retrieved_docs: list[str]
    system_prompt: str
    few_shots: list[dict[str, str]]
    history: list[dict[str, str]]


class DialogService:
    """
    Main dialog orchestration service.

    Implements the RAG (Retrieval-Augmented Generation) flow:
    1. Retrieve relevant documents from vector store
    2. Build prompt with context
    3. Generate LLM response
    4. Extract actions and format response
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        vector_adapter: VectorDBAdapter,
        session_service: SessionService,
    ) -> None:
        self.llm = llm_adapter
        self.vector_db = vector_adapter
        self.sessions = session_service
        self._settings = get_settings()

        # Default system prompt for tenants without custom prompts
        self._default_system_prompt = """You are a helpful AI voice assistant for a business.
Your role is to assist customers with inquiries, bookings, and information requests.

Guidelines:
- Be friendly, professional, and concise
- Ask clarifying questions when needed
- Provide accurate information based on the context provided
- If you cannot help with something, politely explain why and suggest alternatives

When you identify an action the user wants to take (like booking an appointment),
indicate it by including [ACTION:action_type:{"param": "value"}] at the end of your response.
"""

    async def process_turn(
        self,
        tenant_id: str,
        session_id: str,
        transcript: str,
        is_final: bool = True,
    ) -> DialogTurnResponse:
        """
        Process a dialog turn and generate response.

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            transcript: User's speech transcript
            is_final: Whether transcript is final (not interim)

        Returns:
            DialogTurnResponse with speak_text and optional action
        """
        logger.info(
            "processing_turn",
            tenant_id=tenant_id,
            session_id=session_id,
            transcript_length=len(transcript),
            is_final=is_final,
        )

        # Get or create session
        session = self.sessions.get_or_create(
            session_id=session_id,
            tenant_id=tenant_id,
            system_prompt=self._default_system_prompt,
        )

        # Skip processing for interim transcripts
        if not is_final:
            return DialogTurnResponse(
                speak_text="",
                action_object=None,
                confidence=0.0,
                session_id=session_id,
                context_used=[],
            )

        # Step 1: Retrieve relevant context from vector store
        retrieved_docs, doc_ids = await self._retrieve_context(tenant_id, transcript)

        # Step 2: Build conversation history
        history = [
            {"role": turn.role, "content": turn.content}
            for turn in session.history[-self._settings.session_history_max_turns * 2 :]
        ]

        # Step 3: Build prompt
        messages = self._build_prompt(
            system_prompt=session.system_prompt or self._default_system_prompt,
            few_shots=session.few_shots,
            retrieved_docs=retrieved_docs,
            history=history,
            user_message=transcript,
        )

        # Step 4: Generate LLM response
        llm_response = await self.llm.complete(
            messages=messages,
            max_tokens=self._settings.llm_max_tokens,
            temperature=self._settings.llm_temperature,
        )

        # Step 5: Parse response and extract action
        speak_text, action_object = self._parse_response(llm_response.text)

        # Step 6: Update session history
        self.sessions.add_turn(session_id, "user", transcript)
        self.sessions.add_turn(
            session_id,
            "assistant",
            speak_text,
            metadata={"action": action_object.model_dump() if action_object else None},
        )

        # Calculate confidence based on LLM response and retrieval
        confidence = self._calculate_confidence(
            llm_response=llm_response.text,
            retrieved_count=len(retrieved_docs),
            action_extracted=action_object is not None,
        )

        logger.info(
            "turn_processed",
            session_id=session_id,
            speak_text_length=len(speak_text),
            has_action=action_object is not None,
            confidence=confidence,
            docs_used=len(doc_ids),
        )

        return DialogTurnResponse(
            speak_text=speak_text,
            action_object=action_object,
            confidence=confidence,
            session_id=session_id,
            context_used=doc_ids,
        )

    async def _retrieve_context(
        self, tenant_id: str, query: str
    ) -> tuple[list[str], list[str]]:
        """
        Retrieve relevant documents from vector store.

        Returns:
            Tuple of (document contents, document IDs)
        """
        try:
            # Generate query embedding
            if isinstance(self.vector_db, LocalVectorAdapter):
                query_embedding = self.vector_db.generate_query_embedding(query)
            else:
                # For other adapters, we'd need an embedding service
                query_embedding = [0.0] * 128  # Placeholder

            # Search vector store
            results = await self.vector_db.search(
                query_embedding=query_embedding,
                top_k=self._settings.rag_top_k,
                filter_metadata={"tenant_id": tenant_id},
            )

            # Filter by similarity threshold
            threshold = self._settings.rag_similarity_threshold
            filtered = [r for r in results if r.score >= threshold]

            doc_contents = [r.content for r in filtered]
            doc_ids = [r.id for r in filtered]

            logger.debug(
                "retrieved_context",
                query_length=len(query),
                total_results=len(results),
                filtered_results=len(filtered),
            )

            return doc_contents, doc_ids

        except Exception as e:
            logger.error("retrieval_error", error=str(e))
            return [], []

    def _build_prompt(
        self,
        system_prompt: str,
        few_shots: list[dict[str, str]],
        retrieved_docs: list[str],
        history: list[dict[str, str]],
        user_message: str,
    ) -> list[LLMMessage]:
        """
        Build the full prompt for LLM.

        Structure:
        1. System message with instructions
        2. Retrieved context
        3. Few-shot examples
        4. Conversation history
        5. Current user message
        """
        messages: list[LLMMessage] = []

        # System message
        system_content = system_prompt

        # Add retrieved context to system message
        if retrieved_docs:
            context_text = "\n\n".join(
                f"Context {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)
            )
            system_content += f"\n\n--- Relevant Information ---\n{context_text}\n--- End of Information ---"

        messages.append(LLMMessage(role="system", content=system_content))

        # Few-shot examples
        for example in few_shots:
            if "user" in example:
                messages.append(LLMMessage(role="user", content=example["user"]))
            if "assistant" in example:
                messages.append(LLMMessage(role="assistant", content=example["assistant"]))

        # Conversation history
        for turn in history:
            messages.append(LLMMessage(role=turn["role"], content=turn["content"]))

        # Current user message
        messages.append(LLMMessage(role="user", content=user_message))

        return messages

    def _parse_response(self, response_text: str) -> tuple[str, ActionObject | None]:
        """
        Parse LLM response and extract action if present.

        Action format: [ACTION:action_type:{"param": "value"}]
        """
        action_object: ActionObject | None = None

        # Extract action if present
        action_pattern = r"\[ACTION:(\w+):(\{[^}]+\})\]"
        match = re.search(action_pattern, response_text)

        if match:
            action_type = match.group(1)
            try:
                action_params = json.loads(match.group(2).replace("'", '"'))
            except json.JSONDecodeError:
                action_params = {}

            action_object = ActionObject(
                action_type=action_type,
                parameters=action_params,
                confidence=0.85,
            )

            # Remove action from speak text
            speak_text = re.sub(action_pattern, "", response_text).strip()
        else:
            speak_text = response_text.strip()

        return speak_text, action_object

    def _calculate_confidence(
        self,
        llm_response: str,
        retrieved_count: int,
        action_extracted: bool,
    ) -> float:
        """
        Calculate overall confidence score.

        Factors:
        - Response length (longer = more detailed = higher confidence)
        - Retrieved documents used
        - Action extraction success
        """
        confidence = 0.7  # Base confidence

        # Boost for longer responses
        if len(llm_response) > 100:
            confidence += 0.1

        # Boost for context usage
        if retrieved_count > 0:
            confidence += min(0.1, retrieved_count * 0.02)

        # Boost for action extraction
        if action_extracted:
            confidence += 0.05

        return min(1.0, confidence)
