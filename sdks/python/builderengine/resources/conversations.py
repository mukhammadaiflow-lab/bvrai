"""
Builder Engine Python SDK - Conversations Resource

This module provides methods for managing conversations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import Conversation, Message
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class ConversationsResource(BaseResource):
    """
    Resource for managing conversations.

    Conversations represent the dialogue between an agent and a user
    during a call. This resource provides access to conversation history,
    messages, and analytics.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> conversation = client.conversations.get("conv_abc123")
        >>> for message in conversation.messages:
        ...     print(f"{message.role}: {message.content}")
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> PaginatedResponse[Conversation]:
        """
        List all conversations.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            call_id: Filter by call ID
            agent_id: Filter by agent ID
            start_date: Filter by start date
            end_date: Filter by end date
            sort_by: Field to sort by
            sort_order: Sort order (asc, desc)

        Returns:
            PaginatedResponse containing Conversation objects
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            call_id=call_id,
            agent_id=agent_id,
            start_date=start_date,
            end_date=end_date,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        response = self._get(Endpoints.CONVERSATIONS, params=params)
        return self._parse_paginated_response(response, Conversation)

    def get(self, conversation_id: str) -> Conversation:
        """
        Get a conversation by ID.

        Args:
            conversation_id: The conversation's unique identifier

        Returns:
            Conversation object with all messages
        """
        path = Endpoints.CONVERSATION.format(conversation_id=conversation_id)
        response = self._get(path)
        return Conversation.from_dict(response)

    def get_messages(
        self,
        conversation_id: str,
        page: int = 1,
        page_size: int = 100,
        role: Optional[str] = None,
    ) -> PaginatedResponse[Message]:
        """
        Get messages for a conversation.

        Args:
            conversation_id: The conversation's unique identifier
            page: Page number
            page_size: Items per page
            role: Filter by message role (user, assistant, system, function)

        Returns:
            PaginatedResponse containing Message objects
        """
        path = Endpoints.CONVERSATION_MESSAGES.format(conversation_id=conversation_id)
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            role=role,
        )
        response = self._get(path, params=params)
        return self._parse_paginated_response(response, Message)

    def get_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get an AI-generated summary of a conversation.

        Args:
            conversation_id: The conversation's unique identifier

        Returns:
            Summary with key points, action items, and sentiment
        """
        path = f"{Endpoints.CONVERSATION.format(conversation_id=conversation_id)}/summary"
        return self._get(path)

    def get_topics(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Extract topics discussed in a conversation.

        Args:
            conversation_id: The conversation's unique identifier

        Returns:
            List of topics with relevance scores
        """
        path = f"{Endpoints.CONVERSATION.format(conversation_id=conversation_id)}/topics"
        response = self._get(path)
        return response.get("topics", [])

    def get_sentiment(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get sentiment analysis for a conversation.

        Args:
            conversation_id: The conversation's unique identifier

        Returns:
            Sentiment data with overall score and timeline
        """
        path = f"{Endpoints.CONVERSATION.format(conversation_id=conversation_id)}/sentiment"
        return self._get(path)

    def get_intents(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get detected intents from a conversation.

        Args:
            conversation_id: The conversation's unique identifier

        Returns:
            List of detected intents with confidence scores
        """
        path = f"{Endpoints.CONVERSATION.format(conversation_id=conversation_id)}/intents"
        response = self._get(path)
        return response.get("intents", [])

    def get_entities(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from a conversation.

        Args:
            conversation_id: The conversation's unique identifier

        Returns:
            List of entities (names, dates, numbers, etc.)
        """
        path = f"{Endpoints.CONVERSATION.format(conversation_id=conversation_id)}/entities"
        response = self._get(path)
        return response.get("entities", [])

    def export(
        self,
        conversation_id: str,
        format: str = "json",
        include_audio: bool = False,
    ) -> Dict[str, Any]:
        """
        Export a conversation.

        Args:
            conversation_id: The conversation's unique identifier
            format: Export format (json, csv, txt)
            include_audio: Include audio URLs in export

        Returns:
            Export data or download URL
        """
        path = f"{Endpoints.CONVERSATION.format(conversation_id=conversation_id)}/export"
        params = {
            "format": format,
            "include_audio": include_audio,
        }
        return self._get(path, params=params)

    def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedResponse[Conversation]:
        """
        Search conversations by content.

        Args:
            query: Search query
            agent_id: Filter by agent ID
            start_date: Filter by start date
            end_date: Filter by end date
            page: Page number
            page_size: Items per page

        Returns:
            PaginatedResponse containing matching Conversation objects
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            q=query,
            agent_id=agent_id,
            start_date=start_date,
            end_date=end_date,
        )
        response = self._get(f"{Endpoints.CONVERSATIONS}/search", params=params)
        return self._parse_paginated_response(response, Conversation)
