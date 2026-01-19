"""
Builder Engine Python SDK - Knowledge Base Resource

This module provides methods for managing knowledge bases and documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List, BinaryIO, Union

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import KnowledgeBase, Document
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class KnowledgeBaseResource(BaseResource):
    """
    Resource for managing knowledge bases.

    Knowledge bases store documents that agents can reference during
    conversations using RAG (Retrieval Augmented Generation). This
    enables agents to answer questions based on your custom content.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> # Create a knowledge base
        >>> kb = client.knowledge_base.create(
        ...     name="Product Documentation",
        ...     description="All product docs and FAQs"
        ... )
        >>> # Add documents
        >>> doc = client.knowledge_base.add_document(
        ...     knowledge_base_id=kb.id,
        ...     name="Getting Started",
        ...     content="Welcome to our product..."
        ... )
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
    ) -> PaginatedResponse[KnowledgeBase]:
        """
        List all knowledge bases.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            search: Search by name or description

        Returns:
            PaginatedResponse containing KnowledgeBase objects
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            search=search,
        )
        response = self._get(Endpoints.KNOWLEDGE_BASES, params=params)
        return self._parse_paginated_response(response, KnowledgeBase)

    def get(self, knowledge_base_id: str) -> KnowledgeBase:
        """
        Get a knowledge base by ID.

        Args:
            knowledge_base_id: The knowledge base's unique identifier

        Returns:
            KnowledgeBase object
        """
        path = Endpoints.KNOWLEDGE_BASE.format(knowledge_base_id=knowledge_base_id)
        response = self._get(path)
        return KnowledgeBase.from_dict(response)

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        """
        Create a new knowledge base.

        Args:
            name: Name of the knowledge base
            description: Description of the content
            embedding_model: Model for text embeddings
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            metadata: Custom metadata

        Returns:
            Created KnowledgeBase object

        Example:
            >>> kb = client.knowledge_base.create(
            ...     name="Customer Support KB",
            ...     description="FAQs and troubleshooting guides",
            ...     chunk_size=800,
            ...     chunk_overlap=100
            ... )
        """
        data: Dict[str, Any] = {
            "name": name,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

        if description:
            data["description"] = description
        if metadata:
            data["metadata"] = metadata

        response = self._post(Endpoints.KNOWLEDGE_BASES, json=data)
        return KnowledgeBase.from_dict(response)

    def update(
        self,
        knowledge_base_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        """
        Update a knowledge base.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            name: New name
            description: New description
            metadata: New metadata

        Returns:
            Updated KnowledgeBase object
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if metadata is not None:
            data["metadata"] = metadata

        path = Endpoints.KNOWLEDGE_BASE.format(knowledge_base_id=knowledge_base_id)
        response = self._patch(path, json=data)
        return KnowledgeBase.from_dict(response)

    def delete(self, knowledge_base_id: str) -> None:
        """
        Delete a knowledge base and all its documents.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
        """
        path = Endpoints.KNOWLEDGE_BASE.format(knowledge_base_id=knowledge_base_id)
        self._delete(path)

    # Document methods

    def list_documents(
        self,
        knowledge_base_id: str,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
    ) -> PaginatedResponse[Document]:
        """
        List documents in a knowledge base.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            page: Page number
            page_size: Items per page
            status: Filter by vector status (pending, processing, ready, failed)

        Returns:
            PaginatedResponse containing Document objects
        """
        path = Endpoints.KNOWLEDGE_BASE_DOCUMENTS.format(knowledge_base_id=knowledge_base_id)
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            status=status,
        )
        response = self._get(path, params=params)
        return self._parse_paginated_response(response, Document)

    def get_document(self, knowledge_base_id: str, document_id: str) -> Document:
        """
        Get a document by ID.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            document_id: The document's unique identifier

        Returns:
            Document object
        """
        path = Endpoints.KNOWLEDGE_BASE_DOCUMENT.format(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id
        )
        response = self._get(path)
        return Document.from_dict(response)

    def add_document(
        self,
        knowledge_base_id: str,
        name: str,
        content: Optional[str] = None,
        file: Optional[BinaryIO] = None,
        file_url: Optional[str] = None,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Add a document to a knowledge base.

        You can provide content directly, upload a file, or provide a URL.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            name: Name of the document
            content: Text content (for direct text input)
            file: File object to upload
            file_url: URL to fetch content from
            content_type: MIME type of the content
            metadata: Custom metadata

        Returns:
            Created Document object

        Example:
            >>> # Add text content
            >>> doc = client.knowledge_base.add_document(
            ...     knowledge_base_id="kb_abc123",
            ...     name="FAQ",
            ...     content="Q: What is Builder Engine?\\nA: It's an AI voice platform."
            ... )
            >>> # Upload a file
            >>> with open("manual.pdf", "rb") as f:
            ...     doc = client.knowledge_base.add_document(
            ...         knowledge_base_id="kb_abc123",
            ...         name="User Manual",
            ...         file=f
            ...     )
        """
        path = Endpoints.KNOWLEDGE_BASE_DOCUMENTS.format(knowledge_base_id=knowledge_base_id)

        if file:
            # File upload
            files = {"file": file}
            data = {"name": name}
            if metadata:
                data["metadata"] = metadata
            response = self._post(path, data=data, files=files)
        else:
            # JSON request
            data: Dict[str, Any] = {
                "name": name,
                "content_type": content_type,
            }
            if content:
                data["content"] = content
            if file_url:
                data["file_url"] = file_url
            if metadata:
                data["metadata"] = metadata
            response = self._post(path, json=data)

        return Document.from_dict(response)

    def update_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Update a document.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            document_id: The document's unique identifier
            name: New name
            content: New content (triggers re-indexing)
            metadata: New metadata

        Returns:
            Updated Document object
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if content is not None:
            data["content"] = content
        if metadata is not None:
            data["metadata"] = metadata

        path = Endpoints.KNOWLEDGE_BASE_DOCUMENT.format(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id
        )
        response = self._patch(path, json=data)
        return Document.from_dict(response)

    def delete_document(self, knowledge_base_id: str, document_id: str) -> None:
        """
        Delete a document.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            document_id: The document's unique identifier
        """
        path = Endpoints.KNOWLEDGE_BASE_DOCUMENT.format(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id
        )
        self._delete(path)

    def query(
        self,
        knowledge_base_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query a knowledge base for relevant content.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            filter_metadata: Filter by document metadata

        Returns:
            List of matching chunks with scores

        Example:
            >>> results = client.knowledge_base.query(
            ...     knowledge_base_id="kb_abc123",
            ...     query="How do I reset my password?",
            ...     top_k=3
            ... )
            >>> for result in results:
            ...     print(f"Score: {result['score']:.2f}")
            ...     print(f"Content: {result['content'][:200]}...")
        """
        path = Endpoints.KNOWLEDGE_BASE_QUERY.format(knowledge_base_id=knowledge_base_id)
        data = {
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
        }
        if filter_metadata:
            data["filter_metadata"] = filter_metadata

        response = self._post(path, json=data)
        return response.get("results", [])

    def sync(self, knowledge_base_id: str) -> Dict[str, Any]:
        """
        Trigger a sync/re-index of all documents.

        Args:
            knowledge_base_id: The knowledge base's unique identifier

        Returns:
            Sync status information
        """
        path = Endpoints.KNOWLEDGE_BASE_SYNC.format(knowledge_base_id=knowledge_base_id)
        return self._post(path)

    def bulk_add_documents(
        self,
        knowledge_base_id: str,
        documents: List[Dict[str, Any]],
    ) -> List[Document]:
        """
        Add multiple documents at once.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            documents: List of document data (name, content, metadata)

        Returns:
            List of created Document objects
        """
        path = f"{Endpoints.KNOWLEDGE_BASE_DOCUMENTS.format(knowledge_base_id=knowledge_base_id)}/bulk"
        response = self._post(path, json={"documents": documents})
        return [Document.from_dict(d) for d in response.get("documents", [])]

    def import_from_url(
        self,
        knowledge_base_id: str,
        urls: List[str],
        crawl_subpages: bool = False,
        max_pages: int = 100,
    ) -> Dict[str, Any]:
        """
        Import documents from URLs.

        Args:
            knowledge_base_id: The knowledge base's unique identifier
            urls: List of URLs to import
            crawl_subpages: Whether to crawl linked pages
            max_pages: Maximum pages to crawl per URL

        Returns:
            Import job status
        """
        path = f"{Endpoints.KNOWLEDGE_BASE.format(knowledge_base_id=knowledge_base_id)}/import"
        response = self._post(path, json={
            "urls": urls,
            "crawl_subpages": crawl_subpages,
            "max_pages": max_pages,
        })
        return response
