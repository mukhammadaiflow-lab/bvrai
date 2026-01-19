"""Knowledge API for Builder Engine."""

from typing import Optional, Dict, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from bvrai_sdk.client import BvraiClient


@dataclass
class KnowledgeBase:
    """A knowledge base for an agent."""
    id: str
    agent_id: str
    name: str
    description: Optional[str] = None
    source_type: str = "text"
    chunk_count: int = 0
    status: str = "pending"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBase":
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            name=data["name"],
            description=data.get("description"),
            source_type=data.get("source_type", "text"),
            chunk_count=data.get("chunk_count", 0),
            status=data.get("status", "pending"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class KnowledgeAPI:
    """
    Knowledge API client.

    Manage knowledge bases for agent RAG.
    """

    def __init__(self, client: "BvraiClient"):
        self._client = client

    async def create(
        self,
        agent_id: str,
        name: str,
        description: Optional[str] = None,
    ) -> KnowledgeBase:
        """Create a new knowledge base."""
        data = {
            "agent_id": agent_id,
            "name": name,
            "description": description,
        }
        response = await self._client.post("/v1/knowledge", data=data)
        return KnowledgeBase.from_dict(response)

    async def get(self, kb_id: str) -> KnowledgeBase:
        """Get a knowledge base."""
        response = await self._client.get(f"/v1/knowledge/{kb_id}")
        return KnowledgeBase.from_dict(response)

    async def list(self, agent_id: Optional[str] = None) -> List[KnowledgeBase]:
        """List knowledge bases."""
        params = {}
        if agent_id:
            params["agent_id"] = agent_id
        response = await self._client.get("/v1/knowledge", params=params)
        return [KnowledgeBase.from_dict(kb) for kb in response.get("knowledge_bases", [])]

    async def delete(self, kb_id: str) -> bool:
        """Delete a knowledge base."""
        await self._client.delete(f"/v1/knowledge/{kb_id}")
        return True

    async def add_text(
        self,
        kb_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add text to knowledge base."""
        data = {"text": text, "metadata": metadata or {}}
        return await self._client.post(f"/v1/knowledge/{kb_id}/documents", data=data)

    async def add_url(
        self,
        kb_id: str,
        url: str,
        crawl_links: bool = False,
    ) -> Dict[str, Any]:
        """Add URL content to knowledge base."""
        data = {"url": url, "crawl_links": crawl_links}
        return await self._client.post(f"/v1/knowledge/{kb_id}/urls", data=data)

    async def add_file(
        self,
        kb_id: str,
        file_path: str,
    ) -> Dict[str, Any]:
        """Add file to knowledge base."""
        # File upload would use multipart form
        raise NotImplementedError("File upload requires multipart support")

    async def search(
        self,
        kb_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search knowledge base."""
        params = {"query": query, "limit": limit}
        response = await self._client.get(f"/v1/knowledge/{kb_id}/search", params=params)
        return response.get("results", [])

    async def sync(self, kb_id: str) -> Dict[str, Any]:
        """Trigger sync for knowledge base."""
        return await self._client.post(f"/v1/knowledge/{kb_id}/sync")
