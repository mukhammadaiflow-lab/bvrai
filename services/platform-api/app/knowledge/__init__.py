"""Knowledge module."""

from app.knowledge.service import KnowledgeService
from app.knowledge.routes import router

__all__ = ["KnowledgeService", "router"]
