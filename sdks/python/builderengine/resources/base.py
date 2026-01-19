"""
Builder Engine Python SDK - Base Resource

This module contains the base class for all API resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List, TypeVar, Generic
from dataclasses import dataclass

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


T = TypeVar("T")


@dataclass
class PaginatedResponse(Generic[T]):
    """
    Paginated response container.

    Attributes:
        items: List of items in the current page
        total: Total number of items across all pages
        page: Current page number (1-indexed)
        page_size: Number of items per page
        has_more: Whether there are more pages available
    """
    items: List[T]
    total: int
    page: int
    page_size: int
    has_more: bool

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class BaseResource:
    """
    Base class for all API resources.

    Provides common functionality for making API requests
    and handling responses.
    """

    def __init__(self, client: "BuilderEngine") -> None:
        """
        Initialize the resource.

        Args:
            client: The BuilderEngine client instance
        """
        self._client = client

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a GET request."""
        return self._client.request("GET", path, params=params)

    def _post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a POST request."""
        return self._client.request("POST", path, json=json, data=data, files=files)

    def _put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a PUT request."""
        return self._client.request("PUT", path, json=json)

    def _patch(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a PATCH request."""
        return self._client.request("PATCH", path, json=json)

    def _delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a DELETE request."""
        return self._client.request("DELETE", path, params=params)

    def _build_pagination_params(
        self,
        page: int = 1,
        page_size: int = 20,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build pagination query parameters."""
        params = {
            "page": page,
            "page_size": min(page_size, 100),  # Cap at 100
        }
        # Add any additional filters
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return params

    def _parse_paginated_response(
        self,
        response: Dict[str, Any],
        model_class: type,
    ) -> PaginatedResponse:
        """Parse a paginated response into a PaginatedResponse object."""
        items = [model_class.from_dict(item) for item in response.get("items", [])]
        return PaginatedResponse(
            items=items,
            total=response.get("total", len(items)),
            page=response.get("page", 1),
            page_size=response.get("page_size", len(items)),
            has_more=response.get("has_more", False),
        )
