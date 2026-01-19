"""
Builder Engine Python SDK - Workflows Resource

This module provides methods for managing workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import Workflow, WorkflowAction, WorkflowTrigger
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class WorkflowsResource(BaseResource):
    """
    Resource for managing workflows.

    Workflows automate actions based on call events. You can create
    workflows to send follow-up emails, update CRM records, trigger
    webhooks, and more.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> workflow = client.workflows.create(
        ...     name="Post-Call Follow-up",
        ...     trigger=WorkflowTrigger.CALL_ENDED,
        ...     actions=[
        ...         {"type": "send_email", "config": {"template": "follow_up"}}
        ...     ]
        ... )
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        enabled: Optional[bool] = None,
        trigger: Optional[WorkflowTrigger] = None,
        agent_id: Optional[str] = None,
    ) -> PaginatedResponse[Workflow]:
        """
        List all workflows.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            enabled: Filter by enabled status
            trigger: Filter by trigger type
            agent_id: Filter by associated agent

        Returns:
            PaginatedResponse containing Workflow objects
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            enabled=enabled,
            trigger=trigger.value if trigger else None,
            agent_id=agent_id,
        )
        response = self._get(Endpoints.WORKFLOWS, params=params)
        return self._parse_paginated_response(response, Workflow)

    def get(self, workflow_id: str) -> Workflow:
        """
        Get a workflow by ID.

        Args:
            workflow_id: The workflow's unique identifier

        Returns:
            Workflow object
        """
        path = Endpoints.WORKFLOW.format(workflow_id=workflow_id)
        response = self._get(path)
        return Workflow.from_dict(response)

    def create(
        self,
        name: str,
        trigger: WorkflowTrigger,
        actions: List[Dict[str, Any]],
        description: Optional[str] = None,
        trigger_config: Optional[Dict[str, Any]] = None,
        agent_ids: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> Workflow:
        """
        Create a new workflow.

        Args:
            name: Name of the workflow
            trigger: Event that triggers the workflow
            actions: List of actions to execute
            description: Description of the workflow
            trigger_config: Configuration for the trigger
            agent_ids: Agents this workflow applies to (empty = all)
            enabled: Whether the workflow is enabled

        Returns:
            Created Workflow object

        Example:
            >>> workflow = client.workflows.create(
            ...     name="Send SMS After Call",
            ...     trigger=WorkflowTrigger.CALL_ENDED,
            ...     trigger_config={"min_duration_seconds": 30},
            ...     actions=[
            ...         {
            ...             "type": "send_sms",
            ...             "config": {
            ...                 "to": "{{call.to_number}}",
            ...                 "message": "Thanks for calling! Reference: {{call.id}}"
            ...             }
            ...         },
            ...         {
            ...             "type": "webhook",
            ...             "config": {
            ...                 "url": "https://api.example.com/call-completed",
            ...                 "method": "POST"
            ...             }
            ...         }
            ...     ],
            ...     agent_ids=["agent_abc123"]
            ... )
        """
        data: Dict[str, Any] = {
            "name": name,
            "trigger": trigger.value,
            "actions": actions,
            "enabled": enabled,
        }

        if description:
            data["description"] = description
        if trigger_config:
            data["trigger_config"] = trigger_config
        if agent_ids:
            data["agent_ids"] = agent_ids

        response = self._post(Endpoints.WORKFLOWS, json=data)
        return Workflow.from_dict(response)

    def update(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        trigger: Optional[WorkflowTrigger] = None,
        trigger_config: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        agent_ids: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
    ) -> Workflow:
        """
        Update a workflow.

        Args:
            workflow_id: The workflow's unique identifier
            name: New name
            description: New description
            trigger: New trigger type
            trigger_config: New trigger configuration
            actions: New actions list
            agent_ids: New agent associations
            enabled: New enabled status

        Returns:
            Updated Workflow object
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if trigger is not None:
            data["trigger"] = trigger.value
        if trigger_config is not None:
            data["trigger_config"] = trigger_config
        if actions is not None:
            data["actions"] = actions
        if agent_ids is not None:
            data["agent_ids"] = agent_ids
        if enabled is not None:
            data["enabled"] = enabled

        path = Endpoints.WORKFLOW.format(workflow_id=workflow_id)
        response = self._patch(path, json=data)
        return Workflow.from_dict(response)

    def delete(self, workflow_id: str) -> None:
        """
        Delete a workflow.

        Args:
            workflow_id: The workflow's unique identifier
        """
        path = Endpoints.WORKFLOW.format(workflow_id=workflow_id)
        self._delete(path)

    def enable(self, workflow_id: str) -> Workflow:
        """
        Enable a workflow.

        Args:
            workflow_id: The workflow's unique identifier

        Returns:
            Updated Workflow object
        """
        path = Endpoints.WORKFLOW_ENABLE.format(workflow_id=workflow_id)
        response = self._post(path)
        return Workflow.from_dict(response)

    def disable(self, workflow_id: str) -> Workflow:
        """
        Disable a workflow.

        Args:
            workflow_id: The workflow's unique identifier

        Returns:
            Updated Workflow object
        """
        path = Endpoints.WORKFLOW_DISABLE.format(workflow_id=workflow_id)
        response = self._post(path)
        return Workflow.from_dict(response)

    def execute(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Manually execute a workflow.

        Args:
            workflow_id: The workflow's unique identifier
            context: Context variables for the execution

        Returns:
            Execution result

        Example:
            >>> result = client.workflows.execute(
            ...     workflow_id="workflow_abc123",
            ...     context={
            ...         "call_id": "call_xyz789",
            ...         "customer_email": "john@example.com"
            ...     }
            ... )
        """
        path = Endpoints.WORKFLOW_EXECUTE.format(workflow_id=workflow_id)
        data = {}
        if context:
            data["context"] = context
        return self._post(path, json=data)

    def get_executions(
        self,
        workflow_id: str,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get execution history for a workflow.

        Args:
            workflow_id: The workflow's unique identifier
            page: Page number
            page_size: Items per page
            status: Filter by status (success, failed, running)
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            Paginated list of executions
        """
        path = Endpoints.WORKFLOW_EXECUTIONS.format(workflow_id=workflow_id)
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )
        return self._get(path, params=params)

    def duplicate(self, workflow_id: str, name: Optional[str] = None) -> Workflow:
        """
        Duplicate a workflow.

        Args:
            workflow_id: The workflow's unique identifier
            name: Name for the new workflow

        Returns:
            New Workflow object
        """
        path = f"{Endpoints.WORKFLOW.format(workflow_id=workflow_id)}/duplicate"
        data = {}
        if name:
            data["name"] = name
        response = self._post(path, json=data)
        return Workflow.from_dict(response)

    @staticmethod
    def get_action_types() -> List[Dict[str, Any]]:
        """
        Get all available workflow action types.

        Returns:
            List of action types with schemas
        """
        return [
            {
                "type": "send_sms",
                "name": "Send SMS",
                "description": "Send an SMS message",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient phone number"},
                        "message": {"type": "string", "description": "Message content"},
                    },
                    "required": ["to", "message"],
                },
            },
            {
                "type": "send_email",
                "name": "Send Email",
                "description": "Send an email",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"},
                        "template": {"type": "string", "description": "Email template ID"},
                    },
                    "required": ["to"],
                },
            },
            {
                "type": "webhook",
                "name": "HTTP Webhook",
                "description": "Send an HTTP request",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Webhook URL"},
                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH"]},
                        "headers": {"type": "object"},
                        "body": {"type": "object"},
                    },
                    "required": ["url"],
                },
            },
            {
                "type": "transfer_call",
                "name": "Transfer Call",
                "description": "Transfer to another number",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Transfer destination"},
                        "announce": {"type": "string", "description": "Announcement message"},
                    },
                    "required": ["to"],
                },
            },
            {
                "type": "end_call",
                "name": "End Call",
                "description": "End the current call",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Goodbye message"},
                    },
                },
            },
            {
                "type": "update_crm",
                "name": "Update CRM",
                "description": "Update CRM record",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "crm": {"type": "string", "enum": ["salesforce", "hubspot", "pipedrive"]},
                        "object_type": {"type": "string"},
                        "fields": {"type": "object"},
                    },
                    "required": ["crm", "object_type"],
                },
            },
            {
                "type": "slack_message",
                "name": "Send Slack Message",
                "description": "Send a Slack notification",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "description": "Slack channel"},
                        "message": {"type": "string", "description": "Message content"},
                    },
                    "required": ["channel", "message"],
                },
            },
            {
                "type": "delay",
                "name": "Delay",
                "description": "Wait before next action",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "integer", "description": "Delay in seconds"},
                    },
                    "required": ["seconds"],
                },
            },
            {
                "type": "condition",
                "name": "Condition",
                "description": "Conditional branching",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "condition": {"type": "string", "description": "Condition expression"},
                        "then_actions": {"type": "array"},
                        "else_actions": {"type": "array"},
                    },
                    "required": ["condition"],
                },
            },
        ]

    @staticmethod
    def get_trigger_types() -> List[Dict[str, Any]]:
        """
        Get all available workflow trigger types.

        Returns:
            List of trigger types with descriptions
        """
        return [
            {"trigger": t.value, "description": t.name.replace("_", " ").title()}
            for t in WorkflowTrigger
        ]
