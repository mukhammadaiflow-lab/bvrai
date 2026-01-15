"""
Flow Builder Service.

Visual flow builder backend for creating AI voice agent workflows.
This service provides:

1. Node Types:
   - Trigger nodes (incoming call, schedule, webhook)
   - Action nodes (speak, transfer, hangup, API call)
   - Logic nodes (condition, switch, loop)
   - AI nodes (intent detection, entity extraction, sentiment)
   - Integration nodes (CRM, calendar, database)

2. Canvas Management:
   - Flow creation, editing, versioning
   - Real-time collaboration support
   - Template library
   - Import/export

3. Flow Execution:
   - Flow validation
   - Dry-run testing
   - Debug mode with step-through
   - Variable substitution

4. Storage:
   - Persistent flow storage
   - Version history
   - Sharing and permissions

API:
   - POST /flows - Create new flow
   - GET /flows - List flows
   - GET /flows/{id} - Get flow
   - PUT /flows/{id} - Update flow
   - DELETE /flows/{id} - Delete flow
   - POST /flows/{id}/validate - Validate flow
   - POST /flows/{id}/execute - Execute flow (dry-run)
   - GET /nodes - List available node types
   - GET /templates - List flow templates
"""

__version__ = "1.0.0"
