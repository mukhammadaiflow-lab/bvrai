"""
Node Type Definitions.

Complete definitions for all available node types in the flow builder.
"""

from ..config import NodeType, NodeCategory, DataType
from ..models import NodeDefinition, PortDefinition, NodeProperty


# =============================================================================
# Trigger Nodes
# =============================================================================

TRIGGER_NODES = [
    NodeDefinition(
        type=NodeType.INCOMING_CALL,
        category=NodeCategory.TRIGGER,
        name="Incoming Call",
        description="Triggered when an incoming call is received",
        icon="phone-incoming",
        inputs=[],
        outputs=[
            PortDefinition(
                id="output",
                name="Call Connected",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="phone_numbers",
                data_type=DataType.ARRAY,
                required=False,
                description="Filter by specific phone numbers (optional)",
            ),
            NodeProperty(
                name="greeting",
                data_type=DataType.STRING,
                required=False,
                default_value="",
                description="Initial greeting message",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.OUTBOUND_CALL,
        category=NodeCategory.TRIGGER,
        name="Outbound Call",
        description="Initiates an outbound call",
        icon="phone-outgoing",
        inputs=[],
        outputs=[
            PortDefinition(
                id="connected",
                name="Call Connected",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="failed",
                name="Call Failed",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="to_number",
                data_type=DataType.STRING,
                required=True,
                description="Phone number to call",
            ),
            NodeProperty(
                name="from_number",
                data_type=DataType.STRING,
                required=True,
                description="Caller ID phone number",
            ),
            NodeProperty(
                name="timeout_seconds",
                data_type=DataType.NUMBER,
                required=False,
                default_value=30,
                description="Ring timeout in seconds",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.WEBHOOK,
        category=NodeCategory.TRIGGER,
        name="Webhook",
        description="Triggered by an external HTTP request",
        icon="webhook",
        inputs=[],
        outputs=[
            PortDefinition(
                id="output",
                name="Request Data",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="path",
                data_type=DataType.STRING,
                required=True,
                description="Webhook path (e.g., /hooks/my-trigger)",
            ),
            NodeProperty(
                name="method",
                data_type=DataType.STRING,
                required=False,
                default_value="POST",
                options=[
                    {"value": "GET", "label": "GET"},
                    {"value": "POST", "label": "POST"},
                ],
            ),
            NodeProperty(
                name="auth_required",
                data_type=DataType.BOOLEAN,
                required=False,
                default_value=True,
                description="Require authentication",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.SCHEDULE,
        category=NodeCategory.TRIGGER,
        name="Schedule",
        description="Triggered on a schedule (cron)",
        icon="calendar-clock",
        inputs=[],
        outputs=[
            PortDefinition(
                id="output",
                name="Trigger",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="cron",
                data_type=DataType.STRING,
                required=True,
                description="Cron expression (e.g., 0 9 * * MON-FRI)",
            ),
            NodeProperty(
                name="timezone",
                data_type=DataType.STRING,
                required=False,
                default_value="UTC",
                description="Timezone for schedule",
            ),
        ],
    ),
]


# =============================================================================
# Action Nodes
# =============================================================================

ACTION_NODES = [
    NodeDefinition(
        type=NodeType.SPEAK,
        category=NodeCategory.ACTION,
        name="Speak",
        description="Speak text to the caller using TTS",
        icon="message-square",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Continue",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="text",
                data_type=DataType.STRING,
                required=True,
                description="Text to speak (supports {{variables}})",
            ),
            NodeProperty(
                name="voice",
                data_type=DataType.STRING,
                required=False,
                default_value="default",
                description="Voice to use",
            ),
            NodeProperty(
                name="language",
                data_type=DataType.STRING,
                required=False,
                default_value="en-US",
                description="Language code",
            ),
            NodeProperty(
                name="speed",
                data_type=DataType.NUMBER,
                required=False,
                default_value=1.0,
                description="Speaking speed (0.5 to 2.0)",
            ),
            NodeProperty(
                name="interruptible",
                data_type=DataType.BOOLEAN,
                required=False,
                default_value=True,
                description="Allow caller to interrupt",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.GATHER_INPUT,
        category=NodeCategory.ACTION,
        name="Gather Input",
        description="Collect voice or DTMF input from caller",
        icon="mic",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="speech",
                name="Speech Input",
                data_type=DataType.STRING,
            ),
            PortDefinition(
                id="dtmf",
                name="DTMF Input",
                data_type=DataType.STRING,
            ),
            PortDefinition(
                id="timeout",
                name="Timeout",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="prompt",
                data_type=DataType.STRING,
                required=False,
                description="Prompt to speak before gathering",
            ),
            NodeProperty(
                name="input_type",
                data_type=DataType.STRING,
                required=False,
                default_value="speech",
                options=[
                    {"value": "speech", "label": "Speech"},
                    {"value": "dtmf", "label": "DTMF"},
                    {"value": "both", "label": "Both"},
                ],
            ),
            NodeProperty(
                name="timeout_seconds",
                data_type=DataType.NUMBER,
                required=False,
                default_value=5,
                description="Wait time for input",
            ),
            NodeProperty(
                name="end_on_silence",
                data_type=DataType.BOOLEAN,
                required=False,
                default_value=True,
                description="End on silence for speech",
            ),
            NodeProperty(
                name="num_digits",
                data_type=DataType.NUMBER,
                required=False,
                description="Expected DTMF digits (for DTMF input)",
            ),
            NodeProperty(
                name="finish_on_key",
                data_type=DataType.STRING,
                required=False,
                default_value="#",
                description="Key to finish DTMF input",
            ),
        ],
        is_async=True,
    ),
    NodeDefinition(
        type=NodeType.TRANSFER,
        category=NodeCategory.ACTION,
        name="Transfer",
        description="Transfer call to another number or agent",
        icon="phone-forwarded",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="connected",
                name="Connected",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="failed",
                name="Failed",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="transfer_to",
                data_type=DataType.STRING,
                required=True,
                description="Number or SIP URI to transfer to",
            ),
            NodeProperty(
                name="transfer_type",
                data_type=DataType.STRING,
                required=False,
                default_value="cold",
                options=[
                    {"value": "cold", "label": "Cold Transfer"},
                    {"value": "warm", "label": "Warm Transfer"},
                    {"value": "blind", "label": "Blind Transfer"},
                ],
            ),
            NodeProperty(
                name="announce_message",
                data_type=DataType.STRING,
                required=False,
                description="Message before transfer",
            ),
            NodeProperty(
                name="timeout_seconds",
                data_type=DataType.NUMBER,
                required=False,
                default_value=30,
                description="Ring timeout",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.HANGUP,
        category=NodeCategory.ACTION,
        name="Hang Up",
        description="End the call",
        icon="phone-off",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[],
        properties=[
            NodeProperty(
                name="message",
                data_type=DataType.STRING,
                required=False,
                description="Optional goodbye message",
            ),
            NodeProperty(
                name="reason",
                data_type=DataType.STRING,
                required=False,
                default_value="completed",
                options=[
                    {"value": "completed", "label": "Completed"},
                    {"value": "busy", "label": "Busy"},
                    {"value": "no-answer", "label": "No Answer"},
                    {"value": "rejected", "label": "Rejected"},
                ],
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.PLAY_AUDIO,
        category=NodeCategory.ACTION,
        name="Play Audio",
        description="Play an audio file",
        icon="play-circle",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Continue",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="audio_url",
                data_type=DataType.STRING,
                required=True,
                description="URL of audio file to play",
            ),
            NodeProperty(
                name="loop",
                data_type=DataType.BOOLEAN,
                required=False,
                default_value=False,
                description="Loop the audio",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.RECORD,
        category=NodeCategory.ACTION,
        name="Record",
        description="Record the call or a message",
        icon="circle",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Recording",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="max_length_seconds",
                data_type=DataType.NUMBER,
                required=False,
                default_value=60,
                description="Maximum recording length",
            ),
            NodeProperty(
                name="beep",
                data_type=DataType.BOOLEAN,
                required=False,
                default_value=True,
                description="Play beep before recording",
            ),
            NodeProperty(
                name="transcribe",
                data_type=DataType.BOOLEAN,
                required=False,
                default_value=True,
                description="Transcribe the recording",
            ),
        ],
        is_async=True,
    ),
]


# =============================================================================
# Logic Nodes
# =============================================================================

LOGIC_NODES = [
    NodeDefinition(
        type=NodeType.CONDITION,
        category=NodeCategory.LOGIC,
        name="Condition",
        description="Branch based on a condition",
        icon="git-branch",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="true",
                name="True",
                data_type=DataType.ANY,
            ),
            PortDefinition(
                id="false",
                name="False",
                data_type=DataType.ANY,
            ),
        ],
        properties=[
            NodeProperty(
                name="condition",
                data_type=DataType.STRING,
                required=True,
                description="Condition expression (e.g., {{input}} > 5)",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.SWITCH,
        category=NodeCategory.LOGIC,
        name="Switch",
        description="Branch to multiple paths based on value",
        icon="list",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="case_0",
                name="Case 1",
                data_type=DataType.ANY,
            ),
            PortDefinition(
                id="case_1",
                name="Case 2",
                data_type=DataType.ANY,
            ),
            PortDefinition(
                id="default",
                name="Default",
                data_type=DataType.ANY,
            ),
        ],
        properties=[
            NodeProperty(
                name="expression",
                data_type=DataType.STRING,
                required=True,
                description="Value to switch on",
            ),
            NodeProperty(
                name="cases",
                data_type=DataType.ARRAY,
                required=True,
                description="Case values to match",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.LOOP,
        category=NodeCategory.LOGIC,
        name="Loop",
        description="Repeat a set of nodes",
        icon="repeat",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
            PortDefinition(
                id="loop_back",
                name="Loop Back",
                data_type=DataType.ANY,
                required=False,
            ),
        ],
        outputs=[
            PortDefinition(
                id="loop_body",
                name="Loop Body",
                data_type=DataType.ANY,
            ),
            PortDefinition(
                id="exit",
                name="Exit",
                data_type=DataType.ANY,
            ),
        ],
        properties=[
            NodeProperty(
                name="loop_type",
                data_type=DataType.STRING,
                required=False,
                default_value="count",
                options=[
                    {"value": "count", "label": "Count"},
                    {"value": "while", "label": "While"},
                    {"value": "for_each", "label": "For Each"},
                ],
            ),
            NodeProperty(
                name="count",
                data_type=DataType.NUMBER,
                required=False,
                default_value=3,
                description="Number of iterations (for count type)",
            ),
            NodeProperty(
                name="condition",
                data_type=DataType.STRING,
                required=False,
                description="Loop condition (for while type)",
            ),
            NodeProperty(
                name="items",
                data_type=DataType.STRING,
                required=False,
                description="Array to iterate (for for_each type)",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.WAIT,
        category=NodeCategory.LOGIC,
        name="Wait",
        description="Pause execution for a duration",
        icon="clock",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Continue",
                data_type=DataType.ANY,
            ),
        ],
        properties=[
            NodeProperty(
                name="duration_ms",
                data_type=DataType.NUMBER,
                required=True,
                default_value=1000,
                description="Wait duration in milliseconds",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.PARALLEL,
        category=NodeCategory.LOGIC,
        name="Parallel",
        description="Execute multiple paths in parallel",
        icon="git-merge",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="branch_1",
                name="Branch 1",
                data_type=DataType.ANY,
            ),
            PortDefinition(
                id="branch_2",
                name="Branch 2",
                data_type=DataType.ANY,
            ),
            PortDefinition(
                id="join",
                name="Join",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="num_branches",
                data_type=DataType.NUMBER,
                required=False,
                default_value=2,
                description="Number of parallel branches",
            ),
            NodeProperty(
                name="wait_for_all",
                data_type=DataType.BOOLEAN,
                required=False,
                default_value=True,
                description="Wait for all branches to complete",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.TRY_CATCH,
        category=NodeCategory.LOGIC,
        name="Try/Catch",
        description="Handle errors in a flow",
        icon="shield",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="try",
                name="Try",
                data_type=DataType.ANY,
            ),
            PortDefinition(
                id="catch",
                name="Catch",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="finally",
                name="Finally",
                data_type=DataType.ANY,
            ),
        ],
        properties=[
            NodeProperty(
                name="error_var",
                data_type=DataType.STRING,
                required=False,
                default_value="error",
                description="Variable name for error",
            ),
        ],
    ),
]


# =============================================================================
# AI Nodes
# =============================================================================

AI_NODES = [
    NodeDefinition(
        type=NodeType.INTENT_DETECTION,
        category=NodeCategory.AI,
        name="Intent Detection",
        description="Detect user intent from speech/text",
        icon="brain",
        inputs=[
            PortDefinition(
                id="input",
                name="Input Text",
                data_type=DataType.STRING,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="detected",
                name="Intent Detected",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="unknown",
                name="Unknown Intent",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="intents",
                data_type=DataType.ARRAY,
                required=True,
                description="List of intents to detect",
            ),
            NodeProperty(
                name="confidence_threshold",
                data_type=DataType.NUMBER,
                required=False,
                default_value=0.7,
                description="Minimum confidence threshold",
            ),
            NodeProperty(
                name="model",
                data_type=DataType.STRING,
                required=False,
                default_value="default",
                description="Intent detection model",
            ),
        ],
        is_async=True,
    ),
    NodeDefinition(
        type=NodeType.ENTITY_EXTRACTION,
        category=NodeCategory.AI,
        name="Entity Extraction",
        description="Extract entities from speech/text",
        icon="tag",
        inputs=[
            PortDefinition(
                id="input",
                name="Input Text",
                data_type=DataType.STRING,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Entities",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="entity_types",
                data_type=DataType.ARRAY,
                required=False,
                description="Entity types to extract (empty = all)",
            ),
            NodeProperty(
                name="custom_entities",
                data_type=DataType.ARRAY,
                required=False,
                description="Custom entity definitions",
            ),
        ],
        is_async=True,
    ),
    NodeDefinition(
        type=NodeType.SENTIMENT_ANALYSIS,
        category=NodeCategory.AI,
        name="Sentiment Analysis",
        description="Analyze sentiment of speech/text",
        icon="smile",
        inputs=[
            PortDefinition(
                id="input",
                name="Input Text",
                data_type=DataType.STRING,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="positive",
                name="Positive",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="negative",
                name="Negative",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="neutral",
                name="Neutral",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="threshold",
                data_type=DataType.NUMBER,
                required=False,
                default_value=0.3,
                description="Threshold for positive/negative",
            ),
        ],
        is_async=True,
    ),
    NodeDefinition(
        type=NodeType.LLM_PROMPT,
        category=NodeCategory.AI,
        name="LLM Prompt",
        description="Send prompt to LLM and get response",
        icon="sparkles",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Response",
                data_type=DataType.STRING,
            ),
            PortDefinition(
                id="error",
                name="Error",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="prompt",
                data_type=DataType.STRING,
                required=True,
                description="Prompt template (supports {{variables}})",
            ),
            NodeProperty(
                name="system_prompt",
                data_type=DataType.STRING,
                required=False,
                description="System prompt for the LLM",
            ),
            NodeProperty(
                name="model",
                data_type=DataType.STRING,
                required=False,
                default_value="claude-3-5-sonnet",
                description="LLM model to use",
            ),
            NodeProperty(
                name="max_tokens",
                data_type=DataType.NUMBER,
                required=False,
                default_value=500,
                description="Maximum response tokens",
            ),
            NodeProperty(
                name="temperature",
                data_type=DataType.NUMBER,
                required=False,
                default_value=0.7,
                description="Temperature (0-1)",
            ),
        ],
        is_async=True,
    ),
    NodeDefinition(
        type=NodeType.KNOWLEDGE_QUERY,
        category=NodeCategory.AI,
        name="Knowledge Query",
        description="Query a knowledge base (RAG)",
        icon="database",
        inputs=[
            PortDefinition(
                id="input",
                name="Query",
                data_type=DataType.STRING,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="found",
                name="Results Found",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="not_found",
                name="Not Found",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="knowledge_base_id",
                data_type=DataType.STRING,
                required=True,
                description="Knowledge base to query",
            ),
            NodeProperty(
                name="num_results",
                data_type=DataType.NUMBER,
                required=False,
                default_value=3,
                description="Number of results to return",
            ),
            NodeProperty(
                name="min_score",
                data_type=DataType.NUMBER,
                required=False,
                default_value=0.5,
                description="Minimum relevance score",
            ),
        ],
        is_async=True,
    ),
]


# =============================================================================
# Integration Nodes
# =============================================================================

INTEGRATION_NODES = [
    NodeDefinition(
        type=NodeType.HTTP_REQUEST,
        category=NodeCategory.INTEGRATION,
        name="HTTP Request",
        description="Make an HTTP request to an API",
        icon="globe",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="success",
                name="Success",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="error",
                name="Error",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="url",
                data_type=DataType.STRING,
                required=True,
                description="Request URL",
            ),
            NodeProperty(
                name="method",
                data_type=DataType.STRING,
                required=False,
                default_value="GET",
                options=[
                    {"value": "GET", "label": "GET"},
                    {"value": "POST", "label": "POST"},
                    {"value": "PUT", "label": "PUT"},
                    {"value": "PATCH", "label": "PATCH"},
                    {"value": "DELETE", "label": "DELETE"},
                ],
            ),
            NodeProperty(
                name="headers",
                data_type=DataType.OBJECT,
                required=False,
                description="Request headers",
            ),
            NodeProperty(
                name="body",
                data_type=DataType.OBJECT,
                required=False,
                description="Request body (for POST/PUT/PATCH)",
            ),
            NodeProperty(
                name="timeout_ms",
                data_type=DataType.NUMBER,
                required=False,
                default_value=10000,
                description="Request timeout",
            ),
        ],
        is_async=True,
    ),
    NodeDefinition(
        type=NodeType.CRM_LOOKUP,
        category=NodeCategory.INTEGRATION,
        name="CRM Lookup",
        description="Look up a contact in CRM",
        icon="users",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="found",
                name="Contact Found",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="not_found",
                name="Not Found",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="crm_connection",
                data_type=DataType.STRING,
                required=True,
                description="CRM connection to use",
            ),
            NodeProperty(
                name="lookup_field",
                data_type=DataType.STRING,
                required=True,
                default_value="phone",
                description="Field to search by",
            ),
            NodeProperty(
                name="lookup_value",
                data_type=DataType.STRING,
                required=True,
                description="Value to search for",
            ),
        ],
        is_async=True,
    ),
    NodeDefinition(
        type=NodeType.CALENDAR_CHECK,
        category=NodeCategory.INTEGRATION,
        name="Calendar Check",
        description="Check calendar availability",
        icon="calendar",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="available",
                name="Available",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="busy",
                name="Busy",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="calendar_connection",
                data_type=DataType.STRING,
                required=True,
                description="Calendar connection to use",
            ),
            NodeProperty(
                name="date_time",
                data_type=DataType.STRING,
                required=True,
                description="Date/time to check",
            ),
            NodeProperty(
                name="duration_minutes",
                data_type=DataType.NUMBER,
                required=False,
                default_value=30,
                description="Duration of appointment",
            ),
        ],
        is_async=True,
    ),
    NodeDefinition(
        type=NodeType.DATABASE_QUERY,
        category=NodeCategory.INTEGRATION,
        name="Database Query",
        description="Query a database",
        icon="database",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="results",
                name="Results",
                data_type=DataType.ARRAY,
            ),
            PortDefinition(
                id="empty",
                name="No Results",
                data_type=DataType.OBJECT,
            ),
            PortDefinition(
                id="error",
                name="Error",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="connection",
                data_type=DataType.STRING,
                required=True,
                description="Database connection",
            ),
            NodeProperty(
                name="query",
                data_type=DataType.STRING,
                required=True,
                description="SQL query (use {{params}} for variables)",
            ),
            NodeProperty(
                name="params",
                data_type=DataType.OBJECT,
                required=False,
                description="Query parameters",
            ),
        ],
        is_async=True,
    ),
]


# =============================================================================
# Utility Nodes
# =============================================================================

UTILITY_NODES = [
    NodeDefinition(
        type=NodeType.SET_VARIABLE,
        category=NodeCategory.UTILITY,
        name="Set Variable",
        description="Set a variable value",
        icon="variable",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Continue",
                data_type=DataType.ANY,
            ),
        ],
        properties=[
            NodeProperty(
                name="variable_name",
                data_type=DataType.STRING,
                required=True,
                description="Variable name to set",
            ),
            NodeProperty(
                name="value",
                data_type=DataType.ANY,
                required=True,
                description="Value to set (supports expressions)",
            ),
            NodeProperty(
                name="scope",
                data_type=DataType.STRING,
                required=False,
                default_value="flow",
                options=[
                    {"value": "flow", "label": "Flow"},
                    {"value": "session", "label": "Session"},
                    {"value": "global", "label": "Global"},
                ],
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.LOG,
        category=NodeCategory.UTILITY,
        name="Log",
        description="Log a message for debugging",
        icon="file-text",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Continue",
                data_type=DataType.ANY,
            ),
        ],
        properties=[
            NodeProperty(
                name="message",
                data_type=DataType.STRING,
                required=True,
                description="Message to log",
            ),
            NodeProperty(
                name="level",
                data_type=DataType.STRING,
                required=False,
                default_value="info",
                options=[
                    {"value": "debug", "label": "Debug"},
                    {"value": "info", "label": "Info"},
                    {"value": "warn", "label": "Warning"},
                    {"value": "error", "label": "Error"},
                ],
            ),
            NodeProperty(
                name="include_data",
                data_type=DataType.BOOLEAN,
                required=False,
                default_value=True,
                description="Include input data in log",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.FUNCTION,
        category=NodeCategory.UTILITY,
        name="Function",
        description="Execute custom JavaScript code",
        icon="code",
        inputs=[
            PortDefinition(
                id="input",
                name="Input",
                data_type=DataType.ANY,
                required=True,
            ),
        ],
        outputs=[
            PortDefinition(
                id="output",
                name="Output",
                data_type=DataType.ANY,
            ),
            PortDefinition(
                id="error",
                name="Error",
                data_type=DataType.OBJECT,
            ),
        ],
        properties=[
            NodeProperty(
                name="code",
                data_type=DataType.STRING,
                required=True,
                description="JavaScript code to execute",
            ),
            NodeProperty(
                name="timeout_ms",
                data_type=DataType.NUMBER,
                required=False,
                default_value=5000,
                description="Execution timeout",
            ),
        ],
    ),
    NodeDefinition(
        type=NodeType.COMMENT,
        category=NodeCategory.UTILITY,
        name="Comment",
        description="Add a comment to the flow (no execution)",
        icon="message-circle",
        inputs=[],
        outputs=[],
        properties=[
            NodeProperty(
                name="text",
                data_type=DataType.STRING,
                required=True,
                description="Comment text",
            ),
            NodeProperty(
                name="color",
                data_type=DataType.STRING,
                required=False,
                default_value="#f0f0f0",
                description="Background color",
            ),
        ],
    ),
]


# =============================================================================
# All Nodes Combined
# =============================================================================

ALL_NODES = (
    TRIGGER_NODES
    + ACTION_NODES
    + LOGIC_NODES
    + AI_NODES
    + INTEGRATION_NODES
    + UTILITY_NODES
)
