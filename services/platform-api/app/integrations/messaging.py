"""Messaging platform integrations (Slack, Microsoft Teams)."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import aiohttp

from app.integrations.base import Integration, IntegrationConfig

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages."""
    TEXT = "text"
    RICH = "rich"
    CARD = "card"
    FILE = "file"
    REACTION = "reaction"


@dataclass
class Message:
    """A messaging platform message."""
    id: str
    channel_id: str
    content: str
    message_type: MessageType = MessageType.TEXT
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    thread_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "content": self.content,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp.isoformat(),
            "attachments": self.attachments,
            "metadata": self.metadata,
        }


@dataclass
class Channel:
    """A messaging channel."""
    id: str
    name: str
    channel_type: str  # "public", "private", "direct"
    members: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "channel_type": self.channel_type,
            "members": self.members,
            "topic": self.topic,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class User:
    """A messaging platform user."""
    id: str
    name: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    is_bot: bool = False
    timezone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "is_bot": self.is_bot,
            "timezone": self.timezone,
        }


class MessagingIntegration(Integration):
    """Base class for messaging integrations."""

    async def send_message(
        self,
        channel_id: str,
        content: str,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Send a message to a channel."""
        raise NotImplementedError

    async def send_rich_message(
        self,
        channel_id: str,
        blocks: List[Dict[str, Any]],
        text: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Message:
        """Send a rich formatted message."""
        raise NotImplementedError

    async def get_channels(self) -> List[Channel]:
        """Get list of channels."""
        raise NotImplementedError

    async def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get a specific channel."""
        raise NotImplementedError

    async def get_users(self) -> List[User]:
        """Get list of users."""
        raise NotImplementedError

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a specific user."""
        raise NotImplementedError

    async def add_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> bool:
        """Add a reaction to a message."""
        raise NotImplementedError

    async def upload_file(
        self,
        channel_id: str,
        file_content: bytes,
        filename: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a file to a channel."""
        raise NotImplementedError


class SlackIntegration(MessagingIntegration):
    """
    Slack integration.

    Usage:
        config = IntegrationConfig(
            credentials={"bot_token": "xoxb-..."}
        )
        slack = SlackIntegration("slack_1", config)
        await slack.connect()

        # Send message
        await slack.send_message("#general", "Hello from voice AI!")

        # Send rich message
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Call Summary*"}
            }
        ]
        await slack.send_rich_message("#calls", blocks)
    """

    name = "Slack"
    description = "Slack workspace integration"
    version = "1.0.0"

    BASE_URL = "https://slack.com/api"

    def __init__(self, integration_id: str, config: IntegrationConfig):
        super().__init__(integration_id, config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._bot_token: Optional[str] = None
        self._user_token: Optional[str] = None
        self._workspace_id: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to Slack."""
        try:
            self._bot_token = self.config.credentials.get("bot_token")
            self._user_token = self.config.credentials.get("user_token")

            if not self._bot_token:
                raise ValueError("Slack bot token required")

            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._bot_token}"}
            )

            # Test connection
            response = await self._api_call("auth.test")
            if response.get("ok"):
                self._workspace_id = response.get("team_id")
                self._connected = True
                logger.info(f"Connected to Slack workspace: {response.get('team')}")
                return True
            else:
                raise Exception(response.get("error", "Unknown error"))

        except Exception as e:
            logger.error(f"Slack connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Slack."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        return True

    async def _api_call(
        self,
        method: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make Slack API call."""
        if not self._session:
            raise RuntimeError("Not connected to Slack")

        url = f"{self.BASE_URL}/{method}"

        try:
            if files:
                # Multipart form data for file uploads
                form = aiohttp.FormData()
                for key, value in (data or {}).items():
                    form.add_field(key, str(value))
                for key, (filename, content, content_type) in files.items():
                    form.add_field(key, content, filename=filename, content_type=content_type)
                async with self._session.post(url, data=form) as response:
                    return await response.json()
            else:
                async with self._session.post(url, json=data or {}) as response:
                    return await response.json()

        except Exception as e:
            logger.error(f"Slack API error: {e}")
            return {"ok": False, "error": str(e)}

    async def send_message(
        self,
        channel_id: str,
        content: str,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Send a message to a Slack channel."""
        data = {
            "channel": channel_id,
            "text": content,
        }

        if thread_id:
            data["thread_ts"] = thread_id

        if attachments:
            data["attachments"] = attachments

        response = await self._api_call("chat.postMessage", data)

        if not response.get("ok"):
            raise Exception(f"Failed to send message: {response.get('error')}")

        return Message(
            id=response["ts"],
            channel_id=response["channel"],
            content=content,
            message_type=MessageType.TEXT,
            thread_id=thread_id,
            timestamp=datetime.utcnow(),
            attachments=attachments or [],
        )

    async def send_rich_message(
        self,
        channel_id: str,
        blocks: List[Dict[str, Any]],
        text: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Message:
        """Send a rich Block Kit message."""
        data = {
            "channel": channel_id,
            "blocks": blocks,
        }

        if text:
            data["text"] = text  # Fallback text

        if thread_id:
            data["thread_ts"] = thread_id

        response = await self._api_call("chat.postMessage", data)

        if not response.get("ok"):
            raise Exception(f"Failed to send rich message: {response.get('error')}")

        return Message(
            id=response["ts"],
            channel_id=response["channel"],
            content=text or "",
            message_type=MessageType.RICH,
            thread_id=thread_id,
            metadata={"blocks": blocks},
        )

    async def get_channels(self) -> List[Channel]:
        """Get list of channels."""
        channels = []
        cursor = None

        while True:
            data = {"limit": 200}
            if cursor:
                data["cursor"] = cursor

            response = await self._api_call("conversations.list", data)

            if not response.get("ok"):
                raise Exception(f"Failed to get channels: {response.get('error')}")

            for ch in response.get("channels", []):
                channel_type = "public"
                if ch.get("is_private"):
                    channel_type = "private"
                elif ch.get("is_im"):
                    channel_type = "direct"

                channels.append(Channel(
                    id=ch["id"],
                    name=ch.get("name", ""),
                    channel_type=channel_type,
                    topic=ch.get("topic", {}).get("value"),
                    created_at=datetime.fromtimestamp(ch.get("created", 0)),
                ))

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return channels

    async def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get a specific channel."""
        response = await self._api_call("conversations.info", {"channel": channel_id})

        if not response.get("ok"):
            return None

        ch = response.get("channel", {})
        channel_type = "public"
        if ch.get("is_private"):
            channel_type = "private"
        elif ch.get("is_im"):
            channel_type = "direct"

        return Channel(
            id=ch["id"],
            name=ch.get("name", ""),
            channel_type=channel_type,
            topic=ch.get("topic", {}).get("value"),
            created_at=datetime.fromtimestamp(ch.get("created", 0)),
        )

    async def get_users(self) -> List[User]:
        """Get list of users."""
        users = []
        cursor = None

        while True:
            data = {"limit": 200}
            if cursor:
                data["cursor"] = cursor

            response = await self._api_call("users.list", data)

            if not response.get("ok"):
                raise Exception(f"Failed to get users: {response.get('error')}")

            for member in response.get("members", []):
                profile = member.get("profile", {})
                users.append(User(
                    id=member["id"],
                    name=member.get("name", ""),
                    email=profile.get("email"),
                    display_name=profile.get("display_name"),
                    avatar_url=profile.get("image_72"),
                    is_bot=member.get("is_bot", False),
                    timezone=member.get("tz"),
                ))

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return users

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a specific user."""
        response = await self._api_call("users.info", {"user": user_id})

        if not response.get("ok"):
            return None

        member = response.get("user", {})
        profile = member.get("profile", {})

        return User(
            id=member["id"],
            name=member.get("name", ""),
            email=profile.get("email"),
            display_name=profile.get("display_name"),
            avatar_url=profile.get("image_72"),
            is_bot=member.get("is_bot", False),
            timezone=member.get("tz"),
        )

    async def add_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> bool:
        """Add a reaction to a message."""
        response = await self._api_call("reactions.add", {
            "channel": channel_id,
            "timestamp": message_id,
            "name": emoji,
        })
        return response.get("ok", False)

    async def upload_file(
        self,
        channel_id: str,
        file_content: bytes,
        filename: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a file to a channel."""
        data = {
            "channels": channel_id,
            "filename": filename,
        }
        if title:
            data["title"] = title

        files = {
            "file": (filename, file_content, "application/octet-stream")
        }

        response = await self._api_call("files.upload", data, files)

        if not response.get("ok"):
            raise Exception(f"Failed to upload file: {response.get('error')}")

        return response.get("file", {})

    async def send_call_notification(
        self,
        channel_id: str,
        call_data: Dict[str, Any],
    ) -> Message:
        """Send a call notification with rich formatting."""
        status = call_data.get("status", "unknown")
        caller = call_data.get("caller", "Unknown")
        duration = call_data.get("duration", 0)

        status_emoji = {
            "completed": ":white_check_mark:",
            "missed": ":x:",
            "voicemail": ":mailbox_with_mail:",
            "transferred": ":arrow_right:",
        }.get(status, ":phone:")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} Call {status.title()}",
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Caller:*\n{caller}"},
                    {"type": "mrkdwn", "text": f"*Duration:*\n{duration}s"},
                ]
            },
        ]

        if call_data.get("summary"):
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Summary:*\n{call_data['summary']}"}
            })

        if call_data.get("recording_url"):
            blocks.append({
                "type": "actions",
                "elements": [{
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Listen to Recording"},
                    "url": call_data["recording_url"],
                }]
            })

        return await self.send_rich_message(channel_id, blocks, f"Call {status} from {caller}")


class TeamsIntegration(MessagingIntegration):
    """
    Microsoft Teams integration.

    Usage:
        config = IntegrationConfig(
            credentials={
                "client_id": "...",
                "client_secret": "...",
                "tenant_id": "..."
            }
        )
        teams = TeamsIntegration("teams_1", config)
        await teams.connect()

        # Send message
        await teams.send_message("channel_id", "Hello from voice AI!")
    """

    name = "Microsoft Teams"
    description = "Microsoft Teams integration"
    version = "1.0.0"

    GRAPH_URL = "https://graph.microsoft.com/v1.0"
    AUTH_URL = "https://login.microsoftonline.com"

    def __init__(self, integration_id: str, config: IntegrationConfig):
        super().__init__(integration_id, config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._tenant_id: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to Microsoft Teams."""
        try:
            client_id = self.config.credentials.get("client_id")
            client_secret = self.config.credentials.get("client_secret")
            self._tenant_id = self.config.credentials.get("tenant_id")

            if not all([client_id, client_secret, self._tenant_id]):
                raise ValueError("Teams credentials incomplete")

            # Get access token
            await self._get_access_token()

            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._access_token}"}
            )

            # Test connection
            async with self._session.get(f"{self.GRAPH_URL}/me") as response:
                if response.status == 200:
                    user_data = await response.json()
                    self._connected = True
                    logger.info(f"Connected to Teams as: {user_data.get('displayName')}")
                    return True
                else:
                    raise Exception(f"Teams auth failed: {response.status}")

        except Exception as e:
            logger.error(f"Teams connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Teams."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        self._access_token = None
        return True

    async def _get_access_token(self) -> str:
        """Get or refresh access token."""
        if self._access_token and self._token_expiry:
            if datetime.utcnow() < self._token_expiry:
                return self._access_token

        client_id = self.config.credentials.get("client_id")
        client_secret = self.config.credentials.get("client_secret")

        token_url = f"{self.AUTH_URL}/{self._tenant_id}/oauth2/v2.0/token"

        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "https://graph.microsoft.com/.default",
            }

            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    raise Exception("Failed to get access token")

                token_data = await response.json()
                self._access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                self._token_expiry = datetime.utcnow() + timedelta(seconds=expires_in - 60)

        return self._access_token

    async def _ensure_token(self) -> None:
        """Ensure we have a valid token."""
        await self._get_access_token()
        if self._session:
            self._session._default_headers["Authorization"] = f"Bearer {self._access_token}"

    async def send_message(
        self,
        channel_id: str,
        content: str,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Send a message to a Teams channel."""
        await self._ensure_token()

        # Parse channel_id (format: team_id/channel_id)
        parts = channel_id.split("/")
        if len(parts) != 2:
            raise ValueError("Channel ID must be in format: team_id/channel_id")

        team_id, ch_id = parts

        url = f"{self.GRAPH_URL}/teams/{team_id}/channels/{ch_id}/messages"

        if thread_id:
            url = f"{self.GRAPH_URL}/teams/{team_id}/channels/{ch_id}/messages/{thread_id}/replies"

        body = {
            "body": {
                "contentType": "text",
                "content": content,
            }
        }

        if attachments:
            body["attachments"] = attachments

        async with self._session.post(url, json=body) as response:
            if response.status not in [200, 201]:
                error = await response.text()
                raise Exception(f"Failed to send message: {error}")

            data = await response.json()

            return Message(
                id=data["id"],
                channel_id=channel_id,
                content=content,
                message_type=MessageType.TEXT,
                thread_id=thread_id,
                timestamp=datetime.fromisoformat(data["createdDateTime"].replace("Z", "+00:00")),
            )

    async def send_rich_message(
        self,
        channel_id: str,
        blocks: List[Dict[str, Any]],
        text: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Message:
        """Send an Adaptive Card message."""
        await self._ensure_token()

        parts = channel_id.split("/")
        if len(parts) != 2:
            raise ValueError("Channel ID must be in format: team_id/channel_id")

        team_id, ch_id = parts

        url = f"{self.GRAPH_URL}/teams/{team_id}/channels/{ch_id}/messages"

        if thread_id:
            url = f"{self.GRAPH_URL}/teams/{team_id}/channels/{ch_id}/messages/{thread_id}/replies"

        # Adaptive Card format
        card = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": blocks,
        }

        body = {
            "body": {
                "contentType": "html",
                "content": text or "",
            },
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": card,
            }]
        }

        async with self._session.post(url, json=body) as response:
            if response.status not in [200, 201]:
                error = await response.text()
                raise Exception(f"Failed to send rich message: {error}")

            data = await response.json()

            return Message(
                id=data["id"],
                channel_id=channel_id,
                content=text or "",
                message_type=MessageType.CARD,
                thread_id=thread_id,
                metadata={"card": card},
            )

    async def get_channels(self) -> List[Channel]:
        """Get list of channels from all teams."""
        await self._ensure_token()

        channels = []

        # Get all teams
        async with self._session.get(f"{self.GRAPH_URL}/me/joinedTeams") as response:
            if response.status != 200:
                raise Exception("Failed to get teams")

            teams_data = await response.json()

            for team in teams_data.get("value", []):
                team_id = team["id"]

                # Get channels for this team
                async with self._session.get(
                    f"{self.GRAPH_URL}/teams/{team_id}/channels"
                ) as ch_response:
                    if ch_response.status == 200:
                        ch_data = await ch_response.json()

                        for ch in ch_data.get("value", []):
                            channel_type = "public"
                            if ch.get("membershipType") == "private":
                                channel_type = "private"

                            channels.append(Channel(
                                id=f"{team_id}/{ch['id']}",
                                name=f"{team['displayName']} / {ch['displayName']}",
                                channel_type=channel_type,
                                topic=ch.get("description"),
                            ))

        return channels

    async def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get a specific channel."""
        await self._ensure_token()

        parts = channel_id.split("/")
        if len(parts) != 2:
            return None

        team_id, ch_id = parts

        async with self._session.get(
            f"{self.GRAPH_URL}/teams/{team_id}/channels/{ch_id}"
        ) as response:
            if response.status != 200:
                return None

            ch = await response.json()

            # Get team name
            async with self._session.get(f"{self.GRAPH_URL}/teams/{team_id}") as team_response:
                team_name = "Unknown"
                if team_response.status == 200:
                    team_data = await team_response.json()
                    team_name = team_data.get("displayName", "Unknown")

            channel_type = "public"
            if ch.get("membershipType") == "private":
                channel_type = "private"

            return Channel(
                id=channel_id,
                name=f"{team_name} / {ch['displayName']}",
                channel_type=channel_type,
                topic=ch.get("description"),
            )

    async def get_users(self) -> List[User]:
        """Get list of users in organization."""
        await self._ensure_token()

        users = []
        url = f"{self.GRAPH_URL}/users"

        while url:
            async with self._session.get(url) as response:
                if response.status != 200:
                    raise Exception("Failed to get users")

                data = await response.json()

                for member in data.get("value", []):
                    users.append(User(
                        id=member["id"],
                        name=member.get("userPrincipalName", ""),
                        email=member.get("mail"),
                        display_name=member.get("displayName"),
                        is_bot=False,
                    ))

                url = data.get("@odata.nextLink")

        return users

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a specific user."""
        await self._ensure_token()

        async with self._session.get(f"{self.GRAPH_URL}/users/{user_id}") as response:
            if response.status != 200:
                return None

            member = await response.json()

            return User(
                id=member["id"],
                name=member.get("userPrincipalName", ""),
                email=member.get("mail"),
                display_name=member.get("displayName"),
                is_bot=False,
            )

    async def add_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> bool:
        """Add a reaction to a message (not fully supported in Graph API)."""
        # Teams Graph API has limited reaction support
        logger.warning("Teams reaction support is limited in Graph API")
        return False

    async def upload_file(
        self,
        channel_id: str,
        file_content: bytes,
        filename: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a file to a channel."""
        await self._ensure_token()

        parts = channel_id.split("/")
        if len(parts) != 2:
            raise ValueError("Channel ID must be in format: team_id/channel_id")

        team_id, ch_id = parts

        # Get the channel's files folder
        async with self._session.get(
            f"{self.GRAPH_URL}/teams/{team_id}/channels/{ch_id}/filesFolder"
        ) as response:
            if response.status != 200:
                raise Exception("Failed to get files folder")

            folder_data = await response.json()
            drive_id = folder_data.get("parentReference", {}).get("driveId")
            folder_id = folder_data.get("id")

        # Upload file
        upload_url = f"{self.GRAPH_URL}/drives/{drive_id}/items/{folder_id}:/{filename}:/content"

        async with self._session.put(
            upload_url,
            data=file_content,
            headers={"Content-Type": "application/octet-stream"}
        ) as response:
            if response.status not in [200, 201]:
                error = await response.text()
                raise Exception(f"Failed to upload file: {error}")

            return await response.json()

    async def send_call_notification(
        self,
        channel_id: str,
        call_data: Dict[str, Any],
    ) -> Message:
        """Send a call notification with Adaptive Card."""
        status = call_data.get("status", "unknown")
        caller = call_data.get("caller", "Unknown")
        duration = call_data.get("duration", 0)

        status_color = {
            "completed": "good",
            "missed": "attention",
            "voicemail": "warning",
            "transferred": "accent",
        }.get(status, "default")

        blocks = [
            {
                "type": "TextBlock",
                "size": "Large",
                "weight": "Bolder",
                "text": f"📞 Call {status.title()}",
                "color": status_color,
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Caller", "value": caller},
                    {"title": "Duration", "value": f"{duration}s"},
                ]
            }
        ]

        if call_data.get("summary"):
            blocks.append({
                "type": "TextBlock",
                "text": call_data["summary"],
                "wrap": True,
            })

        return await self.send_rich_message(
            channel_id,
            blocks,
            f"Call {status} from {caller}",
        )


# Import for timedelta
from datetime import timedelta


class WebhookIntegration(MessagingIntegration):
    """
    Generic webhook integration for custom messaging endpoints.

    Usage:
        config = IntegrationConfig(
            settings={"webhook_url": "https://example.com/webhook"}
        )
        webhook = WebhookIntegration("webhook_1", config)

        await webhook.send_message("channel", "Hello!")
    """

    name = "Webhook"
    description = "Generic webhook integration"
    version = "1.0.0"

    def __init__(self, integration_id: str, config: IntegrationConfig):
        super().__init__(integration_id, config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._webhook_url: Optional[str] = None

    async def connect(self) -> bool:
        """Initialize webhook integration."""
        try:
            self._webhook_url = self.config.settings.get("webhook_url")

            if not self._webhook_url:
                raise ValueError("Webhook URL required")

            self._session = aiohttp.ClientSession()

            # Optional auth headers
            auth_header = self.config.credentials.get("auth_header")
            if auth_header:
                self._session._default_headers["Authorization"] = auth_header

            self._connected = True
            logger.info(f"Webhook integration initialized: {self._webhook_url}")
            return True

        except Exception as e:
            logger.error(f"Webhook initialization failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> bool:
        """Close webhook session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        return True

    async def send_message(
        self,
        channel_id: str,
        content: str,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Send message via webhook."""
        if not self._session or not self._webhook_url:
            raise RuntimeError("Webhook not connected")

        payload = {
            "channel": channel_id,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if thread_id:
            payload["thread_id"] = thread_id

        if attachments:
            payload["attachments"] = attachments

        async with self._session.post(self._webhook_url, json=payload) as response:
            response_data = {}
            try:
                response_data = await response.json()
            except Exception:
                pass

            message_id = response_data.get("id", str(datetime.utcnow().timestamp()))

            return Message(
                id=message_id,
                channel_id=channel_id,
                content=content,
                message_type=MessageType.TEXT,
                thread_id=thread_id,
                attachments=attachments or [],
            )

    async def send_rich_message(
        self,
        channel_id: str,
        blocks: List[Dict[str, Any]],
        text: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Message:
        """Send rich message via webhook."""
        if not self._session or not self._webhook_url:
            raise RuntimeError("Webhook not connected")

        payload = {
            "channel": channel_id,
            "blocks": blocks,
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if thread_id:
            payload["thread_id"] = thread_id

        async with self._session.post(self._webhook_url, json=payload) as response:
            response_data = {}
            try:
                response_data = await response.json()
            except Exception:
                pass

            message_id = response_data.get("id", str(datetime.utcnow().timestamp()))

            return Message(
                id=message_id,
                channel_id=channel_id,
                content=text or "",
                message_type=MessageType.RICH,
                thread_id=thread_id,
                metadata={"blocks": blocks},
            )

    async def get_channels(self) -> List[Channel]:
        """Not applicable for webhook."""
        return []

    async def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Not applicable for webhook."""
        return None

    async def get_users(self) -> List[User]:
        """Not applicable for webhook."""
        return []

    async def get_user(self, user_id: str) -> Optional[User]:
        """Not applicable for webhook."""
        return None

    async def add_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> bool:
        """Not applicable for webhook."""
        return False

    async def upload_file(
        self,
        channel_id: str,
        file_content: bytes,
        filename: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload file via webhook (if supported)."""
        if not self._session or not self._webhook_url:
            raise RuntimeError("Webhook not connected")

        import base64

        payload = {
            "channel": channel_id,
            "file": {
                "filename": filename,
                "content": base64.b64encode(file_content).decode(),
                "title": title,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        async with self._session.post(self._webhook_url, json=payload) as response:
            try:
                return await response.json()
            except Exception:
                return {"status": "sent"}
