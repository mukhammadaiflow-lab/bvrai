"""
Configuration Version Management Module

This module provides comprehensive version control for agent configurations,
including change tracking, diff generation, rollback capabilities, and audit logging.
"""

import copy
import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .base import (
    AgentConfiguration,
    ConfigVersion,
    ConfigStatus,
    VersionError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class ChangeType(str, Enum):
    """Types of configuration changes."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class RollbackStrategy(str, Enum):
    """Strategies for rollback operations."""

    FULL = "full"  # Complete rollback to previous version
    SELECTIVE = "selective"  # Rollback only specific fields
    MERGE = "merge"  # Merge previous values with current


# =============================================================================
# Change Detection and Diff
# =============================================================================


class ConfigDiff:
    """
    Generates and represents differences between configurations.
    """

    def __init__(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
    ):
        """
        Initialize diff.

        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        self.old_config = old_config
        self.new_config = new_config
        self.changes: List[Dict[str, Any]] = []
        self._compute_diff()

    def _compute_diff(self) -> None:
        """Compute differences between configurations."""
        self._diff_objects(self.old_config, self.new_config, "")

    def _diff_objects(
        self,
        old: Any,
        new: Any,
        path: str,
    ) -> None:
        """Recursively diff objects."""
        # Handle None cases
        if old is None and new is None:
            return
        if old is None:
            self.changes.append({
                "path": path,
                "type": ChangeType.ADDED.value,
                "old_value": None,
                "new_value": new,
            })
            return
        if new is None:
            self.changes.append({
                "path": path,
                "type": ChangeType.REMOVED.value,
                "old_value": old,
                "new_value": None,
            })
            return

        # Handle different types
        if type(old) != type(new):
            self.changes.append({
                "path": path,
                "type": ChangeType.MODIFIED.value,
                "old_value": old,
                "new_value": new,
            })
            return

        # Handle dictionaries
        if isinstance(old, dict):
            all_keys = set(old.keys()) | set(new.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in old:
                    self.changes.append({
                        "path": new_path,
                        "type": ChangeType.ADDED.value,
                        "old_value": None,
                        "new_value": new[key],
                    })
                elif key not in new:
                    self.changes.append({
                        "path": new_path,
                        "type": ChangeType.REMOVED.value,
                        "old_value": old[key],
                        "new_value": None,
                    })
                else:
                    self._diff_objects(old[key], new[key], new_path)
            return

        # Handle lists
        if isinstance(old, list):
            if old != new:
                self.changes.append({
                    "path": path,
                    "type": ChangeType.MODIFIED.value,
                    "old_value": old,
                    "new_value": new,
                })
            return

        # Handle primitives
        if old != new:
            self.changes.append({
                "path": path,
                "type": ChangeType.MODIFIED.value,
                "old_value": old,
                "new_value": new,
            })

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return len(self.changes) > 0

    def get_changes_by_type(self, change_type: ChangeType) -> List[Dict[str, Any]]:
        """Get changes of a specific type."""
        return [c for c in self.changes if c["type"] == change_type.value]

    def get_modified_paths(self) -> List[str]:
        """Get list of modified paths."""
        return [c["path"] for c in self.changes]

    def get_summary(self) -> str:
        """Generate human-readable summary of changes."""
        added = len(self.get_changes_by_type(ChangeType.ADDED))
        removed = len(self.get_changes_by_type(ChangeType.REMOVED))
        modified = len(self.get_changes_by_type(ChangeType.MODIFIED))

        parts = []
        if added > 0:
            parts.append(f"{added} added")
        if removed > 0:
            parts.append(f"{removed} removed")
        if modified > 0:
            parts.append(f"{modified} modified")

        if not parts:
            return "No changes"

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert diff to dictionary."""
        return {
            "changes": self.changes,
            "summary": self.get_summary(),
            "has_changes": self.has_changes(),
            "added_count": len(self.get_changes_by_type(ChangeType.ADDED)),
            "removed_count": len(self.get_changes_by_type(ChangeType.REMOVED)),
            "modified_count": len(self.get_changes_by_type(ChangeType.MODIFIED)),
        }


# =============================================================================
# Version Storage Interface
# =============================================================================


class VersionStorage(ABC):
    """Abstract base class for version storage backends."""

    @abstractmethod
    async def save_version(self, version: ConfigVersion) -> None:
        """Save a version record."""
        pass

    @abstractmethod
    async def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """Get a version by ID."""
        pass

    @abstractmethod
    async def get_latest_version(self, config_id: str) -> Optional[ConfigVersion]:
        """Get latest version for a configuration."""
        pass

    @abstractmethod
    async def list_versions(
        self,
        config_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ConfigVersion], int]:
        """List versions for a configuration."""
        pass

    @abstractmethod
    async def get_version_by_number(
        self,
        config_id: str,
        version_number: int,
    ) -> Optional[ConfigVersion]:
        """Get version by version number."""
        pass

    @abstractmethod
    async def delete_versions(
        self,
        config_id: str,
        keep_latest: int = 0,
    ) -> int:
        """Delete versions, optionally keeping latest N."""
        pass


class InMemoryVersionStorage(VersionStorage):
    """In-memory version storage for testing and development."""

    def __init__(self):
        self._versions: Dict[str, ConfigVersion] = {}
        self._config_versions: Dict[str, List[str]] = {}

    async def save_version(self, version: ConfigVersion) -> None:
        """Save a version to memory."""
        self._versions[version.id] = version

        if version.config_id not in self._config_versions:
            self._config_versions[version.config_id] = []
        self._config_versions[version.config_id].append(version.id)

    async def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """Get a version by ID."""
        return self._versions.get(version_id)

    async def get_latest_version(self, config_id: str) -> Optional[ConfigVersion]:
        """Get latest version for a configuration."""
        if config_id not in self._config_versions:
            return None

        version_ids = self._config_versions[config_id]
        if not version_ids:
            return None

        # Get all versions and find highest version number
        versions = [self._versions[vid] for vid in version_ids if vid in self._versions]
        if not versions:
            return None

        return max(versions, key=lambda v: v.version_number)

    async def list_versions(
        self,
        config_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ConfigVersion], int]:
        """List versions for a configuration."""
        if config_id not in self._config_versions:
            return [], 0

        version_ids = self._config_versions[config_id]
        versions = [self._versions[vid] for vid in version_ids if vid in self._versions]

        # Sort by version number descending
        versions.sort(key=lambda v: v.version_number, reverse=True)

        total = len(versions)
        versions = versions[offset:offset + limit]

        return versions, total

    async def get_version_by_number(
        self,
        config_id: str,
        version_number: int,
    ) -> Optional[ConfigVersion]:
        """Get version by version number."""
        if config_id not in self._config_versions:
            return None

        for vid in self._config_versions[config_id]:
            version = self._versions.get(vid)
            if version and version.version_number == version_number:
                return version

        return None

    async def delete_versions(
        self,
        config_id: str,
        keep_latest: int = 0,
    ) -> int:
        """Delete versions, optionally keeping latest N."""
        if config_id not in self._config_versions:
            return 0

        version_ids = self._config_versions[config_id]
        versions = [self._versions[vid] for vid in version_ids if vid in self._versions]

        # Sort by version number descending
        versions.sort(key=lambda v: v.version_number, reverse=True)

        # Determine which to delete
        to_delete = versions[keep_latest:] if keep_latest > 0 else versions
        deleted_count = 0

        for version in to_delete:
            if version.id in self._versions:
                del self._versions[version.id]
                self._config_versions[config_id].remove(version.id)
                deleted_count += 1

        return deleted_count


# =============================================================================
# Version Manager
# =============================================================================


class VersionManager:
    """
    Manages configuration versions with change tracking and rollback capabilities.
    """

    def __init__(
        self,
        storage: Optional[VersionStorage] = None,
        max_versions_per_config: int = 100,
    ):
        """
        Initialize version manager.

        Args:
            storage: Version storage backend
            max_versions_per_config: Maximum versions to keep per configuration
        """
        self.storage = storage or InMemoryVersionStorage()
        self.max_versions = max_versions_per_config

    async def create_version(
        self,
        config: AgentConfiguration,
        change_summary: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> ConfigVersion:
        """
        Create a new version for a configuration.

        Args:
            config: Configuration to version
            change_summary: Description of changes
            created_by: User creating the version

        Returns:
            Created version
        """
        # Get latest version to determine version number
        latest = await self.storage.get_latest_version(config.id)
        version_number = (latest.version_number + 1) if latest else 1

        # Generate change list
        changes = []
        if latest:
            diff = ConfigDiff(latest.config_snapshot, config.to_dict())
            changes = diff.changes
            if not change_summary:
                change_summary = diff.get_summary()

        version = ConfigVersion(
            id=f"ver_{uuid.uuid4().hex[:24]}",
            config_id=config.id,
            version_number=version_number,
            config_snapshot=copy.deepcopy(config.to_dict()),
            change_summary=change_summary or "Initial version",
            changes=changes,
            created_at=datetime.utcnow(),
            created_by=created_by,
        )

        # Save version
        await self.storage.save_version(version)

        # Clean up old versions if needed
        await self._cleanup_old_versions(config.id)

        logger.info(
            f"Created version {version_number} for config {config.id}: {change_summary}"
        )
        return version

    async def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """
        Get a specific version.

        Args:
            version_id: Version ID

        Returns:
            Version or None
        """
        return await self.storage.get_version(version_id)

    async def get_latest_version(
        self,
        config_id: str,
    ) -> Optional[ConfigVersion]:
        """
        Get the latest version for a configuration.

        Args:
            config_id: Configuration ID

        Returns:
            Latest version or None
        """
        return await self.storage.get_latest_version(config_id)

    async def get_version_by_number(
        self,
        config_id: str,
        version_number: int,
    ) -> Optional[ConfigVersion]:
        """
        Get a specific version by number.

        Args:
            config_id: Configuration ID
            version_number: Version number

        Returns:
            Version or None
        """
        return await self.storage.get_version_by_number(config_id, version_number)

    async def list_versions(
        self,
        config_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ConfigVersion], int]:
        """
        List versions for a configuration.

        Args:
            config_id: Configuration ID
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (versions, total_count)
        """
        return await self.storage.list_versions(config_id, offset, limit)

    async def compare_versions(
        self,
        config_id: str,
        version_a: int,
        version_b: int,
    ) -> ConfigDiff:
        """
        Compare two versions of a configuration.

        Args:
            config_id: Configuration ID
            version_a: First version number
            version_b: Second version number

        Returns:
            Diff between versions
        """
        ver_a = await self.storage.get_version_by_number(config_id, version_a)
        ver_b = await self.storage.get_version_by_number(config_id, version_b)

        if not ver_a:
            raise VersionError(f"Version {version_a} not found for config {config_id}")
        if not ver_b:
            raise VersionError(f"Version {version_b} not found for config {config_id}")

        return ConfigDiff(ver_a.config_snapshot, ver_b.config_snapshot)

    async def rollback(
        self,
        config_id: str,
        target_version: int,
        strategy: RollbackStrategy = RollbackStrategy.FULL,
        fields_to_rollback: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Rollback to a previous version.

        Args:
            config_id: Configuration ID
            target_version: Version number to rollback to
            strategy: Rollback strategy
            fields_to_rollback: Fields to rollback (for selective strategy)
            created_by: User performing rollback

        Returns:
            Rolled back configuration
        """
        target = await self.storage.get_version_by_number(config_id, target_version)
        if not target:
            raise VersionError(f"Version {target_version} not found for config {config_id}")

        current = await self.storage.get_latest_version(config_id)
        if not current:
            raise VersionError(f"No current version found for config {config_id}")

        # Perform rollback based on strategy
        if strategy == RollbackStrategy.FULL:
            rolled_back_config = AgentConfiguration.from_dict(target.config_snapshot)

        elif strategy == RollbackStrategy.SELECTIVE:
            if not fields_to_rollback:
                raise VersionError("Fields must be specified for selective rollback")

            rolled_back_config = AgentConfiguration.from_dict(current.config_snapshot)
            for field in fields_to_rollback:
                self._apply_field_value(
                    rolled_back_config,
                    field,
                    self._get_field_value(target.config_snapshot, field),
                )

        elif strategy == RollbackStrategy.MERGE:
            rolled_back_config = self._merge_configs(
                AgentConfiguration.from_dict(current.config_snapshot),
                target.config_snapshot,
            )

        else:
            raise VersionError(f"Unknown rollback strategy: {strategy}")

        # Update version info
        rolled_back_config.version = current.version_number + 1
        rolled_back_config.updated_at = datetime.utcnow()

        # Create new version recording the rollback
        rollback_version = ConfigVersion(
            id=f"ver_{uuid.uuid4().hex[:24]}",
            config_id=config_id,
            version_number=current.version_number + 1,
            config_snapshot=rolled_back_config.to_dict(),
            change_summary=f"Rollback to version {target_version}",
            changes=[],
            created_at=datetime.utcnow(),
            created_by=created_by,
            is_rollback=True,
            rolled_back_from=current.id,
        )

        await self.storage.save_version(rollback_version)

        logger.info(
            f"Rolled back config {config_id} from v{current.version_number} "
            f"to v{target_version} (now v{rollback_version.version_number})"
        )

        return rolled_back_config

    async def restore_version(
        self,
        version_id: str,
        created_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Restore configuration from a specific version.

        Args:
            version_id: Version ID to restore
            created_by: User performing restore

        Returns:
            Restored configuration
        """
        version = await self.storage.get_version(version_id)
        if not version:
            raise VersionError(f"Version not found: {version_id}")

        return await self.rollback(
            config_id=version.config_id,
            target_version=version.version_number,
            created_by=created_by,
        )

    def _get_field_value(
        self,
        config_dict: Dict[str, Any],
        field_path: str,
    ) -> Any:
        """Get a nested field value from config dict."""
        parts = field_path.split(".")
        value = config_dict
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

    def _apply_field_value(
        self,
        config: AgentConfiguration,
        field_path: str,
        value: Any,
    ) -> None:
        """Apply a value to a field in the configuration."""
        parts = field_path.split(".")
        obj = config

        # Navigate to parent
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return

        # Set the value
        final_part = parts[-1]
        if hasattr(obj, final_part):
            setattr(obj, final_part, value)

    def _merge_configs(
        self,
        current: AgentConfiguration,
        old_snapshot: Dict[str, Any],
    ) -> AgentConfiguration:
        """Merge old configuration values into current."""
        # For merge, we keep current values but restore any removed fields
        # from the old version that don't exist in current
        current_dict = current.to_dict()

        def merge_dicts(current_d: Dict, old_d: Dict) -> Dict:
            result = copy.deepcopy(current_d)
            for key, old_val in old_d.items():
                if key not in result:
                    result[key] = old_val
                elif isinstance(old_val, dict) and isinstance(result[key], dict):
                    result[key] = merge_dicts(result[key], old_val)
            return result

        merged = merge_dicts(current_dict, old_snapshot)
        return AgentConfiguration.from_dict(merged)

    async def _cleanup_old_versions(self, config_id: str) -> None:
        """Clean up old versions exceeding maximum."""
        versions, total = await self.storage.list_versions(config_id)
        if total > self.max_versions:
            deleted = await self.storage.delete_versions(
                config_id,
                keep_latest=self.max_versions,
            )
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old versions for config {config_id}")


# =============================================================================
# Audit Log
# =============================================================================


class AuditAction(str, Enum):
    """Types of audit actions."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    ROLLBACK = "rollback"
    CLONE = "clone"
    EXPORT = "export"
    IMPORT = "import"


class AuditEntry:
    """An entry in the audit log."""

    def __init__(
        self,
        id: str,
        entity_type: str,
        entity_id: str,
        action: AuditAction,
        organization_id: str,
        actor_id: Optional[str] = None,
        actor_email: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        self.id = id
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.action = action
        self.organization_id = organization_id
        self.actor_id = actor_id
        self.actor_email = actor_email
        self.changes = changes or {}
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.ip_address = ip_address
        self.user_agent = user_agent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action.value,
            "organization_id": self.organization_id,
            "actor_id": self.actor_id,
            "actor_email": self.actor_email,
            "changes": self.changes,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }


class AuditLogger:
    """
    Logs audit events for configuration changes.
    """

    def __init__(self):
        self._entries: List[AuditEntry] = []
        self._entity_index: Dict[str, List[str]] = {}
        self._org_index: Dict[str, List[str]] = {}

    async def log(
        self,
        entity_type: str,
        entity_id: str,
        action: AuditAction,
        organization_id: str,
        actor_id: Optional[str] = None,
        actor_email: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEntry:
        """
        Log an audit event.

        Args:
            entity_type: Type of entity (config, template, profile)
            entity_id: Entity ID
            action: Action performed
            organization_id: Organization ID
            actor_id: ID of user performing action
            actor_email: Email of user
            changes: Changes made
            metadata: Additional metadata
            ip_address: Client IP
            user_agent: Client user agent

        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            id=f"audit_{uuid.uuid4().hex[:24]}",
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            organization_id=organization_id,
            actor_id=actor_id,
            actor_email=actor_email,
            changes=changes,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self._entries.append(entry)

        # Update indices
        entity_key = f"{entity_type}:{entity_id}"
        if entity_key not in self._entity_index:
            self._entity_index[entity_key] = []
        self._entity_index[entity_key].append(entry.id)

        if organization_id not in self._org_index:
            self._org_index[organization_id] = []
        self._org_index[organization_id].append(entry.id)

        logger.debug(
            f"Audit: {action.value} on {entity_type} {entity_id} by {actor_email or actor_id}"
        )

        return entry

    async def get_entity_history(
        self,
        entity_type: str,
        entity_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[AuditEntry], int]:
        """
        Get audit history for an entity.

        Args:
            entity_type: Entity type
            entity_id: Entity ID
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (entries, total_count)
        """
        entity_key = f"{entity_type}:{entity_id}"
        if entity_key not in self._entity_index:
            return [], 0

        entry_ids = self._entity_index[entity_key]
        entries = [e for e in self._entries if e.id in entry_ids]

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        total = len(entries)
        entries = entries[offset:offset + limit]

        return entries, total

    async def get_organization_history(
        self,
        organization_id: str,
        entity_type: Optional[str] = None,
        action: Optional[AuditAction] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[AuditEntry], int]:
        """
        Get audit history for an organization.

        Args:
            organization_id: Organization ID
            entity_type: Filter by entity type
            action: Filter by action
            start_time: Filter by start time
            end_time: Filter by end time
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (entries, total_count)
        """
        if organization_id not in self._org_index:
            return [], 0

        entry_ids = self._org_index[organization_id]
        entries = [e for e in self._entries if e.id in entry_ids]

        # Apply filters
        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]
        if action:
            entries = [e for e in entries if e.action == action]
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        total = len(entries)
        entries = entries[offset:offset + limit]

        return entries, total

    async def search(
        self,
        organization_id: str,
        actor_id: Optional[str] = None,
        actor_email: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        actions: Optional[List[AuditAction]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[AuditEntry], int]:
        """
        Search audit logs with multiple filters.

        Args:
            organization_id: Organization ID
            actor_id: Filter by actor ID
            actor_email: Filter by actor email
            entity_types: Filter by entity types
            actions: Filter by actions
            start_time: Filter by start time
            end_time: Filter by end time
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (entries, total_count)
        """
        if organization_id not in self._org_index:
            return [], 0

        entry_ids = self._org_index[organization_id]
        entries = [e for e in self._entries if e.id in entry_ids]

        # Apply filters
        if actor_id:
            entries = [e for e in entries if e.actor_id == actor_id]
        if actor_email:
            entries = [e for e in entries if e.actor_email == actor_email]
        if entity_types:
            entries = [e for e in entries if e.entity_type in entity_types]
        if actions:
            entries = [e for e in entries if e.action in actions]
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        total = len(entries)
        entries = entries[offset:offset + limit]

        return entries, total


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ChangeType",
    "RollbackStrategy",
    "ConfigDiff",
    "VersionStorage",
    "InMemoryVersionStorage",
    "VersionManager",
    "AuditAction",
    "AuditEntry",
    "AuditLogger",
]
