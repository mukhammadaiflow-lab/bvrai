"""
Plugin Architecture
===================

Extensible plugin system for adding custom functionality to the platform.

Features:
- Plugin lifecycle management
- Hook system for intercepting events
- Dependency injection
- Configuration management
- Hot reloading support
- Plugin marketplace integration
- Sandboxed execution

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import tempfile
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")
PluginT = TypeVar("PluginT", bound="Plugin")


# =============================================================================
# ENUMS
# =============================================================================


class PluginState(str, Enum):
    """Plugin lifecycle states"""

    DISCOVERED = "discovered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    DISABLED = "disabled"


class PluginType(str, Enum):
    """Types of plugins"""

    # Core extensions
    SERVICE = "service"
    MIDDLEWARE = "middleware"
    PROVIDER = "provider"

    # Integration plugins
    CRM = "crm"
    CALENDAR = "calendar"
    STORAGE = "storage"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"

    # Voice plugins
    STT = "stt"
    TTS = "tts"
    LLM = "llm"
    NLU = "nlu"

    # Custom
    CUSTOM = "custom"


class HookType(str, Enum):
    """Types of plugin hooks"""

    # Lifecycle hooks
    BEFORE_START = "before_start"
    AFTER_START = "after_start"
    BEFORE_STOP = "before_stop"
    AFTER_STOP = "after_stop"

    # Request hooks
    BEFORE_REQUEST = "before_request"
    AFTER_REQUEST = "after_request"
    ON_REQUEST_ERROR = "on_request_error"

    # Call hooks
    BEFORE_CALL = "before_call"
    AFTER_CALL = "after_call"
    ON_CALL_START = "on_call_start"
    ON_CALL_END = "on_call_end"

    # Conversation hooks
    BEFORE_MESSAGE = "before_message"
    AFTER_MESSAGE = "after_message"
    ON_INTENT_DETECTED = "on_intent_detected"
    ON_ENTITY_EXTRACTED = "on_entity_extracted"

    # Agent hooks
    ON_AGENT_CREATE = "on_agent_create"
    ON_AGENT_UPDATE = "on_agent_update"
    ON_AGENT_DELETE = "on_agent_delete"

    # LLM hooks
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    ON_STREAM_TOKEN = "on_stream_token"

    # Custom hooks
    CUSTOM = "custom"


class PluginPriority(int, Enum):
    """Plugin execution priority"""

    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 100
    BACKGROUND = 200


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class PluginDependency(BaseModel):
    """Plugin dependency definition"""

    name: str
    version: str = "*"
    optional: bool = False


class PluginPermission(BaseModel):
    """Plugin permission definition"""

    resource: str
    actions: List[str] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)


class PluginConfig(BaseModel):
    """Plugin configuration"""

    # Identity
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    license: str = "MIT"
    homepage: str = ""

    # Classification
    type: PluginType = PluginType.CUSTOM
    tags: List[str] = Field(default_factory=list)
    category: str = "general"

    # Dependencies
    dependencies: List[PluginDependency] = Field(default_factory=list)
    platform_version: str = ">=2.0.0"

    # Permissions
    permissions: List[PluginPermission] = Field(default_factory=list)

    # Execution
    priority: PluginPriority = PluginPriority.NORMAL
    enabled: bool = True
    auto_start: bool = True
    sandboxed: bool = False

    # Configuration
    settings: Dict[str, Any] = Field(default_factory=dict)
    settings_schema: Optional[Dict[str, Any]] = None

    # Resources
    icon: Optional[str] = None
    readme: Optional[str] = None


class PluginSettings(BaseModel):
    """Runtime plugin settings"""

    plugin_id: str
    organization_id: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PluginContext:
    """Context provided to plugins during execution"""

    plugin_id: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    agent_id: Optional[str] = None
    call_id: Optional[str] = None
    conversation_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


@dataclass
class HookResult:
    """Result of a hook execution"""

    success: bool = True
    data: Any = None
    error: Optional[str] = None
    modified: bool = False
    stop_propagation: bool = False
    execution_time_ms: float = 0.0


@dataclass
class PluginHook:
    """Registered plugin hook"""

    id: str = field(default_factory=lambda: str(uuid4()))
    plugin_id: str = ""
    hook_type: HookType = HookType.CUSTOM
    handler: Callable[[PluginContext], Awaitable[HookResult]] = None
    priority: int = 0
    enabled: bool = True
    filter_expression: Optional[str] = None
    timeout_seconds: float = 30.0
    async_execution: bool = False
    registered_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PluginEvent:
    """Event emitted by a plugin"""

    id: str = field(default_factory=lambda: str(uuid4()))
    plugin_id: str = ""
    event_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PluginMetrics:
    """Plugin execution metrics"""

    plugin_id: str
    hooks_executed: int = 0
    hooks_failed: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    events_emitted: int = 0
    last_execution: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# ABSTRACT PLUGIN BASE
# =============================================================================


class Plugin(ABC):
    """
    Abstract base class for all plugins.

    Plugins must implement initialization, activation, and deactivation
    methods. They can register hooks, emit events, and access platform
    services through the plugin context.

    Usage:
        class MyPlugin(Plugin):
            async def on_initialize(self) -> None:
                self.register_hook(
                    HookType.BEFORE_CALL,
                    self.handle_before_call
                )

            async def on_activate(self) -> None:
                self.logger.info("Plugin activated")

            async def on_deactivate(self) -> None:
                self.logger.info("Plugin deactivated")

            async def handle_before_call(self, ctx: PluginContext) -> HookResult:
                # Custom logic
                return HookResult(success=True)
    """

    def __init__(self, config: PluginConfig):
        self.config = config
        self._state = PluginState.DISCOVERED
        self._logger = structlog.get_logger(f"plugin.{config.name}")
        self._manager: Optional["PluginManager"] = None
        self._hooks: List[PluginHook] = []
        self._metrics = PluginMetrics(plugin_id=config.id)
        self._settings: Dict[str, Any] = config.settings.copy()
        self._context: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def state(self) -> PluginState:
        return self._state

    @state.setter
    def state(self, value: PluginState) -> None:
        old_state = self._state
        self._state = value
        self._logger.info(
            "plugin_state_changed",
            old_state=old_state,
            new_state=value
        )

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def settings(self) -> Dict[str, Any]:
        return self._settings

    @property
    def metrics(self) -> PluginMetrics:
        return self._metrics

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def on_initialize(self) -> None:
        """
        Initialize the plugin.
        Called once when the plugin is first loaded.
        Register hooks and set up resources here.
        """
        pass

    @abstractmethod
    async def on_activate(self) -> None:
        """
        Activate the plugin.
        Called when the plugin is enabled/started.
        """
        pass

    @abstractmethod
    async def on_deactivate(self) -> None:
        """
        Deactivate the plugin.
        Called when the plugin is disabled/stopped.
        Clean up resources here.
        """
        pass

    # -------------------------------------------------------------------------
    # Optional Lifecycle Methods
    # -------------------------------------------------------------------------

    async def on_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Called when plugin settings are updated"""
        self._settings = settings

    async def on_upgrade(self, from_version: str) -> None:
        """Called when plugin is upgraded from a previous version"""
        pass

    async def on_uninstall(self) -> None:
        """Called when plugin is being uninstalled"""
        pass

    # -------------------------------------------------------------------------
    # Hook Registration
    # -------------------------------------------------------------------------

    def register_hook(
        self,
        hook_type: HookType,
        handler: Callable[[PluginContext], Awaitable[HookResult]],
        priority: int = 0,
        filter_expression: Optional[str] = None,
        timeout_seconds: float = 30.0,
        async_execution: bool = False
    ) -> str:
        """
        Register a hook handler.

        Args:
            hook_type: Type of hook to register
            handler: Async function to handle the hook
            priority: Execution priority (lower = earlier)
            filter_expression: Optional filter for when to execute
            timeout_seconds: Maximum execution time
            async_execution: Whether to execute asynchronously

        Returns:
            Hook ID
        """
        hook = PluginHook(
            plugin_id=self.id,
            hook_type=hook_type,
            handler=handler,
            priority=priority,
            filter_expression=filter_expression,
            timeout_seconds=timeout_seconds,
            async_execution=async_execution
        )

        self._hooks.append(hook)

        if self._manager:
            self._manager._register_hook(hook)

        self._logger.debug(
            "hook_registered",
            hook_type=hook_type.value,
            hook_id=hook.id
        )

        return hook.id

    def unregister_hook(self, hook_id: str) -> bool:
        """Unregister a hook handler"""
        for i, hook in enumerate(self._hooks):
            if hook.id == hook_id:
                self._hooks.pop(i)
                if self._manager:
                    self._manager._unregister_hook(hook_id)
                return True
        return False

    # -------------------------------------------------------------------------
    # Event Emission
    # -------------------------------------------------------------------------

    async def emit_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Emit an event from this plugin"""
        event = PluginEvent(
            plugin_id=self.id,
            event_type=event_type,
            data=data
        )

        if self._manager:
            await self._manager._emit_plugin_event(event)

        self._metrics.events_emitted += 1

    # -------------------------------------------------------------------------
    # Service Access
    # -------------------------------------------------------------------------

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a platform service by name"""
        if self._manager:
            return self._manager.get_service(service_name)
        return None

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a plugin setting value"""
        return self._settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a plugin setting value"""
        self._settings[key] = value

    # -------------------------------------------------------------------------
    # Storage Access
    # -------------------------------------------------------------------------

    async def get_storage(self, key: str) -> Optional[Any]:
        """Get a value from plugin storage"""
        if self._manager:
            return await self._manager.get_plugin_storage(self.id, key)
        return None

    async def set_storage(self, key: str, value: Any) -> None:
        """Set a value in plugin storage"""
        if self._manager:
            await self._manager.set_plugin_storage(self.id, key, value)


# =============================================================================
# PLUGIN REGISTRY
# =============================================================================


class PluginRegistry:
    """
    Registry for discovering and managing plugin definitions.
    """

    def __init__(self):
        self._plugins: Dict[str, Type[Plugin]] = {}
        self._configs: Dict[str, PluginConfig] = {}
        self._search_paths: List[Path] = []
        self._logger = structlog.get_logger("plugin_registry")

    def register(
        self,
        plugin_class: Type[Plugin],
        config: PluginConfig
    ) -> None:
        """Register a plugin class"""
        self._plugins[config.id] = plugin_class
        self._configs[config.id] = config
        self._logger.info(
            "plugin_registered",
            plugin_id=config.id,
            name=config.name,
            version=config.version
        )

    def unregister(self, plugin_id: str) -> bool:
        """Unregister a plugin"""
        if plugin_id in self._plugins:
            del self._plugins[plugin_id]
            del self._configs[plugin_id]
            return True
        return False

    def get_class(self, plugin_id: str) -> Optional[Type[Plugin]]:
        """Get a plugin class by ID"""
        return self._plugins.get(plugin_id)

    def get_config(self, plugin_id: str) -> Optional[PluginConfig]:
        """Get a plugin config by ID"""
        return self._configs.get(plugin_id)

    def list_all(self) -> List[PluginConfig]:
        """List all registered plugins"""
        return list(self._configs.values())

    def list_by_type(self, plugin_type: PluginType) -> List[PluginConfig]:
        """List plugins by type"""
        return [
            c for c in self._configs.values()
            if c.type == plugin_type
        ]

    def add_search_path(self, path: Path) -> None:
        """Add a path to search for plugins"""
        if path not in self._search_paths:
            self._search_paths.append(path)

    def discover(self) -> List[PluginConfig]:
        """Discover plugins in search paths"""
        discovered = []

        for search_path in self._search_paths:
            if not search_path.exists():
                continue

            for plugin_path in search_path.iterdir():
                if plugin_path.is_dir() and (plugin_path / "plugin.json").exists():
                    try:
                        config = self._load_plugin_config(plugin_path)
                        if config:
                            discovered.append(config)
                    except Exception as e:
                        self._logger.error(
                            "plugin_discovery_error",
                            path=str(plugin_path),
                            error=str(e)
                        )

        return discovered

    def _load_plugin_config(self, plugin_path: Path) -> Optional[PluginConfig]:
        """Load plugin configuration from directory"""
        config_file = plugin_path / "plugin.json"

        if not config_file.exists():
            return None

        with open(config_file) as f:
            data = json.load(f)

        return PluginConfig(**data)

    def load_plugin_class(
        self,
        plugin_path: Path,
        class_name: str
    ) -> Optional[Type[Plugin]]:
        """Dynamically load a plugin class from path"""
        main_file = plugin_path / "main.py"

        if not main_file.exists():
            return None

        spec = importlib.util.spec_from_file_location(
            f"plugin_{plugin_path.name}",
            main_file
        )

        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        plugin_class = getattr(module, class_name, None)

        if plugin_class and issubclass(plugin_class, Plugin):
            return plugin_class

        return None


# =============================================================================
# HOOK EXECUTOR
# =============================================================================


class HookExecutor:
    """
    Executes plugin hooks with proper ordering and error handling.
    """

    def __init__(self):
        self._hooks: Dict[HookType, List[PluginHook]] = defaultdict(list)
        self._logger = structlog.get_logger("hook_executor")

    def register(self, hook: PluginHook) -> None:
        """Register a hook"""
        self._hooks[hook.hook_type].append(hook)
        # Sort by priority
        self._hooks[hook.hook_type].sort(key=lambda h: h.priority)

    def unregister(self, hook_id: str) -> bool:
        """Unregister a hook"""
        for hook_type, hooks in self._hooks.items():
            for i, hook in enumerate(hooks):
                if hook.id == hook_id:
                    hooks.pop(i)
                    return True
        return False

    async def execute(
        self,
        hook_type: HookType,
        context: PluginContext
    ) -> List[HookResult]:
        """Execute all hooks of a type"""
        hooks = self._hooks.get(hook_type, [])
        results = []

        for hook in hooks:
            if not hook.enabled:
                continue

            # Check filter
            if hook.filter_expression:
                if not self._evaluate_filter(hook.filter_expression, context):
                    continue

            try:
                import time
                start_time = time.time()

                if hook.async_execution:
                    # Fire and forget
                    asyncio.create_task(hook.handler(context))
                    result = HookResult(success=True)
                else:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        hook.handler(context),
                        timeout=hook.timeout_seconds
                    )

                result.execution_time_ms = (time.time() - start_time) * 1000

            except asyncio.TimeoutError:
                result = HookResult(
                    success=False,
                    error="Hook execution timed out"
                )
                self._logger.warning(
                    "hook_timeout",
                    hook_id=hook.id,
                    plugin_id=hook.plugin_id
                )

            except Exception as e:
                result = HookResult(
                    success=False,
                    error=str(e)
                )
                self._logger.error(
                    "hook_execution_error",
                    hook_id=hook.id,
                    plugin_id=hook.plugin_id,
                    error=str(e)
                )

            results.append(result)

            # Stop propagation if requested
            if result.stop_propagation:
                break

        return results

    async def execute_single(
        self,
        hook_type: HookType,
        context: PluginContext,
        plugin_id: str
    ) -> Optional[HookResult]:
        """Execute hooks from a specific plugin"""
        hooks = [
            h for h in self._hooks.get(hook_type, [])
            if h.plugin_id == plugin_id and h.enabled
        ]

        if not hooks:
            return None

        results = []
        for hook in hooks:
            try:
                result = await asyncio.wait_for(
                    hook.handler(context),
                    timeout=hook.timeout_seconds
                )
                results.append(result)
            except Exception as e:
                results.append(HookResult(success=False, error=str(e)))

        return results[0] if results else None

    def _evaluate_filter(
        self,
        expression: str,
        context: PluginContext
    ) -> bool:
        """Evaluate a filter expression"""
        try:
            eval_context = {
                "ctx": context,
                "data": context.data,
                "user_id": context.user_id,
                "org_id": context.organization_id,
                "agent_id": context.agent_id,
            }
            return bool(eval(expression, {"__builtins__": {}}, eval_context))
        except Exception:
            return True


# =============================================================================
# PLUGIN MANAGER
# =============================================================================


class PluginManager:
    """
    Central plugin management system.

    Manages plugin lifecycle, hook execution, event emission,
    and plugin configuration.

    Usage:
        manager = PluginManager()
        await manager.start()

        # Load and activate a plugin
        await manager.load_plugin(MyPluginClass, config)
        await manager.activate_plugin(plugin_id)

        # Execute hooks
        results = await manager.execute_hook(HookType.BEFORE_CALL, context)

        await manager.stop()
    """

    def __init__(
        self,
        plugins_dir: Optional[Path] = None,
        auto_discover: bool = True
    ):
        self._registry = PluginRegistry()
        self._hook_executor = HookExecutor()
        self._instances: Dict[str, Plugin] = {}
        self._services: Dict[str, Any] = {}
        self._storage: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("plugin_manager")

        # Auto-discovery
        if plugins_dir:
            self._registry.add_search_path(plugins_dir)
        self._auto_discover = auto_discover

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the plugin manager"""
        self._running = True

        # Discover plugins
        if self._auto_discover:
            discovered = self._registry.discover()
            self._logger.info("plugins_discovered", count=len(discovered))

        # Auto-start enabled plugins
        for config in self._registry.list_all():
            if config.auto_start and config.enabled:
                try:
                    await self.activate_plugin(config.id)
                except Exception as e:
                    self._logger.error(
                        "plugin_auto_start_failed",
                        plugin_id=config.id,
                        error=str(e)
                    )

        self._logger.info("plugin_manager_started")

    async def stop(self) -> None:
        """Stop the plugin manager"""
        # Deactivate all plugins
        for plugin_id in list(self._instances.keys()):
            try:
                await self.deactivate_plugin(plugin_id)
            except Exception as e:
                self._logger.error(
                    "plugin_deactivation_error",
                    plugin_id=plugin_id,
                    error=str(e)
                )

        self._running = False
        self._logger.info("plugin_manager_stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Plugin Loading
    # -------------------------------------------------------------------------

    async def load_plugin(
        self,
        plugin_class: Type[Plugin],
        config: PluginConfig
    ) -> str:
        """
        Load a plugin class.

        Args:
            plugin_class: The plugin class to load
            config: Plugin configuration

        Returns:
            Plugin ID
        """
        # Register in registry
        self._registry.register(plugin_class, config)

        # Check dependencies
        if not self._check_dependencies(config):
            raise RuntimeError(
                f"Plugin {config.name} has unmet dependencies"
            )

        # Create instance
        instance = plugin_class(config)
        instance._manager = self

        # Initialize
        instance.state = PluginState.INITIALIZING
        await instance.on_initialize()

        # Register hooks
        for hook in instance._hooks:
            self._hook_executor.register(hook)

        instance.state = PluginState.REGISTERED
        self._instances[config.id] = instance

        self._logger.info(
            "plugin_loaded",
            plugin_id=config.id,
            name=config.name,
            version=config.version
        )

        return config.id

    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        instance = self._instances.get(plugin_id)

        if not instance:
            return False

        # Deactivate if active
        if instance.state == PluginState.ACTIVE:
            await self.deactivate_plugin(plugin_id)

        # Unregister hooks
        for hook in instance._hooks:
            self._hook_executor.unregister(hook.id)

        # Call uninstall hook
        await instance.on_uninstall()

        # Remove instance
        del self._instances[plugin_id]
        self._registry.unregister(plugin_id)

        self._logger.info("plugin_unloaded", plugin_id=plugin_id)

        return True

    def _check_dependencies(self, config: PluginConfig) -> bool:
        """Check if plugin dependencies are met"""
        for dep in config.dependencies:
            if dep.optional:
                continue

            # Check if dependency is loaded
            found = False
            for other_config in self._registry.list_all():
                if other_config.name == dep.name:
                    # Check version
                    if dep.version == "*" or other_config.version == dep.version:
                        found = True
                        break

            if not found:
                self._logger.warning(
                    "missing_dependency",
                    plugin=config.name,
                    dependency=dep.name
                )
                return False

        return True

    # -------------------------------------------------------------------------
    # Plugin Activation
    # -------------------------------------------------------------------------

    async def activate_plugin(self, plugin_id: str) -> bool:
        """Activate a plugin"""
        instance = self._instances.get(plugin_id)

        if not instance:
            # Try to create instance from registry
            plugin_class = self._registry.get_class(plugin_id)
            config = self._registry.get_config(plugin_id)

            if plugin_class and config:
                await self.load_plugin(plugin_class, config)
                instance = self._instances.get(plugin_id)

        if not instance:
            self._logger.error("plugin_not_found", plugin_id=plugin_id)
            return False

        if instance.state == PluginState.ACTIVE:
            return True

        try:
            await instance.on_activate()
            instance.state = PluginState.ACTIVE

            self._logger.info("plugin_activated", plugin_id=plugin_id)
            return True

        except Exception as e:
            instance.state = PluginState.FAILED
            self._logger.error(
                "plugin_activation_failed",
                plugin_id=plugin_id,
                error=str(e)
            )
            return False

    async def deactivate_plugin(self, plugin_id: str) -> bool:
        """Deactivate a plugin"""
        instance = self._instances.get(plugin_id)

        if not instance:
            return False

        if instance.state != PluginState.ACTIVE:
            return True

        try:
            instance.state = PluginState.STOPPING
            await instance.on_deactivate()
            instance.state = PluginState.STOPPED

            self._logger.info("plugin_deactivated", plugin_id=plugin_id)
            return True

        except Exception as e:
            instance.state = PluginState.FAILED
            self._logger.error(
                "plugin_deactivation_failed",
                plugin_id=plugin_id,
                error=str(e)
            )
            return False

    async def suspend_plugin(self, plugin_id: str) -> bool:
        """Temporarily suspend a plugin"""
        instance = self._instances.get(plugin_id)

        if not instance or instance.state != PluginState.ACTIVE:
            return False

        instance.state = PluginState.SUSPENDED
        self._logger.info("plugin_suspended", plugin_id=plugin_id)
        return True

    async def resume_plugin(self, plugin_id: str) -> bool:
        """Resume a suspended plugin"""
        instance = self._instances.get(plugin_id)

        if not instance or instance.state != PluginState.SUSPENDED:
            return False

        instance.state = PluginState.ACTIVE
        self._logger.info("plugin_resumed", plugin_id=plugin_id)
        return True

    # -------------------------------------------------------------------------
    # Hook Execution
    # -------------------------------------------------------------------------

    async def execute_hook(
        self,
        hook_type: HookType,
        context: Optional[PluginContext] = None
    ) -> List[HookResult]:
        """Execute all hooks of a type"""
        context = context or PluginContext(plugin_id="")
        return await self._hook_executor.execute(hook_type, context)

    async def execute_plugin_hook(
        self,
        hook_type: HookType,
        plugin_id: str,
        context: Optional[PluginContext] = None
    ) -> Optional[HookResult]:
        """Execute hooks from a specific plugin"""
        context = context or PluginContext(plugin_id=plugin_id)
        return await self._hook_executor.execute_single(
            hook_type, context, plugin_id
        )

    def _register_hook(self, hook: PluginHook) -> None:
        """Internal: Register a hook"""
        self._hook_executor.register(hook)

    def _unregister_hook(self, hook_id: str) -> None:
        """Internal: Unregister a hook"""
        self._hook_executor.unregister(hook_id)

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def on_plugin_event(
        self,
        event_type: str,
        handler: Callable[[PluginEvent], Awaitable[None]]
    ) -> None:
        """Register a handler for plugin events"""
        self._event_handlers[event_type].append(handler)

    async def _emit_plugin_event(self, event: PluginEvent) -> None:
        """Internal: Emit a plugin event"""
        handlers = self._event_handlers.get(event.event_type, [])
        handlers.extend(self._event_handlers.get("*", []))  # Wildcard

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                self._logger.error(
                    "event_handler_error",
                    event_type=event.event_type,
                    error=str(e)
                )

    # -------------------------------------------------------------------------
    # Services
    # -------------------------------------------------------------------------

    def register_service(self, name: str, service: Any) -> None:
        """Register a service for plugin access"""
        self._services[name] = service

    def get_service(self, name: str) -> Optional[Any]:
        """Get a registered service"""
        return self._services.get(name)

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    async def get_plugin_storage(
        self,
        plugin_id: str,
        key: str
    ) -> Optional[Any]:
        """Get a value from plugin storage"""
        return self._storage[plugin_id].get(key)

    async def set_plugin_storage(
        self,
        plugin_id: str,
        key: str,
        value: Any
    ) -> None:
        """Set a value in plugin storage"""
        self._storage[plugin_id][key] = value

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    async def update_plugin_settings(
        self,
        plugin_id: str,
        settings: Dict[str, Any]
    ) -> bool:
        """Update plugin settings"""
        instance = self._instances.get(plugin_id)

        if not instance:
            return False

        await instance.on_settings_changed(settings)
        return True

    def get_plugin_settings(
        self,
        plugin_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get plugin settings"""
        instance = self._instances.get(plugin_id)
        return instance.settings if instance else None

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin instance"""
        return self._instances.get(plugin_id)

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        state: Optional[PluginState] = None
    ) -> List[Plugin]:
        """List plugins with optional filtering"""
        plugins = list(self._instances.values())

        if plugin_type:
            plugins = [p for p in plugins if p.config.type == plugin_type]

        if state:
            plugins = [p for p in plugins if p.state == state]

        return plugins

    def get_active_plugins(self) -> List[Plugin]:
        """Get all active plugins"""
        return self.list_plugins(state=PluginState.ACTIVE)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get plugin manager status"""
        return {
            "running": self._running,
            "plugins_total": len(self._instances),
            "plugins_active": len(self.get_active_plugins()),
            "plugins": {
                plugin_id: {
                    "name": plugin.name,
                    "version": plugin.version,
                    "type": plugin.config.type.value,
                    "state": plugin.state.value,
                    "hooks_count": len(plugin._hooks),
                    "metrics": {
                        "hooks_executed": plugin.metrics.hooks_executed,
                        "hooks_failed": plugin.metrics.hooks_failed,
                        "events_emitted": plugin.metrics.events_emitted
                    }
                }
                for plugin_id, plugin in self._instances.items()
            }
        }


# =============================================================================
# DECORATORS
# =============================================================================


def plugin(
    name: str,
    version: str = "1.0.0",
    plugin_type: PluginType = PluginType.CUSTOM,
    **config_kwargs
) -> Callable[[Type[Plugin]], Type[Plugin]]:
    """
    Decorator to configure a plugin class.

    Usage:
        @plugin("my-plugin", version="1.0.0", type=PluginType.SERVICE)
        class MyPlugin(Plugin):
            ...
    """
    def decorator(cls: Type[Plugin]) -> Type[Plugin]:
        # Store config on class
        cls._default_config = PluginConfig(
            name=name,
            version=version,
            type=plugin_type,
            **config_kwargs
        )
        return cls

    return decorator


def hook(
    hook_type: HookType,
    priority: int = 0,
    filter_expression: Optional[str] = None
) -> Callable:
    """
    Decorator to mark a method as a hook handler.

    Usage:
        class MyPlugin(Plugin):
            @hook(HookType.BEFORE_CALL)
            async def handle_before_call(self, ctx: PluginContext) -> HookResult:
                return HookResult(success=True)
    """
    def decorator(func: Callable) -> Callable:
        func._hook_config = {
            "hook_type": hook_type,
            "priority": priority,
            "filter_expression": filter_expression
        }
        return func

    return decorator
