"""Audio splitter for routing to multiple destinations."""

import asyncio
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
import structlog
import copy


logger = structlog.get_logger()


@dataclass
class SplitterOutput:
    """Splitter output configuration."""
    id: str
    callback: Optional[Callable[[bytes], Any]] = None
    queue: Optional[asyncio.Queue] = None
    enabled: bool = True
    transform: Optional[Callable[[bytes], bytes]] = None


class AudioSplitter:
    """
    Splits audio stream to multiple destinations.

    Use cases:
    - Send to ASR and recording simultaneously
    - Send to multiple monitors
    - Fork for A/B testing
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._outputs: Dict[str, SplitterOutput] = {}

        # Statistics
        self._frames_split = 0
        self._bytes_distributed = 0

    def add_output(
        self,
        output_id: str,
        callback: Optional[Callable[[bytes], Any]] = None,
        queue: Optional[asyncio.Queue] = None,
        transform: Optional[Callable[[bytes], bytes]] = None,
    ) -> bool:
        """
        Add an output destination.

        Args:
            output_id: Unique output identifier
            callback: Async callback function
            queue: Queue to put audio into
            transform: Optional transform function

        Returns:
            True if output added
        """
        if output_id in self._outputs:
            return False

        self._outputs[output_id] = SplitterOutput(
            id=output_id,
            callback=callback,
            queue=queue,
            transform=transform,
        )

        logger.debug(
            "splitter_output_added",
            session_id=self.session_id,
            output_id=output_id,
        )
        return True

    def remove_output(self, output_id: str) -> bool:
        """Remove an output destination."""
        if output_id in self._outputs:
            del self._outputs[output_id]
            return True
        return False

    async def split(self, audio_data: bytes) -> Dict[str, bool]:
        """
        Split audio to all outputs.

        Args:
            audio_data: Audio data to split

        Returns:
            Dict of output_id -> success
        """
        results = {}
        self._frames_split += 1

        for output_id, output in self._outputs.items():
            if not output.enabled:
                results[output_id] = False
                continue

            try:
                # Apply transform if present
                data = output.transform(audio_data) if output.transform else audio_data

                # Send to destination
                if output.callback:
                    result = output.callback(data)
                    if asyncio.iscoroutine(result):
                        await result
                    results[output_id] = True
                elif output.queue:
                    try:
                        output.queue.put_nowait(data)
                        results[output_id] = True
                    except asyncio.QueueFull:
                        results[output_id] = False
                else:
                    results[output_id] = False

                self._bytes_distributed += len(data)

            except Exception as e:
                logger.error(
                    "splitter_output_error",
                    session_id=self.session_id,
                    output_id=output_id,
                    error=str(e),
                )
                results[output_id] = False

        return results

    def set_output_enabled(self, output_id: str, enabled: bool) -> bool:
        """Enable or disable an output."""
        if output_id in self._outputs:
            self._outputs[output_id].enabled = enabled
            return True
        return False

    def get_statistics(self) -> dict:
        """Get splitter statistics."""
        return {
            "session_id": self.session_id,
            "output_count": len(self._outputs),
            "frames_split": self._frames_split,
            "bytes_distributed": self._bytes_distributed,
            "outputs": {
                oid: {"enabled": o.enabled}
                for oid, o in self._outputs.items()
            },
        }


class ConditionalSplitter:
    """
    Conditional audio splitter based on audio properties.

    Routes audio based on conditions like:
    - Voice activity
    - Energy level
    - Custom predicates
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._routes: Dict[str, tuple] = {}  # id -> (condition, destination)

    def add_route(
        self,
        route_id: str,
        condition: Callable[[bytes], bool],
        destination: Callable[[bytes], Any],
    ) -> None:
        """
        Add a conditional route.

        Args:
            route_id: Unique route identifier
            condition: Function that returns True if audio should be routed
            destination: Async function to send audio to
        """
        self._routes[route_id] = (condition, destination)

    def remove_route(self, route_id: str) -> bool:
        """Remove a route."""
        if route_id in self._routes:
            del self._routes[route_id]
            return True
        return False

    async def route(self, audio_data: bytes) -> List[str]:
        """
        Route audio based on conditions.

        Args:
            audio_data: Audio to route

        Returns:
            List of route IDs that were triggered
        """
        triggered = []

        for route_id, (condition, destination) in self._routes.items():
            try:
                if condition(audio_data):
                    result = destination(audio_data)
                    if asyncio.iscoroutine(result):
                        await result
                    triggered.append(route_id)
            except Exception as e:
                logger.error(
                    "conditional_route_error",
                    session_id=self.session_id,
                    route_id=route_id,
                    error=str(e),
                )

        return triggered
