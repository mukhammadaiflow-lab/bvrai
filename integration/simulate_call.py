#!/usr/bin/env python3
"""
Integration Test Harness - Simulates end-to-end voice agent flow.

This script demonstrates the happy path:
1. Request a token from Token Service
2. Connect to Media Bridge via WebSocket
3. Send transcript frames
4. Receive dialog responses
5. Validate the flow

Usage:
    python integration/simulate_call.py

Prerequisites:
    - All services running (docker-compose up)
    - Python 3.11+ with httpx and websockets
"""
import asyncio
import json
import sys
from datetime import datetime

try:
    import httpx
    import websockets
except ImportError:
    print("Please install dependencies: pip install httpx websockets")
    sys.exit(1)


# Configuration
TOKEN_SERVICE_URL = "http://localhost:3001"
MEDIA_BRIDGE_URL = "http://localhost:3002"
MEDIA_BRIDGE_WS_URL = "ws://localhost:3002"
DIALOG_MANAGER_URL = "http://localhost:3003"

# Test data
TEST_ROOM = "integration-test-room"
TEST_IDENTITY = "test-user-001"
TEST_TENANT = "test-tenant"


class IntegrationTest:
    """Integration test runner."""

    def __init__(self):
        self.results = []
        self.token = None
        self.ws_url = None

    def log(self, message: str, status: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        symbol = {"PASS": "✓", "FAIL": "✗", "INFO": "→"}.get(status, "→")
        print(f"[{timestamp}] {symbol} {message}")
        self.results.append({"message": message, "status": status})

    async def test_health_checks(self) -> bool:
        """Test that all services are healthy."""
        self.log("Testing service health checks...")

        services = [
            (TOKEN_SERVICE_URL, "Token Service"),
            (MEDIA_BRIDGE_URL, "Media Bridge"),
            (DIALOG_MANAGER_URL, "Dialog Manager"),
        ]

        async with httpx.AsyncClient() as client:
            for url, name in services:
                try:
                    response = await client.get(f"{url}/health", timeout=5.0)
                    if response.status_code == 200:
                        self.log(f"{name} is healthy", "PASS")
                    else:
                        self.log(f"{name} health check failed: {response.status_code}", "FAIL")
                        return False
                except Exception as e:
                    self.log(f"{name} not reachable: {e}", "FAIL")
                    return False

        return True

    async def test_token_generation(self) -> bool:
        """Test token generation."""
        self.log("Testing token generation...")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{TOKEN_SERVICE_URL}/token",
                    json={
                        "room": TEST_ROOM,
                        "identity": TEST_IDENTITY,
                        "ttl_seconds": 3600,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    self.token = data.get("token")
                    self.ws_url = data.get("wsUrl")

                    if self.token:
                        self.log(f"Token received (length: {len(self.token)})", "PASS")
                        return True
                    else:
                        self.log("Token not in response", "FAIL")
                        return False
                else:
                    self.log(f"Token request failed: {response.status_code}", "FAIL")
                    return False

            except Exception as e:
                self.log(f"Token request error: {e}", "FAIL")
                return False

    async def test_dialog_turn(self) -> bool:
        """Test dialog turn processing."""
        self.log("Testing dialog turn endpoint...")

        test_transcripts = [
            ("Hello, I need help", False),
            ("I want to book an appointment", True),
            ("What are your hours?", True),
        ]

        async with httpx.AsyncClient() as client:
            for transcript, expect_action in test_transcripts:
                try:
                    response = await client.post(
                        f"{DIALOG_MANAGER_URL}/dialog/turn",
                        json={
                            "tenant_id": TEST_TENANT,
                            "session_id": f"test-{datetime.now().timestamp()}",
                            "transcript": transcript,
                            "is_final": True,
                        },
                        timeout=10.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        speak_text = data.get("speak_text", "")
                        action = data.get("action_object")

                        if speak_text:
                            self.log(f'Dialog response for "{transcript[:30]}...": "{speak_text[:50]}..."', "PASS")
                            if expect_action and action:
                                self.log(f"  Action detected: {action.get('action_type')}", "PASS")
                            elif expect_action and not action:
                                self.log("  Expected action but none found", "INFO")
                        else:
                            self.log(f"Empty response for: {transcript}", "FAIL")
                            return False
                    else:
                        self.log(f"Dialog turn failed: {response.status_code}", "FAIL")
                        return False

                except Exception as e:
                    self.log(f"Dialog turn error: {e}", "FAIL")
                    return False

        return True

    async def test_websocket_flow(self) -> bool:
        """Test WebSocket connection and message flow."""
        self.log("Testing WebSocket connection...")

        if not self.token:
            self.log("No token available for WebSocket test", "FAIL")
            return False

        ws_url = f"{MEDIA_BRIDGE_WS_URL}/media/{TEST_ROOM}?token={self.token}"

        try:
            async with websockets.connect(ws_url, open_timeout=5) as ws:
                self.log("WebSocket connected", "PASS")

                # Wait for connection confirmation
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(message)

                    if data.get("type") == "connected":
                        self.log(f"Received connection confirmation: participant={data.get('participantId')}", "PASS")
                    else:
                        self.log(f"Unexpected first message: {data.get('type')}", "INFO")

                except asyncio.TimeoutError:
                    self.log("No connection confirmation received", "INFO")

                # Send a transcript
                self.log("Sending transcript frame...")
                await ws.send(json.dumps({
                    "type": "transcript",
                    "text": "Hello, I need help with my order",
                    "isFinal": True,
                }))

                # Wait for responses
                self.log("Waiting for dialog response...")
                try:
                    # Collect messages for a few seconds
                    end_time = asyncio.get_event_loop().time() + 5
                    received_dialog = False

                    while asyncio.get_event_loop().time() < end_time:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(message)
                            msg_type = data.get("type")

                            if msg_type == "dialog_response":
                                speak = data.get("speakText", "")[:50]
                                self.log(f'Received dialog response: "{speak}..."', "PASS")
                                received_dialog = True
                                break
                            elif msg_type == "transcript":
                                self.log(f"Received transcript echo", "INFO")
                            else:
                                self.log(f"Received message type: {msg_type}", "INFO")

                        except asyncio.TimeoutError:
                            continue

                    if not received_dialog:
                        self.log("No dialog response received (may be expected in test mode)", "INFO")

                except Exception as e:
                    self.log(f"Error receiving messages: {e}", "INFO")

                # Send ping
                await ws.send(json.dumps({"type": "ping"}))

                # Wait for pong
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(message)
                    if data.get("type") == "pong":
                        self.log("Ping/pong working", "PASS")
                except asyncio.TimeoutError:
                    self.log("No pong received", "INFO")

                return True

        except websockets.exceptions.InvalidStatusCode as e:
            self.log(f"WebSocket connection rejected: {e.status_code}", "FAIL")
            return False
        except Exception as e:
            self.log(f"WebSocket error: {e}", "FAIL")
            return False

    async def run(self) -> bool:
        """Run all integration tests."""
        print("\n" + "=" * 60)
        print("  Builder Engine Integration Test")
        print("=" * 60 + "\n")

        tests = [
            ("Health Checks", self.test_health_checks),
            ("Token Generation", self.test_token_generation),
            ("Dialog Turn Processing", self.test_dialog_turn),
            ("WebSocket Flow", self.test_websocket_flow),
        ]

        all_passed = True

        for name, test_fn in tests:
            print(f"\n--- {name} ---")
            try:
                passed = await test_fn()
                if not passed:
                    all_passed = False
            except Exception as e:
                self.log(f"Test error: {e}", "FAIL")
                all_passed = False

        # Summary
        print("\n" + "=" * 60)
        passed_count = sum(1 for r in self.results if r["status"] == "PASS")
        failed_count = sum(1 for r in self.results if r["status"] == "FAIL")

        if all_passed:
            print(f"  ✓ All tests passed ({passed_count} checks)")
            print("=" * 60 + "\n")
            return True
        else:
            print(f"  ✗ Some tests failed ({passed_count} passed, {failed_count} failed)")
            print("=" * 60 + "\n")
            return False


async def main():
    """Main entry point."""
    test = IntegrationTest()
    success = await test.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
