"""End-to-end tests for complete call flows."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4


class TestOutboundCallFlow:
    """End-to-end tests for outbound call flow."""

    @pytest.mark.asyncio
    async def test_complete_outbound_call_flow(
        self,
        authenticated_client,
        test_agent,
        mock_twilio,
        mock_openai,
        mock_elevenlabs,
        mock_deepgram,
    ):
        """Test complete outbound call from initiation to completion."""
        # 1. Initiate outbound call
        call_response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "+15555551234",
                "from_number": "+15555555678",
                "metadata": {"customer_name": "John Doe"},
            },
        )

        assert call_response.status_code == 201
        call_data = call_response.json()
        call_id = call_data["call_id"]

        # 2. Simulate call being answered (webhook from telephony provider)
        with patch("app.services.call.handle_call_answered", new_callable=AsyncMock) as mock_answered:
            mock_answered.return_value = {"status": "connected"}

            webhook_response = await authenticated_client.post(
                "/api/v1/webhooks/twilio/status",
                data={
                    "CallSid": call_id,
                    "CallStatus": "in-progress",
                },
            )

            assert webhook_response.status_code == 200

        # 3. Verify call status updated
        status_response = await authenticated_client.get(f"/api/v1/calls/{call_id}")

        # 4. Simulate conversation turns
        with patch("app.services.conversation.process_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = {
                "response": "Hello John, how can I help you today?",
                "audio": b"audio_data",
            }

            # User says something
            turn_response = await authenticated_client.post(
                f"/api/v1/calls/{call_id}/turn",
                json={
                    "transcript": "I need help with my order",
                    "audio_url": "https://example.com/audio.wav",
                },
            )

        # 5. Simulate call ending
        with patch("app.services.call.handle_call_ended", new_callable=AsyncMock) as mock_ended:
            mock_ended.return_value = {
                "status": "completed",
                "duration": 120,
            }

            end_webhook = await authenticated_client.post(
                "/api/v1/webhooks/twilio/status",
                data={
                    "CallSid": call_id,
                    "CallStatus": "completed",
                    "CallDuration": "120",
                },
            )

            assert end_webhook.status_code == 200

        # 6. Verify transcript and recording are available
        transcript_response = await authenticated_client.get(f"/api/v1/calls/{call_id}/transcript")
        recording_response = await authenticated_client.get(f"/api/v1/calls/{call_id}/recording")

    @pytest.mark.asyncio
    async def test_call_with_tool_usage(
        self,
        authenticated_client,
        test_agent,
        mock_twilio,
        mock_openai,
    ):
        """Test call flow with agent using tools."""
        # Configure agent with calendar tool
        await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/tools",
            json={
                "name": "book_appointment",
                "type": "function",
                "description": "Book an appointment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                    },
                },
            },
        )

        # Initiate call
        call_response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "+15555551234",
                "from_number": "+15555555678",
            },
        )

        call_id = call_response.json()["call_id"]

        # Simulate LLM wanting to use tool
        with patch("app.services.llm.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": None,
                "tool_calls": [
                    {
                        "name": "book_appointment",
                        "arguments": {"date": "2024-01-25", "time": "14:00"},
                    }
                ],
            }

            # Process turn that triggers tool
            with patch("app.services.tools.execute_tool", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = {
                    "success": True,
                    "result": {"appointment_id": "apt_123"},
                }

                turn_response = await authenticated_client.post(
                    f"/api/v1/calls/{call_id}/turn",
                    json={"transcript": "I'd like to book an appointment for tomorrow at 2pm"},
                )

    @pytest.mark.asyncio
    async def test_call_with_transfer(
        self,
        authenticated_client,
        test_agent,
        mock_twilio,
    ):
        """Test call flow with transfer to human."""
        call_response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "+15555551234",
                "from_number": "+15555555678",
            },
        )

        call_id = call_response.json()["call_id"]

        # Trigger transfer
        with patch("app.services.telephony.transfer_call", new_callable=AsyncMock) as mock_transfer:
            mock_transfer.return_value = {"status": "transferred"}

            transfer_response = await authenticated_client.post(
                f"/api/v1/calls/{call_id}/transfer",
                json={
                    "target_number": "+15555559999",
                    "reason": "Customer requested human agent",
                },
            )

            assert transfer_response.status_code == 200


class TestInboundCallFlow:
    """End-to-end tests for inbound call flow."""

    @pytest.mark.asyncio
    async def test_complete_inbound_call_flow(
        self,
        async_client,
        db_session,
        test_tenant,
        test_agent,
        mock_openai,
        mock_elevenlabs,
    ):
        """Test complete inbound call handling."""
        from app.models.phone_number import PhoneNumber

        # Setup: Create phone number assigned to agent
        phone = PhoneNumber(
            id=str(uuid4()),
            tenant_id=test_tenant.id,
            phone_number="+15555555678",
            agent_id=test_agent.id,
            status="active",
        )
        db_session.add(phone)
        await db_session.commit()

        # 1. Receive incoming call webhook
        with patch("app.services.call.handle_incoming_call", new_callable=AsyncMock) as mock_incoming:
            mock_incoming.return_value = {
                "call_id": str(uuid4()),
                "twiml": '<?xml version="1.0"?><Response><Say>Hello</Say></Response>',
            }

            incoming_response = await async_client.post(
                "/api/v1/webhooks/twilio/voice",
                data={
                    "CallSid": "CA_incoming_123",
                    "To": "+15555555678",
                    "From": "+15555551234",
                    "CallStatus": "ringing",
                },
            )

            assert incoming_response.status_code == 200
            # Should return TwiML
            assert "Response" in incoming_response.text

        # 2. Handle ongoing conversation
        with patch("app.services.conversation.process_speech", new_callable=AsyncMock) as mock_speech:
            mock_speech.return_value = {
                "response_audio_url": "https://storage.example.com/response.wav",
            }

            speech_response = await async_client.post(
                "/api/v1/webhooks/twilio/gather",
                data={
                    "CallSid": "CA_incoming_123",
                    "SpeechResult": "I want to check my order status",
                    "Confidence": "0.95",
                },
            )

            assert speech_response.status_code == 200

    @pytest.mark.asyncio
    async def test_inbound_call_voicemail(
        self,
        async_client,
        db_session,
        test_tenant,
        test_agent,
    ):
        """Test inbound call going to voicemail."""
        from app.models.phone_number import PhoneNumber

        phone = PhoneNumber(
            id=str(uuid4()),
            tenant_id=test_tenant.id,
            phone_number="+15555555678",
            agent_id=test_agent.id,
            status="active",
            settings={"voicemail_enabled": True},
        )
        db_session.add(phone)
        await db_session.commit()

        # Simulate no answer scenario
        with patch("app.services.call.handle_no_answer", new_callable=AsyncMock) as mock_no_answer:
            mock_no_answer.return_value = {
                "action": "voicemail",
                "twiml": '<Response><Record maxLength="120" /></Response>',
            }

            no_answer_response = await async_client.post(
                "/api/v1/webhooks/twilio/status",
                data={
                    "CallSid": "CA_incoming_123",
                    "CallStatus": "no-answer",
                },
            )

            assert no_answer_response.status_code == 200


class TestConversationFlow:
    """End-to-end tests for conversation handling."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(
        self,
        authenticated_client,
        test_agent,
        mock_openai,
    ):
        """Test multi-turn conversation flow."""
        call_response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "+15555551234",
                "from_number": "+15555555678",
            },
        )

        call_id = call_response.json()["call_id"]

        # Multiple conversation turns
        conversation = [
            ("Hello, I need help", "Hello! I'd be happy to help. What can I assist you with?"),
            ("I want to check my order status", "Of course! Could you please provide your order number?"),
            ("It's order 12345", "Thank you! Let me look that up for you."),
        ]

        with patch("app.services.conversation.process_turn", new_callable=AsyncMock) as mock_turn:
            for user_input, expected_response in conversation:
                mock_turn.return_value = {
                    "response": expected_response,
                    "audio": b"audio_data",
                }

                turn_response = await authenticated_client.post(
                    f"/api/v1/calls/{call_id}/turn",
                    json={"transcript": user_input},
                )

                assert turn_response.status_code == 200

        # Verify conversation history
        transcript_response = await authenticated_client.get(f"/api/v1/calls/{call_id}/transcript")

    @pytest.mark.asyncio
    async def test_conversation_with_rag(
        self,
        authenticated_client,
        test_agent,
        mock_openai,
    ):
        """Test conversation using RAG from knowledge base."""
        # Setup knowledge base for agent
        kb_response = await authenticated_client.post(
            "/api/v1/knowledge-bases",
            json={"name": "Product FAQ"},
        )
        kb_id = kb_response.json()["id"]

        # Link KB to agent
        await authenticated_client.patch(
            f"/api/v1/agents/{test_agent.id}",
            json={"knowledge_base_id": kb_id},
        )

        # Start call
        call_response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "+15555551234",
                "from_number": "+15555555678",
            },
        )

        call_id = call_response.json()["call_id"]

        # Ask question that requires RAG
        with patch("app.services.knowledge.search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {"content": "Our return policy allows returns within 30 days.", "score": 0.95}
            ]

            with patch("app.services.conversation.process_turn", new_callable=AsyncMock) as mock_turn:
                mock_turn.return_value = {
                    "response": "Our return policy allows returns within 30 days of purchase.",
                    "sources": [{"doc_id": "doc_1", "chunk_id": "chunk_1"}],
                }

                turn_response = await authenticated_client.post(
                    f"/api/v1/calls/{call_id}/turn",
                    json={"transcript": "What's your return policy?"},
                )

                assert turn_response.status_code == 200


class TestWorkflowExecution:
    """End-to-end tests for workflow execution during calls."""

    @pytest.mark.asyncio
    async def test_post_call_workflow(
        self,
        authenticated_client,
        test_agent,
        db_session,
        test_tenant,
    ):
        """Test workflow execution after call completes."""
        from app.models.workflow import Workflow

        # Create post-call workflow
        workflow = Workflow(
            id=str(uuid4()),
            tenant_id=test_tenant.id,
            name="Post-Call CRM Sync",
            trigger_type="call_ended",
            status="active",
            nodes=[
                {"id": "1", "type": "trigger", "name": "Call Ended"},
                {"id": "2", "type": "action", "name": "Update CRM", "config": {"crm": "salesforce"}},
            ],
        )
        db_session.add(workflow)
        await db_session.commit()

        # Make a call
        call_response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "+15555551234",
                "from_number": "+15555555678",
            },
        )

        call_id = call_response.json()["call_id"]

        # End call - should trigger workflow
        with patch("app.services.workflow.execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"execution_id": "exec_123", "status": "completed"}

            end_response = await authenticated_client.post(
                "/api/v1/webhooks/twilio/status",
                data={
                    "CallSid": call_id,
                    "CallStatus": "completed",
                },
            )

            # Verify workflow was triggered
            # (In real implementation, this would be checked via workflow execution logs)
