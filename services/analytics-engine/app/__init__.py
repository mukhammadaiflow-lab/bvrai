"""
Analytics Engine Service.

Advanced analytics platform for voice AI applications providing:

1. Call Analytics:
   - Call volume, duration, outcomes
   - Peak time analysis
   - Geographic distribution
   - Retry and callback patterns

2. Conversation Analytics:
   - Intent distribution
   - Sentiment trends
   - Topic analysis
   - Conversation flow analysis
   - Drop-off points

3. Performance Metrics:
   - Latency tracking (ASR, LLM, TTS)
   - Success rates by component
   - Error analysis
   - System health

4. Business Intelligence:
   - Conversion rates
   - Goal completion
   - Customer satisfaction (CSAT)
   - Cost analysis
   - ROI tracking

5. Real-Time Dashboards:
   - Live call monitoring
   - Alert system
   - Custom dashboards
   - Export capabilities

API:
   - GET /metrics/{metric_type} - Get specific metrics
   - POST /events - Ingest analytics events
   - GET /dashboards - Dashboard configurations
   - GET /reports - Generate reports
   - WebSocket /ws/live - Real-time data stream
"""

__version__ = "1.0.0"
