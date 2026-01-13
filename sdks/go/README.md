# Builder Engine Go SDK

Official Go SDK for the Builder Engine AI Voice Agent Platform.

[![Go Reference](https://pkg.go.dev/badge/github.com/builderengine/sdk-go.svg)](https://pkg.go.dev/github.com/builderengine/sdk-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/builderengine/sdk-go)](https://goreportcard.com/report/github.com/builderengine/sdk-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
go get github.com/builderengine/sdk-go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/builderengine/sdk-go/builderengine"
)

func main() {
    // Create client
    client, err := builderengine.NewClient("your-api-key")
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    // Create an AI voice agent
    agent, err := client.Agents.Create(ctx, builderengine.AgentCreateParams{
        Name:         "Support Agent",
        Voice:        "nova",
        SystemPrompt: "You are a helpful customer support agent.",
        Model:        "gpt-4-turbo",
        Language:     "en-US",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Make an outbound call
    call, err := client.Calls.Create(ctx, builderengine.CallCreateParams{
        AgentID:  agent.ID,
        ToNumber: "+14155551234",
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Call started: %s\n", call.ID)
}
```

## Features

- **Full API Coverage**: Access all Builder Engine APIs
- **Type Safe**: Full Go type definitions
- **Context Support**: Full context.Context support for cancellation
- **Retry Logic**: Built-in retry with exponential backoff
- **WebSocket Streaming**: Real-time event streaming
- **Error Handling**: Typed errors for different scenarios

## Usage Examples

### Managing Agents

```go
ctx := context.Background()

// List all agents
agents, err := client.Agents.List(ctx, builderengine.AgentListParams{
    ListParams: builderengine.ListParams{Limit: 10},
    Status:     builderengine.AgentStatusActive,
})

// Get a specific agent
agent, err := client.Agents.Get(ctx, "agent_abc123")

// Update an agent
temperature := 0.8
updated, err := client.Agents.Update(ctx, "agent_abc123", builderengine.AgentUpdateParams{
    Temperature: &temperature,
})

// Delete an agent
err = client.Agents.Delete(ctx, "agent_abc123")
```

### Making Calls

```go
ctx := context.Background()

// Create an outbound call
call, err := client.Calls.Create(ctx, builderengine.CallCreateParams{
    AgentID:    "agent_abc123",
    ToNumber:   "+14155551234",
    FromNumber: "+14155550000",
    Metadata: map[string]interface{}{
        "customer_id": "cust_123",
    },
})

// Get call status
status, err := client.Calls.Get(ctx, call.ID)
fmt.Printf("Call status: %s, duration: %d\n", status.Status, status.Duration)

// List recent calls
calls, err := client.Calls.List(ctx, builderengine.CallListParams{
    Status: builderengine.CallStatusCompleted,
    ListParams: builderengine.ListParams{Limit: 50},
})

// Get call transcript
transcript, err := client.Calls.GetTranscript(ctx, call.ID)
for _, msg := range transcript.Messages {
    fmt.Printf("[%s]: %s\n", msg.Role, msg.Content)
}

// End an active call
err = client.Calls.End(ctx, call.ID)
```

### Real-time Streaming

```go
// Create streaming client
streaming := client.Streaming()

// Register event handlers
streaming.On(builderengine.StreamEventTranscriptionFinal, func(event builderengine.StreamEvent) {
    fmt.Printf("User said: %v\n", event.Data["text"])
})

streaming.On(builderengine.StreamEventAgentSpeechStart, func(event builderengine.StreamEvent) {
    fmt.Println("Agent is speaking...")
})

streaming.On(builderengine.StreamEventCallEnded, func(event builderengine.StreamEvent) {
    fmt.Printf("Call ended: %v\n", event.Data["reason"])
})

// Connect to a specific call
err := streaming.Connect(builderengine.ConnectOptions{
    CallID: "call_abc123",
})

// Or subscribe to all events
err := streaming.Connect(builderengine.ConnectOptions{
    SubscribeAll: true,
})

// Inject text for agent to speak
streaming.InjectText("call_abc123", "Please hold while I check that.", true)

// Disconnect when done
streaming.Disconnect()
```

### Phone Numbers

```go
ctx := context.Background()

// List available numbers
available, err := client.PhoneNumbers.ListAvailable(ctx, builderengine.AvailableNumbersParams{
    Country:  "US",
    AreaCode: "415",
})

// Purchase a number
number, err := client.PhoneNumbers.Purchase(ctx, builderengine.PurchaseParams{
    PhoneNumber:  "+14155551234",
    FriendlyName: "Support Line",
})

// Configure with an agent
agentID := "agent_abc123"
updated, err := client.PhoneNumbers.Update(ctx, number.ID, builderengine.PhoneNumberUpdateParams{
    AgentID: &agentID,
})

// Release a number
err = client.PhoneNumbers.Release(ctx, number.ID)
```

### Campaigns (Batch Calls)

```go
ctx := context.Background()

// Create a campaign
campaign, err := client.Campaigns.Create(ctx, builderengine.CampaignCreateParams{
    Name:       "January Outreach",
    AgentID:    "agent_abc123",
    FromNumber: "+14155550000",
    Contacts: []builderengine.ContactInput{
        {PhoneNumber: "+14155551111", Name: "John Doe"},
        {PhoneNumber: "+14155552222", Name: "Jane Smith"},
    },
    Schedule: &builderengine.CampaignSchedule{
        StartTime:     "2024-01-15T09:00:00Z",
        EndTime:       "2024-01-15T17:00:00Z",
        Timezone:      "America/Los_Angeles",
        MaxConcurrent: 10,
    },
})

// Start the campaign
started, err := client.Campaigns.Start(ctx, campaign.ID)

// Get progress
progress, err := client.Campaigns.GetProgress(ctx, campaign.ID)
fmt.Printf("Completed: %d/%d\n", progress.Completed, progress.Total)

// Pause/resume
client.Campaigns.Pause(ctx, campaign.ID)
client.Campaigns.Resume(ctx, campaign.ID)

// Cancel
client.Campaigns.Cancel(ctx, campaign.ID)
```

### Knowledge Base

```go
ctx := context.Background()

// Create a knowledge base
kb, err := client.KnowledgeBase.Create(ctx, builderengine.KnowledgeBaseCreateParams{
    Name:        "Product Documentation",
    Description: "Company product docs and FAQs",
})

// Add a document
doc, err := client.KnowledgeBase.AddDocument(ctx, kb.ID, builderengine.DocumentCreateParams{
    Title:   "Return Policy",
    Content: "Our return policy allows returns within 30 days...",
    Metadata: map[string]interface{}{
        "category": "policies",
    },
})

// Search the knowledge base
results, err := client.KnowledgeBase.Search(ctx, kb.ID, builderengine.SearchParams{
    Query: "How do I return a product?",
    Limit: 5,
})

for _, result := range results {
    fmt.Printf("Score: %.2f - %s\n", result.Score, result.Title)
}
```

### Analytics

```go
ctx := context.Background()

// Get overview analytics
overview, err := client.Analytics.GetOverview(ctx, builderengine.AnalyticsParams{
    Period:    "week",
    StartDate: "2024-01-01",
    EndDate:   "2024-01-07",
})

fmt.Printf("Total calls: %d\n", overview.TotalCalls)
fmt.Printf("Success rate: %.1f%%\n", overview.SuccessRate*100)

// Get usage metrics
usage, err := client.Analytics.GetUsage(ctx, "month")
fmt.Printf("Minutes used: %.1f\n", usage.Minutes)

// Export a report
report, err := client.Analytics.Export(ctx, builderengine.ExportParams{
    ReportType: "calls",
    Period:     "month",
    Format:     "csv",
})
```

## Error Handling

The SDK provides typed errors for different scenarios:

```go
call, err := client.Calls.Create(ctx, builderengine.CallCreateParams{
    AgentID: "invalid",
    ToNumber: "+14155551234",
})

if err != nil {
    switch {
    case builderengine.IsAuthenticationError(err):
        log.Println("Invalid API key")
    case builderengine.IsRateLimitError(err):
        if rlErr, ok := err.(*builderengine.RateLimitError); ok {
            log.Printf("Rate limited. Retry after %d seconds", rlErr.RetryAfter)
        }
    case builderengine.IsValidationError(err):
        if vErr, ok := err.(*builderengine.ValidationError); ok {
            log.Printf("Validation failed: %v", vErr.Errors)
        }
    case builderengine.IsNotFoundError(err):
        log.Println("Resource not found")
    case builderengine.IsInsufficientCreditsError(err):
        log.Println("Add credits to your account")
    default:
        log.Printf("API error: %v", err)
    }
}
```

## Configuration Options

```go
client, err := builderengine.NewClient("your-api-key",
    // Custom base URL (for enterprise/self-hosted)
    builderengine.WithBaseURL("https://api.your-domain.com"),

    // Request timeout
    builderengine.WithTimeout(60 * time.Second),

    // Maximum retry attempts
    builderengine.WithMaxRetries(5),

    // Custom HTTP client
    builderengine.WithHTTPClient(&http.Client{
        Transport: &http.Transport{
            MaxIdleConns: 100,
        },
    }),
)
```

## Context Support

All API methods accept a `context.Context` for cancellation and timeouts:

```go
// With timeout
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

agent, err := client.Agents.Get(ctx, "agent_123")

// With cancellation
ctx, cancel := context.WithCancel(context.Background())
go func() {
    time.Sleep(5 * time.Second)
    cancel()
}()

calls, err := client.Calls.List(ctx, builderengine.CallListParams{})
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- Documentation: https://docs.builderengine.io
- API Reference: https://docs.builderengine.io/api
- Discord: https://discord.gg/builderengine
- Email: support@builderengine.io
