package builderengine

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	t.Run("creates client with API key", func(t *testing.T) {
		client, err := NewClient("test-api-key")
		require.NoError(t, err)
		assert.NotNil(t, client)
	})

	t.Run("returns error without API key", func(t *testing.T) {
		client, err := NewClient("")
		assert.Error(t, err)
		assert.Nil(t, client)
	})

	t.Run("applies custom options", func(t *testing.T) {
		client, err := NewClient("test-api-key",
			WithBaseURL("https://custom.api.com"),
			WithTimeout(60*time.Second),
			WithMaxRetries(5),
		)
		require.NoError(t, err)
		assert.NotNil(t, client)
	})
}

func TestClientResources(t *testing.T) {
	client, err := NewClient("test-api-key")
	require.NoError(t, err)

	t.Run("has agents resource", func(t *testing.T) {
		assert.NotNil(t, client.Agents)
	})

	t.Run("has calls resource", func(t *testing.T) {
		assert.NotNil(t, client.Calls)
	})

	t.Run("has conversations resource", func(t *testing.T) {
		assert.NotNil(t, client.Conversations)
	})

	t.Run("has phoneNumbers resource", func(t *testing.T) {
		assert.NotNil(t, client.PhoneNumbers)
	})

	t.Run("has voices resource", func(t *testing.T) {
		assert.NotNil(t, client.Voices)
	})

	t.Run("has webhooks resource", func(t *testing.T) {
		assert.NotNil(t, client.Webhooks)
	})

	t.Run("has knowledgeBase resource", func(t *testing.T) {
		assert.NotNil(t, client.KnowledgeBase)
	})

	t.Run("has workflows resource", func(t *testing.T) {
		assert.NotNil(t, client.Workflows)
	})

	t.Run("has campaigns resource", func(t *testing.T) {
		assert.NotNil(t, client.Campaigns)
	})

	t.Run("has analytics resource", func(t *testing.T) {
		assert.NotNil(t, client.Analytics)
	})

	t.Run("has organizations resource", func(t *testing.T) {
		assert.NotNil(t, client.Organizations)
	})

	t.Run("has users resource", func(t *testing.T) {
		assert.NotNil(t, client.Users)
	})

	t.Run("has apiKeys resource", func(t *testing.T) {
		assert.NotNil(t, client.APIKeys)
	})

	t.Run("has billing resource", func(t *testing.T) {
		assert.NotNil(t, client.Billing)
	})
}

func TestClientRequest(t *testing.T) {
	t.Run("makes successful GET request", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "GET", r.Method)
			assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
			assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":   "agent_123",
				"name": "Test Agent",
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		var result map[string]interface{}
		err := client.Request(context.Background(), RequestOptions{
			Method: "GET",
			Path:   "/api/v1/agents/agent_123",
		}, &result)

		require.NoError(t, err)
		assert.Equal(t, "agent_123", result["id"])
	})

	t.Run("makes successful POST request", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)

			var body map[string]interface{}
			json.NewDecoder(r.Body).Decode(&body)
			assert.Equal(t, "New Agent", body["name"])

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":   "agent_new",
				"name": "New Agent",
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		var result map[string]interface{}
		err := client.Request(context.Background(), RequestOptions{
			Method: "POST",
			Path:   "/api/v1/agents",
			Body:   map[string]string{"name": "New Agent"},
		}, &result)

		require.NoError(t, err)
		assert.Equal(t, "agent_new", result["id"])
	})

	t.Run("includes query params", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "10", r.URL.Query().Get("limit"))
			assert.Equal(t, "active", r.URL.Query().Get("status"))

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"data": []interface{}{}})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		var result map[string]interface{}
		err := client.Request(context.Background(), RequestOptions{
			Method: "GET",
			Path:   "/api/v1/agents",
			Params: map[string]string{"limit": "10", "status": "active"},
		}, &result)

		require.NoError(t, err)
	})
}

func TestErrorHandling(t *testing.T) {
	t.Run("handles 401 authentication error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": "Invalid API key",
					"code":    "invalid_api_key",
				},
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		var result map[string]interface{}
		err := client.Request(context.Background(), RequestOptions{
			Method: "GET",
			Path:   "/api/v1/agents",
		}, &result)

		require.Error(t, err)
		assert.True(t, IsAuthenticationError(err))
	})

	t.Run("handles 404 not found error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": "Agent not found",
				},
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		var result map[string]interface{}
		err := client.Request(context.Background(), RequestOptions{
			Method: "GET",
			Path:   "/api/v1/agents/nonexistent",
		}, &result)

		require.Error(t, err)
		assert.True(t, IsNotFoundError(err))
	})

	t.Run("handles 422 validation error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnprocessableEntity)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": "Validation failed",
					"details": map[string]interface{}{
						"name": "Name is required",
					},
				},
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		var result map[string]interface{}
		err := client.Request(context.Background(), RequestOptions{
			Method: "POST",
			Path:   "/api/v1/agents",
			Body:   map[string]string{},
		}, &result)

		require.Error(t, err)
		assert.True(t, IsValidationError(err))
	})

	t.Run("handles 429 rate limit error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Retry-After", "30")
			w.WriteHeader(http.StatusTooManyRequests)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": "Rate limit exceeded",
				},
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL), WithMaxRetries(0))

		var result map[string]interface{}
		err := client.Request(context.Background(), RequestOptions{
			Method: "GET",
			Path:   "/api/v1/agents",
		}, &result)

		require.Error(t, err)
		assert.True(t, IsRateLimitError(err))
		if rlErr, ok := err.(*RateLimitError); ok {
			assert.Equal(t, 30, rlErr.RetryAfter)
		}
	})
}

func TestAgentsService(t *testing.T) {
	t.Run("lists agents", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data":   []map[string]interface{}{{"id": "agent_1"}, {"id": "agent_2"}},
				"total":  2,
				"limit":  10,
				"offset": 0,
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		result, err := client.Agents.List(context.Background(), AgentListParams{})

		require.NoError(t, err)
		assert.Len(t, result.Data, 2)
		assert.Equal(t, 2, result.Total)
	})

	t.Run("creates agent", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]interface{}
			json.NewDecoder(r.Body).Decode(&body)
			assert.Equal(t, "Test Agent", body["name"])

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":   "agent_new",
				"name": "Test Agent",
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		result, err := client.Agents.Create(context.Background(), AgentCreateParams{
			Name:         "Test Agent",
			Voice:        "nova",
			SystemPrompt: "You are helpful.",
		})

		require.NoError(t, err)
		assert.Equal(t, "agent_new", result.ID)
	})

	t.Run("gets agent", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":   "agent_123",
				"name": "Test Agent",
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		result, err := client.Agents.Get(context.Background(), "agent_123")

		require.NoError(t, err)
		assert.Equal(t, "agent_123", result.ID)
	})

	t.Run("deletes agent", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "DELETE", r.Method)
			w.WriteHeader(http.StatusNoContent)
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		err := client.Agents.Delete(context.Background(), "agent_123")

		require.NoError(t, err)
	})
}

func TestCallsService(t *testing.T) {
	t.Run("creates call", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":     "call_123",
				"status": "queued",
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		result, err := client.Calls.Create(context.Background(), CallCreateParams{
			AgentID:  "agent_123",
			ToNumber: "+14155551234",
		})

		require.NoError(t, err)
		assert.Equal(t, "call_123", result.ID)
		assert.Equal(t, CallStatus("queued"), result.Status)
	})

	t.Run("ends call", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Contains(t, r.URL.Path, "/end")
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		err := client.Calls.End(context.Background(), "call_123")

		require.NoError(t, err)
	})

	t.Run("gets transcript", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"call_id": "call_123",
				"messages": []map[string]interface{}{
					{"role": "assistant", "content": "Hello!"},
					{"role": "user", "content": "Hi there"},
				},
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		result, err := client.Calls.GetTranscript(context.Background(), "call_123")

		require.NoError(t, err)
		assert.Equal(t, "call_123", result.CallID)
		assert.Len(t, result.Messages, 2)
	})
}

func TestCampaignsService(t *testing.T) {
	t.Run("creates campaign", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":     "camp_123",
				"name":   "Test Campaign",
				"status": "draft",
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		result, err := client.Campaigns.Create(context.Background(), CampaignCreateParams{
			Name:    "Test Campaign",
			AgentID: "agent_123",
		})

		require.NoError(t, err)
		assert.Equal(t, "camp_123", result.ID)
		assert.Equal(t, CampaignStatus("draft"), result.Status)
	})

	t.Run("starts campaign", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Contains(t, r.URL.Path, "/start")
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":     "camp_123",
				"status": "running",
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		result, err := client.Campaigns.Start(context.Background(), "camp_123")

		require.NoError(t, err)
		assert.Equal(t, CampaignStatus("running"), result.Status)
	})

	t.Run("gets progress", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"total":      100,
				"completed":  45,
				"successful": 40,
				"failed":     5,
				"pending":    55,
			})
		}))
		defer server.Close()

		client, _ := NewClient("test-api-key", WithBaseURL(server.URL))

		result, err := client.Campaigns.GetProgress(context.Background(), "camp_123")

		require.NoError(t, err)
		assert.Equal(t, 100, result.Total)
		assert.Equal(t, 45, result.Completed)
	})
}

func TestStreaming(t *testing.T) {
	t.Run("creates streaming client", func(t *testing.T) {
		client, _ := NewClient("test-api-key")
		streaming := client.Streaming()

		assert.NotNil(t, streaming)
	})
}
