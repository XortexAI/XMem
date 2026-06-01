package api

import (
	"bytes"
	"context"
	json "github.com/goccy/go-json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/xortexai/xmem-go/internal/agents"
	"github.com/xortexai/xmem-go/internal/config"
	"github.com/xortexai/xmem-go/internal/database"
	"github.com/xortexai/xmem-go/internal/graph"
	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/pipelines"
	"github.com/xortexai/xmem-go/internal/storage"
	"github.com/xortexai/xmem-go/internal/weaver"
)

type testModel struct{}

func (testModel) Name() string { return "test-model" }
func (testModel) Generate(_ context.Context, prompt string) (models.Response, error) {
	return models.Response{Content: prompt, ModelName: "test-model"}, nil
}
func (testModel) GenerateWithMessages(_ context.Context, msgs []models.Message) (models.Response, error) {
	content := ""
	for _, m := range msgs {
		content += m.Content + "\n"
	}
	return models.Response{Content: strings.TrimSpace(content), ModelName: "test-model"}, nil
}
func (testModel) GenerateVision(_ context.Context, _ string, userText string, _ string) (models.Response, error) {
	return models.Response{Content: userText, ModelName: "test-model"}, nil
}
func (testModel) SelectTools(_ context.Context, query string, _ []map[string]string) (models.Response, error) {
	return models.Response{
		ToolCalls: []models.ToolCall{{ID: "call-1", Name: "search_summary", Args: map[string]any{"query": query}}},
		ModelName: "test-model",
	}, nil
}

func testHandler() http.Handler {
	settings := config.Settings{
		APIHost:             "127.0.0.1",
		APIPort:             8081,
		APIKeys:             []string{"test-key"},
		RateLimit:           60,
		MaxRequestBodyBytes: 10 * 1024 * 1024,
		CORSOrigins:         []string{"http://localhost:5173"},
		JWTSecretKey:        "test-secret",
		JWTAlgorithm:        "HS256",
	}
	vectorStore := storage.NewMemoryVectorStore()
	snippetStore := storage.NewMemoryVectorStore()
	temporalStore := graph.NewMemoryTemporalStore()
	model := testModel{}
	w := &weaver.Weaver{
		VectorStore:        vectorStore,
		SnippetVectorStore: snippetStore,
		Embedder:           storage.HashEmbedder{Dimension: 64},
		TemporalStore:      temporalStore,
	}
	ingest := &pipelines.IngestPipeline{
		ModelName:  model.Name(),
		Weaver:     w,
		Classifier: agents.ClassifierAgent{Model: model},
		Profiler:   agents.ProfilerAgent{Model: model},
		Temporal:   agents.TemporalAgent{Model: model},
		Summarizer: agents.SummarizerAgent{Model: model},
		Image:      agents.ImageAgent{Model: model},
		Snippet:    agents.SnippetAgent{Model: model},
		Judge:      agents.JudgeAgent{Model: model, VectorStore: vectorStore, TemporalStore: temporalStore, TopK: 3},
	}
	retrieval := &pipelines.RetrievalPipeline{
		Model:         model,
		VectorStore:   vectorStore,
		SnippetStore:  snippetStore,
		TemporalStore: temporalStore,
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	return NewServer(settings, logger, ingest, retrieval, database.NewMemoryAPIKeyStore()).Handler()
}

func TestHealthAndAuth(t *testing.T) {
	handler := testHandler()
	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/health", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("health status = %d body=%s", health.Code, health.Body.String())
	}

	missing := httptest.NewRecorder()
	handler.ServeHTTP(missing, httptest.NewRequest(http.MethodPost, "/v1/memory/search", strings.NewReader(`{}`)))
	if missing.Code != http.StatusUnauthorized {
		t.Fatalf("missing auth status = %d", missing.Code)
	}

	invalid := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/memory/search", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer wrong")
	handler.ServeHTTP(invalid, req)
	if invalid.Code != http.StatusForbidden {
		t.Fatalf("invalid auth status = %d", invalid.Code)
	}
}

func TestMemoryFlowAndExcludedRoutes(t *testing.T) {
	handler := testHandler()
	ingestBody := `{"user_query":"My name is Alice and I work at XMem. I have a demo tomorrow at 10:00.","user_id":"alice","session_datetime":"2026-05-21T09:00:00Z"}`
	ingest := authedJSON(handler, http.MethodPost, "/v1/memory/ingest", ingestBody)
	if ingest.Code != http.StatusOK {
		t.Fatalf("ingest status=%d body=%s", ingest.Code, ingest.Body.String())
	}
	var ingestEnvelope APIResponse
	if err := json.Unmarshal(ingest.Body.Bytes(), &ingestEnvelope); err != nil {
		t.Fatal(err)
	}
	if ingestEnvelope.Status != "ok" || ingestEnvelope.Data == nil {
		t.Fatalf("bad ingest envelope: %#v", ingestEnvelope)
	}
	ingestData := ingestEnvelope.Data.(map[string]any)
	jobID, _ := ingestData["job_id"].(string)
	if jobID == "" {
		t.Fatalf("ingest response missing job_id: %#v", ingestData)
	}
	waitForJob(t, handler, "/v1/memory/ingest/"+jobID+"/status")

	retrieve := authedJSON(handler, http.MethodPost, "/v1/memory/retrieve", `{"query":"Where do I work and what is upcoming?","user_id":"alice","top_k":5}`)
	if retrieve.Code != http.StatusOK {
		t.Fatalf("retrieve status=%d body=%s", retrieve.Code, retrieve.Body.String())
	}
	if !strings.Contains(retrieve.Body.String(), "XMem") {
		t.Fatalf("retrieve should mention stored memory, body=%s", retrieve.Body.String())
	}

	for _, path := range []string{"/v1/memory/scrape", "/v1/memory/parse_transcript", "/context", "/v1/scanner/status"} {
		resp := authedJSON(handler, http.MethodPost, path, `{}`)
		if resp.Code != http.StatusNotFound && resp.Code != http.StatusMethodNotAllowed {
			t.Fatalf("%s should not exist, got %d body=%s", path, resp.Code, resp.Body.String())
		}
	}
}

func TestBatchIngestAndSearch(t *testing.T) {
	handler := testHandler()
	batch := authedJSON(handler, http.MethodPost, "/v1/memory/batch-ingest", `{"items":[{"user_query":"I love pizza.","user_id":"alice"},{"user_query":"I enjoy hiking on weekends.","user_id":"alice"}]}`)
	if batch.Code != http.StatusOK {
		t.Fatalf("batch status=%d body=%s", batch.Code, batch.Body.String())
	}
	var batchEnvelope APIResponse
	if err := json.Unmarshal(batch.Body.Bytes(), &batchEnvelope); err != nil {
		t.Fatal(err)
	}
	batchData := batchEnvelope.Data.(map[string]any)
	jobID, _ := batchData["job_id"].(string)
	if jobID == "" {
		t.Fatalf("batch response missing job_id: %#v", batchData)
	}
	waitForJob(t, handler, "/v1/memory/jobs/"+jobID+"/status")
	search := authedJSON(handler, http.MethodPost, "/v1/memory/search", `{"query":"pizza","user_id":"alice","domains":["profile","summary"],"top_k":10}`)
	if search.Code != http.StatusOK {
		t.Fatalf("search status=%d body=%s", search.Code, search.Body.String())
	}
	if !strings.Contains(search.Body.String(), `"total"`) {
		t.Fatalf("search missing total: %s", search.Body.String())
	}
}

func authedJSON(handler http.Handler, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	return resp
}

func waitForJob(t *testing.T, handler http.Handler, statusPath string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		resp := authedJSON(handler, http.MethodGet, statusPath, ``)
		if resp.Code != http.StatusOK {
			t.Fatalf("status path=%s code=%d body=%s", statusPath, resp.Code, resp.Body.String())
		}
		var envelope APIResponse
		if err := json.Unmarshal(resp.Body.Bytes(), &envelope); err != nil {
			t.Fatal(err)
		}
		data, ok := envelope.Data.(map[string]any)
		if !ok {
			t.Fatalf("status data missing: %#v", envelope.Data)
		}
		if data["status"] == "succeeded" {
			return
		}
		if data["status"] == "dead_letter" {
			t.Fatalf("job dead-lettered: %s", resp.Body.String())
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not finish: %s", statusPath)
}
