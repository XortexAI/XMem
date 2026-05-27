//go:build ignore
// +build ignore

package main

import (
	"context"
	json "github.com/goccy/go-json"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/xortexai/xmem-go/internal/agents"
	"github.com/xortexai/xmem-go/internal/config"
	"github.com/xortexai/xmem-go/internal/contracts"
	"github.com/xortexai/xmem-go/internal/graph"
	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/pipelines"
	"github.com/xortexai/xmem-go/internal/storage"
	"github.com/xortexai/xmem-go/internal/weaver"
)

// ── Metrics Collector ─────────────────────────────────────────────────────

type MetricsEntry struct {
	Agent             string
	CallCount         int
	LLMTime           time.Duration
	TotalInputTokens  int
	TotalOutputTokens int
}

type PipelineTiming struct {
	Name         string
	WallClock    time.Duration
	LLMTime      time.Duration
	Calls        int
	InputTokens  int
	OutputTokens int
}

type MetricsCollector struct {
	mu              sync.Mutex
	entries         map[string]*MetricsEntry
	pipelineTimings []PipelineTiming
	queryTokens     int
}

func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{entries: make(map[string]*MetricsEntry)}
}

func (mc *MetricsCollector) Record(agent string, latency time.Duration, inputTokens, outputTokens int) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	entry, ok := mc.entries[agent]
	if !ok {
		entry = &MetricsEntry{Agent: agent}
		mc.entries[agent] = entry
	}
	entry.CallCount++
	entry.LLMTime += latency
	entry.TotalInputTokens += inputTokens
	entry.TotalOutputTokens += outputTokens
}

func (mc *MetricsCollector) TotalLLMTime() time.Duration {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	var total time.Duration
	for _, e := range mc.entries {
		total += e.LLMTime
	}
	return total
}

func (mc *MetricsCollector) TotalCalls() int {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	total := 0
	for _, e := range mc.entries {
		total += e.CallCount
	}
	return total
}

func (mc *MetricsCollector) RecordPipeline(name string, wallClock time.Duration, llmTime time.Duration, calls int, inputTokens int, outputTokens int) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.pipelineTimings = append(mc.pipelineTimings, PipelineTiming{Name: name, WallClock: wallClock, LLMTime: llmTime, Calls: calls, InputTokens: inputTokens, OutputTokens: outputTokens})
}

func (mc *MetricsCollector) TotalInputTokens() int {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	total := 0
	for _, e := range mc.entries {
		total += e.TotalInputTokens
	}
	return total
}

func (mc *MetricsCollector) TotalOutputTokens() int {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	total := 0
	for _, e := range mc.entries {
		total += e.TotalOutputTokens
	}
	return total
}

func (mc *MetricsCollector) PrintSummary() {
	agents := make([]string, 0, len(mc.entries))
	for name := range mc.entries {
		agents = append(agents, name)
	}
	sort.Strings(agents)

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              XMem-Go Pipeline Metrics Summary                                  ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  %-20s %6s %12s %12s %12s %12s ║\n", "Agent", "Calls", "LLM Time", "Overhead", "In Tokens", "Out Tokens")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, name := range agents {
		entry := mc.entries[name]
		fmt.Printf("║  %-20s %6d %12s %12s %12d %12d ║\n",
			entry.Agent,
			entry.CallCount,
			entry.LLMTime.Round(time.Millisecond),
			"—",
			entry.TotalInputTokens,
			entry.TotalOutputTokens,
		)
	}
	if len(mc.pipelineTimings) > 0 {
		fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
		fmt.Printf("║  %-20s %6s %12s %12s %12s %12s ║\n", "Pipeline", "Calls", "LLM Time", "Overhead", "Wall Clock", "")
		fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
		for _, pt := range mc.pipelineTimings {
			overhead := pt.WallClock - pt.LLMTime
			if overhead < 0 {
				overhead = 0
			}
			fmt.Printf("║  %-20s %6d %12s %12s %12s %12s ║\n",
				pt.Name,
				pt.Calls,
				pt.LLMTime.Round(time.Millisecond),
				overhead.Round(time.Millisecond),
				pt.WallClock.Round(time.Millisecond),
				"",
			)
		}
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝")

	if len(mc.pipelineTimings) > 0 {
		fmt.Println()
		totalIn := mc.TotalInputTokens()
		totalOut := mc.TotalOutputTokens()

		var ingestPT, retrievePT *PipelineTiming
		for i := range mc.pipelineTimings {
			if strings.Contains(mc.pipelineTimings[i].Name, "Ingest") {
				ingestPT = &mc.pipelineTimings[i]
			} else if strings.Contains(mc.pipelineTimings[i].Name, "Retrieval") {
				retrievePT = &mc.pipelineTimings[i]
			}
		}

		inputPrice := 0.15 / 1_000_000.0
		outputPrice := 0.60 / 1_000_000.0

		fmt.Printf("  User Query Tokens:    ~%d\n", mc.queryTokens)
		fmt.Printf("  Total Input Tokens:    %d\n", totalIn)
		fmt.Printf("  Total Output Tokens:   %d\n", totalOut)
		if ingestPT != nil {
			cost := float64(ingestPT.InputTokens)*inputPrice + float64(ingestPT.OutputTokens)*outputPrice
			fmt.Printf("  Cost to Ingest:        $%.6f  (%d in / %d out)\n", cost, ingestPT.InputTokens, ingestPT.OutputTokens)
		}
		if retrievePT != nil {
			cost := float64(retrievePT.InputTokens)*inputPrice + float64(retrievePT.OutputTokens)*outputPrice
			fmt.Printf("  Cost to Retrieve:      $%.6f  (%d in / %d out)\n", cost, retrievePT.InputTokens, retrievePT.OutputTokens)
		}
		fmt.Println()
		fmt.Println("  Cost estimate based on gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output).")
		fmt.Println("  Actual cost varies by provider/model. Tokens may be estimated if provider didn't return usage.")
	}
}

// TracingModel intercepts ChatModel calls and prints the user message and response (system prompts are hidden).
// Metrics are recorded in a shared MetricsCollector for the final summary table.
type TracingModel struct {
	inner   models.ChatModel
	agent   string
	mu      *sync.Mutex
	metrics *MetricsCollector
}

func NewTracingModel(inner models.ChatModel) *TracingModel {
	return &TracingModel{inner: inner, agent: "model", mu: &sync.Mutex{}, metrics: NewMetricsCollector()}
}

func (t *TracingModel) ForAgent(agent string) *TracingModel {
	return &TracingModel{inner: t.inner, agent: agent, mu: t.mu, metrics: t.metrics}
}

func (t *TracingModel) Metrics() *MetricsCollector {
	return t.metrics
}

func (t *TracingModel) Name() string {
	return t.inner.Name()
}

func indentLines(str string, spaces int) string {
	indent := strings.Repeat(" ", spaces)
	lines := strings.Split(str, "\n")
	for i, line := range lines {
		lines[i] = indent + line
	}
	return strings.Join(lines, "\n")
}

func estimateTokens(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	// Good enough for trace readability without provider-specific tokenizers.
	return (len([]rune(text)) + 3) / 4
}

func estimateMessageTokens(msgs []models.Message) int {
	total := 0
	for _, msg := range msgs {
		total += estimateTokens(msg.Role) + estimateTokens(msg.Content)
	}
	return total
}

func tokenCounts(resp models.Response, estimatedInput int, estimatedOutput int) (int, int, bool) {
	if resp.InputTokens > 0 || resp.OutputTokens > 0 {
		return resp.InputTokens, resp.OutputTokens, false
	}
	return estimatedInput, estimatedOutput, true
}

func (t *TracingModel) Generate(ctx context.Context, prompt string) (models.Response, error) {
	start := time.Now()
	resp, err := t.inner.Generate(ctx, prompt)
	elapsed := time.Since(start)

	inputTokens, outputTokens, _ := tokenCounts(resp, estimateTokens(prompt), estimateTokens(resp.Content))
	t.metrics.Record(t.agent, elapsed, inputTokens, outputTokens)

	t.mu.Lock()
	defer t.mu.Unlock()

	fmt.Println()
	fmt.Printf("  \x1b[35m┌─── [LLM Call: %s / Generate] ────────────────────────────────────────────────┐\x1b[0m\n", t.agent)
	fmt.Println("  \x1b[35m│\x1b[0m \x1b[1;33mPrompt:\x1b[0m")
	fmt.Println(indentLines(prompt, 4))
	fmt.Println("  \x1b[35m├─────────────────────────────────────────────────────────────────────────────────┤\x1b[0m")
	if err != nil {
		fmt.Printf("  \x1b[35m│\x1b[0m \x1b[1;31mError:\x1b[0m %v\n", err)
	} else {
		fmt.Println("  \x1b[35m│\x1b[0m \x1b[1;32mResponse:\x1b[0m")
		fmt.Println(indentLines(resp.Content, 4))
	}
	fmt.Println("  \x1b[35m└─────────────────────────────────────────────────────────────────────────────────┘\x1b[0m")

	return resp, err
}

func (t *TracingModel) GenerateWithMessages(ctx context.Context, msgs []models.Message) (models.Response, error) {
	start := time.Now()
	resp, err := t.inner.GenerateWithMessages(ctx, msgs)
	elapsed := time.Since(start)

	inputTokens, outputTokens, _ := tokenCounts(resp, estimateMessageTokens(msgs), estimateTokens(resp.Content))
	t.metrics.Record(t.agent, elapsed, inputTokens, outputTokens)

	t.mu.Lock()
	defer t.mu.Unlock()

	fmt.Println()
	fmt.Printf("  \x1b[35m┌─── [LLM Call: %s / GenerateWithMessages] ───────────────────────────────────┐\x1b[0m\n", t.agent)
	for _, m := range msgs {
		if m.Role == "system" {
			continue // Skip printing system prompts — they are extremely long
		}
		roleColor := "\x1b[1;33m" // user -> Yellow
		if m.Role == "assistant" || m.Role == "model" {
			roleColor = "\x1b[1;32m" // assistant -> Green
		}
		fmt.Printf("  \x1b[35m│\x1b[0m %s%s:\x1b[0m\n", roleColor, strings.ToUpper(m.Role))
		fmt.Println(indentLines(m.Content, 4))
	}
	fmt.Println("  \x1b[35m├─────────────────────────────────────────────────────────────────────────────────┤\x1b[0m")
	if err != nil {
		fmt.Printf("  \x1b[35m│\x1b[0m \x1b[1;31mError:\x1b[0m %v\n", err)
	} else {
		fmt.Println("  \x1b[35m│\x1b[0m \x1b[1;32mResponse:\x1b[0m")
		fmt.Println(indentLines(resp.Content, 4))
	}
	fmt.Println("  \x1b[35m└─────────────────────────────────────────────────────────────────────────────────┘\x1b[0m")

	return resp, err
}

func (t *TracingModel) GenerateVision(ctx context.Context, systemPrompt string, userText string, imageURL string) (models.Response, error) {
	start := time.Now()
	resp, err := t.inner.GenerateVision(ctx, systemPrompt, userText, imageURL)
	elapsed := time.Since(start)
	inputTokens, outputTokens, _ := tokenCounts(resp, estimateTokens(userText)+estimateTokens(systemPrompt), estimateTokens(resp.Content))
	t.metrics.Record(t.agent, elapsed, inputTokens, outputTokens)

	t.mu.Lock()
	defer t.mu.Unlock()
	fmt.Println()
	fmt.Printf("  \x1b[35m┌─── [LLM Call: %s / GenerateVision] ───────────────────────────────────┐\x1b[0m\n", t.agent)
	fmt.Printf("  \x1b[35m│\x1b[0m \x1b[1;33mUser Text:\x1b[0m %s\n", userText[:min(80, len(userText))])
	fmt.Printf("  \x1b[35m│\x1b[0m \x1b[1;34mImage URL:\x1b[0m %s\n", imageURL[:min(80, len(imageURL))])
	fmt.Println("  \x1b[35m├─────────────────────────────────────────────────────────────────────────────────┤\x1b[0m")
	if err != nil {
		fmt.Printf("  \x1b[35m│\x1b[0m \x1b[1;31mError:\x1b[0m %v\n", err)
	} else {
		fmt.Println("  \x1b[35m│\x1b[0m \x1b[1;32mResponse:\x1b[0m")
		fmt.Println(indentLines(resp.Content, 4))
	}
	fmt.Println("  \x1b[35m└─────────────────────────────────────────────────────────────────────────────────┘\x1b[0m")
	return resp, err
}

func (t *TracingModel) SelectTools(ctx context.Context, query string, catalog []map[string]string) (models.Response, error) {
	start := time.Now()
	resp, err := t.inner.SelectTools(ctx, query, catalog)
	elapsed := time.Since(start)

	toolCallsJSON, _ := json.Marshal(resp.ToolCalls)
	catalogJSON, _ := json.Marshal(catalog)
	inputTokens, outputTokens, _ := tokenCounts(resp, estimateTokens(query)+estimateTokens(string(catalogJSON)), estimateTokens(string(toolCallsJSON)))
	t.metrics.Record(t.agent, elapsed, inputTokens, outputTokens)

	t.mu.Lock()
	defer t.mu.Unlock()

	fmt.Println()
	fmt.Println("  \x1b[35m┌─── [LLM Call: SelectTools] ─────────────────────────────────────────────────────┐\x1b[0m")
	fmt.Println("  \x1b[35m│\x1b[0m \x1b[1;33mQuery:\x1b[0m", query)
	fmt.Println("  \x1b[35m│\x1b[0m \x1b[1;34mProfile Catalog:\x1b[0m", string(catalogJSON))
	fmt.Println("  \x1b[35m├─────────────────────────────────────────────────────────────────────────────────┤\x1b[0m")
	if err != nil {
		fmt.Printf("  \x1b[35m│\x1b[0m \x1b[1;31mError:\x1b[0m %v\n", err)
	} else {
		fmt.Println("  \x1b[35m│\x1b[0m \x1b[1;32mSelected Tools:\x1b[0m")
		prettyToolCallsJSON, _ := json.MarshalIndent(resp.ToolCalls, "    ", "  ")
		fmt.Println(indentLines(string(prettyToolCallsJSON), 2))
	}
	fmt.Println("  \x1b[35m└─────────────────────────────────────────────────────────────────────────────────┘\x1b[0m")

	return resp, err
}

func printBanner(title string) {
	border := strings.Repeat("═", len(title)+8)
	fmt.Printf("\n\x1b[1;36m╔%s╗\x1b[0m\n", border)
	fmt.Printf("\x1b[1;36m║    %s    ║\x1b[0m\n", title)
	fmt.Printf("\x1b[1;36m╚%s╝\x1b[0m\n\n", border)
}

func printStep(step string) {
	fmt.Printf("\n\x1b[1;34m─── %s ──────────────────────────────────────────────────────────\x1b[0m\n", step)
}

// buildRealStores uses the configured cloud stores. It intentionally fails hard
// instead of falling back to memory so benchmark runs cannot silently avoid cloud DBs.
func buildRealStores(ctx context.Context, settings config.Settings, logger *slog.Logger) (storage.Embedder, storage.VectorStore, storage.VectorStore, graph.TemporalStore, error) {
	// --- Embedder ---
	var embedder storage.Embedder
	if settings.EmbeddingProvider == "openai" {
		oai, err := storage.NewOpenAIEmbedder(settings)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("openai embedder unavailable: %w", err)
		}
		embedder = oai
		logger.Info("using OpenAI embedder", "model", settings.OpenAIEmbeddingModel, "dimension", settings.PineconeDimension)
	} else {
		return nil, nil, nil, nil, fmt.Errorf("cloud benchmark requires EMBEDDING_PROVIDER=openai, got %q", settings.EmbeddingProvider)
	}

	// --- Vector stores ---
	if !strings.EqualFold(settings.VectorStoreProvider, "pinecone") {
		return nil, nil, nil, nil, fmt.Errorf("cloud benchmark requires VECTOR_STORE_PROVIDER=pinecone, got %q", settings.VectorStoreProvider)
	}
	vectorStore, err := storage.NewPineconeVectorStore(ctx, settings, embedder, settings.PineconeNamespace)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("pinecone vector store unavailable: %w", err)
	}
	logger.Info("using Pinecone vector store", "namespace", settings.PineconeNamespace)

	snippetNS := settings.PineconeNamespace + "-snippets"
	snippetStore, err := storage.NewPineconeVectorStore(ctx, settings, embedder, snippetNS)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("pinecone snippet store unavailable: %w", err)
	}
	logger.Info("using Pinecone snippet store", "namespace", snippetNS)

	// --- Temporal store ---
	if settings.Neo4jPassword == "" {
		return nil, nil, nil, nil, fmt.Errorf("cloud benchmark requires NEO4J_PASSWORD")
	}
	temporalStore, err := graph.NewNeo4jTemporalStore(ctx, settings, embedder)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("neo4j temporal store unavailable: %w", err)
	}
	logger.Info("using Neo4j temporal store")

	return embedder, vectorStore, snippetStore, temporalStore, nil
}

func main() {
	settings, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	ctx := context.Background()

	realModel, err := models.NewRegistry(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "model registry error: %v\n", err)
		os.Exit(1)
	}
	tracingModel := NewTracingModel(realModel)

	// Build real stores from .env (Pinecone, OpenAI embeddings, Neo4j)
	embedder, vectorStore, snippetStore, temporalStore, err := buildRealStores(ctx, settings, logger)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cloud store error: %v\n", err)
		os.Exit(1)
	}

	printBanner(fmt.Sprintf("XMem-Go Pipeline Flow Trace (Model: %s)", realModel.Name()))

	userID := "trace-user"

	makePipeline := func() *pipelines.IngestPipeline {
		return &pipelines.IngestPipeline{
			ModelName:  tracingModel.Name(),
			Weaver:     &weaver.Weaver{VectorStore: vectorStore, SnippetVectorStore: snippetStore, Embedder: embedder, TemporalStore: temporalStore},
			Classifier: agents.ClassifierAgent{Model: tracingModel.ForAgent("classifier")},
			Profiler:   agents.ProfilerAgent{Model: tracingModel.ForAgent("profiler")},
			Temporal:   agents.TemporalAgent{Model: tracingModel.ForAgent("temporal")},
			Summarizer: agents.SummarizerAgent{Model: tracingModel.ForAgent("summarizer")},
			Image:      agents.ImageAgent{Model: tracingModel.ForAgent("image")},
			Code:       agents.CodeAgent{Model: tracingModel.ForAgent("code")},
			Snippet:    agents.SnippetAgent{Model: tracingModel.ForAgent("snippet")},
			Judge:      agents.JudgeAgent{Model: tracingModel.ForAgent("judge"), VectorStore: vectorStore, TemporalStore: temporalStore, TopK: 3},
		}
	}

	// ──────────────────────────────────────────────────────────────────────
	// 1. Full Ingest Pipeline Flow
	// ──────────────────────────────────────────────────────────────────────
	printStep("1. Full Ingest Pipeline Flow")
	{
		pipeline := makePipeline()

		req := contracts.IngestRequest{
			UserQuery:       "My name is Bob, and I started a new job at Vercel as a frontend developer today!",
			AgentResponse:   "Congratulations on your new role Bob! That's wonderful news.",
			SessionDatetime: "4:00 pm on 20 May, 2026",
		}

		tracingModel.Metrics().queryTokens = estimateTokens(req.UserQuery)

		fmt.Printf("\x1b[1;33mIngest Request:\x1b[0m\n  User Query:  %s\n  Response:    %s\n", req.UserQuery, req.AgentResponse)
		fmt.Println("\nExecuting Ingest Pipeline (data goes to Pinecone / Neo4j)...")

		llmBefore := tracingModel.Metrics().TotalLLMTime()
		callsBefore := tracingModel.Metrics().TotalCalls()
		inBefore := tracingModel.Metrics().TotalInputTokens()
		outBefore := tracingModel.Metrics().TotalOutputTokens()
		ingestStart := time.Now()
		resp, err := pipeline.Run(ctx, req, userID)
		ingestWall := time.Since(ingestStart)
		ingestLLM := tracingModel.Metrics().TotalLLMTime() - llmBefore
		ingestCalls := tracingModel.Metrics().TotalCalls() - callsBefore
		ingestIn := tracingModel.Metrics().TotalInputTokens() - inBefore
		ingestOut := tracingModel.Metrics().TotalOutputTokens() - outBefore
		tracingModel.Metrics().RecordPipeline("Full Ingest", ingestWall, ingestLLM, ingestCalls, ingestIn, ingestOut)

		if err != nil {
			fmt.Printf("\x1b[1;31mPipeline Error:\x1b[0m %v\n", err)
		} else {
			fmt.Println("\n\x1b[1;32mIngest Pipeline Completed. Response:\x1b[0m")
			respBytes, _ := json.MarshalIndent(resp, "", "  ")
			fmt.Println(string(respBytes))
		}
	}

	// ──────────────────────────────────────────────────────────────────────
	// 2. Full Retrieval Pipeline Flow
	// ──────────────────────────────────────────────────────────────────────
	printStep("2. Full Retrieval Pipeline Flow")
	{
		pipeline := &pipelines.RetrievalPipeline{
			Model:         tracingModel.ForAgent("retrieval"),
			VectorStore:   vectorStore,
			SnippetStore:  snippetStore,
			TemporalStore: temporalStore,
		}

		req := contracts.RetrieveRequest{
			Query: "What is my name and where do I work?",
		}

		fmt.Printf("\x1b[1;33mRetrieval Request Query:\x1b[0m %s\n", req.Query)
		fmt.Println("\nExecuting Retrieval Pipeline (querying Pinecone / Neo4j)...")

		llmBefore := tracingModel.Metrics().TotalLLMTime()
		callsBefore := tracingModel.Metrics().TotalCalls()
		inBefore := tracingModel.Metrics().TotalInputTokens()
		outBefore := tracingModel.Metrics().TotalOutputTokens()
		retrieveStart := time.Now()
		resp, err := pipeline.Run(ctx, req, userID)
		retrieveWall := time.Since(retrieveStart)
		retrieveLLM := tracingModel.Metrics().TotalLLMTime() - llmBefore
		retrieveCalls := tracingModel.Metrics().TotalCalls() - callsBefore
		retrieveIn := tracingModel.Metrics().TotalInputTokens() - inBefore
		retrieveOut := tracingModel.Metrics().TotalOutputTokens() - outBefore
		tracingModel.Metrics().RecordPipeline("Full Retrieval", retrieveWall, retrieveLLM, retrieveCalls, retrieveIn, retrieveOut)

		if err != nil {
			fmt.Printf("\x1b[1;31mPipeline Error:\x1b[0m %v\n", err)
		} else {
			fmt.Println("\n\x1b[1;32mRetrieval Response:\x1b[0m")
			fmt.Printf("  Answer:     %s\n", resp.Answer)
			fmt.Printf("  Confidence: %.2f\n", resp.Confidence)
			fmt.Printf("  Sources retrieved: %d\n", len(resp.Sources))
			for i, src := range resp.Sources {
				fmt.Printf("    [%d] Domain: %s | Score: %.3f | Content: %s\n", i+1, src.Domain, src.Score, src.Content)
			}
		}
	}

	tracingModel.Metrics().PrintSummary()
}
