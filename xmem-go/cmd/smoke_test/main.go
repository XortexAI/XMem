package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
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

// TimedModel wraps a real ChatModel and tracks cumulative LLM call time.
type TimedModel struct {
	inner   models.ChatModel
	llmTime int64 // atomic nanoseconds
	calls   int64 // atomic count
}

func NewTimedModel(inner models.ChatModel) *TimedModel {
	return &TimedModel{inner: inner}
}

func (t *TimedModel) Name() string { return t.inner.Name() }

func (t *TimedModel) Generate(ctx context.Context, prompt string) (models.Response, error) {
	start := time.Now()
	resp, err := t.inner.Generate(ctx, prompt)
	atomic.AddInt64(&t.llmTime, int64(time.Since(start)))
	atomic.AddInt64(&t.calls, 1)
	return resp, err
}

func (t *TimedModel) GenerateWithMessages(ctx context.Context, msgs []models.Message) (models.Response, error) {
	start := time.Now()
	resp, err := t.inner.GenerateWithMessages(ctx, msgs)
	atomic.AddInt64(&t.llmTime, int64(time.Since(start)))
	atomic.AddInt64(&t.calls, 1)
	return resp, err
}

func (t *TimedModel) GenerateVision(ctx context.Context, systemPrompt string, userText string, imageURL string) (models.Response, error) {
	start := time.Now()
	resp, err := t.inner.GenerateVision(ctx, systemPrompt, userText, imageURL)
	atomic.AddInt64(&t.llmTime, int64(time.Since(start)))
	atomic.AddInt64(&t.calls, 1)
	return resp, err
}

func (t *TimedModel) SelectTools(ctx context.Context, query string, catalog []map[string]string) (models.Response, error) {
	start := time.Now()
	resp, err := t.inner.SelectTools(ctx, query, catalog)
	atomic.AddInt64(&t.llmTime, int64(time.Since(start)))
	atomic.AddInt64(&t.calls, 1)
	return resp, err
}

func (t *TimedModel) LLMDuration() time.Duration {
	return time.Duration(atomic.LoadInt64(&t.llmTime))
}

func (t *TimedModel) CallCount() int64 {
	return atomic.LoadInt64(&t.calls)
}

func (t *TimedModel) Reset() {
	atomic.StoreInt64(&t.llmTime, 0)
	atomic.StoreInt64(&t.calls, 0)
}

type timing struct {
	name       string
	total      time.Duration
	llm        time.Duration
	overhead   time.Duration
	calls      int64
	concurrent bool
}

func main() {
	settings, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	realModel, err := models.NewRegistry(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "model registry error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Model: %s\n\n", realModel.Name())

	ctx := context.Background()
	testQuery := "My name is Alice and I work at Google as a senior software engineer. My birthday is April 5th. I love sushi and hiking on weekends."
	testResponse := "Nice to meet you Alice! That sounds like a great lifestyle."
	sessionDT := "4:04 pm on 20 January, 2025"

	var timings []timing

	// --- Individual Agent Benchmarks ---

	fmt.Println("Running individual agent benchmarks (real LLM calls)...")
	fmt.Println(strings.Repeat("─", 70))

	// Classifier
	{
		tm := NewTimedModel(realModel)
		agent := agents.ClassifierAgent{Model: tm}
		start := time.Now()
		result := agent.Run(ctx, testQuery, "")
		total := time.Since(start)
		t := timing{"Classifier Agent", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), false}
		timings = append(timings, t)
		fmt.Printf("  %-30s results=%d\n", t.name, len(result))
	}

	// Profiler
	{
		tm := NewTimedModel(realModel)
		agent := agents.ProfilerAgent{Model: tm}
		start := time.Now()
		result := agent.Run(ctx, testQuery)
		total := time.Since(start)
		t := timing{"Profiler Agent", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), false}
		timings = append(timings, t)
		fmt.Printf("  %-30s facts=%d\n", t.name, len(result))
	}

	// Temporal
	{
		tm := NewTimedModel(realModel)
		agent := agents.TemporalAgent{Model: tm}
		start := time.Now()
		result := agent.Run(ctx, testQuery, sessionDT)
		total := time.Since(start)
		t := timing{"Temporal Agent", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), false}
		timings = append(timings, t)
		fmt.Printf("  %-30s events=%d\n", t.name, len(result))
	}

	// Summarizer
	{
		tm := NewTimedModel(realModel)
		agent := agents.SummarizerAgent{Model: tm}
		start := time.Now()
		result := agent.Run(ctx, testQuery, testResponse)
		total := time.Since(start)
		t := timing{"Summarizer Agent", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), false}
		timings = append(timings, t)
		fmt.Printf("  %-30s bullets=%d\n", t.name, len(result))
	}

	// Judge (deterministic profile path — no LLM)
	{
		tm := NewTimedModel(realModel)
		agent := agents.JudgeAgent{Model: tm}
		facts := []agents.ProfileFact{
			{Topic: "basic_info", SubTopic: "name", Memo: "Alice"},
			{Topic: "work", SubTopic: "company", Memo: "Google"},
			{Topic: "work", SubTopic: "title", Memo: "Senior Software Engineer"},
		}
		start := time.Now()
		result := agent.JudgeProfile(ctx, facts, "bench-user")
		total := time.Since(start)
		t := timing{"Judge (deterministic)", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), false}
		timings = append(timings, t)
		fmt.Printf("  %-30s ops=%d\n", t.name, len(result.Operations))
	}

	// Judge (LLM path — summary domain)
	{
		tm := NewTimedModel(realModel)
		agent := agents.JudgeAgent{Model: tm}
		items := []string{
			"User's name is Alice and works at Google as a senior software engineer",
			"User's birthday is April 5th",
			"User loves sushi and hiking on weekends",
		}
		start := time.Now()
		result := agent.JudgeItems(ctx, weaver.DomainSummary, items, "bench-user", 0.8)
		total := time.Since(start)
		t := timing{"Judge (LLM)", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), false}
		timings = append(timings, t)
		fmt.Printf("  %-30s ops=%d\n", t.name, len(result.Operations))
	}

	fmt.Println(strings.Repeat("─", 70))

	// Shared stores: ingest writes here, retrieval reads from the same data.
	memStore := storage.NewMemoryVectorStore()
	tempStore := graph.NewMemoryTemporalStore()
	embedder := storage.HashEmbedder{Dimension: settings.PineconeDimension}
	benchUserID := "bench-user"

	// --- Full Ingest Pipeline ---

	fmt.Println("\nRunning full ingest pipeline...")
	{
		tm := NewTimedModel(realModel)

		pipeline := &pipelines.IngestPipeline{
			ModelName:  tm.Name(),
			Weaver:     &weaver.Weaver{VectorStore: memStore, SnippetVectorStore: memStore, Embedder: embedder, TemporalStore: tempStore},
			Classifier: agents.ClassifierAgent{Model: tm},
			Profiler:   agents.ProfilerAgent{Model: tm},
			Temporal:   agents.TemporalAgent{Model: tm},
			Summarizer: agents.SummarizerAgent{Model: tm},
			Image:      agents.ImageAgent{Model: tm},
			Snippet:    agents.SnippetAgent{Model: tm},
			Judge:      agents.JudgeAgent{Model: tm, VectorStore: memStore, TemporalStore: tempStore, TopK: 3},
		}

		req := contracts.IngestRequest{
			UserQuery:       testQuery,
			AgentResponse:   testResponse,
			SessionDatetime: sessionDT,
		}

		start := time.Now()
		resp, err := pipeline.Run(ctx, req, benchUserID)
		total := time.Since(start)
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
		} else {
			t := timing{"Full Ingest Pipeline", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), true}
			timings = append(timings, t)
			fmt.Printf("  %-30s calls=%d (parallel)\n", t.name, t.calls)
			fmt.Printf("  classifications=%d", len(resp.Classification))
			if resp.Profile != nil {
				fmt.Printf("  profile_ops=%d", len(resp.Profile.Operations))
			}
			if resp.Temporal != nil {
				fmt.Printf("  temporal_ops=%d", len(resp.Temporal.Operations))
			}
			if resp.Summary != nil {
				fmt.Printf("  summary_ops=%d", len(resp.Summary.Operations))
			}
			fmt.Println()
		}
	}

	// --- Full Retrieval Pipeline (uses memories from ingest above) ---

	fmt.Println("\nRunning full retrieval pipeline (after ingest)...")
	{
		tm := NewTimedModel(realModel)

		pipeline := &pipelines.RetrievalPipeline{
			Model:         tm,
			VectorStore:   memStore,
			SnippetStore:  memStore,
			TemporalStore: tempStore,
		}

		req := contracts.RetrieveRequest{
			Query: "What is my name and where do I work?",
		}

		start := time.Now()
		resp, err := pipeline.Run(ctx, req, benchUserID)
		total := time.Since(start)
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
		} else {
			t := timing{"Full Retrieval Pipeline", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), false}
			timings = append(timings, t)
			fmt.Printf("  %-30s calls=%d\n", t.name, t.calls)
			fmt.Printf("  answer=%q  sources=%d  confidence=%.2f\n", truncate(resp.Answer, 80), len(resp.Sources), resp.Confidence)
		}
	}

	// --- Concurrent Agent Benchmark (simulates real pipeline parallelism) ---

	fmt.Println("\nRunning concurrent agent benchmark (classifier → profiler+temporal+summarizer in parallel)...")
	{
		tm := NewTimedModel(realModel)
		start := time.Now()

		// Step 1: Classifier (sequential)
		classifier := agents.ClassifierAgent{Model: tm}
		_ = classifier.Run(ctx, testQuery, "")

		// Step 2: Parallel agents
		var wg sync.WaitGroup
		for _, fn := range []func(){
			func() { agents.ProfilerAgent{Model: tm}.Run(ctx, testQuery) },
			func() { agents.TemporalAgent{Model: tm}.Run(ctx, testQuery, sessionDT) },
			func() { agents.SummarizerAgent{Model: tm}.Run(ctx, testQuery, testResponse) },
		} {
			wg.Add(1)
			fn := fn
			go func() { defer wg.Done(); fn() }()
		}
		wg.Wait()

		total := time.Since(start)
		t := timing{"Concurrent Pipeline Sim", total, tm.LLMDuration(), total - tm.LLMDuration(), tm.CallCount(), true}
		timings = append(timings, t)
		fmt.Printf("  %-30s calls=%d (parallel)\n", t.name, t.calls)
	}

	// --- Summary Table ---

	fmt.Println()
	fmt.Println("╔════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        XMem-Go Benchmark Summary                                      ║")
	fmt.Printf("║  Model: %-77s║\n", realModel.Name())
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  %-30s %10s %12s %10s %6s %8s ║\n", "Component", "Total", "LLM Time", "Overhead", "Calls", "")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, t := range timings {
		if t.concurrent {
			saved := t.llm - t.total
			if saved < 0 {
				saved = 0
			}
			fmt.Printf("║  %-30s %10s %10s† %10s %6d %8s ║\n",
				t.name,
				t.total.Round(time.Millisecond),
				t.llm.Round(time.Millisecond),
				saved.Round(time.Millisecond),
				t.calls,
				"parallel",
			)
		} else {
			fmt.Printf("║  %-30s %10s %12s %10s %6d %8s ║\n",
				t.name,
				t.total.Round(time.Millisecond),
				t.llm.Round(time.Millisecond),
				t.overhead.Round(time.Microsecond),
				t.calls,
				"",
			)
		}
	}
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Sequential agents:  Overhead = Total - LLM Time (prompt building, parsing, etc.)")
	fmt.Println("Parallel agents:    LLM Time† = cumulative across goroutines; Overhead = time saved by concurrency")
	fmt.Println("Compare sequential 'Overhead' column with Python's to see Go's speed advantage.")
}

func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}
