package pipelines

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/xortexai/xmem-go/internal/contracts"
	"github.com/xortexai/xmem-go/internal/graph"
	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
	"github.com/xortexai/xmem-go/internal/storage"
)

type RetrievalPipeline struct {
	Model         models.ChatModel
	VectorStore   storage.VectorStore
	SnippetStore  storage.VectorStore
	TemporalStore graph.TemporalStore
}

func (p *RetrievalPipeline) Run(ctx context.Context, req contracts.RetrieveRequest, userID string) (contracts.RetrieveResponse, error) {
	if req.TopK == 0 {
		req.TopK = 5
	}
	catalog, profileRecords := p.fetchProfileCatalog(ctx, userID)

	catalogStr := formatProfileCatalog(catalog)
	systemPrompt := prompts.BuildRetrievalSystemPrompt(catalogStr)
	toolResp, err := p.Model.SelectTools(ctx, req.Query, catalog)
	if err != nil {
		return contracts.RetrieveResponse{}, err
	}
	_ = systemPrompt

	var mu sync.Mutex
	var wg sync.WaitGroup
	sources := []contracts.SourceRecord{}
	calledSummary := false
	for _, call := range toolResp.ToolCalls {
		call := call
		if normalizeToolName(call.Name) == "searchsummary" {
			calledSummary = true
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			records := p.executeTool(ctx, call, req.Query, userID, req.TopK, profileRecords)
			mu.Lock()
			sources = append(sources, records...)
			mu.Unlock()
		}()
	}
	wg.Wait()
	if !calledSummary {
		sources = append(sources, p.searchSummary(ctx, req.Query, userID, 20)...)
	}

	contextText := formatSources(sources)
	answerPrompt := prompts.BuildAnswerPrompt(contextText, req.Query)
	answerResp, err := p.Model.GenerateWithMessages(ctx, []models.Message{
		{Role: "user", Content: answerPrompt},
	})
	if err != nil {
		return contracts.RetrieveResponse{}, err
	}
	confidence := 0.1
	if len(sources) > 0 {
		confidence = math.Min(1, float64(len(sources))*0.2)
	}
	return contracts.RetrieveResponse{Model: p.Model.Name(), Answer: answerResp.Content, Sources: sources, Confidence: confidence}, nil
}

func (p *RetrievalPipeline) Search(ctx context.Context, req contracts.SearchRequest, userID string) (contracts.SearchResponse, error) {
	if req.TopK == 0 {
		req.TopK = 10
	}
	if len(req.Domains) == 0 {
		req.Domains = []string{"profile", "temporal", "summary"}
	}
	results := []contracts.SourceRecord{}
	for _, domain := range req.Domains {
		switch domain {
		case "profile":
			records, _ := p.VectorStore.SearchByMetadata(ctx, map[string]any{"user_id": userID, "domain": "profile"}, 100)
			results = append(results, toSources("profile", records)...)
		case "temporal":
			events, _ := p.TemporalStore.SearchEventsByEmbedding(ctx, userID, req.Query, req.TopK, 0.15)
			results = append(results, eventsToSources(events)...)
		case "summary":
			results = append(results, p.searchSummary(ctx, req.Query, userID, req.TopK)...)
		}
	}
	return contracts.SearchResponse{Results: results, Total: len(results)}, nil
}

func (p *RetrievalPipeline) executeTool(ctx context.Context, call models.ToolCall, query string, userID string, topK int, profileRecords []storage.SearchResult) []contracts.SourceRecord {
	switch normalizeToolName(call.Name) {
	case "searchprofile":
		topic, _ := call.Args["topic"].(string)
		return searchProfile(topic, profileRecords)
	case "searchtemporal":
		q, _ := call.Args["query"].(string)
		if q == "" {
			q = query
		}
		events, _ := p.TemporalStore.SearchEventsByEmbedding(ctx, userID, q, 10, 0.15)
		return eventsToSources(events)
	case "searchsummary":
		q, _ := call.Args["query"].(string)
		if q == "" {
			q = query
		}
		return p.searchSummary(ctx, q, userID, 15)
	case "searchsnippet":
		q, _ := call.Args["query"].(string)
		if q == "" {
			q = query
		}
		store := p.SnippetStore
		if store == nil {
			store = p.VectorStore
		}
		records, _ := store.SearchByText(ctx, q, 5, map[string]any{"domain": "snippet"})
		return toSources("snippet", records)
	default:
		return nil
	}
}

func (p *RetrievalPipeline) fetchProfileCatalog(ctx context.Context, userID string) ([]map[string]string, []storage.SearchResult) {
	records, err := p.VectorStore.SearchByMetadata(ctx, map[string]any{"user_id": userID, "domain": "profile"}, 100)
	if err != nil {
		return nil, nil
	}
	seen := map[string]bool{}
	catalog := []map[string]string{}
	for _, record := range records {
		main, _ := record.Metadata["main_content"].(string)
		if main == "" || seen[main] {
			continue
		}
		seen[main] = true
		parts := strings.SplitN(main, "_", 2)
		item := map[string]string{"topic": parts[0], "sub_topic": ""}
		if len(parts) == 2 {
			item["sub_topic"] = parts[1]
		}
		catalog = append(catalog, item)
	}
	return catalog, records
}

func searchProfile(topic string, records []storage.SearchResult) []contracts.SourceRecord {
	topicPrefix := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(topic)), " ", "_")
	out := []contracts.SourceRecord{}
	for _, record := range records {
		main := fmt.Sprint(record.Metadata["main_content"])
		if topicPrefix != "" && !strings.HasPrefix(main, topicPrefix) {
			continue
		}
		meta := cloneMeta(record.Metadata)
		meta["id"] = record.ID
		out = append(out, contracts.SourceRecord{Domain: "profile", Content: record.Content, Score: round3(record.Score), Metadata: meta})
	}
	return out
}

func (p *RetrievalPipeline) searchSummary(ctx context.Context, query string, userID string, topK int) []contracts.SourceRecord {
	records, err := p.VectorStore.SearchByText(ctx, query, topK, map[string]any{"user_id": userID, "domain": "summary"})
	if err != nil {
		return nil
	}
	return toSources("summary", records)
}

func toSources(domain string, records []storage.SearchResult) []contracts.SourceRecord {
	out := make([]contracts.SourceRecord, 0, len(records))
	for _, record := range records {
		meta := cloneMeta(record.Metadata)
		meta["id"] = record.ID
		out = append(out, contracts.SourceRecord{Domain: domain, Content: record.Content, Score: round3(record.Score), Metadata: meta})
	}
	return out
}

func eventsToSources(events []graph.Event) []contracts.SourceRecord {
	out := make([]contracts.SourceRecord, 0, len(events))
	for _, ev := range events {
		parts := []string{}
		if ev.Date != "" {
			date := ev.Date
			if ev.Year != "" {
				date += ", " + ev.Year
			}
			parts = append(parts, "Date: "+date)
		}
		if ev.EventName != "" {
			parts = append(parts, "Event: "+ev.EventName)
		}
		if ev.Description != "" {
			parts = append(parts, "Description: "+ev.Description)
		}
		if ev.Time != "" {
			parts = append(parts, "Time: "+ev.Time)
		}
		out = append(out, contracts.SourceRecord{
			Domain:  "temporal",
			Content: strings.Join(parts, " | "),
			Score:   round3(ev.SimilarityScore),
			Metadata: map[string]any{
				"date": ev.Date, "event_name": ev.EventName, "desc": ev.Description,
				"year": ev.Year, "time": ev.Time, "date_expression": ev.DateExpression,
				"similarity_score": ev.SimilarityScore,
			},
		})
	}
	return out
}

func formatSources(sources []contracts.SourceRecord) string {
	if len(sources) == 0 {
		return "No results found."
	}
	lines := make([]string, 0, len(sources))
	for i, src := range sources {
		lines = append(lines, fmt.Sprintf("%d. [%s] %s", i+1, src.Domain, src.Content))
	}
	return strings.Join(lines, "\n")
}

func normalizeToolName(name string) string {
	return strings.ToLower(strings.ReplaceAll(name, "_", ""))
}

func cloneMeta(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func round3(v float64) float64 {
	return math.Round(v*1000) / 1000
}

func formatProfileCatalog(catalog []map[string]string) string {
	if len(catalog) == 0 {
		return "(No profile data stored yet)"
	}
	lines := make([]string, 0, len(catalog))
	for _, item := range catalog {
		topic := item["topic"]
		subTopic := item["sub_topic"]
		if subTopic != "" {
			lines = append(lines, fmt.Sprintf("  - %s / %s", topic, subTopic))
		} else {
			lines = append(lines, fmt.Sprintf("  - %s", topic))
		}
	}
	return strings.Join(lines, "\n")
}
