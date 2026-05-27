package pipelines

import (
	"context"
	json "github.com/goccy/go-json"
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
	toolResp, err := selectToolsWithRetrievalPrompt(ctx, p.Model, req.Query, catalog, systemPrompt)
	if err != nil {
		return contracts.RetrieveResponse{}, err
	}

	var wg sync.WaitGroup
	sources := []contracts.SourceRecord{}
	toolResults := make([][]contracts.SourceRecord, len(toolResp.ToolCalls))
	calledSummary := false
	for i, call := range toolResp.ToolCalls {
		i := i
		call := call
		if normalizeToolName(call.Name) == "searchsummary" {
			calledSummary = true
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			toolResults[i] = p.executeTool(ctx, call, req.Query, userID, req.TopK, profileRecords)
		}()
	}
	wg.Wait()
	if len(toolResp.ToolCalls) == 0 {
		answer := strings.TrimSpace(toolResp.Content)
		return contracts.RetrieveResponse{Model: p.Model.Name(), Answer: answer, Sources: sources, Confidence: 0.1}, nil
	}

	contextBlocks := make([]string, 0, len(toolResults)+1)
	for _, records := range toolResults {
		sources = append(sources, records...)
		contextBlocks = append(contextBlocks, formatToolResults(records))
	}
	if !calledSummary {
		extra := p.searchSummary(ctx, req.Query, userID, 20)
		sources = append(sources, extra...)
		contextBlocks = append(contextBlocks, "[Auto-fetched summary context]\n"+formatToolResults(extra))
	}

	contextText := strings.Join(contextBlocks, "\n")
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

func selectToolsWithRetrievalPrompt(ctx context.Context, model models.ChatModel, query string, catalog []map[string]string, systemPrompt string) (models.Response, error) {
	messages := []models.Message{
		{Role: "system", Content: systemPrompt + "\n\n" + toolSelectionOutputInstructions},
		{Role: "user", Content: query},
	}
	resp, err := model.GenerateWithMessages(ctx, messages)
	if err != nil {
		return model.SelectTools(ctx, query, catalog)
	}

	toolCalls, directAnswer, ok := parseToolCalls(resp.Content)
	if !ok {
		fallbackResp, fallbackErr := model.SelectTools(ctx, query, catalog)
		if fallbackErr != nil {
			return resp, nil
		}
		return fallbackResp, nil
	}
	resp.ToolCalls = toolCalls
	if directAnswer != "" {
		resp.Content = directAnswer
	}
	return resp, nil
}

const toolSelectionOutputInstructions = `You have access to these retrieval tools: search_profile, search_temporal, search_summary, search_snippet.
Return only JSON in this exact shape, with no markdown or commentary:
{"tool_calls":[{"name":"search_profile","args":{"topic":"work"}},{"name":"search_temporal","args":{"query":"dentist appointment"}},{"name":"search_summary","args":{"query":"..."}},{"name":"search_snippet","args":{"query":"..."}}]}
If you can answer directly without searching, return {"tool_calls":[],"answer":"..."}.`

func parseToolCalls(text string) ([]models.ToolCall, string, bool) {
	if strings.Contains(text, "Return only JSON in this exact shape") {
		return nil, "", false
	}

	var parsed struct {
		ToolCalls []models.ToolCall `json:"tool_calls"`
		Answer    string            `json:"answer"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(text)), &parsed); err != nil {
		return nil, "", false
	}
	for i := range parsed.ToolCalls {
		if parsed.ToolCalls[i].ID == "" {
			parsed.ToolCalls[i].ID = fmt.Sprintf("call-%d", i+1)
		}
	}
	return parsed.ToolCalls, parsed.Answer, true
}

func extractJSONObject(text string) string {
	text = strings.TrimSpace(text)
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start >= 0 && end > start {
		return text[start : end+1]
	}
	return ""
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
		return toSnippetSources(records)
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
		meta["topic"] = topic
		parts := strings.SplitN(main, "_", 2)
		if len(parts) == 2 {
			meta["sub_topic"] = parts[1]
		} else {
			meta["sub_topic"] = ""
		}
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

func toSnippetSources(records []storage.SearchResult) []contracts.SourceRecord {
	out := make([]contracts.SourceRecord, 0, len(records))
	for _, record := range records {
		meta := cloneMeta(record.Metadata)
		meta["id"] = record.ID
		content := record.Content
		if snippet, ok := record.Metadata["code_snippet"].(string); ok && snippet != "" {
			lang, _ := record.Metadata["language"].(string)
			content += fmt.Sprintf("\n```%s\n%s\n```", lang, snippet)
		}
		out = append(out, contracts.SourceRecord{Domain: "snippet", Content: content, Score: round3(record.Score), Metadata: meta})
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

func formatToolResults(records []contracts.SourceRecord) string {
	if len(records) == 0 {
		return "No results found."
	}
	lines := make([]string, 0, len(records))
	for i, rec := range records {
		score := ""
		if rec.Score > 0 {
			score = fmt.Sprintf(" (score: %.2f)", rec.Score)
		}
		lines = append(lines, fmt.Sprintf("%d. [%s]%s %s", i+1, rec.Domain, score, rec.Content))
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
		lines = append(lines, fmt.Sprintf("  - %s / %s", topic, subTopic))
	}
	return strings.Join(lines, "\n")
}
