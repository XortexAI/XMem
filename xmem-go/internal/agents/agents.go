package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
	"github.com/xortexai/xmem-go/internal/storage"
	"github.com/xortexai/xmem-go/internal/utils"
	"github.com/xortexai/xmem-go/internal/weaver"
)

const summaryJudgeSimilarityThreshold = 0.4

// ---------- shared LLM call helper ----------

func callModel(ctx context.Context, model models.ChatModel, systemPrompt, userMessage string) (string, error) {
	messages := []models.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userMessage},
	}
	resp, err := model.GenerateWithMessages(ctx, messages)
	if err != nil {
		return "", fmt.Errorf("LLM call failed (%s): %w", model.Name(), err)
	}
	return resp.Content, nil
}

// ---------- Classification ----------

type Classification struct {
	Source string `json:"source"`
	Query  string `json:"query"`
}

type ClassifierAgent struct {
	Model models.ChatModel
}

func (a ClassifierAgent) Run(ctx context.Context, userQuery string, imageURL string) []Classification {
	systemPrompt := prompts.BuildClassifierSystemPrompt()
	userMessage := prompts.PackClassificationQuery(userQuery)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	parsed := utils.ParseRawResponseToClassifications(raw)

	out := make([]Classification, 0, len(parsed))
	for _, c := range parsed {
		out = append(out, Classification{Source: c.Source, Query: c.Query})
	}

	if strings.TrimSpace(imageURL) != "" {
		hasImage := false
		for _, c := range out {
			if c.Source == "image" {
				hasImage = true
				break
			}
		}
		if !hasImage {
			out = append(out, Classification{Source: "image", Query: "Analyze this image for memory-relevant details."})
		}
	}

	return out
}

// ---------- Profile ----------

type ProfileFact struct {
	Topic    string
	SubTopic string
	Memo     string
}

type ProfilerAgent struct {
	Model models.ChatModel
}

func (a ProfilerAgent) Run(ctx context.Context, text string) []ProfileFact {
	systemPrompt := prompts.BuildProfilerSystemPrompt()
	userMessage := prompts.PackProfilerQuery(text)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	parsed := utils.ParseRawResponseToProfiles(raw)

	facts := make([]ProfileFact, 0, len(parsed))
	for _, f := range parsed {
		facts = append(facts, ProfileFact{
			Topic:    f.Topic,
			SubTopic: f.SubTopic,
			Memo:     f.Memo,
		})
	}
	return facts
}

// ---------- Temporal ----------

type Event struct {
	Date           string
	EventName      string
	Desc           string
	Year           string
	Time           string
	DateExpression string
}

type TemporalAgent struct {
	Model models.ChatModel
}

func (a TemporalAgent) Run(ctx context.Context, text string, sessionDatetime string) []Event {
	systemPrompt := prompts.BuildTemporalSystemPrompt()
	userMessage := prompts.PackTemporalQuery(text, sessionDatetime)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	parsed := utils.ParseRawResponseToEvents(raw)

	events := make([]Event, 0, len(parsed))
	for _, e := range parsed {
		if !isValidDate(e.Date) {
			continue
		}
		events = append(events, Event{
			Date:           e.Date,
			EventName:      e.EventName,
			Desc:           e.Desc,
			Year:           e.Year,
			Time:           e.Time,
			DateExpression: e.DateExpression,
		})
	}
	return events
}

func isValidDate(date string) bool {
	parts := strings.SplitN(date, "-", 2)
	if len(parts) != 2 {
		return false
	}
	month, err := strconv.Atoi(parts[0])
	if err != nil || month < 1 || month > 12 {
		return false
	}
	day, err := strconv.Atoi(parts[1])
	if err != nil || day < 1 || day > 31 {
		return false
	}
	return true
}

// ---------- Summarizer ----------

type SummarizerAgent struct {
	Model models.ChatModel
}

var emptySentinels = map[string]struct{}{
	`""`:             {},
	`''`:             {},
	"empty":          {},
	"(empty)":        {},
	"(empty string)": {},
}

func (a SummarizerAgent) Run(ctx context.Context, userQuery string, agentResponse string) []string {
	systemPrompt := prompts.BuildSummarizerSystemPrompt()
	userMessage := prompts.PackSummaryQuery(userQuery, agentResponse)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	trimmed := strings.TrimSpace(raw)
	if _, isEmpty := emptySentinels[strings.ToLower(trimmed)]; isEmpty || trimmed == "" {
		return nil
	}

	lines := strings.Split(trimmed, "\n")
	bullets := make([]string, 0, len(lines))
	for _, line := range lines {
		cleaned := strings.TrimSpace(line)
		cleaned = strings.TrimLeft(cleaned, "-•*")
		cleaned = strings.TrimSpace(cleaned)
		if cleaned != "" {
			bullets = append(bullets, cleaned)
		}
	}
	return bullets
}

// ---------- Image ----------

type ImageAgent struct {
	Model models.ChatModel
}

func (a ImageAgent) Run(ctx context.Context, imageURL string) []string {
	if strings.TrimSpace(imageURL) == "" {
		return nil
	}

	systemPrompt := prompts.BuildImageSystemPrompt()
	userMessage := prompts.PackImageQuery("", imageURL)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	result := utils.ParseRawResponseToImage(raw)

	observations := make([]string, 0, len(result.Observations))
	for _, obs := range result.Observations {
		entry := "[" + obs.Category + "] " + obs.Description
		if obs.Confidence != "" {
			entry += " (confidence: " + obs.Confidence + ")"
		}
		observations = append(observations, entry)
	}
	if len(observations) == 0 && result.Description != "" {
		observations = append(observations, result.Description)
	}
	return observations
}

// ---------- Snippet ----------

type SnippetAgent struct {
	Model models.ChatModel
}

type snippetJSON struct {
	Content     string `json:"content"`
	CodeSnippet string `json:"code_snippet"`
	Language    string `json:"language"`
	SnippetType string `json:"snippet_type"`
	Tags        string `json:"tags"`
}

type snippetsResponse struct {
	Snippets []snippetJSON `json:"snippets"`
}

func (a SnippetAgent) Run(ctx context.Context, text string) []string {
	systemPrompt := prompts.BuildSnippetSystemPrompt()
	userMessage := prompts.PackSnippetQuery(text)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	items := parseSnippetResponse(raw)
	results := make([]string, 0, len(items))
	for _, s := range items {
		line := joinPipe(s.Content, s.CodeSnippet, s.Language, s.SnippetType, s.Tags)
		results = append(results, line)
	}
	return results
}

func parseSnippetResponse(raw string) []snippetJSON {
	jsonStr := extractJSONObject(raw)
	if jsonStr == "" {
		jsonStr = extractJSONArray(raw)
	}

	var resp snippetsResponse
	if err := json.Unmarshal([]byte(jsonStr), &resp); err == nil && len(resp.Snippets) > 0 {
		return resp.Snippets
	}

	var arr []snippetJSON
	if err := json.Unmarshal([]byte(jsonStr), &arr); err == nil && len(arr) > 0 {
		return arr
	}

	var single snippetJSON
	if err := json.Unmarshal([]byte(jsonStr), &single); err == nil && single.Content != "" {
		return []snippetJSON{single}
	}

	return nil
}

// ---------- Judge ----------

type JudgeAgent struct {
	Model       models.ChatModel
	VectorStore storage.VectorStore
	TopK        int
}

func (a JudgeAgent) judgeTopK() int {
	if a.TopK <= 0 {
		return 3
	}
	return a.TopK
}

func (a JudgeAgent) JudgeItems(ctx context.Context, domain weaver.JudgeDomain, items []string, userID string, confidence float64) weaver.JudgeResult {
	if len(items) == 0 {
		return weaver.JudgeResult{}
	}

	if domain == weaver.DomainSummary {
		matches := a.fetchSimilarSummaries(ctx, items, userID)
		if !hasSummaryJudgeCandidates(matches) {
			return judgeDeterministicSummary(items, confidence)
		}
		similarBlock := formatSummarySimilarBlock(items, filterMatchesByThreshold(matches, summaryJudgeSimilarityThreshold))
		return a.judgeItemsWithLLM(ctx, domain, items, similarBlock, confidence)
	}

	return a.judgeItemsWithLLM(ctx, domain, items, nil, confidence)
}

func (a JudgeAgent) judgeItemsWithLLM(ctx context.Context, domain weaver.JudgeDomain, items []string, similarLines []string, confidence float64) weaver.JudgeResult {
	systemPrompt := prompts.BuildJudgeSystemPrompt()
	userMessage := prompts.PackJudgeQuery(items, similarLines, string(domain))

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return judgeFallback(items, confidence)
	}

	result, ok := parseJudgeResponse(raw)
	if !ok || len(result.Operations) == 0 {
		return judgeFallback(items, confidence)
	}

	if confidence > 0 {
		result.Confidence = confidence
	}
	return result
}

func (a JudgeAgent) JudgeProfile(ctx context.Context, facts []ProfileFact) weaver.JudgeResult {
	items := make([]string, 0, len(facts))
	for _, fact := range facts {
		items = append(items, fact.Topic+" / "+fact.SubTopic+" = "+fact.Memo)
	}
	return judgeDeterministic(items, 1.0)
}

func (a JudgeAgent) JudgeTemporal(ctx context.Context, events []Event) weaver.JudgeResult {
	items := make([]string, 0, len(events))
	for _, event := range events {
		items = append(items, strings.Join([]string{
			event.Date, event.EventName, event.Desc, event.Year, event.Time, event.DateExpression,
		}, " | "))
	}
	return judgeDeterministic(items, 1.0)
}

func judgeDeterministic(items []string, confidence float64) weaver.JudgeResult {
	ops := make([]weaver.Operation, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		ops = append(ops, weaver.Operation{
			Type:    weaver.OperationAdd,
			Content: item,
			Reason:  "Deterministic extraction — no deduplication needed.",
		})
	}
	return weaver.JudgeResult{Operations: ops, Confidence: confidence}
}

func judgeDeterministicSummary(items []string, confidence float64) weaver.JudgeResult {
	ops := make([]weaver.Operation, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		ops = append(ops, weaver.Operation{
			Type:    weaver.OperationAdd,
			Content: item,
			Reason:  "No similar summary at or above 0.4 — defaulting to ADD.",
		})
	}
	if confidence == 0 {
		confidence = 0.8
	}
	return weaver.JudgeResult{Operations: ops, Confidence: confidence}
}

func (a JudgeAgent) fetchSimilarSummaries(ctx context.Context, items []string, userID string) map[string][]storage.SearchResult {
	out := make(map[string][]storage.SearchResult, len(items))
	if a.VectorStore == nil {
		return out
	}
	filters := map[string]any{"domain": string(weaver.DomainSummary)}
	if strings.TrimSpace(userID) != "" {
		filters["user_id"] = userID
	}
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		results, err := a.VectorStore.SearchByText(ctx, item, a.judgeTopK(), filters)
		if err != nil {
			out[item] = nil
			continue
		}
		out[item] = results
	}
	return out
}

func hasSummaryJudgeCandidates(matches map[string][]storage.SearchResult) bool {
	for _, results := range matches {
		for _, result := range results {
			if result.Score >= summaryJudgeSimilarityThreshold {
				return true
			}
		}
	}
	return false
}

func filterMatchesByThreshold(matches map[string][]storage.SearchResult, threshold float64) map[string][]storage.SearchResult {
	out := make(map[string][]storage.SearchResult, len(matches))
	for item, results := range matches {
		filtered := make([]storage.SearchResult, 0, len(results))
		for _, result := range results {
			if result.Score >= threshold {
				filtered = append(filtered, result)
			}
		}
		out[item] = filtered
	}
	return out
}

func formatSummarySimilarBlock(items []string, matches map[string][]storage.SearchResult) []string {
	if len(matches) == 0 {
		return nil
	}
	lines := make([]string, 0)
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		results := matches[item]
		lines = append(lines, fmt.Sprintf("For item: %q", item))
		if len(results) == 0 {
			lines = append(lines, "  - (no similar records above threshold)")
			continue
		}
		for _, result := range results {
			lines = append(lines, fmt.Sprintf("  - ID: %s | Score: %.2f | %q", result.ID, result.Score, result.Content))
		}
	}
	return lines
}

func judgeFallback(items []string, confidence float64) weaver.JudgeResult {
	ops := make([]weaver.Operation, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		ops = append(ops, weaver.Operation{
			Type:    weaver.OperationAdd,
			Content: item,
			Reason:  "LLM judge unavailable — defaulting to ADD.",
		})
	}
	if confidence == 0 {
		confidence = 0.8
	}
	return weaver.JudgeResult{Operations: ops, Confidence: confidence}
}

type judgeResponse struct {
	Operations []struct {
		Type        string `json:"type"`
		Content     string `json:"content"`
		EmbeddingID string `json:"embedding_id"`
		Reason      string `json:"reason"`
	} `json:"operations"`
	Confidence float64 `json:"confidence"`
}

func parseJudgeResponse(raw string) (weaver.JudgeResult, bool) {
	jsonStr := extractJSONObject(raw)
	if jsonStr == "" {
		return weaver.JudgeResult{}, false
	}

	var resp judgeResponse
	if err := json.Unmarshal([]byte(jsonStr), &resp); err != nil {
		return weaver.JudgeResult{}, false
	}

	ops := make([]weaver.Operation, 0, len(resp.Operations))
	for _, o := range resp.Operations {
		opType := weaver.OperationType(strings.ToUpper(strings.TrimSpace(o.Type)))
		switch opType {
		case weaver.OperationAdd, weaver.OperationUpdate, weaver.OperationDelete, weaver.OperationNoop:
		default:
			opType = weaver.OperationAdd
		}
		if strings.TrimSpace(o.Content) == "" && opType != weaver.OperationDelete {
			continue
		}
		ops = append(ops, weaver.Operation{
			Type:        opType,
			Content:     o.Content,
			EmbeddingID: o.EmbeddingID,
			Reason:      o.Reason,
		})
	}

	conf := resp.Confidence
	if conf <= 0 || conf > 1 {
		conf = 0.8
	}
	return weaver.JudgeResult{Operations: ops, Confidence: conf}, len(ops) > 0
}

// ---------- helpers ----------

func joinPipe(parts ...string) string {
	return strings.Join(parts, " | ")
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

func extractJSONArray(text string) string {
	text = strings.TrimSpace(text)
	start := strings.Index(text, "[")
	end := strings.LastIndex(text, "]")
	if start >= 0 && end > start {
		return text[start : end+1]
	}
	return ""
}
