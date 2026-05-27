package agents

import (
	"context"
	json "github.com/goccy/go-json"
	"fmt"
	"strings"

	"github.com/xortexai/xmem-go/internal/graph"
	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
	"github.com/xortexai/xmem-go/internal/storage"
	"github.com/xortexai/xmem-go/internal/weaver"
)

const summaryJudgeSimilarityThreshold = 0.4

type JudgeAgent struct {
	Model         models.ChatModel
	VectorStore   storage.VectorStore
	TemporalStore graph.TemporalStore
	TopK          int
}

func (a JudgeAgent) judgeTopK() int {
	if a.TopK <= 0 {
		return 1
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
		return judgeParseFallback(items)
	}

	if confidence > 0 {
		result.Confidence = confidence
	}
	return result
}

func (a JudgeAgent) JudgeProfile(ctx context.Context, facts []ProfileFact, userID string) weaver.JudgeResult {
	dedupedFacts := dedupeProfileItems(facts)
	ops := make([]weaver.Operation, 0, len(dedupedFacts))

	for _, fact := range dedupedFacts {
		itemStr := fact.Topic + " / " + fact.SubTopic + " = " + fact.Memo
		key := buildProfileMetadataKey(fact)
		if key == "" || a.VectorStore == nil {
			ops = append(ops, weaver.Operation{Type: weaver.OperationAdd, Content: itemStr, Reason: "No vector store or invalid metadata key — defaulting to ADD."})
			continue
		}

		filters := map[string]any{"domain": "profile", "main_content": key}
		if userID != "" {
			filters["user_id"] = userID
		}

		results, err := a.VectorStore.SearchByMetadata(ctx, filters, a.judgeTopK())
		if err != nil || len(results) == 0 {
			ops = append(ops, weaver.Operation{Type: weaver.OperationAdd, Content: itemStr, Reason: "No profile record with the same topic/sub_topic."})
			continue
		}

		match := results[0]
		incomingMemo := profileMemoFromContent(itemStr)
		existingMemo := profileMemoFromMatch(match)
		if normText(incomingMemo) == normText(existingMemo) {
			ops = append(ops, weaver.Operation{Type: weaver.OperationNoop, Content: itemStr, EmbeddingID: match.ID, Reason: "Existing profile fact is unchanged."})
		} else {
			ops = append(ops, weaver.Operation{Type: weaver.OperationUpdate, Content: itemStr, EmbeddingID: match.ID, Reason: "Existing profile fact has new content."})
		}
	}

	return weaver.JudgeResult{Operations: ops, Confidence: 1.0}
}

func (a JudgeAgent) JudgeTemporal(ctx context.Context, events []Event, userID string) weaver.JudgeResult {
	dedupedEvents := dedupeTemporalItems(events)
	ops := make([]weaver.Operation, 0, len(dedupedEvents))

	for _, event := range dedupedEvents {
		itemStr := strings.Join([]string{event.Date, event.EventName, event.Desc, event.Year, event.Time, event.DateExpression}, " | ")
		if event.EventName == "" || a.TemporalStore == nil {
			ops = append(ops, weaver.Operation{Type: weaver.OperationAdd, Content: itemStr, Reason: "No temporal store or invalid event name — defaulting to ADD."})
			continue
		}

		results, err := a.TemporalStore.SearchEventsByName(ctx, event.EventName, userID, a.judgeTopK())
		if err != nil || len(results) == 0 {
			ops = append(ops, weaver.Operation{Type: weaver.OperationAdd, Content: itemStr, Reason: "No temporal event with the same event_name."})
			continue
		}

		match := results[0]
		incoming := temporalFieldsFromContent(itemStr)
		existing := temporalFieldsFromMatch(match)
		if sameTemporalEvent(incoming, existing) {
			ops = append(ops, weaver.Operation{Type: weaver.OperationNoop, Content: itemStr, EmbeddingID: match.EmbeddingID(), Reason: "Existing temporal event is unchanged."})
		} else if normText(incoming.Date) != normText(existing.Date) {
			ops = append(ops, weaver.Operation{Type: weaver.OperationDelete, Content: strings.Join([]string{match.Date, match.EventName, match.Description, match.Year, match.Time, match.DateExpression}, " | "), EmbeddingID: match.EmbeddingID(), Reason: "Existing temporal event moved to a different date."})
			ops = append(ops, weaver.Operation{Type: weaver.OperationAdd, Content: itemStr, Reason: "Re-created temporal event on the updated date."})
		} else {
			ops = append(ops, weaver.Operation{Type: weaver.OperationUpdate, Content: itemStr, EmbeddingID: match.EmbeddingID(), Reason: "Existing temporal event has new content."})
		}
	}

	return weaver.JudgeResult{Operations: ops, Confidence: 1.0}
}

func judgeDeterministicSummary(items []string, confidence float64) weaver.JudgeResult {
	ops := make([]weaver.Operation, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		ops = append(ops, weaver.Operation{Type: weaver.OperationAdd, Content: item, Reason: "No similar summary at or above 0.4 — defaulting to ADD."})
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
			lines = append(lines, "  - (no similar records)")
			continue
		}
		for _, result := range results {
			lines = append(lines, fmt.Sprintf("  - ID: %s | Score: %.2f | %q", result.ID, result.Score, result.Content))
		}
	}
	return lines
}

func judgeParseFallback(items []string) weaver.JudgeResult {
	ops := make([]weaver.Operation, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		ops = append(ops, weaver.Operation{Type: weaver.OperationAdd, Content: item, Reason: "Fallback — JSON parse failed"})
	}
	return weaver.JudgeResult{Operations: ops, Confidence: 0.5}
}

func judgeFallback(items []string, confidence float64) weaver.JudgeResult {
	ops := make([]weaver.Operation, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		ops = append(ops, weaver.Operation{Type: weaver.OperationAdd, Content: item, Reason: "LLM judge unavailable — defaulting to ADD."})
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
		if strings.TrimSpace(o.Content) == "" && opType != weaver.OperationDelete && opType != weaver.OperationNoop {
			continue
		}
		ops = append(ops, weaver.Operation{Type: opType, Content: o.Content, EmbeddingID: o.EmbeddingID, Reason: o.Reason})
	}

	conf := resp.Confidence
	if conf <= 0 || conf > 1 {
		conf = 0.8
	}
	return weaver.JudgeResult{Operations: ops, Confidence: conf}, len(ops) > 0
}
