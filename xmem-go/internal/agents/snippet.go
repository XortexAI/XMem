package agents

import (
	"context"
	json "github.com/goccy/go-json"
	"fmt"
	"strings"

	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
)

type SnippetAgent struct {
	Model models.ChatModel
}

type snippetJSON struct {
	Content     string `json:"content"`
	CodeSnippet string `json:"code_snippet"`
	Language    string `json:"language"`
	SnippetType string `json:"snippet_type"`
	Tags        any    `json:"tags"`
}

type snippetsResponse struct {
	Snippets []snippetJSON `json:"snippets"`
}

func (a SnippetAgent) Run(ctx context.Context, text string) []string {
	if strings.TrimSpace(text) == "" {
		return nil
	}

	systemPrompt := prompts.BuildSnippetSystemPrompt()
	userMessage := prompts.PackSnippetQuery(text)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	items := parseSnippetResponse(raw)
	results := make([]string, 0, len(items))
	for _, s := range items {
		content := strings.TrimSpace(s.Content)
		if content == "" {
			continue
		}
		line := joinPipe(content, s.CodeSnippet, s.Language, defaultString(s.SnippetType, "algorithm"), normalizeTags(s.Tags))
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
	return nil
}

func normalizeTags(tags any) string {
	switch v := tags.(type) {
	case nil:
		return ""
	case string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, tag := range v {
			cleaned := strings.TrimSpace(fmt.Sprintf("%v", tag))
			if cleaned != "" {
				out = append(out, cleaned)
			}
		}
		if len(out) > 10 {
			out = out[:10]
		}
		return strings.Join(out, ",")
	default:
		return strings.TrimSpace(fmt.Sprintf("%v", v))
	}
}
