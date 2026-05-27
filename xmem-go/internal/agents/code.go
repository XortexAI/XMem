package agents

import (
	"context"
	json "github.com/goccy/go-json"
	"strings"

	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
)

type CodeAgent struct {
	Model models.ChatModel
}

type codeAnnotationJSON struct {
	TargetSymbol   string `json:"target_symbol"`
	TargetFile     string `json:"target_file"`
	Content        string `json:"content"`
	AnnotationType string `json:"annotation_type"`
	Severity       string `json:"severity"`
	Repo           string `json:"repo"`
	AssignedToName string `json:"assigned_to_name"`
}

type codeAnnotationsResponse struct {
	Annotations []codeAnnotationJSON `json:"annotations"`
}

func (a CodeAgent) Run(ctx context.Context, text string) []string {
	if strings.TrimSpace(text) == "" {
		return nil
	}

	systemPrompt := prompts.BuildCodeSystemPrompt()
	userMessage := prompts.PackCodeQuery(text)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	items := parseCodeResponse(raw)
	results := make([]string, 0, len(items))
	for _, ann := range items {
		content := strings.TrimSpace(ann.Content)
		if content == "" {
			continue
		}
		line := joinPipe(
			defaultString(ann.AnnotationType, "explanation"),
			ann.TargetSymbol,
			ann.TargetFile,
			ann.Repo,
			ann.Severity,
			content,
		)
		results = append(results, line)
	}
	return results
}

func parseCodeResponse(raw string) []codeAnnotationJSON {
	jsonStr := extractJSONObject(raw)
	if jsonStr == "" {
		jsonStr = extractJSONArray(raw)
	}

	var resp codeAnnotationsResponse
	if err := json.Unmarshal([]byte(jsonStr), &resp); err == nil && len(resp.Annotations) > 0 {
		return resp.Annotations
	}
	return nil
}
