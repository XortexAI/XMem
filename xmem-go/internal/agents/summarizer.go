package agents

import (
	"context"
	"strings"

	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
)

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
	if strings.TrimSpace(userQuery) == "" && strings.TrimSpace(agentResponse) == "" {
		return nil
	}

	systemPrompt := prompts.BuildSummarizerSystemPrompt()
	userMessage := prompts.PackSummaryQuery(userQuery, agentResponse)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	trimmed := strings.TrimSpace(raw)
	if _, isEmpty := emptySentinels[trimmed]; isEmpty || trimmed == "" {
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
