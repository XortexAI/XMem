package agents

import (
	"context"

	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
	"github.com/xortexai/xmem-go/internal/utils"
)

type Classification struct {
	Source string `json:"source"`
	Query  string `json:"query"`
}

type ClassifierAgent struct {
	Model models.ChatModel
}

func (a ClassifierAgent) Run(ctx context.Context, userQuery string, _ string) []Classification {
	if userQuery == "" {
		return nil
	}

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

	return out
}
