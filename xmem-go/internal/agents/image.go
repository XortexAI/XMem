package agents

import (
	"context"
	"strings"

	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
	"github.com/xortexai/xmem-go/internal/utils"
)

type ImageAgent struct {
	Model models.ChatModel
}

func (a ImageAgent) Run(ctx context.Context, query string, imageURL string) []string {
	if strings.TrimSpace(query) == "" && strings.TrimSpace(imageURL) == "" {
		return nil
	}

	systemPrompt := prompts.BuildImageSystemPrompt()
	userText := prompts.PackImageQuery(query, "")

	var raw string
	var err error
	if strings.TrimSpace(imageURL) != "" {
		ctx, cancel := context.WithTimeout(ctx, llmCallTimeout)
		defer cancel()
		resp, visionErr := a.Model.GenerateVision(ctx, systemPrompt, userText, imageURL)
		if visionErr != nil {
			return nil
		}
		raw = resp.Content
	} else {
		raw, err = callModel(ctx, a.Model, systemPrompt, userText)
		if err != nil {
			return nil
		}
	}

	result := utils.ParseRawResponseToImage(raw)
	items := make([]string, 0, len(result.Observations)+1)
	if strings.TrimSpace(result.Description) != "" {
		items = append(items, "[Image] "+result.Description)
	}
	for _, obs := range result.Observations {
		entry := "[Image/" + obs.Category + "] " + obs.Description
		if obs.Confidence != "" {
			entry += " (" + obs.Confidence + ")"
		}
		items = append(items, entry)
	}
	return items
}
