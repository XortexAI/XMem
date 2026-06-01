package agents

import (
	"context"
	"strings"

	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
	"github.com/xortexai/xmem-go/internal/utils"
)

type ProfileFact struct {
	Topic    string
	SubTopic string
	Memo     string
}

type ProfilerAgent struct {
	Model models.ChatModel
}

func (a ProfilerAgent) Run(ctx context.Context, text string) []ProfileFact {
	if strings.TrimSpace(text) == "" {
		return nil
	}

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
