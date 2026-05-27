package agents

import (
	"context"
	"fmt"
	"time"

	"github.com/xortexai/xmem-go/internal/models"
)

const llmCallTimeout = 45 * time.Second

func callModel(ctx context.Context, model models.ChatModel, systemPrompt, userMessage string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, llmCallTimeout)
	defer cancel()
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
