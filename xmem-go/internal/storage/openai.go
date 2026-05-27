package storage

import (
	"bytes"
	"context"
	json "github.com/goccy/go-json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/xortexai/xmem-go/internal/config"
)

type OpenAIEmbedder struct {
	apiKey     string
	model      string
	dimensions int
	client     *http.Client
}

func NewOpenAIEmbedder(settings config.Settings) (*OpenAIEmbedder, error) {
	if settings.OpenAIAPIKey == "" {
		return nil, errors.New("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
	}
	model := settings.OpenAIEmbeddingModel
	if model == "" {
		model = settings.EmbeddingModel
	}
	if model == "" {
		model = "text-embedding-3-small"
	}
	return &OpenAIEmbedder{
		apiKey:     settings.OpenAIAPIKey,
		model:      model,
		dimensions: settings.PineconeDimension,
		client:     &http.Client{Timeout: 60 * time.Second},
	}, nil
}

func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	body := map[string]any{
		"model": e.model,
		"input": text,
	}
	if e.dimensions > 0 {
		body["dimensions"] = e.dimensions
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/embeddings", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+e.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai embedding failed: %s: %s", resp.Status, string(b))
	}
	var out struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 || len(out.Data[0].Embedding) == 0 {
		return nil, errors.New("openai embedding response was empty")
	}
	return out.Data[0].Embedding, nil
}
