package storage

import "context"

type SearchResult struct {
	ID       string
	Content  string
	Score    float64
	Metadata map[string]any
}

type VectorDocument struct {
	ID        string
	Text      string
	Embedding []float64
	Metadata  map[string]any
}

type VectorStore interface {
	Add(ctx context.Context, docs []VectorDocument) ([]string, error)
	Update(ctx context.Context, id string, doc VectorDocument) (bool, error)
	Delete(ctx context.Context, ids []string) (bool, error)
	SearchByText(ctx context.Context, query string, topK int, filters map[string]any) ([]SearchResult, error)
	SearchByMetadata(ctx context.Context, filters map[string]any, topK int) ([]SearchResult, error)
}

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float64, error)
}
