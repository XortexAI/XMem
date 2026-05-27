package graph

import "context"

type Event struct {
	UserID          string
	Date            string
	EventName       string
	Description     string
	Year            string
	Time            string
	DateExpression  string
	Embedding       []float64
	SimilarityScore float64
}

type TemporalStore interface {
	CreateEvent(ctx context.Context, userID string, date string, event Event) error
	UpdateEvent(ctx context.Context, userID string, date string, event Event) error
	DeleteEvent(ctx context.Context, userID string, embeddingID string) error
	SearchEventsByName(ctx context.Context, eventName string, userID string, topK int) ([]Event, error)
	SearchEventsByEmbedding(ctx context.Context, userID string, queryText string, topK int, threshold float64) ([]Event, error)
}
