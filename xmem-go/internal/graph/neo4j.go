package graph

import (
	"context"
	"strings"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/xortexai/xmem-go/internal/config"
	"github.com/xortexai/xmem-go/internal/storage"
	"github.com/xortexai/xmem-go/internal/utils"
)

type Neo4jTemporalStore struct {
	driver   neo4j.DriverWithContext
	embedder storage.Embedder
}

func NewNeo4jTemporalStore(ctx context.Context, settings config.Settings, embedder storage.Embedder) (*Neo4jTemporalStore, error) {
	driver, err := neo4j.NewDriverWithContext(
		settings.Neo4jURI,
		neo4j.BasicAuth(settings.Neo4jUsername, settings.Neo4jPassword, ""),
	)
	if err != nil {
		return nil, err
	}
	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := driver.VerifyConnectivity(pingCtx); err != nil {
		_ = driver.Close(context.Background())
		return nil, err
	}
	store := &Neo4jTemporalStore{driver: driver, embedder: embedder}
	store.initSchema(ctx)
	return store, nil
}

func (s *Neo4jTemporalStore) initSchema(ctx context.Context) {
	constraints := []string{
		"CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
		"CREATE CONSTRAINT date_val_unique IF NOT EXISTS FOR (d:Date) REQUIRE d.date IS UNIQUE",
	}
	for _, q := range constraints {
		_, _ = neo4j.ExecuteQuery(ctx, s.driver, q, nil,
			neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithWritersRouting())
	}
}

func (s *Neo4jTemporalStore) Close(ctx context.Context) error {
	return s.driver.Close(ctx)
}

func (s *Neo4jTemporalStore) CreateEvent(ctx context.Context, userID string, date string, event Event) error {
	return s.withRetry(ctx, func() error {
		return s.createEventOnce(ctx, userID, date, event)
	})
}

func (s *Neo4jTemporalStore) createEventOnce(ctx context.Context, userID string, date string, event Event) error {
	props := s.buildEventProps(ctx, event)

	_, err := neo4j.ExecuteQuery(ctx, s.driver, `
		MERGE (u:User {user_id: $user_id})
		MERGE (d:Date {date: $date_str})
		CREATE (u)-[r:HAS_EVENT]->(d)
		SET r += $props
	`, map[string]any{
		"user_id":  userID,
		"date_str": date,
		"props":    props,
	}, neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithWritersRouting())
	return err
}

func (s *Neo4jTemporalStore) UpdateEvent(ctx context.Context, userID string, date string, event Event) error {
	return s.withRetry(ctx, func() error {
		return s.updateEventOnce(ctx, userID, date, event)
	})
}

func (s *Neo4jTemporalStore) updateEventOnce(ctx context.Context, userID string, date string, event Event) error {
	props := s.buildEventProps(ctx, event)

	_, err := neo4j.ExecuteQuery(ctx, s.driver, `
		MATCH (u:User {user_id: $user_id})
		      -[r:HAS_EVENT]->
		      (d:Date {date: $date_str})
		SET r += $props
	`, map[string]any{
		"user_id":  userID,
		"date_str": date,
		"props":    props,
	}, neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithWritersRouting())
	return err
}

func (s *Neo4jTemporalStore) DeleteEvent(ctx context.Context, userID string, embeddingID string) error {
	return s.withRetry(ctx, func() error {
		return s.deleteEventOnce(ctx, userID, embeddingID)
	})
}

func (s *Neo4jTemporalStore) deleteEventOnce(ctx context.Context, userID string, embeddingID string) error {
	date, name := splitEmbeddingID(embeddingID)
	if name != "" {
		_, err := neo4j.ExecuteQuery(ctx, s.driver, `
			MATCH (u:User {user_id: $user_id})
			      -[r:HAS_EVENT {event_name: $event_name}]->
			      (d:Date {date: $date_str})
			DELETE r
		`, map[string]any{"user_id": userID, "date_str": date, "event_name": name},
			neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithWritersRouting())
		return err
	}
	_, err := neo4j.ExecuteQuery(ctx, s.driver, `
		MATCH (u:User {user_id: $user_id})
		      -[r:HAS_EVENT]->
		      (d:Date {date: $date_str})
		DELETE r
	`, map[string]any{"user_id": userID, "date_str": date},
		neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithWritersRouting())
	return err
}

func (s *Neo4jTemporalStore) SearchEventsByName(ctx context.Context, eventName string, userID string, topK int) ([]Event, error) {
	if topK <= 0 {
		topK = 10
	}
	var result *neo4j.EagerResult
	var err error
	retryErr := s.withRetry(ctx, func() error {
		result, err = neo4j.ExecuteQuery(ctx, s.driver, `
			MATCH (u:User {user_id: $user_id})
			      -[r:HAS_EVENT]->
			      (d:Date)
			WHERE toLower(r.event_name) = toLower($event_name)
			RETURN r.event_name AS event_name, r.desc AS desc, r.year AS year,
			       r.time AS time, r.date_expression AS date_expression,
			       d.date AS date
			LIMIT $top_k
		`, map[string]any{"user_id": userID, "event_name": eventName, "top_k": topK},
			neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithReadersRouting())
		return err
	})
	if retryErr != nil {
		return nil, retryErr
	}
	return recordsToEvents(result.Records), nil
}

func (s *Neo4jTemporalStore) SearchEventsByEmbedding(ctx context.Context, userID string, queryText string, topK int, threshold float64) ([]Event, error) {
	if topK <= 0 {
		topK = 10
	}

	if s.embedder == nil {
		return s.searchEventsByKeyword(ctx, userID, queryText, topK)
	}

	queryEmbedding, err := s.embedder.Embed(ctx, queryText)
	if err != nil || len(queryEmbedding) == 0 {
		return s.searchEventsByKeyword(ctx, userID, queryText, topK)
	}

	var result *neo4j.EagerResult
	retryErr := s.withRetry(ctx, func() error {
		result, err = neo4j.ExecuteQuery(ctx, s.driver, `
			MATCH (u:User {user_id: $user_id})
			      -[r:HAS_EVENT]->
			      (d:Date)
			WHERE r.embedding IS NOT NULL
			  AND size(r.embedding) = size($query_embedding)
			WITH r, d,
			     2.0 * vector.similarity.cosine(r.embedding, $query_embedding) - 1.0
			     AS similarity_score
			WHERE similarity_score >= $similarity_threshold
			RETURN r.event_name AS event_name, r.desc AS desc, r.year AS year,
			       r.time AS time, r.date_expression AS date_expression,
			       d.date AS date, similarity_score
			ORDER BY similarity_score DESC
			LIMIT $top_k
		`, map[string]any{
			"user_id":              userID,
			"query_embedding":      queryEmbedding,
			"similarity_threshold": threshold,
			"top_k":                topK,
		}, neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithReadersRouting())
		return err
	})
	if retryErr != nil {
		return nil, retryErr
	}
	return recordsToEvents(result.Records), nil
}

func (s *Neo4jTemporalStore) searchEventsByKeyword(ctx context.Context, userID string, queryText string, topK int) ([]Event, error) {
	terms := strings.ToLower(queryText)
	var result *neo4j.EagerResult
	var err error
	retryErr := s.withRetry(ctx, func() error {
		result, err = neo4j.ExecuteQuery(ctx, s.driver, `
			MATCH (u:User {user_id: $user_id})
			      -[r:HAS_EVENT]->
			      (d:Date)
			WITH r, d,
			     CASE
			       WHEN toLower(coalesce(r.event_name, '') + ' ' + coalesce(r.desc, '') + ' ' + coalesce(r.date_expression, '')) CONTAINS $query THEN 1.0
			       ELSE 0.25
			     END AS similarity_score
			WHERE similarity_score > 0.25
			RETURN r.event_name AS event_name, r.desc AS desc, r.year AS year,
			       r.time AS time, r.date_expression AS date_expression,
			       d.date AS date, similarity_score
			ORDER BY similarity_score DESC
			LIMIT $top_k
		`, map[string]any{"user_id": userID, "query": terms, "top_k": topK},
			neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithReadersRouting())
		return err
	})
	if retryErr != nil {
		return nil, retryErr
	}
	return recordsToEvents(result.Records), nil
}

func (s *Neo4jTemporalStore) buildEventProps(ctx context.Context, event Event) map[string]any {
	props := map[string]any{
		"event_name":      event.EventName,
		"desc":            event.Description,
		"year":            event.Year,
		"time":            event.Time,
		"date_expression": event.DateExpression,
	}

	if s.embedder != nil {
		searchable := event.EventName
		if event.Description != "" {
			searchable = event.EventName + ": " + event.Description
		}
		if searchable != "" {
			if embedding, err := s.embedder.Embed(ctx, searchable); err == nil {
				props["embedding"] = embedding
			}
		}
	}

	return props
}

func (s *Neo4jTemporalStore) withRetry(ctx context.Context, fn func() error) error {
	return utils.RetryWithBackoff(ctx, 3, time.Second, fn)
}

func recordsToEvents(records []*neo4j.Record) []Event {
	out := make([]Event, 0, len(records))
	for _, record := range records {
		out = append(out, Event{
			Date:            asString(record, "date"),
			EventName:       asString(record, "event_name"),
			Description:     asString(record, "desc"),
			Year:            asString(record, "year"),
			Time:            asString(record, "time"),
			DateExpression:  asString(record, "date_expression"),
			SimilarityScore: asFloat(record, "similarity_score"),
		})
	}
	return out
}

func asString(record *neo4j.Record, key string) string {
	value, _ := record.Get(key)
	if value == nil {
		return ""
	}
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}

func asFloat(record *neo4j.Record, key string) float64 {
	value, _ := record.Get(key)
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int64:
		return float64(v)
	case int:
		return float64(v)
	default:
		return 0
	}
}
