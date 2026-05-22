package graph

import (
	"context"
	"strings"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/xortexai/xmem-go/internal/config"
)

type Neo4jTemporalStore struct {
	driver neo4j.DriverWithContext
	db     string
}

func NewNeo4jTemporalStore(ctx context.Context, settings config.Settings) (*Neo4jTemporalStore, error) {
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
	store := &Neo4jTemporalStore{driver: driver}
	_, _ = neo4j.ExecuteQuery(ctx, driver,
		"CREATE CONSTRAINT xmem_go_event_key IF NOT EXISTS FOR (e:XMemGoEvent) REQUIRE (e.user_id, e.date, e.event_name) IS UNIQUE",
		nil, neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithWritersRouting())
	return store, nil
}

func (s *Neo4jTemporalStore) Close(ctx context.Context) error {
	return s.driver.Close(ctx)
}

func (s *Neo4jTemporalStore) CreateEvent(ctx context.Context, userID string, date string, event Event) error {
	return s.upsertEvent(ctx, userID, date, event)
}

func (s *Neo4jTemporalStore) UpdateEvent(ctx context.Context, userID string, date string, event Event) error {
	return s.upsertEvent(ctx, userID, date, event)
}

func (s *Neo4jTemporalStore) DeleteEvent(ctx context.Context, userID string, embeddingID string) error {
	date, name := splitEmbeddingID(embeddingID)
	_, err := neo4j.ExecuteQuery(ctx, s.driver, `
		MATCH (e:XMemGoEvent {user_id: $user_id, date: $date, event_name: $event_name})
		DELETE e
	`, map[string]any{"user_id": userID, "date": date, "event_name": name},
		neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithWritersRouting())
	return err
}

func (s *Neo4jTemporalStore) SearchEventsByName(ctx context.Context, eventName string, userID string, topK int) ([]Event, error) {
	if topK <= 0 {
		topK = 10
	}
	result, err := neo4j.ExecuteQuery(ctx, s.driver, `
		MATCH (e:XMemGoEvent {user_id: $user_id})
		WHERE toLower(e.event_name) CONTAINS toLower($event_name)
		RETURN e.date AS date, e.event_name AS event_name, e.desc AS desc, e.year AS year,
		       e.time AS time, e.date_expression AS date_expression, 1.0 AS score
		LIMIT $top_k
	`, map[string]any{"user_id": userID, "event_name": eventName, "top_k": topK},
		neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithReadersRouting())
	if err != nil {
		return nil, err
	}
	return recordsToEvents(result.Records), nil
}

func (s *Neo4jTemporalStore) SearchEventsByEmbedding(ctx context.Context, userID string, queryText string, topK int, _ float64) ([]Event, error) {
	if topK <= 0 {
		topK = 10
	}
	terms := strings.ToLower(queryText)
	result, err := neo4j.ExecuteQuery(ctx, s.driver, `
		MATCH (e:XMemGoEvent {user_id: $user_id})
		WITH e,
		     CASE
		       WHEN toLower(coalesce(e.event_name, '') + ' ' + coalesce(e.desc, '') + ' ' + coalesce(e.date_expression, '')) CONTAINS $query THEN 1.0
		       ELSE 0.25
		     END AS score
		RETURN e.date AS date, e.event_name AS event_name, e.desc AS desc, e.year AS year,
		       e.time AS time, e.date_expression AS date_expression, score
		ORDER BY score DESC
		LIMIT $top_k
	`, map[string]any{"user_id": userID, "query": terms, "top_k": topK},
		neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithReadersRouting())
	if err != nil {
		return nil, err
	}
	return recordsToEvents(result.Records), nil
}

func (s *Neo4jTemporalStore) upsertEvent(ctx context.Context, userID string, date string, event Event) error {
	_, err := neo4j.ExecuteQuery(ctx, s.driver, `
		MERGE (e:XMemGoEvent {user_id: $user_id, date: $date, event_name: $event_name})
		SET e.desc = $desc,
		    e.year = $year,
		    e.time = $time,
		    e.date_expression = $date_expression,
		    e.updated_at = datetime()
	`, map[string]any{
		"user_id": userID, "date": date, "event_name": event.EventName,
		"desc": event.Description, "year": event.Year, "time": event.Time,
		"date_expression": event.DateExpression,
	}, neo4j.EagerResultTransformer, neo4j.ExecuteQueryWithWritersRouting())
	return err
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
			SimilarityScore: asFloat(record, "score"),
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
