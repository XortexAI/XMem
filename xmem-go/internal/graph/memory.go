package graph

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
)

type MemoryTemporalStore struct {
	mu     sync.RWMutex
	events map[string]Event
}

func NewMemoryTemporalStore() *MemoryTemporalStore {
	return &MemoryTemporalStore{events: map[string]Event{}}
}

func (s *MemoryTemporalStore) CreateEvent(_ context.Context, userID string, date string, event Event) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	event.UserID = userID
	event.Date = date
	s.events[key(userID, date, event.EventName)] = event
	return nil
}

func (s *MemoryTemporalStore) UpdateEvent(ctx context.Context, userID string, date string, event Event) error {
	return s.CreateEvent(ctx, userID, date, event)
}

func (s *MemoryTemporalStore) DeleteEvent(_ context.Context, userID string, embeddingID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	date, name := splitEmbeddingID(embeddingID)
	delete(s.events, key(userID, date, name))
	return nil
}

func (s *MemoryTemporalStore) SearchEventsByName(_ context.Context, eventName string, userID string, topK int) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	eventName = strings.ToLower(strings.TrimSpace(eventName))
	out := []Event{}
	for _, event := range s.events {
		if event.UserID != userID {
			continue
		}
		name := strings.ToLower(event.EventName)
		if eventName == "" || strings.Contains(name, eventName) || strings.Contains(eventName, name) {
			event.SimilarityScore = 1
			out = append(out, event)
		}
	}
	return limitEvents(out, topK), nil
}

func (s *MemoryTemporalStore) SearchEventsByEmbedding(_ context.Context, userID string, queryText string, topK int, _ float64) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	query := strings.ToLower(queryText)
	out := []Event{}
	for _, event := range s.events {
		if event.UserID != userID {
			continue
		}
		text := strings.ToLower(event.EventName + " " + event.Description + " " + event.DateExpression)
		score := overlap(query, text)
		if score == 0 && query != "" {
			continue
		}
		event.SimilarityScore = score
		if event.SimilarityScore == 0 {
			event.SimilarityScore = 0.1
		}
		out = append(out, event)
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].SimilarityScore > out[j].SimilarityScore
	})
	return limitEvents(out, topK), nil
}

func key(userID, date, name string) string {
	return userID + "|" + date + "|" + strings.ToLower(strings.TrimSpace(name))
}

func splitEmbeddingID(id string) (string, string) {
	parts := strings.SplitN(id, "_", 2)
	if len(parts) == 1 {
		return parts[0], ""
	}
	return parts[0], parts[1]
}

func limitEvents(events []Event, topK int) []Event {
	if topK <= 0 || topK > len(events) {
		topK = len(events)
	}
	return append([]Event(nil), events[:topK]...)
}

func overlap(a, b string) float64 {
	aw := wordSet(a)
	bw := wordSet(b)
	if len(aw) == 0 || len(bw) == 0 {
		return 0
	}
	matches := 0
	for w := range aw {
		if bw[w] {
			matches++
		}
	}
	return float64(matches) / float64(len(aw)+len(bw)-matches)
}

func wordSet(s string) map[string]bool {
	out := map[string]bool{}
	for _, word := range strings.FieldsFunc(s, func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	}) {
		if len(word) > 2 {
			out[strings.ToLower(word)] = true
		}
	}
	return out
}

func (e Event) EmbeddingID() string {
	return fmt.Sprintf("%s_%s", e.Date, e.EventName)
}
