package storage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
)

type HashEmbedder struct {
	Dimension int
}

func (e HashEmbedder) Embed(_ context.Context, text string) ([]float64, error) {
	dim := e.Dimension
	if dim <= 0 {
		dim = 64
	}
	vec := make([]float64, dim)
	words := strings.Fields(strings.ToLower(text))
	if len(words) == 0 {
		words = []string{text}
	}
	for _, word := range words {
		sum := sha256.Sum256([]byte(word))
		idx := int(binary.BigEndian.Uint64(sum[:8]) % uint64(dim))
		vec[idx] += 1
	}
	normalise(vec)
	return vec, nil
}

type MemoryVectorStore struct {
	mu      sync.RWMutex
	nextID  int
	records map[string]VectorDocument
}

func NewMemoryVectorStore() *MemoryVectorStore {
	return &MemoryVectorStore{nextID: 1, records: map[string]VectorDocument{}}
}

func (s *MemoryVectorStore) Add(_ context.Context, docs []VectorDocument) ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	ids := make([]string, 0, len(docs))
	for _, doc := range docs {
		id := doc.ID
		if id == "" {
			id = fmt.Sprintf("vec-%d", s.nextID)
			s.nextID++
		}
		doc.ID = id
		doc.Metadata = cloneMap(doc.Metadata)
		s.records[id] = doc
		ids = append(ids, id)
	}
	return ids, nil
}

func (s *MemoryVectorStore) Update(_ context.Context, id string, doc VectorDocument) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.records[id]; !ok {
		return false, nil
	}
	doc.ID = id
	doc.Metadata = cloneMap(doc.Metadata)
	s.records[id] = doc
	return true, nil
}

func (s *MemoryVectorStore) Delete(_ context.Context, ids []string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range ids {
		delete(s.records, id)
	}
	return true, nil
}

func (s *MemoryVectorStore) SearchByText(ctx context.Context, query string, topK int, filters map[string]any) ([]SearchResult, error) {
	embedder := HashEmbedder{Dimension: 64}
	queryVec, _ := embedder.Embed(ctx, query)

	s.mu.RLock()
	defer s.mu.RUnlock()

	results := make([]SearchResult, 0)
	for id, record := range s.records {
		if !metadataMatches(record.Metadata, filters) {
			continue
		}
		score := cosine(queryVec, record.Embedding)
		if score == 0 && strings.Contains(strings.ToLower(record.Text), strings.ToLower(query)) {
			score = 1
		}
		results = append(results, SearchResult{
			ID:       id,
			Content:  record.Text,
			Score:    score,
			Metadata: cloneMap(record.Metadata),
		})
	}
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	return limitResults(results, topK), nil
}

func (s *MemoryVectorStore) SearchByMetadata(_ context.Context, filters map[string]any, topK int) ([]SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	results := make([]SearchResult, 0)
	for id, record := range s.records {
		if !metadataMatches(record.Metadata, filters) {
			continue
		}
		results = append(results, SearchResult{
			ID:       id,
			Content:  record.Text,
			Score:    1,
			Metadata: cloneMap(record.Metadata),
		})
	}
	return limitResults(results, topK), nil
}

func metadataMatches(metadata map[string]any, filters map[string]any) bool {
	for key, expected := range filters {
		actual, ok := metadata[key]
		if !ok {
			return false
		}
		if fmt.Sprint(actual) != fmt.Sprint(expected) {
			return false
		}
	}
	return true
}

func limitResults(results []SearchResult, topK int) []SearchResult {
	if topK <= 0 || topK > len(results) {
		topK = len(results)
	}
	return append([]SearchResult(nil), results[:topK]...)
}

func cloneMap(input map[string]any) map[string]any {
	out := make(map[string]any, len(input))
	for k, v := range input {
		out[k] = v
	}
	return out
}

func cosine(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	score := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if score < 0 {
		return 0
	}
	if score > 1 {
		return 1
	}
	return score
}

func normalise(vec []float64) {
	var sum float64
	for _, v := range vec {
		sum += v * v
	}
	if sum == 0 {
		return
	}
	norm := math.Sqrt(sum)
	for i := range vec {
		vec[i] /= norm
	}
}
