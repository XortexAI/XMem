package storage

import (
	"bytes"
	"context"
	json "github.com/goccy/go-json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/xortexai/xmem-go/internal/config"
	"github.com/xortexai/xmem-go/internal/utils"
)

type PineconeVectorStore struct {
	apiKey    string
	host      string
	indexName string
	namespace string
	dimension int
	embedder  Embedder
	client    *http.Client
}

func NewPineconeVectorStore(ctx context.Context, settings config.Settings, embedder Embedder, namespace string) (*PineconeVectorStore, error) {
	if settings.PineconeAPIKey == "" {
		return nil, errors.New("PINECONE_API_KEY is required")
	}
	if namespace == "" {
		namespace = settings.PineconeNamespace
	}
	store := &PineconeVectorStore{
		apiKey:    settings.PineconeAPIKey,
		host:      strings.TrimRight(settings.PineconeHost, "/"),
		indexName: settings.PineconeIndexName,
		namespace: namespace,
		dimension: settings.PineconeDimension,
		embedder:  embedder,
		client:    &http.Client{Timeout: 30 * time.Second},
	}
	if store.host == "" {
		host, err := store.resolveHost(ctx)
		if err != nil {
			return nil, err
		}
		store.host = host
	}
	if !strings.HasPrefix(store.host, "http") {
		store.host = "https://" + store.host
	}
	return store, nil
}

func (s *PineconeVectorStore) Add(ctx context.Context, docs []VectorDocument) ([]string, error) {
	vectors := make([]map[string]any, 0, len(docs))
	ids := make([]string, 0, len(docs))
	for i, doc := range docs {
		id := doc.ID
		if id == "" {
			id = fmt.Sprintf("vec-%d-%d", time.Now().UnixNano(), i)
		}
		meta := cloneMap(doc.Metadata)
		meta["content"] = doc.Text
		vectors = append(vectors, map[string]any{"id": id, "values": doc.Embedding, "metadata": meta})
		ids = append(ids, id)
	}
	var out map[string]any
	if err := s.do(ctx, http.MethodPost, "/vectors/upsert", map[string]any{
		"namespace": s.namespace,
		"vectors":   vectors,
	}, &out); err != nil {
		return nil, err
	}
	return ids, nil
}

func (s *PineconeVectorStore) Update(ctx context.Context, id string, doc VectorDocument) (bool, error) {
	meta := cloneMap(doc.Metadata)
	meta["content"] = doc.Text
	err := s.do(ctx, http.MethodPost, "/vectors/update", map[string]any{
		"namespace":   s.namespace,
		"id":          id,
		"values":      doc.Embedding,
		"setMetadata": meta,
	}, nil)
	return err == nil, err
}

func (s *PineconeVectorStore) Delete(ctx context.Context, ids []string) (bool, error) {
	err := s.do(ctx, http.MethodPost, "/vectors/delete", map[string]any{
		"namespace": s.namespace,
		"ids":       ids,
	}, nil)
	return err == nil, err
}

func (s *PineconeVectorStore) SearchByText(ctx context.Context, query string, topK int, filters map[string]any) ([]SearchResult, error) {
	if s.embedder == nil {
		return nil, errors.New("pinecone search requires an embedder")
	}
	vector, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}
	return s.query(ctx, vector, topK, filters)
}

func (s *PineconeVectorStore) SearchByMetadata(ctx context.Context, filters map[string]any, topK int) ([]SearchResult, error) {
	vector := make([]float64, s.dimension)
	return s.query(ctx, vector, topK, filters)
}

func (s *PineconeVectorStore) query(ctx context.Context, vector []float64, topK int, filters map[string]any) ([]SearchResult, error) {
	if topK <= 0 {
		topK = 10
	}
	var out struct {
		Matches []struct {
			ID       string         `json:"id"`
			Score    float64        `json:"score"`
			Metadata map[string]any `json:"metadata"`
		} `json:"matches"`
	}
	body := map[string]any{
		"namespace":       s.namespace,
		"vector":          vector,
		"topK":            topK,
		"includeMetadata": true,
		"filter":          pineconeFilter(filters),
	}
	if err := s.do(ctx, http.MethodPost, "/query", body, &out); err != nil {
		return nil, err
	}
	results := make([]SearchResult, 0, len(out.Matches))
	for _, match := range out.Matches {
		content, _ := match.Metadata["content"].(string)
		delete(match.Metadata, "content")
		results = append(results, SearchResult{ID: match.ID, Content: content, Score: match.Score, Metadata: match.Metadata})
	}
	return results, nil
}

func (s *PineconeVectorStore) resolveHost(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.pinecone.io/indexes/"+s.indexName, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Api-Key", s.apiKey)
	req.Header.Set("X-Pinecone-API-Version", "2025-01")
	resp, err := s.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("pinecone host lookup failed: %s: %s", resp.Status, string(b))
	}
	var data struct {
		Host string `json:"host"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", err
	}
	if data.Host == "" {
		return "", errors.New("pinecone host lookup returned empty host")
	}
	return "https://" + strings.TrimRight(data.Host, "/"), nil
}

func (s *PineconeVectorStore) do(ctx context.Context, method string, path string, body any, out any) error {
	return utils.RetryWithBackoff(ctx, 3, time.Second, func() error {
		return s.doOnce(ctx, method, path, body, out)
	})
}

func (s *PineconeVectorStore) doOnce(ctx context.Context, method string, path string, body any, out any) error {
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, s.host+path, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Api-Key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Pinecone-API-Version", "2025-01")
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("pinecone %s failed: %s: %s", path, resp.Status, string(b))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func pineconeFilter(filters map[string]any) map[string]any {
	if len(filters) == 0 {
		return nil
	}
	out := map[string]any{}
	for k, v := range filters {
		out[k] = map[string]any{"$eq": v}
	}
	return out
}
