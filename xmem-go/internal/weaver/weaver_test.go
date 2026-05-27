package weaver

import (
	"context"
	"testing"

	"github.com/xortexai/xmem-go/internal/graph"
	"github.com/xortexai/xmem-go/internal/storage"
)

func TestWeaverProfileAddStoresStructuredMetadata(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryVectorStore()
	w := Weaver{VectorStore: store, Embedder: storage.HashEmbedder{Dimension: 16}}

	result := w.Execute(ctx, JudgeResult{Operations: []Operation{{
		Type: OperationAdd, Content: "work / company = XMem", Reason: "test",
	}}, Confidence: 1}, DomainProfile, "alice")

	if result.Succeeded() != 1 {
		t.Fatalf("expected one success, got %#v", result)
	}
	records, err := store.SearchByMetadata(ctx, map[string]any{"user_id": "alice", "domain": "profile", "main_content": "work_company"}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 || records[0].Metadata["subcontent"] != "XMem" {
		t.Fatalf("unexpected records: %#v", records)
	}
}

func TestWeaverTemporalAddAndDelete(t *testing.T) {
	ctx := context.Background()
	graphStore := graph.NewMemoryTemporalStore()
	w := Weaver{TemporalStore: graphStore}

	added := w.Execute(ctx, JudgeResult{Operations: []Operation{{
		Type: OperationAdd, Content: "05-21 | Demo | Product demo | 2026 | 10:00 | today",
	}}}, DomainTemporal, "alice")
	if added.Succeeded() != 1 {
		t.Fatalf("expected add success: %#v", added)
	}
	events, _ := graphStore.SearchEventsByName(ctx, "Demo", "alice", 10)
	if len(events) != 1 {
		t.Fatalf("expected event to be stored, got %#v", events)
	}
	deleted := w.Execute(ctx, JudgeResult{Operations: []Operation{{
		Type: OperationDelete, EmbeddingID: "05-21_Demo",
	}}}, DomainTemporal, "alice")
	if deleted.Succeeded() != 1 {
		t.Fatalf("expected delete success: %#v", deleted)
	}
}

func TestMissingUpdateIDBecomesAdd(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryVectorStore()
	w := Weaver{VectorStore: store, Embedder: storage.HashEmbedder{Dimension: 16}}
	result := w.Execute(ctx, JudgeResult{Operations: []Operation{{
		Type: OperationUpdate, Content: "User likes coffee",
	}}}, DomainSummary, "alice")
	if result.Succeeded() != 1 || result.Executed[0].Type != OperationAdd {
		t.Fatalf("expected missing-id update to become add: %#v", result)
	}
}
