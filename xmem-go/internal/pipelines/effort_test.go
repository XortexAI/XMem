package pipelines

import "testing"

func TestHighEffortChunkTextProducesOverlappingChunks(t *testing.T) {
	text := "This is the first sentence. This is the second sentence. This is the third sentence with extra words."
	chunks := ChunkText(text, 8, 2)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	if chunks[0] == "" || chunks[1] == "" {
		t.Fatalf("chunks must not be empty: %#v", chunks)
	}
}
