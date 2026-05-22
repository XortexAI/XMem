package weaver

import (
	"context"
	"errors"
	"strings"

	"github.com/xortexai/xmem-go/internal/graph"
	"github.com/xortexai/xmem-go/internal/storage"
)

type Weaver struct {
	VectorStore        storage.VectorStore
	CodeVectorStore    storage.VectorStore
	SnippetVectorStore storage.VectorStore
	Embedder           storage.Embedder
	TemporalStore      graph.TemporalStore
}

func (w *Weaver) Execute(ctx context.Context, judge JudgeResult, domain JudgeDomain, userID string) WeaverResult {
	result := WeaverResult{}
	if len(judge.Operations) == 0 || !judge.HasWrites() {
		return result
	}

	if isBatchedVectorDomain(domain) && w.VectorStore != nil {
		result.Executed = append(result.Executed, w.executeBatchedVector(ctx, judge.Operations, domain, userID)...)
		return result
	}

	for _, op := range judge.Operations {
		result.Executed = append(result.Executed, w.executeOne(ctx, op, domain, userID))
	}
	return result
}

func (w *Weaver) executeBatchedVector(ctx context.Context, ops []Operation, domain JudgeDomain, userID string) []ExecutedOp {
	executed := []ExecutedOp{}
	addBatch := []Operation{}
	deleteBatch := []Operation{}

	flushAdd := func() {
		if len(addBatch) == 0 {
			return
		}
		if w.Embedder == nil {
			for _, op := range addBatch {
				executed = append(executed, failed(op, "No embedder provided"))
			}
			addBatch = nil
			return
		}
		if w.VectorStore == nil {
			for _, op := range addBatch {
				executed = append(executed, failed(op, "No vector store attached"))
			}
			addBatch = nil
			return
		}

		validOps := []Operation{}
		docs := []storage.VectorDocument{}
		for _, op := range addBatch {
			if strings.TrimSpace(op.Content) == "" {
				executed = append(executed, skipped(op, "ADD requires content"))
				continue
			}
			embedding, err := w.Embedder.Embed(ctx, op.Content)
			if err != nil {
				executed = append(executed, failed(op, err.Error()))
				continue
			}
			meta := vectorMetadata(domain, userID, op.Content)
			validOps = append(validOps, op)
			docs = append(docs, storage.VectorDocument{Text: op.Content, Embedding: embedding, Metadata: meta})
		}
		if len(docs) == 0 {
			addBatch = nil
			return
		}
		ids, err := w.VectorStore.Add(ctx, docs)
		if err != nil {
			for _, op := range validOps {
				executed = append(executed, failed(op, err.Error()))
			}
			addBatch = nil
			return
		}
		for i, op := range validOps {
			newID := ""
			if i < len(ids) {
				newID = ids[i]
			}
			executed = append(executed, successWithNewID(op, newID))
		}
		addBatch = nil
	}

	flushDelete := func() {
		if len(deleteBatch) == 0 {
			return
		}
		if w.VectorStore == nil {
			for _, op := range deleteBatch {
				executed = append(executed, failed(op, "No vector store attached"))
			}
			deleteBatch = nil
			return
		}
		validOps := []Operation{}
		ids := []string{}
		for _, op := range deleteBatch {
			if op.EmbeddingID == "" {
				executed = append(executed, failed(op, "DELETE missing embedding_id"))
				continue
			}
			validOps = append(validOps, op)
			ids = append(ids, op.EmbeddingID)
		}
		ok, err := w.VectorStore.Delete(ctx, ids)
		for _, op := range validOps {
			if err != nil {
				executed = append(executed, failed(op, err.Error()))
			} else if ok {
				executed = append(executed, success(op))
			} else {
				executed = append(executed, failed(op, "delete failed"))
			}
		}
		deleteBatch = nil
	}

	for _, original := range ops {
		op := normalizeMissingID(original)
		switch op.Type {
		case OperationNoop:
			flushAdd()
			flushDelete()
			executed = append(executed, skipped(op, ""))
		case OperationAdd:
			flushDelete()
			addBatch = append(addBatch, op)
		case OperationDelete:
			flushAdd()
			deleteBatch = append(deleteBatch, op)
		case OperationUpdate:
			flushAdd()
			flushDelete()
			executed = append(executed, w.executeOne(ctx, op, domain, userID))
		default:
			flushAdd()
			flushDelete()
			executed = append(executed, failed(op, "unknown operation type"))
		}
	}
	flushAdd()
	flushDelete()
	return executed
}

func (w *Weaver) executeOne(ctx context.Context, op Operation, domain JudgeDomain, userID string) ExecutedOp {
	if op.Type == OperationNoop {
		return skipped(op, "")
	}
	if op.Type == OperationAdd && strings.TrimSpace(op.Content) == "" {
		return skipped(op, "ADD requires content")
	}
	op = normalizeMissingID(op)

	switch domain {
	case DomainTemporal:
		return w.executeTemporal(ctx, op, userID)
	case DomainCode:
		return w.executeCode(ctx, op, userID)
	case DomainSnippet:
		return w.executeSnippet(ctx, op, userID)
	default:
		return w.executeVector(ctx, op, domain, userID)
	}
}

func (w *Weaver) executeVector(ctx context.Context, op Operation, domain JudgeDomain, userID string) ExecutedOp {
	if w.VectorStore == nil {
		return failed(op, "No vector store attached")
	}
	return vectorOp(ctx, w.VectorStore, w.Embedder, op, domain, userID)
}

func (w *Weaver) executeTemporal(ctx context.Context, op Operation, userID string) ExecutedOp {
	if w.TemporalStore == nil {
		return failed(op, "No temporal graph store attached")
	}
	event := parseTemporalContent(op.Content)
	if op.Type == OperationDelete {
		if err := w.TemporalStore.DeleteEvent(ctx, userID, op.EmbeddingID); err != nil {
			return failed(op, err.Error())
		}
		return success(op)
	}
	if event.Date == "" {
		return failed(op, "No date found in temporal content")
	}
	var err error
	switch op.Type {
	case OperationAdd:
		err = w.TemporalStore.CreateEvent(ctx, userID, event.Date, event)
	case OperationUpdate:
		err = w.TemporalStore.UpdateEvent(ctx, userID, event.Date, event)
	default:
		return skipped(op, "")
	}
	if err != nil {
		return failed(op, err.Error())
	}
	return success(op)
}

func (w *Weaver) executeCode(ctx context.Context, op Operation, userID string) ExecutedOp {
	store := w.CodeVectorStore
	if store == nil {
		store = w.VectorStore
	}
	if store == nil {
		return failed(op, "No vector store for code domain")
	}
	parsed := parseCodeAnnotationContent(op.Content)
	meta := map[string]any{
		"user_id":         userID,
		"domain":          "code",
		"annotation_type": parsed["annotation_type"],
		"target_symbol":   parsed["target_symbol"],
		"target_file":     parsed["target_file"],
		"repo":            parsed["repo"],
		"severity":        parsed["severity"],
	}
	return vectorOpWithMetadata(ctx, store, w.Embedder, op, DomainCode, userID, meta)
}

func (w *Weaver) executeSnippet(ctx context.Context, op Operation, userID string) ExecutedOp {
	store := w.SnippetVectorStore
	if store == nil {
		store = w.VectorStore
	}
	if store == nil {
		return failed(op, "No vector store for snippet domain")
	}
	parsed := parseSnippetContent(op.Content)
	searchable := parsed["content"]
	if searchable == "" {
		searchable = op.Content
	}
	meta := map[string]any{
		"user_id":      userID,
		"domain":       "snippet",
		"code_snippet": parsed["code_snippet"],
		"language":     parsed["language"],
		"snippet_type": parsed["snippet_type"],
		"tags":         parsed["tags"],
		"source":       "chat",
	}
	op.Content = searchable
	return vectorOpWithMetadata(ctx, store, w.Embedder, op, DomainSnippet, userID, meta)
}

func vectorOp(ctx context.Context, store storage.VectorStore, embedder storage.Embedder, op Operation, domain JudgeDomain, userID string) ExecutedOp {
	return vectorOpWithMetadata(ctx, store, embedder, op, domain, userID, vectorMetadata(domain, userID, op.Content))
}

func vectorOpWithMetadata(ctx context.Context, store storage.VectorStore, embedder storage.Embedder, op Operation, domain JudgeDomain, userID string, meta map[string]any) ExecutedOp {
	if embedder == nil && op.Type != OperationDelete {
		return failed(op, "No embedder provided")
	}
	switch op.Type {
	case OperationAdd:
		embedding, err := embedder.Embed(ctx, op.Content)
		if err != nil {
			return failed(op, err.Error())
		}
		ids, err := store.Add(ctx, []storage.VectorDocument{{Text: op.Content, Embedding: embedding, Metadata: meta}})
		if err != nil {
			return failed(op, err.Error())
		}
		newID := ""
		if len(ids) > 0 {
			newID = ids[0]
		}
		return successWithNewID(op, newID)
	case OperationUpdate:
		embedding, err := embedder.Embed(ctx, op.Content)
		if err != nil {
			return failed(op, err.Error())
		}
		ok, err := store.Update(ctx, op.EmbeddingID, storage.VectorDocument{Text: op.Content, Embedding: embedding, Metadata: meta})
		if err != nil {
			return failed(op, err.Error())
		}
		if ok {
			return success(op)
		}
		op.Type = OperationAdd
		op.EmbeddingID = ""
		return vectorOpWithMetadata(ctx, store, embedder, op, domain, userID, meta)
	case OperationDelete:
		ok, err := store.Delete(ctx, []string{op.EmbeddingID})
		if err != nil {
			return failed(op, err.Error())
		}
		if !ok {
			return failed(op, "delete failed")
		}
		return success(op)
	default:
		return skipped(op, "")
	}
}

func normalizeMissingID(op Operation) Operation {
	if (op.Type == OperationUpdate || op.Type == OperationDelete) && op.EmbeddingID == "" {
		op.Type = OperationAdd
	}
	return op
}

func isBatchedVectorDomain(domain JudgeDomain) bool {
	return domain == DomainProfile || domain == DomainSummary || domain == DomainImage
}

func vectorMetadata(domain JudgeDomain, userID string, content string) map[string]any {
	meta := map[string]any{"user_id": userID, "domain": string(domain)}
	for k, v := range ExtractStructuredMetadata(content) {
		meta[k] = v
	}
	return meta
}

func ExtractStructuredMetadata(content string) map[string]string {
	result := map[string]string{}
	if strings.Contains(content, " = ") {
		parts := strings.SplitN(content, " = ", 2)
		if strings.Contains(parts[0], " / ") {
			key := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(strings.TrimSpace(parts[0]), " / ", "_"), " ", "_"))
			result["main_content"] = key
			result["subcontent"] = strings.TrimSpace(parts[1])
			return result
		}
	}
	result["main_content"] = ""
	result["subcontent"] = strings.TrimSpace(content)
	return result
}

func parseTemporalContent(content string) graph.Event {
	parts := splitPipe(content)
	event := graph.Event{}
	if len(parts) > 0 {
		event.Date = parts[0]
	}
	if len(parts) > 1 {
		event.EventName = parts[1]
	}
	if len(parts) > 2 {
		event.Description = parts[2]
	}
	if len(parts) > 3 {
		event.Year = parts[3]
	}
	if len(parts) > 4 {
		event.Time = parts[4]
	}
	if len(parts) > 5 {
		event.DateExpression = parts[5]
	}
	return event
}

func parseSnippetContent(content string) map[string]string {
	parts := splitSpacedPipe(content)
	result := map[string]string{"snippet_type": "algorithm"}
	if len(parts) >= 5 {
		result["content"] = parts[0]
		result["code_snippet"] = parts[1]
		result["language"] = parts[2]
		result["snippet_type"] = parts[3]
		result["tags"] = parts[4]
	} else if len(parts) >= 3 {
		result["content"] = parts[0]
		result["code_snippet"] = parts[1]
		result["language"] = parts[2]
		result["tags"] = ""
	} else {
		result["content"] = content
		result["code_snippet"] = ""
		result["language"] = ""
		result["tags"] = ""
	}
	return result
}

func parseCodeAnnotationContent(content string) map[string]string {
	parts := splitPipe(content)
	result := map[string]string{"annotation_type": "explanation"}
	if len(parts) >= 6 {
		result["annotation_type"] = defaultString(parts[0], "explanation")
		result["target_symbol"] = parts[1]
		result["target_file"] = parts[2]
		result["repo"] = parts[3]
		result["severity"] = parts[4]
		result["content"] = parts[5]
	} else if len(parts) >= 2 {
		result["annotation_type"] = defaultString(parts[0], "explanation")
		result["content"] = strings.Join(parts[1:], " | ")
	} else {
		result["content"] = content
	}
	return result
}

func splitPipe(content string) []string {
	raw := strings.Split(content, "|")
	out := make([]string, 0, len(raw))
	for _, p := range raw {
		out = append(out, strings.TrimSpace(p))
	}
	return out
}

func splitSpacedPipe(content string) []string {
	if strings.Contains(content, " | ") {
		return splitPipe(content)
	}
	return []string{strings.TrimSpace(content)}
}

func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return strings.TrimSpace(value)
}

func success(op Operation) ExecutedOp {
	return ExecutedOp{Type: op.Type, Status: StatusSuccess, Content: op.Content, EmbeddingID: op.EmbeddingID}
}

func successWithNewID(op Operation, id string) ExecutedOp {
	ex := success(op)
	ex.NewID = id
	return ex
}

func skipped(op Operation, msg string) ExecutedOp {
	return ExecutedOp{Type: op.Type, Status: StatusSkipped, Content: op.Content, EmbeddingID: op.EmbeddingID, Error: msg}
}

func failed(op Operation, msg string) ExecutedOp {
	if msg == "" {
		msg = errors.New("operation failed").Error()
	}
	return ExecutedOp{Type: op.Type, Status: StatusFailed, Content: op.Content, EmbeddingID: op.EmbeddingID, Error: msg}
}
