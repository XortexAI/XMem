package weaver

type OperationType string

const (
	OperationAdd    OperationType = "ADD"
	OperationUpdate OperationType = "UPDATE"
	OperationDelete OperationType = "DELETE"
	OperationNoop   OperationType = "NOOP"
)

type JudgeDomain string

const (
	DomainProfile  JudgeDomain = "profile"
	DomainTemporal JudgeDomain = "temporal"
	DomainSummary  JudgeDomain = "summary"
	DomainImage    JudgeDomain = "image"
	DomainCode     JudgeDomain = "code"
	DomainSnippet  JudgeDomain = "snippet"
)

type Operation struct {
	Type        OperationType `json:"type"`
	Content     string        `json:"content"`
	EmbeddingID string        `json:"embedding_id,omitempty"`
	Reason      string        `json:"reason"`
}

type JudgeResult struct {
	Operations []Operation `json:"operations"`
	Confidence float64     `json:"confidence"`
}

func (r JudgeResult) HasWrites() bool {
	for _, op := range r.Operations {
		if op.Type != OperationNoop {
			return true
		}
	}
	return false
}

type OpStatus string

const (
	StatusSuccess OpStatus = "success"
	StatusSkipped OpStatus = "skipped"
	StatusFailed  OpStatus = "failed"
)

type ExecutedOp struct {
	Type        OperationType `json:"type"`
	Status      OpStatus      `json:"status"`
	Content     string        `json:"content,omitempty"`
	EmbeddingID string        `json:"embedding_id,omitempty"`
	NewID       string        `json:"new_id,omitempty"`
	Error       string        `json:"error,omitempty"`
}

type WeaverResult struct {
	Executed []ExecutedOp `json:"executed"`
}

func (r WeaverResult) Total() int { return len(r.Executed) }

func (r WeaverResult) Succeeded() int {
	return r.count(StatusSuccess)
}

func (r WeaverResult) Skipped() int {
	return r.count(StatusSkipped)
}

func (r WeaverResult) Failed() int {
	return r.count(StatusFailed)
}

func (r WeaverResult) count(status OpStatus) int {
	n := 0
	for _, op := range r.Executed {
		if op.Status == status {
			n++
		}
	}
	return n
}
