package contracts

type APIResponse struct {
	Status    string   `json:"status"`
	RequestID string   `json:"request_id,omitempty"`
	Data      any      `json:"data,omitempty"`
	Error     string   `json:"error,omitempty"`
	ElapsedMS *float64 `json:"elapsed_ms,omitempty"`
}

type HealthResponse struct {
	Status         string   `json:"status"`
	PipelinesReady bool     `json:"pipelines_ready"`
	Version        string   `json:"version"`
	UptimeSeconds  *float64 `json:"uptime_seconds,omitempty"`
	Error          string   `json:"error,omitempty"`
}

type IngestRequest struct {
	UserQuery       string `json:"user_query"`
	AgentResponse   string `json:"agent_response,omitempty"`
	UserID          string `json:"user_id"`
	SessionDatetime string `json:"session_datetime,omitempty"`
	ImageURL        string `json:"image_url,omitempty"`
	EffortLevel     string `json:"effort_level,omitempty"`
}

type OperationDetail struct {
	Type    string `json:"type"`
	Content string `json:"content"`
	Reason  string `json:"reason"`
}

type WeaverSummary struct {
	Succeeded int `json:"succeeded"`
	Skipped   int `json:"skipped"`
	Failed    int `json:"failed"`
}

type DomainResult struct {
	Confidence float64           `json:"confidence"`
	Operations []OperationDetail `json:"operations"`
	Weaver     *WeaverSummary    `json:"weaver,omitempty"`
}

type IngestResponse struct {
	Model          string        `json:"model"`
	Classification []any         `json:"classification"`
	Profile        *DomainResult `json:"profile,omitempty"`
	Temporal       *DomainResult `json:"temporal,omitempty"`
	Summary        *DomainResult `json:"summary,omitempty"`
	Image          *DomainResult `json:"image,omitempty"`
}

type BatchIngestRequest struct {
	Items []IngestRequest `json:"items"`
}

type BatchIngestResponse struct {
	Results []IngestResponse `json:"results"`
}

type JobAcceptedResponse struct {
	JobID     string `json:"job_id"`
	Status    string `json:"status"`
	Created   bool   `json:"created"`
	StatusURL string `json:"status_url"`
}

type RetrieveRequest struct {
	Query  string `json:"query"`
	UserID string `json:"user_id"`
	TopK   int    `json:"top_k,omitempty"`
}

type SourceRecord struct {
	Domain   string         `json:"domain"`
	Content  string         `json:"content"`
	Score    float64        `json:"score"`
	Metadata map[string]any `json:"metadata"`
}

type RetrieveResponse struct {
	Model      string         `json:"model"`
	Answer     string         `json:"answer"`
	Sources    []SourceRecord `json:"sources"`
	Confidence float64        `json:"confidence"`
}

type SearchRequest struct {
	Query   string   `json:"query"`
	UserID  string   `json:"user_id"`
	Domains []string `json:"domains,omitempty"`
	TopK    int      `json:"top_k,omitempty"`
}

type SearchResponse struct {
	Results []SourceRecord `json:"results"`
	Total   int            `json:"total"`
}

type User struct {
	ID       string
	Name     string
	Email    string
	Username string
	APIKey   map[string]any
}
