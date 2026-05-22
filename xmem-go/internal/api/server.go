package api

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/xortexai/xmem-go/internal/config"
	"github.com/xortexai/xmem-go/internal/database"
	"github.com/xortexai/xmem-go/internal/jobs"
	"github.com/xortexai/xmem-go/internal/pipelines"
)

type Server struct {
	settings  config.Settings
	logger    *slog.Logger
	startedAt time.Time
	ready     bool
	initError string

	ingest    *pipelines.IngestPipeline
	retrieval *pipelines.RetrievalPipeline
	keys      database.APIKeyStore
	jobStore  jobs.Store
	limiter   *RateLimiter
}

func NewServer(settings config.Settings, logger *slog.Logger, ingest *pipelines.IngestPipeline, retrieval *pipelines.RetrievalPipeline, keys database.APIKeyStore, jobStores ...jobs.Store) *Server {
	jobStore := jobs.Store(jobs.NewMemoryStore())
	if len(jobStores) > 0 && jobStores[0] != nil {
		jobStore = jobStores[0]
	}
	return &Server{
		settings:  settings,
		logger:    logger,
		startedAt: time.Now(),
		ready:     true,
		ingest:    ingest,
		retrieval: retrieval,
		keys:      keys,
		jobStore:  jobStore,
		limiter:   NewRateLimiter(settings.RateLimit, time.Minute),
	}
}

func (s *Server) Handler() http.Handler {
	router := chi.NewRouter()
	router.Get("/health", s.health)
	router.With(s.memoryMiddleware).Post("/v1/memory/ingest", s.ingestMemory)
	router.With(s.memoryMiddleware).Get("/v1/memory/ingest/{jobID}/status", s.ingestJobStatus)
	router.With(s.memoryMiddleware).Post("/v1/memory/batch-ingest", s.batchIngestMemory)
	router.With(s.memoryMiddleware).Get("/v1/memory/jobs/{jobID}/status", s.memoryJobStatus)
	router.With(s.memoryMiddleware).Post("/v1/memory/retrieve", s.retrieveMemory)
	router.With(s.memoryMiddleware).Post("/v1/memory/search", s.searchMemory)
	return s.requestContext(s.securityHeaders(s.cors(s.maxBody(router))))
}
