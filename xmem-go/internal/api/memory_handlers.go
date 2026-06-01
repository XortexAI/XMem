package api

import (
	"context"
	"math"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/xortexai/xmem-go/internal/jobs"
)

func (s *Server) ingestMemory(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	user := userFromRequest(r)
	var req IngestRequest
	if !decodeJSON(w, r, &req, start) {
		return
	}
	if err := validateIngest(req); err != nil {
		writeError(w, r, http.StatusUnprocessableEntity, err.Error(), start)
		return
	}
	userID := effectiveUserID(user)
	job, created, err := s.jobStore.Enqueue(r.Context(), jobs.EnqueueInput{
		JobType: "memory_ingest",
		Payload: map[string]any{
			"user_query":        req.UserQuery,
			"agent_response":    req.AgentResponse,
			"user_id":           userID,
			"session_datetime":  req.SessionDatetime,
			"image_url":         req.ImageURL,
			"effort_level":      req.EffortLevel,
			"request_user_id":   req.UserID,
			"authenticated_uid": userID,
		},
		IdempotencyFields: map[string]any{
			"user_id":          userID,
			"user_query":       req.UserQuery,
			"agent_response":   req.AgentResponse,
			"session_datetime": req.SessionDatetime,
			"image_url":        req.ImageURL,
			"effort_level":     req.EffortLevel,
		},
		UserID:         userID,
		TimeoutSeconds: 120,
		MaxAttempts:    3,
	})
	if err != nil {
		writeError(w, r, http.StatusInternalServerError, err.Error(), start)
		return
	}
	s.scheduleJob(job, func(ctx context.Context) (any, error) {
		return s.ingest.Run(ctx, req, userID)
	})
	writeData(w, r, JobAcceptedResponse{
		JobID:     job.ID,
		Status:    string(job.Status),
		Created:   created,
		StatusURL: "/v1/memory/ingest/" + job.ID + "/status",
	}, start)
}

func (s *Server) batchIngestMemory(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	user := userFromRequest(r)
	var req BatchIngestRequest
	if !decodeJSON(w, r, &req, start) {
		return
	}
	if len(req.Items) == 0 || len(req.Items) > 100 {
		writeError(w, r, http.StatusUnprocessableEntity, "items must contain between 1 and 100 ingest requests", start)
		return
	}
	for _, item := range req.Items {
		if err := validateIngest(item); err != nil {
			writeError(w, r, http.StatusUnprocessableEntity, err.Error(), start)
			return
		}
	}
	userID := effectiveUserID(user)
	timeoutSeconds := math.Max(120, math.Min(float64(len(req.Items))*120, 3600))
	job, created, err := s.jobStore.Enqueue(r.Context(), jobs.EnqueueInput{
		JobType: "memory_batch_ingest",
		Payload: map[string]any{
			"items":             req.Items,
			"user_id":           userID,
			"authenticated_uid": userID,
		},
		IdempotencyFields: map[string]any{
			"user_id": userID,
			"items":   req.Items,
		},
		UserID:         userID,
		TimeoutSeconds: timeoutSeconds,
		MaxAttempts:    3,
	})
	if err != nil {
		writeError(w, r, http.StatusInternalServerError, err.Error(), start)
		return
	}
	s.scheduleJob(job, func(ctx context.Context) (any, error) {
		results := make([]IngestResponse, 0, len(req.Items))
		for _, item := range req.Items {
			data, err := s.ingest.Run(ctx, item, userID)
			if err != nil {
				return nil, err
			}
			results = append(results, data)
		}
		return BatchIngestResponse{Results: results}, nil
	})
	writeData(w, r, JobAcceptedResponse{
		JobID:     job.ID,
		Status:    string(job.Status),
		Created:   created,
		StatusURL: "/v1/memory/jobs/" + job.ID + "/status",
	}, start)
}

func (s *Server) ingestJobStatus(w http.ResponseWriter, r *http.Request) {
	s.memoryJobStatus(w, r)
}

func (s *Server) memoryJobStatus(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := effectiveUserID(userFromRequest(r))
	jobID := chi.URLParam(r, "jobID")
	job, ok, err := s.jobStore.Get(r.Context(), jobID)
	if err != nil {
		writeError(w, r, http.StatusInternalServerError, err.Error(), start)
		return
	}
	if !ok || job.UserID != userID {
		writeError(w, r, http.StatusNotFound, "Job not found.", start)
		return
	}
	writeData(w, r, jobs.Public(job), start)
}

func (s *Server) scheduleJob(job jobs.Job, handler jobs.Handler) {
	if job.Status != jobs.StatusQueued {
		return
	}
	go jobs.Run(context.Background(), s.jobStore, s.logger, job.ID, handler)
}

func (s *Server) retrieveMemory(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	user := userFromRequest(r)
	var req RetrieveRequest
	if !decodeJSON(w, r, &req, start) {
		return
	}
	if err := validateRetrieve(req); err != nil {
		writeError(w, r, http.StatusUnprocessableEntity, err.Error(), start)
		return
	}
	data, err := s.retrieval.Run(r.Context(), req, effectiveUserID(user))
	if err != nil {
		writeError(w, r, http.StatusInternalServerError, err.Error(), start)
		return
	}
	writeData(w, r, data, start)
}

func (s *Server) searchMemory(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	user := userFromRequest(r)
	var req SearchRequest
	if !decodeJSON(w, r, &req, start) {
		return
	}
	if err := validateSearch(req); err != nil {
		writeError(w, r, http.StatusUnprocessableEntity, err.Error(), start)
		return
	}
	data, err := s.retrieval.Search(r.Context(), req, effectiveUserID(user))
	if err != nil {
		writeError(w, r, http.StatusInternalServerError, err.Error(), start)
		return
	}
	writeData(w, r, data, start)
}
