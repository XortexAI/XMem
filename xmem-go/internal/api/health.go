package api

import (
	"fmt"
	"net/http"
	"time"
)

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	uptime := time.Since(s.startedAt).Seconds()
	status := "ready"
	code := http.StatusOK
	envelopeStatus := "ok"
	if !s.ready {
		status = "loading"
		code = http.StatusServiceUnavailable
		envelopeStatus = "error"
	}
	if s.initError != "" {
		status = "error"
		code = http.StatusServiceUnavailable
		envelopeStatus = "error"
	}
	w.Header().Set("X-Response-Time-Ms", fmt.Sprintf("%.2f", elapsedMS(start)))
	writeJSON(w, r, code, APIResponse{
		Status: envelopeStatus,
		Data: HealthResponse{
			Status:         status,
			PipelinesReady: s.ready && s.initError == "",
			Version:        "1.0.0",
			UptimeSeconds:  &uptime,
			Error:          s.initError,
		},
	})
}
