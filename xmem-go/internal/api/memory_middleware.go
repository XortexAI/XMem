package api

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

func (s *Server) memoryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		if !s.ready || s.initError != "" {
			msg := "Pipelines are still loading. Retry shortly."
			if s.initError != "" {
				msg = "Pipeline initialisation failed: " + s.initError
			}
			writeError(w, r, http.StatusServiceUnavailable, msg, start)
			return
		}
		user, code, msg := s.authenticate(r)
		if code != http.StatusOK {
			writeError(w, r, code, msg, start)
			return
		}
		allowed, remaining := s.limiter.Check(user.ID)
		r = r.WithContext(context.WithValue(r.Context(), rateRemainingKey{}, remaining))
		if !allowed {
			w.Header().Set("Retry-After", "60")
			w.Header().Set("X-RateLimit-Limit", fmt.Sprint(s.settings.RateLimit))
			w.Header().Set("X-RateLimit-Remaining", "0")
			writeError(w, r, http.StatusTooManyRequests, "Rate limit exceeded. Try again later.", start)
			return
		}
		r = r.WithContext(context.WithValue(r.Context(), userKey{}, user))
		next.ServeHTTP(w, r)
	})
}
