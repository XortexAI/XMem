package api

import (
	json "github.com/goccy/go-json"
	"fmt"
	"net/http"
	"time"
)

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any, start time.Time) bool {
	defer r.Body.Close()
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		writeError(w, r, http.StatusUnprocessableEntity, err.Error(), start)
		return false
	}
	return true
}

func writeData(w http.ResponseWriter, r *http.Request, data any, start time.Time) {
	elapsed := elapsedMS(start)
	w.Header().Set("X-Response-Time-Ms", fmt.Sprintf("%.2f", elapsed))
	writeJSON(w, r, http.StatusOK, APIResponse{Status: "ok", RequestID: requestID(r), Data: data, ElapsedMS: &elapsed})
}

func writeError(w http.ResponseWriter, r *http.Request, code int, msg string, start time.Time) {
	elapsed := elapsedMS(start)
	w.Header().Set("X-Response-Time-Ms", fmt.Sprintf("%.2f", elapsed))
	writeJSON(w, r, code, APIResponse{Status: "error", RequestID: requestID(r), Error: msg, ElapsedMS: &elapsed})
}

func writeJSON(w http.ResponseWriter, r *http.Request, code int, body APIResponse) {
	if body.RequestID == "" {
		body.RequestID = requestID(r)
	}
	if remaining, ok := r.Context().Value(rateRemainingKey{}).(int); ok {
		w.Header().Set("X-RateLimit-Remaining", fmt.Sprint(remaining))
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(body)
}

func elapsedMS(start time.Time) float64 {
	return float64(time.Since(start).Microseconds()) / 1000
}

type requestIDKey struct{}
type rateRemainingKey struct{}
type userKey struct{}

func requestID(r *http.Request) string {
	id, _ := r.Context().Value(requestIDKey{}).(string)
	return id
}

func userFromRequest(r *http.Request) User {
	user, _ := r.Context().Value(userKey{}).(User)
	return user
}
