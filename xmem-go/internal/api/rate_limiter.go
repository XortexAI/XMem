package api

import (
	"sync"
	"time"
)

type RateLimiter struct {
	mu       sync.Mutex
	limit    int
	window   time.Duration
	requests map[string][]time.Time
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	if limit <= 0 {
		limit = 60
	}
	return &RateLimiter{limit: limit, window: window, requests: map[string][]time.Time{}}
}

func (l *RateLimiter) Check(key string) (bool, int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	cutoff := now.Add(-l.window)
	hits := l.requests[key]
	kept := hits[:0]
	for _, hit := range hits {
		if hit.After(cutoff) {
			kept = append(kept, hit)
		}
	}
	if len(kept) >= l.limit {
		l.requests[key] = kept
		return false, 0
	}
	kept = append(kept, now)
	l.requests[key] = kept
	return true, l.limit - len(kept)
}
