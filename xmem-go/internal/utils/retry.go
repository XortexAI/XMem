package utils

import (
	"context"
	"strings"
	"time"
)

func RetryWithBackoff(ctx context.Context, maxRetries int, baseDelay time.Duration, fn func() error) error {
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		lastErr = fn()
		if lastErr == nil {
			return nil
		}
		if !isRetryable(lastErr) {
			return lastErr
		}
		if attempt < maxRetries-1 {
			delay := baseDelay * (1 << uint(attempt))
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}
	}
	return lastErr
}

func isRetryable(err error) bool {
	msg := strings.ToLower(err.Error())
	retryableKeywords := []string{
		"connection", "timeout", "eof", "reset", "refused",
		"ssl", "routing", "temporary", "unavailable", "503", "429",
	}
	for _, kw := range retryableKeywords {
		if strings.Contains(msg, kw) {
			return true
		}
	}
	return false
}
