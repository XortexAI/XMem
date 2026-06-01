package jobs

import (
	"context"
	"time"
)

type attemptResult struct {
	result any
	err    error
}

func runAttempt(ctx context.Context, timeout time.Duration, handler Handler) (any, error) {
	attemptCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	done := make(chan attemptResult, 1)
	go func() {
		result, err := handler(attemptCtx)
		done <- attemptResult{result: result, err: err}
	}()

	select {
	case result := <-done:
		return result.result, result.err
	case <-attemptCtx.Done():
		return nil, attemptCtx.Err()
	}
}
