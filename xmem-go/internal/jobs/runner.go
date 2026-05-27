package jobs

import (
	"context"
	"log/slog"
	"math"
	"time"
)

func Run(ctx context.Context, store Store, logger *slog.Logger, jobID string, handler Handler) {
	if logger == nil {
		logger = slog.Default()
	}
	for {
		job, ok, err := store.Get(ctx, jobID)
		if err != nil {
			logger.Error("durable job lookup failed", "job_id", jobID, "error", err)
			return
		}
		if !ok {
			logger.Warn("durable job disappeared before execution", "job_id", jobID)
			return
		}
		if terminalStatuses[job.Status] {
			return
		}

		claimed, ok, err := store.ClaimForRun(ctx, jobID)
		if err != nil {
			logger.Error("durable job claim failed", "job_id", jobID, "error", err)
			return
		}
		if !ok {
			logger.Info("durable job already claimed; skipping duplicate runner", "job_id", jobID)
			return
		}

		timeout := time.Duration(claimed.TimeoutSeconds * float64(time.Second))
		if timeout <= 0 {
			timeout = 120 * time.Second
		}
		started := time.Now()
		result, runErr := runAttempt(ctx, timeout, handler)
		if runErr == nil {
			payload := toMap(result)
			payload["elapsed_ms"] = float64(time.Since(started).Microseconds()) / 1000
			if err := store.MarkSucceeded(ctx, jobID, payload); err != nil {
				logger.Error("durable job success update failed", "job_id", jobID, "error", err)
			}
			return
		}

		status, err := store.MarkFailed(ctx, jobID, runErr.Error())
		if err != nil {
			logger.Error("durable job failure update failed", "job_id", jobID, "error", err)
			return
		}
		if status == StatusDeadLetter {
			logger.Error("durable job dead-lettered", "job_id", jobID, "error", runErr)
			return
		}
		delay := retryDelay(claimed.RetryCount)
		logger.Warn("durable job failed; retrying", "job_id", jobID, "delay", delay.String(), "error", runErr)
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return
		}
		if err := store.ResetForRetry(ctx, jobID); err != nil {
			logger.Error("durable job retry reset failed", "job_id", jobID, "error", err)
			return
		}
	}
}

func retryDelay(retryCount int) time.Duration {
	if retryCount < 1 {
		retryCount = 1
	}
	seconds := math.Pow(2, float64(retryCount-1))
	if seconds > 30 {
		seconds = 30
	}
	return time.Duration(seconds * float64(time.Second))
}
