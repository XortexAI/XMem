package jobs

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestIdempotencyKeyIsStable(t *testing.T) {
	left := IdempotencyKey("memory_ingest", map[string]any{"b": 2, "a": map[string]any{"z": 1, "y": 0}})
	right := IdempotencyKey("memory_ingest", map[string]any{"a": map[string]any{"y": 0, "z": 1}, "b": 2})
	if left != right {
		t.Fatalf("idempotency key should be stable: %s != %s", left, right)
	}
}

func TestRunDuplicateRunnersOnlyExecuteOnce(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	job, _, err := store.Enqueue(ctx, EnqueueInput{
		JobType:           "memory_ingest",
		Payload:           map[string]any{"user_query": "hello"},
		IdempotencyFields: map[string]any{"user_query": "hello"},
		UserID:            "alice",
		TimeoutSeconds:    1,
		MaxAttempts:       1,
	})
	if err != nil {
		t.Fatal(err)
	}

	started := make(chan struct{})
	release := make(chan struct{})
	done := make(chan struct{}, 2)
	attempts := 0
	handler := func(ctx context.Context) (any, error) {
		attempts++
		close(started)
		select {
		case <-release:
			return map[string]any{"ok": true}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	go func() {
		Run(ctx, store, nil, job.ID, handler)
		done <- struct{}{}
	}()
	<-started
	go func() {
		Run(ctx, store, nil, job.ID, handler)
		done <- struct{}{}
	}()
	time.Sleep(10 * time.Millisecond)
	close(release)
	<-done
	<-done

	if attempts != 1 {
		t.Fatalf("handler attempts = %d, want 1", attempts)
	}
	stored, ok, err := store.Get(ctx, job.ID)
	if err != nil || !ok {
		t.Fatalf("stored job missing err=%v ok=%v", err, ok)
	}
	if stored.Status != StatusSucceeded || stored.RetryCount != 1 {
		t.Fatalf("status=%s retry_count=%d", stored.Status, stored.RetryCount)
	}
}

func TestRunRetriesAndDeadLetters(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	job, _, err := store.Enqueue(ctx, EnqueueInput{
		JobType:           "memory_ingest",
		Payload:           map[string]any{"user_query": "hello"},
		IdempotencyFields: map[string]any{"user_query": "hello"},
		UserID:            "alice",
		TimeoutSeconds:    1,
		MaxAttempts:       2,
	})
	if err != nil {
		t.Fatal(err)
	}

	attempts := 0
	Run(ctx, store, nil, job.ID, func(context.Context) (any, error) {
		attempts++
		return nil, errors.New("still broken")
	})

	if attempts != 2 {
		t.Fatalf("attempts=%d, want 2", attempts)
	}
	stored, ok, err := store.Get(ctx, job.ID)
	if err != nil || !ok {
		t.Fatalf("stored job missing err=%v ok=%v", err, ok)
	}
	if stored.Status != StatusDeadLetter || stored.Error != "still broken" {
		t.Fatalf("status=%s error=%q", stored.Status, stored.Error)
	}
}

func TestRunDeadLettersTimedOutHandler(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	job, _, err := store.Enqueue(ctx, EnqueueInput{
		JobType:           "memory_ingest",
		Payload:           map[string]any{"user_query": "hello"},
		IdempotencyFields: map[string]any{"user_query": "hello"},
		UserID:            "alice",
		TimeoutSeconds:    0.01,
		MaxAttempts:       1,
	})
	if err != nil {
		t.Fatal(err)
	}

	Run(ctx, store, nil, job.ID, func(context.Context) (any, error) {
		time.Sleep(50 * time.Millisecond)
		return map[string]any{"too": "late"}, nil
	})

	stored, ok, err := store.Get(ctx, job.ID)
	if err != nil || !ok {
		t.Fatalf("stored job missing err=%v ok=%v", err, ok)
	}
	if stored.Status != StatusDeadLetter || stored.Error == "" {
		t.Fatalf("status=%s error=%q", stored.Status, stored.Error)
	}
}
