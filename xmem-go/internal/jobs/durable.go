package jobs

import (
	"context"
	"time"
)

type Status string

const (
	StatusQueued     Status = "queued"
	StatusRunning    Status = "running"
	StatusSucceeded  Status = "succeeded"
	StatusFailed     Status = "failed"
	StatusDeadLetter Status = "dead_letter"
)

var terminalStatuses = map[Status]bool{
	StatusSucceeded:  true,
	StatusDeadLetter: true,
}

type Job struct {
	ID             string         `json:"job_id" bson:"job_id"`
	Type           string         `json:"job_type" bson:"job_type"`
	IdempotencyKey string         `json:"idempotency_key" bson:"idempotency_key"`
	UserID         string         `json:"user_id" bson:"user_id"`
	Payload        map[string]any `json:"payload,omitempty" bson:"payload,omitempty"`
	Status         Status         `json:"status" bson:"status"`
	RetryCount     int            `json:"retry_count" bson:"retry_count"`
	MaxAttempts    int            `json:"max_attempts" bson:"max_attempts"`
	TimeoutSeconds float64        `json:"timeout_seconds" bson:"timeout_seconds"`
	Error          string         `json:"error,omitempty" bson:"error,omitempty"`
	ErrorState     map[string]any `json:"error_state,omitempty" bson:"error_state,omitempty"`
	Result         any            `json:"result,omitempty" bson:"result,omitempty"`
	CreatedAt      time.Time      `json:"created_at" bson:"created_at"`
	UpdatedAt      time.Time      `json:"updated_at" bson:"updated_at"`
	StartedAt      *time.Time     `json:"started_at,omitempty" bson:"started_at,omitempty"`
	CompletedAt    *time.Time     `json:"completed_at,omitempty" bson:"completed_at,omitempty"`
	DeadLetteredAt *time.Time     `json:"dead_lettered_at,omitempty" bson:"dead_lettered_at,omitempty"`
}

type EnqueueInput struct {
	JobType           string
	Payload           any
	IdempotencyFields any
	UserID            string
	TimeoutSeconds    float64
	MaxAttempts       int
}

type Store interface {
	Enqueue(ctx context.Context, input EnqueueInput) (Job, bool, error)
	Get(ctx context.Context, jobID string) (Job, bool, error)
	ClaimForRun(ctx context.Context, jobID string) (Job, bool, error)
	MarkSucceeded(ctx context.Context, jobID string, result any) error
	MarkFailed(ctx context.Context, jobID string, err string) (Status, error)
	ResetForRetry(ctx context.Context, jobID string) error
}

type Handler func(context.Context) (any, error)

func Public(job Job) map[string]any {
	return map[string]any{
		"job_id":           job.ID,
		"job_type":         job.Type,
		"status":           job.Status,
		"retry_count":      job.RetryCount,
		"max_attempts":     job.MaxAttempts,
		"timeout_seconds":  job.TimeoutSeconds,
		"error":            job.Error,
		"error_state":      job.ErrorState,
		"result":           job.Result,
		"created_at":       job.CreatedAt,
		"updated_at":       job.UpdatedAt,
		"started_at":       job.StartedAt,
		"completed_at":     job.CompletedAt,
		"dead_lettered_at": job.DeadLetteredAt,
	}
}

func IdempotencyKey(jobType string, fields any) string {
	return stableHash(map[string]any{"job_type": jobType, "fields": fields})
}

func NewJob(input EnqueueInput) Job {
	key := IdempotencyKey(input.JobType, input.IdempotencyFields)
	timeout := input.TimeoutSeconds
	if timeout <= 0 {
		timeout = 120
	}
	maxAttempts := input.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 3
	}
	now := time.Now().UTC()
	return Job{
		ID:             input.JobType + ":" + key,
		Type:           input.JobType,
		IdempotencyKey: key,
		UserID:         input.UserID,
		Payload:        Redact(toMap(input.Payload)),
		Status:         StatusQueued,
		RetryCount:     0,
		MaxAttempts:    maxAttempts,
		TimeoutSeconds: timeout,
		CreatedAt:      now,
		UpdatedAt:      now,
	}
}
