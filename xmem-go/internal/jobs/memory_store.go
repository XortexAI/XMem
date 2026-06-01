package jobs

import (
	"context"
	"errors"
	"sync"
	"time"
)

type MemoryStore struct {
	mu      sync.Mutex
	jobs    map[string]Job
	byIdemp map[string]string
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		jobs:    map[string]Job{},
		byIdemp: map[string]string{},
	}
}

func (s *MemoryStore) Enqueue(_ context.Context, input EnqueueInput) (Job, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job := NewJob(input)
	idempKey := job.Type + ":" + job.IdempotencyKey
	if existingID, ok := s.byIdemp[idempKey]; ok {
		return s.jobs[existingID], false, nil
	}
	s.jobs[job.ID] = job
	s.byIdemp[idempKey] = job.ID
	return job, true, nil
}

func (s *MemoryStore) Get(_ context.Context, jobID string) (Job, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[jobID]
	return job, ok, nil
}

func (s *MemoryStore) ClaimForRun(_ context.Context, jobID string) (Job, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[jobID]
	if !ok || job.Status != StatusQueued {
		return Job{}, false, nil
	}
	now := time.Now().UTC()
	job.Status = StatusRunning
	job.StartedAt = &now
	job.UpdatedAt = now
	job.Error = ""
	job.ErrorState = nil
	job.RetryCount++
	s.jobs[jobID] = job
	return job, true, nil
}

func (s *MemoryStore) MarkSucceeded(_ context.Context, jobID string, result any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[jobID]
	if !ok {
		return errors.New("job not found")
	}
	now := time.Now().UTC()
	job.Status = StatusSucceeded
	job.Result = result
	job.Error = ""
	job.ErrorState = nil
	job.CompletedAt = &now
	job.UpdatedAt = now
	s.jobs[jobID] = job
	return nil
}

func (s *MemoryStore) MarkFailed(_ context.Context, jobID string, message string) (Status, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[jobID]
	if !ok {
		return "", errors.New("job not found")
	}
	now := time.Now().UTC()
	status := StatusFailed
	if job.RetryCount >= job.MaxAttempts {
		status = StatusDeadLetter
		job.DeadLetteredAt = &now
		job.CompletedAt = &now
	}
	job.Status = status
	job.Error = message
	job.ErrorState = map[string]any{
		"message":   message,
		"failed_at": now,
		"attempt":   job.RetryCount,
	}
	job.UpdatedAt = now
	s.jobs[jobID] = job
	return status, nil
}

func (s *MemoryStore) ResetForRetry(_ context.Context, jobID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[jobID]
	if !ok {
		return errors.New("job not found")
	}
	job.Status = StatusQueued
	job.UpdatedAt = time.Now().UTC()
	s.jobs[jobID] = job
	return nil
}
