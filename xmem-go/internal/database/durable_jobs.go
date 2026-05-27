package database

import (
	"context"
	"errors"
	"time"

	"github.com/xortexai/xmem-go/internal/config"
	"github.com/xortexai/xmem-go/internal/jobs"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
	"go.mongodb.org/mongo-driver/v2/mongo/readpref"
)

type MongoDurableJobStore struct {
	client *mongo.Client
	jobs   *mongo.Collection
}

func NewMongoDurableJobStore(ctx context.Context, settings config.Settings) (*MongoDurableJobStore, error) {
	client, err := mongo.Connect(options.Client().ApplyURI(settings.MongoDBURI))
	if err != nil {
		return nil, err
	}
	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := client.Ping(pingCtx, readpref.Primary()); err != nil {
		_ = client.Disconnect(context.Background())
		return nil, err
	}
	store := &MongoDurableJobStore{
		client: client,
		jobs:   client.Database(settings.MongoDBDatabase).Collection("durable_jobs"),
	}
	if err := store.ensureIndexes(ctx); err != nil {
		_ = client.Disconnect(context.Background())
		return nil, err
	}
	return store, nil
}

func (s *MongoDurableJobStore) Close(ctx context.Context) error {
	return s.client.Disconnect(ctx)
}

func (s *MongoDurableJobStore) ensureIndexes(ctx context.Context) error {
	_, err := s.jobs.Indexes().CreateMany(ctx, []mongo.IndexModel{
		{Keys: bson.D{{Key: "job_id", Value: 1}}, Options: options.Index().SetUnique(true)},
		{Keys: bson.D{{Key: "job_type", Value: 1}, {Key: "idempotency_key", Value: 1}}, Options: options.Index().SetUnique(true)},
		{Keys: bson.D{{Key: "user_id", Value: 1}, {Key: "updated_at", Value: -1}}},
		{Keys: bson.D{{Key: "status", Value: 1}, {Key: "updated_at", Value: 1}}},
	})
	return err
}

func (s *MongoDurableJobStore) Enqueue(ctx context.Context, input jobs.EnqueueInput) (jobs.Job, bool, error) {
	job := jobs.NewJob(input)
	_, err := s.jobs.InsertOne(ctx, job)
	if err == nil {
		return job, true, nil
	}
	if !mongo.IsDuplicateKeyError(err) {
		return jobs.Job{}, false, err
	}
	existing, ok, getErr := s.Get(ctx, job.ID)
	if getErr != nil {
		return jobs.Job{}, false, getErr
	}
	if ok {
		return existing, false, nil
	}
	var byIdempotency jobs.Job
	err = s.jobs.FindOne(ctx, bson.M{
		"job_type":        job.Type,
		"idempotency_key": job.IdempotencyKey,
	}).Decode(&byIdempotency)
	if err != nil {
		return jobs.Job{}, false, err
	}
	return byIdempotency, false, nil
}

func (s *MongoDurableJobStore) Get(ctx context.Context, jobID string) (jobs.Job, bool, error) {
	var job jobs.Job
	err := s.jobs.FindOne(ctx, bson.M{"job_id": jobID}).Decode(&job)
	if errors.Is(err, mongo.ErrNoDocuments) {
		return jobs.Job{}, false, nil
	}
	if err != nil {
		return jobs.Job{}, false, err
	}
	return job, true, nil
}

func (s *MongoDurableJobStore) ClaimForRun(ctx context.Context, jobID string) (jobs.Job, bool, error) {
	now := time.Now().UTC()
	result, err := s.jobs.UpdateOne(
		ctx,
		bson.M{"job_id": jobID, "status": jobs.StatusQueued},
		bson.M{
			"$set": bson.M{
				"status":      jobs.StatusRunning,
				"started_at":  now,
				"updated_at":  now,
				"error":       "",
				"error_state": nil,
			},
			"$inc": bson.M{"retry_count": 1},
		},
	)
	if err != nil {
		return jobs.Job{}, false, err
	}
	if result.ModifiedCount != 1 {
		return jobs.Job{}, false, nil
	}
	return s.Get(ctx, jobID)
}

func (s *MongoDurableJobStore) MarkSucceeded(ctx context.Context, jobID string, result any) error {
	now := time.Now().UTC()
	_, err := s.jobs.UpdateOne(ctx, bson.M{"job_id": jobID}, bson.M{"$set": bson.M{
		"status":       jobs.StatusSucceeded,
		"result":       result,
		"error":        "",
		"error_state":  nil,
		"completed_at": now,
		"updated_at":   now,
	}})
	return err
}

func (s *MongoDurableJobStore) MarkFailed(ctx context.Context, jobID string, message string) (jobs.Status, error) {
	job, ok, err := s.Get(ctx, jobID)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", errors.New("job not found")
	}
	now := time.Now().UTC()
	status := jobs.StatusFailed
	update := bson.M{
		"status": jobs.StatusFailed,
		"error":  message,
		"error_state": bson.M{
			"message":   message,
			"failed_at": now,
			"attempt":   job.RetryCount,
		},
		"updated_at": now,
	}
	if job.RetryCount >= job.MaxAttempts {
		status = jobs.StatusDeadLetter
		update["status"] = jobs.StatusDeadLetter
		update["dead_lettered_at"] = now
		update["completed_at"] = now
	}
	_, err = s.jobs.UpdateOne(ctx, bson.M{"job_id": jobID}, bson.M{"$set": update})
	return status, err
}

func (s *MongoDurableJobStore) ResetForRetry(ctx context.Context, jobID string) error {
	_, err := s.jobs.UpdateOne(ctx, bson.M{"job_id": jobID}, bson.M{"$set": bson.M{
		"status":     jobs.StatusQueued,
		"updated_at": time.Now().UTC(),
	}})
	return err
}
