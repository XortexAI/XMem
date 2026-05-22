package database

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/xortexai/xmem-go/internal/config"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
	"go.mongodb.org/mongo-driver/v2/mongo/readpref"
)

type MongoAPIKeyStore struct {
	client  *mongo.Client
	apiKeys *mongo.Collection
	users   *mongo.Collection
}

func NewMongoAPIKeyStore(ctx context.Context, settings config.Settings) (*MongoAPIKeyStore, error) {
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
	db := client.Database(settings.MongoDBDatabase)
	return &MongoAPIKeyStore{
		client:  client,
		apiKeys: db.Collection("api_keys"),
		users:   db.Collection("users"),
	}, nil
}

func (s *MongoAPIKeyStore) Close(ctx context.Context) error {
	return s.client.Disconnect(ctx)
}

func (s *MongoAPIKeyStore) ValidateAPIKey(token string) (*APIKeyDoc, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var doc struct {
		ID        bson.ObjectID `bson:"_id"`
		UserID    string        `bson:"user_id"`
		Scopes    []string      `bson:"scopes"`
		OrgID     string        `bson:"org_id"`
		ProjectID string        `bson:"project_id"`
		ExpiresAt *time.Time    `bson:"expires_at"`
	}
	err := s.apiKeys.FindOne(ctx, bson.M{
		"key_hash":  hashToken(token),
		"is_active": true,
	}).Decode(&doc)
	if err != nil {
		return nil, false
	}
	if doc.ExpiresAt != nil && time.Now().After(*doc.ExpiresAt) {
		_, _ = s.apiKeys.UpdateByID(ctx, doc.ID, bson.M{"$set": bson.M{"is_active": false}})
		return nil, false
	}
	_, _ = s.apiKeys.UpdateByID(ctx, doc.ID, bson.M{"$set": bson.M{"last_used": time.Now().UTC()}})
	if len(doc.Scopes) == 0 {
		doc.Scopes = []string{"*"}
	}
	return &APIKeyDoc{
		ID:        doc.ID.Hex(),
		UserID:    doc.UserID,
		KeyHash:   sha256Hex(token),
		Scopes:    doc.Scopes,
		OrgID:     doc.OrgID,
		ProjectID: doc.ProjectID,
	}, true
}

func (s *MongoAPIKeyStore) GetUserByID(id string) (*UserDoc, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	filter := bson.M{"_id": id}
	if oid, err := bson.ObjectIDFromHex(id); err == nil {
		filter = bson.M{"_id": oid}
	}

	var doc struct {
		ID       any    `bson:"_id"`
		Name     string `bson:"name"`
		Email    string `bson:"email"`
		Username string `bson:"username"`
	}
	if err := s.users.FindOne(ctx, filter).Decode(&doc); err != nil {
		return nil, false
	}
	return &UserDoc{
		ID:       fmt.Sprint(doc.ID),
		Name:     doc.Name,
		Email:    doc.Email,
		Username: doc.Username,
	}, true
}

func sha256Hex(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}
