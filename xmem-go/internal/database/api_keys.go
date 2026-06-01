package database

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"sync"
)

type APIKeyDoc struct {
	ID        string
	UserID    string
	KeyHash   string
	Scopes    []string
	OrgID     string
	ProjectID string
}

type UserDoc struct {
	ID       string
	Name     string
	Email    string
	Username string
}

type APIKeyStore interface {
	ValidateAPIKey(token string) (*APIKeyDoc, bool)
	GetUserByID(id string) (*UserDoc, bool)
}

type MemoryAPIKeyStore struct {
	mu    sync.RWMutex
	keys  map[string]APIKeyDoc
	users map[string]UserDoc
}

func NewMemoryAPIKeyStore() *MemoryAPIKeyStore {
	return &MemoryAPIKeyStore{keys: map[string]APIKeyDoc{}, users: map[string]UserDoc{}}
}

func (s *MemoryAPIKeyStore) AddUser(user UserDoc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.users[user.ID] = user
}

func (s *MemoryAPIKeyStore) AddKey(id, token, userID string, scopes []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.keys[hashToken(token)] = APIKeyDoc{ID: id, UserID: userID, KeyHash: hashToken(token), Scopes: scopes}
}

func (s *MemoryAPIKeyStore) ValidateAPIKey(token string) (*APIKeyDoc, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	doc, ok := s.keys[hashToken(token)]
	if !ok {
		return nil, false
	}
	return &doc, true
}

func (s *MemoryAPIKeyStore) GetUserByID(id string) (*UserDoc, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	user, ok := s.users[id]
	if !ok {
		return nil, false
	}
	return &user, true
}

func ConstantTimeEqual(a, b string) bool {
	return hmac.Equal([]byte(a), []byte(b))
}

func StaticUserID(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])[:16]
}

func hashToken(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}
