package api

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	json "github.com/goccy/go-json"
	"net/http"
	"strings"

	"github.com/xortexai/xmem-go/internal/database"
)

func (s *Server) authenticate(r *http.Request) (User, int, string) {
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		return User{}, http.StatusUnauthorized, "Missing API key. Provide a Bearer token in the Authorization header."
	}
	token := strings.TrimSpace(strings.TrimPrefix(auth, "Bearer "))
	if token == "" {
		return User{}, http.StatusUnauthorized, "Missing API key. Provide a Bearer token in the Authorization header."
	}
	if !strings.HasPrefix(token, "xmem_") {
		if user, ok := s.validateJWT(token); ok {
			return user, http.StatusOK, ""
		}
	}
	if doc, ok := s.keys.ValidateAPIKey(token); ok {
		if userDoc, exists := s.keys.GetUserByID(doc.UserID); exists {
			return User{ID: userDoc.ID, Name: userDoc.Name, Email: userDoc.Email, Username: userDoc.Username, APIKey: map[string]any{"id": doc.ID, "scopes": doc.Scopes, "org_id": doc.OrgID, "project_id": doc.ProjectID}}, http.StatusOK, ""
		}
	}
	for _, key := range s.settings.APIKeys {
		if database.ConstantTimeEqual(token, key) {
			return User{ID: database.StaticUserID(token), Name: "Static Key User", Email: "static@xmem.ai"}, http.StatusOK, ""
		}
	}
	return User{}, http.StatusForbidden, "Invalid API key or token."
}

func (s *Server) validateJWT(token string) (User, bool) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 || s.settings.JWTAlgorithm != "HS256" {
		return User{}, false
	}
	signingInput := parts[0] + "." + parts[1]
	mac := hmac.New(sha256.New, []byte(s.settings.JWTSecretKey))
	_, _ = mac.Write([]byte(signingInput))
	expected := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(expected), []byte(parts[2])) {
		return User{}, false
	}
	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return User{}, false
	}
	var payload map[string]any
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return User{}, false
	}
	if payload["type"] != "access" {
		return User{}, false
	}
	sub, _ := payload["sub"].(string)
	if sub == "" {
		return User{}, false
	}
	if userDoc, exists := s.keys.GetUserByID(sub); exists {
		return User{ID: userDoc.ID, Name: userDoc.Name, Email: userDoc.Email, Username: userDoc.Username}, true
	}
	return User{ID: sub, Name: sub, Email: ""}, true
}

func effectiveUserID(user User) string {
	if user.Username != "" {
		return user.Username
	}
	if user.Name != "" {
		return user.Name
	}
	return user.ID
}
