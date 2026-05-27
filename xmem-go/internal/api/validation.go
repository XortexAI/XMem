package api

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
)

func validateIngest(req IngestRequest) error {
	req.UserQuery = strings.TrimSpace(req.UserQuery)
	if req.UserQuery == "" || len(req.UserQuery) > 10000 {
		return errors.New("user_query must be between 1 and 10000 characters")
	}
	if len(req.AgentResponse) > 10000 {
		return errors.New("agent_response must be at most 10000 characters")
	}
	if !validUserID(req.UserID) {
		return errors.New("user_id is required and may contain only letters, numbers, dots, hyphens, underscores, and @")
	}
	if len(req.ImageURL) > 50000 {
		return errors.New("image_url must be at most 50000 characters")
	}
	if req.EffortLevel != "" && req.EffortLevel != "low" && req.EffortLevel != "high" {
		return errors.New("effort_level must be 'low' or 'high'")
	}
	return nil
}

func validateRetrieve(req RetrieveRequest) error {
	if strings.TrimSpace(req.Query) == "" || len(req.Query) > 5000 {
		return errors.New("query must be between 1 and 5000 characters")
	}
	if !validUserID(req.UserID) {
		return errors.New("user_id is required and may contain only letters, numbers, dots, hyphens, underscores, and @")
	}
	if req.TopK < 0 || req.TopK > 50 {
		return errors.New("top_k must be between 1 and 50")
	}
	return nil
}

func validateSearch(req SearchRequest) error {
	if strings.TrimSpace(req.Query) == "" || len(req.Query) > 5000 {
		return errors.New("query must be between 1 and 5000 characters")
	}
	if !validUserID(req.UserID) {
		return errors.New("user_id is required and may contain only letters, numbers, dots, hyphens, underscores, and @")
	}
	if req.TopK < 0 || req.TopK > 100 {
		return errors.New("top_k must be between 1 and 100")
	}
	allowed := map[string]bool{"profile": true, "temporal": true, "summary": true}
	for _, domain := range req.Domains {
		if !allowed[domain] {
			return fmt.Errorf("invalid domain %q", domain)
		}
	}
	return nil
}

func validUserID(id string) bool {
	if id == "" || len(id) > 256 {
		return false
	}
	return regexp.MustCompile(`^[\w.\-@]+$`).MatchString(id)
}
