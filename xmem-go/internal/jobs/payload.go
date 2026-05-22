package jobs

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"
)

func Redact(payload map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range payload {
		lowered := strings.ToLower(key)
		if strings.Contains(lowered, "authorization") ||
			strings.Contains(lowered, "cookie") ||
			strings.Contains(lowered, "password") ||
			strings.Contains(lowered, "secret") ||
			strings.Contains(lowered, "token") ||
			strings.Contains(lowered, "pat") {
			out[key] = "[redacted]"
			continue
		}
		if nested, ok := value.(map[string]any); ok {
			out[key] = Redact(nested)
			continue
		}
		if items, ok := value.([]any); ok {
			redacted := make([]any, 0, len(items))
			for _, item := range items {
				if nested, ok := item.(map[string]any); ok {
					redacted = append(redacted, Redact(nested))
				} else {
					redacted = append(redacted, item)
				}
			}
			out[key] = redacted
			continue
		}
		out[key] = value
	}
	return out
}

func stableHash(value any) string {
	encoded, _ := json.Marshal(value)
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

func toMap(value any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	if m, ok := value.(map[string]any); ok {
		return m
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(encoded, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func SortedKeys(values map[string]any) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
