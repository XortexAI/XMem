package jobs

import (
	"crypto/sha256"
	"encoding/hex"
	json "github.com/goccy/go-json"
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
	encoded := canonicalJSON(value)
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

func canonicalJSON(v any) []byte {
	switch val := v.(type) {
	case map[string]any:
		keys := SortedKeys(val)
		buf := []byte("{")
		for i, k := range keys {
			if i > 0 {
				buf = append(buf, ',')
			}
			keyBytes, _ := json.Marshal(k)
			buf = append(buf, keyBytes...)
			buf = append(buf, ':')
			buf = append(buf, canonicalJSON(val[k])...)
		}
		buf = append(buf, '}')
		return buf
	case []any:
		buf := []byte("[")
		for i, item := range val {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = append(buf, canonicalJSON(item)...)
		}
		buf = append(buf, ']')
		return buf
	default:
		b, _ := json.Marshal(v)
		return b
	}
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
