package agents

import (
	"fmt"
	"strings"

	"github.com/xortexai/xmem-go/internal/graph"
	"github.com/xortexai/xmem-go/internal/storage"
)

func joinPipe(parts ...string) string {
	return strings.Join(parts, " | ")
}

func extractJSONObject(text string) string {
	text = strings.TrimSpace(text)
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start >= 0 && end > start {
		return text[start : end+1]
	}
	return ""
}

func extractJSONArray(text string) string {
	text = strings.TrimSpace(text)
	start := strings.Index(text, "[")
	end := strings.LastIndex(text, "]")
	if start >= 0 && end > start {
		return text[start : end+1]
	}
	return ""
}

func buildProfileMetadataKey(fact ProfileFact) string {
	topic := strings.TrimSpace(fact.Topic)
	subTopic := strings.TrimSpace(fact.SubTopic)
	if topic == "" || subTopic == "" {
		return ""
	}
	key := topic + "_" + subTopic
	key = strings.ReplaceAll(key, " ", "_")
	return strings.ToLower(key)
}

func dedupeProfileItems(facts []ProfileFact) []ProfileFact {
	latest := make(map[string]ProfileFact)
	order := []string{}
	passthrough := []ProfileFact{}
	for _, fact := range facts {
		key := buildProfileMetadataKey(fact)
		if key != "" {
			if _, exists := latest[key]; !exists {
				order = append(order, key)
			}
			latest[key] = fact
		} else {
			passthrough = append(passthrough, fact)
		}
	}
	out := make([]ProfileFact, 0, len(latest)+len(passthrough))
	for _, key := range order {
		out = append(out, latest[key])
	}
	out = append(out, passthrough...)
	return out
}

func dedupeTemporalItems(events []Event) []Event {
	latest := make(map[string]Event)
	order := []string{}
	passthrough := []Event{}
	for _, event := range events {
		name := normText(event.EventName)
		if name != "" {
			if _, exists := latest[name]; !exists {
				order = append(order, name)
			}
			latest[name] = event
		} else {
			passthrough = append(passthrough, event)
		}
	}
	out := make([]Event, 0, len(latest)+len(passthrough))
	for _, key := range order {
		out = append(out, latest[key])
	}
	out = append(out, passthrough...)
	return out
}

func normText(val string) string {
	fields := strings.Fields(strings.ToLower(strings.TrimSpace(val)))
	return strings.Join(fields, " ")
}

func profileMemoFromContent(content string) string {
	if !strings.Contains(content, " = ") {
		return content
	}
	parts := strings.SplitN(content, " = ", 2)
	return strings.TrimSpace(parts[1])
}

func profileMemoFromMatch(match storage.SearchResult) string {
	if match.Metadata != nil {
		if sub, ok := match.Metadata["subcontent"]; ok {
			return fmt.Sprintf("%v", sub)
		}
	}
	return profileMemoFromContent(match.Content)
}

func temporalFieldsFromContent(content string) Event {
	parts := strings.Split(content, " | ")
	for i, p := range parts {
		parts[i] = strings.TrimSpace(p)
	}
	e := Event{}
	if len(parts) > 0 {
		e.Date = parts[0]
	}
	if len(parts) > 1 {
		e.EventName = parts[1]
	}
	if len(parts) > 2 {
		e.Desc = parts[2]
	}
	if len(parts) > 3 {
		e.Year = parts[3]
	}
	if len(parts) > 4 {
		e.Time = parts[4]
	}
	if len(parts) > 5 {
		e.DateExpression = parts[5]
	}
	return e
}

func temporalFieldsFromMatch(match graph.Event) Event {
	return Event{
		Date:           match.Date,
		EventName:      match.EventName,
		Desc:           match.Description,
		Year:           match.Year,
		Time:           match.Time,
		DateExpression: match.DateExpression,
	}
}

func sameTemporalEvent(incoming, existing Event) bool {
	return normText(incoming.Date) == normText(existing.Date) &&
		normText(incoming.EventName) == normText(existing.EventName) &&
		normText(incoming.Desc) == normText(existing.Desc) &&
		normText(incoming.Year) == normText(existing.Year) &&
		normText(incoming.Time) == normText(existing.Time) &&
		normText(incoming.DateExpression) == normText(existing.DateExpression)
}

func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return strings.TrimSpace(value)
}
