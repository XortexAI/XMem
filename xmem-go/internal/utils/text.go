package utils

import (
	"strconv"
	"strings"
)

const LLMTabSeparator = "::"

var validSources = map[string]struct{}{
	"code":    {},
	"profile": {},
	"event":   {},
	"image":   {},
}

type Classification struct {
	Source string
	Query  string
}

type ProfileFact struct {
	Topic    string
	SubTopic string
	Memo     string
}

type EventData struct {
	Date           string
	EventName      string
	Year           string
	Desc           string
	Time           string
	DateExpression string
}

type ImageObservation struct {
	Category    string
	Description string
	Confidence  string
}

type ImageAnalysisResult struct {
	Description  string
	Observations []ImageObservation
}

func AttributeUnify(value string) string {
	return strings.ToLower(strings.ReplaceAll(value, " ", "_"))
}

func PackClassificationsIntoString(classifications []Classification) string {
	lines := make([]string, 0, len(classifications))
	for _, c := range classifications {
		lines = append(lines, c.Source+LLMTabSeparator+c.Query)
	}
	return strings.Join(lines, "\n")
}

func ParseRawResponseToClassifications(content string) []Classification {
	classifications := []Classification{}
	for _, line := range strings.Split(strings.TrimSpace(content), "\n") {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, LLMTabSeparator) {
			continue
		}
		parts := strings.SplitN(line, LLMTabSeparator, 2)
		if len(parts) < 2 {
			continue
		}
		source := strings.ToLower(strings.TrimSpace(parts[0]))
		query := strings.TrimSpace(parts[1])
		if _, ok := validSources[source]; ok && query != "" {
			classifications = append(classifications, Classification{Source: source, Query: query})
		}
	}
	return classifications
}

func PackProfilesIntoString(facts []ProfileFact) string {
	lines := make([]string, 0, len(facts))
	for _, f := range facts {
		lines = append(lines,
			AttributeUnify(f.Topic)+LLMTabSeparator+
				AttributeUnify(f.SubTopic)+LLMTabSeparator+
				strings.TrimSpace(f.Memo))
	}
	if len(lines) == 0 {
		return "NONE"
	}
	return strings.Join(lines, "\n")
}

func ParseRawResponseToProfiles(content string) []ProfileFact {
	facts := []ProfileFact{}
	if strings.Contains(content, "---") {
		parts := strings.SplitN(content, "---", 2)
		content = parts[1]
	}
	for _, line := range strings.Split(strings.TrimSpace(content), "\n") {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, LLMTabSeparator) {
			continue
		}
		parts := strings.Split(line, LLMTabSeparator)
		if len(parts) >= 3 {
			topic := strings.ToLower(strings.TrimSpace(parts[0]))
			subTopic := strings.ToLower(strings.TrimSpace(parts[1]))
			memo := strings.TrimSpace(strings.Join(parts[2:], LLMTabSeparator))
			if topic != "" && subTopic != "" && memo != "" {
				facts = append(facts, ProfileFact{
					Topic:    topic,
					SubTopic: subTopic,
					Memo:     memo,
				})
			}
		}
	}
	return facts
}

func ParseRawResponseToEvents(content string) []EventData {
	content = strings.TrimSpace(content)
	if strings.Contains(strings.ToUpper(content), "NO_EVENT") {
		return []EventData{}
	}

	rawBlocks := strings.Split(content, "---")
	blocks := make([]string, 0, len(rawBlocks))
	for _, b := range rawBlocks {
		trimmed := strings.TrimSpace(b)
		if trimmed != "" {
			blocks = append(blocks, trimmed)
		}
	}

	events := []EventData{}
	for _, block := range blocks {
		if event, ok := parseSingleEventBlock(block); ok {
			events = append(events, event)
		}
	}
	if len(events) == 0 {
		if event, ok := parseSingleEventBlock(content); ok {
			events = append(events, event)
		}
	}
	return events
}

func parseSingleEventBlock(block string) (EventData, bool) {
	if strings.Contains(strings.ToUpper(block), "NO_EVENT") {
		return EventData{}, false
	}
	eventData := EventData{}
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		upper := strings.ToUpper(line)

		if strings.HasPrefix(upper, "DATE:") && !strings.HasPrefix(upper, "DATE_EXPRESSION:") {
			value := strings.TrimSpace(line[5:])
			if value != "" {
				eventData.Date = value
			}
		} else if strings.HasPrefix(upper, "EVENT_NAME:") {
			value := strings.TrimSpace(line[11:])
			if value != "" {
				eventData.EventName = value
			}
		} else if strings.HasPrefix(upper, "YEAR:") {
			value := strings.TrimSpace(line[5:])
			if value != "" {
				if _, err := strconv.Atoi(value); err == nil {
					eventData.Year = value
				} else {
					eventData.Year = value
				}
			}
		} else if strings.HasPrefix(upper, "DESC:") {
			value := strings.TrimSpace(line[5:])
			if value != "" {
				eventData.Desc = value
			}
		} else if strings.HasPrefix(upper, "TIME:") {
			value := strings.TrimSpace(line[5:])
			if value != "" {
				eventData.Time = value
			}
		} else if strings.HasPrefix(upper, "DATE_EXPRESSION:") {
			value := strings.TrimSpace(line[16:])
			if value != "" {
				eventData.DateExpression = value
			}
		}
	}
	if eventData.Date == "" {
		return EventData{}, false
	}
	return eventData, true
}

func ParseRawResponseToImage(content string) ImageAnalysisResult {
	content = strings.TrimSpace(content)
	result := ImageAnalysisResult{
		Description:  "",
		Observations: []ImageObservation{},
	}

	for _, line := range strings.Split(content, "\n") {
		stripped := strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(stripped), "DESCRIPTION:") {
			result.Description = strings.TrimSpace(stripped[len("DESCRIPTION:"):])
			break
		}
	}

	inObservations := false
	for _, line := range strings.Split(content, "\n") {
		stripped := strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(stripped), "OBSERVATIONS:") {
			inObservations = true
			continue
		}
		if !inObservations || !strings.HasPrefix(stripped, "-") {
			continue
		}
		entry := strings.TrimSpace(stripped[1:])
		category := "other"
		confidence := ""
		description := entry

		if strings.HasPrefix(entry, "[") && strings.Contains(entry, "]") {
			bracketEnd := strings.Index(entry, "]")
			category = strings.ToLower(entry[1:bracketEnd])
			description = strings.TrimSpace(entry[bracketEnd+1:])
		}

		lowerDesc := strings.ToLower(description)
		if strings.Contains(lowerDesc, "(confidence:") {
			idx := strings.Index(lowerDesc, "(confidence:")
			confPart := description[idx:]
			description = strings.TrimSpace(description[:idx])
			confPart = strings.Trim(confPart, "()")
			if strings.Contains(confPart, ":") {
				parts := strings.SplitN(confPart, ":", 2)
				confidence = strings.ToLower(strings.TrimSpace(parts[1]))
			}
		}

		if description != "" {
			result.Observations = append(result.Observations, ImageObservation{
				Category:    category,
				Description: description,
				Confidence:  confidence,
			})
		}
	}

	return result
}
