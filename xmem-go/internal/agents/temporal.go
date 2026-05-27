package agents

import (
	"context"
	"strconv"
	"strings"

	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/prompts"
	"github.com/xortexai/xmem-go/internal/utils"
)

var daysInMonth = map[int]int{
	1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
	7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

type Event struct {
	Date           string
	EventName      string
	Desc           string
	Year           string
	Time           string
	DateExpression string
}

type TemporalAgent struct {
	Model models.ChatModel
}

func (a TemporalAgent) Run(ctx context.Context, text string, sessionDatetime string) []Event {
	if strings.TrimSpace(text) == "" {
		return nil
	}

	systemPrompt := prompts.BuildTemporalSystemPrompt()
	userMessage := prompts.PackTemporalQuery(text, sessionDatetime)

	raw, err := callModel(ctx, a.Model, systemPrompt, userMessage)
	if err != nil {
		return nil
	}

	parsed := utils.ParseRawResponseToEvents(raw)
	events := make([]Event, 0, len(parsed))
	for _, e := range parsed {
		if !isValidDate(e.Date) {
			continue
		}
		events = append(events, Event{
			Date:           e.Date,
			EventName:      e.EventName,
			Desc:           e.Desc,
			Year:           e.Year,
			Time:           e.Time,
			DateExpression: e.DateExpression,
		})
	}
	return events
}

func isValidDate(date string) bool {
	if len(date) != 5 {
		return false
	}
	parts := strings.Split(date, "-")
	if len(parts) != 2 {
		return false
	}
	month, err := strconv.Atoi(parts[0])
	if err != nil || month < 1 || month > 12 {
		return false
	}
	day, err := strconv.Atoi(parts[1])
	if err != nil || day < 1 || day > daysInMonth[month] {
		return false
	}
	return true
}
