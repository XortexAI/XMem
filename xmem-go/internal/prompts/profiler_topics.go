package prompts

import (
	"fmt"
	"strings"
)

type SubTopic struct {
	Name        string
	Description string
}

type ProfileTopic struct {
	Topic       string
	SubTopics   []SubTopic
	Description string
}

var ProfileTopics = []ProfileTopic{
	{
		Topic: "basic_info",
		SubTopics: []SubTopic{
			{Name: "Name"},
			{Name: "Age", Description: "integer"},
			{Name: "Gender"},
			{Name: "birth_date"},
			{Name: "nationality"},
			{Name: "ethnicity"},
			{Name: "language_spoken"},
		},
	},
	{
		Topic: "contact_info",
		SubTopics: []SubTopic{
			{Name: "email"},
			{Name: "phone"},
			{Name: "city"},
			{Name: "country"},
		},
	},
	{
		Topic: "education",
		SubTopics: []SubTopic{
			{Name: "school"},
			{Name: "degree"},
			{Name: "major"},
		},
	},
	{
		Topic: "demographics",
		SubTopics: []SubTopic{
			{Name: "marital_status"},
			{Name: "spouse_name"},
			{Name: "number_of_children"},
			{Name: "household_income"},
			{Name: "relationship"},
		},
	},
	{
		Topic: "work",
		SubTopics: []SubTopic{
			{Name: "company"},
			{Name: "title"},
			{Name: "working_industry"},
			{Name: "previous_projects"},
			{Name: "work_skills"},
		},
	},
	{
		Topic: "interest",
		SubTopics: []SubTopic{
			{Name: "books"},
			{Name: "movies"},
			{Name: "music"},
			{Name: "foods"},
			{Name: "sports"},
			{Name: "hobbies"},
			{Name: "art"},
			{Name: "travel"},
			{Name: "games"},
		},
	},
	{
		Topic: "life_event",
		SubTopics: []SubTopic{
			{Name: "marriage"},
			{Name: "relocation"},
			{Name: "retirement"},
			{Name: "health"},
			{Name: "achievement"},
		},
	},
}

func FormatTopicsForPrompt(topics []ProfileTopic) string {
	if topics == nil {
		topics = ProfileTopics
	}
	var lines []string
	for _, t := range topics {
		desc := ""
		if t.Description != "" {
			desc = fmt.Sprintf(" (%s)", t.Description)
		}
		lines = append(lines, fmt.Sprintf("- %s%s", t.Topic, desc))
		for _, st := range t.SubTopics {
			stDesc := ""
			if st.Description != "" {
				stDesc = fmt.Sprintf("(%s)", st.Description)
			}
			lines = append(lines, fmt.Sprintf("  - %s%s", st.Name, stDesc))
		}
	}
	lines = append(lines, "...")
	return strings.Join(lines, "\n")
}
