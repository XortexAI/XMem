package prompts

import (
	"fmt"
	"strings"
	"sync"
)

type Classification struct {
	Source string
	Query  string
}

var classificationExamples = []struct {
	Input           string
	Classifications []Classification
}{
	{"Thank you so much!", nil},
	{"Hi, how are you?", nil},
	{"Great, thanks!", nil},
	{"Debug this error: TypeError: 'int' object is not iterable", []Classification{
		{Source: "code", Query: "Debug this error: TypeError: 'int' object is not iterable"},
	}},
	{"Explain how the asyncio event loop works in Python", []Classification{
		{Source: "code", Query: "Explain how the asyncio event loop works in Python"},
	}},
	{"Help me write a function to reverse a linked list", []Classification{
		{Source: "code", Query: "Help me write a function to reverse a linked list"},
	}},
	{"I prefer dark mode in all my applications", []Classification{
		{Source: "profile", Query: "I prefer dark mode in all my applications"},
	}},
	{"My name is Alice and I work at Google", []Classification{
		{Source: "profile", Query: "My name is Alice and I work at Google"},
	}},
	{"I'm a vegetarian and love Italian food", []Classification{
		{Source: "profile", Query: "I'm a vegetarian and love Italian food"},
	}},
	{"Our wedding anniversary is July 22nd, 2019", []Classification{
		{Source: "event", Query: "Our wedding anniversary is July 22nd, 2019"},
	}},
	{"I have a dentist appointment on January 10th at 2:30 PM", []Classification{
		{Source: "event", Query: "I have a dentist appointment on January 10th at 2:30 PM"},
	}},
	{"My daughter's birthday is December 25th, she was born in 2015", []Classification{
		{Source: "event", Query: "My daughter's birthday is December 25th, she was born in 2015"},
	}},
	{"My name is Alice and I want to write a python script to hello world", []Classification{
		{Source: "profile", Query: "My name is Alice"},
		{Source: "code", Query: "I want to write a python script to hello world"},
	}},
	{"I'm learning Rust. How do I print variables in Rust?", []Classification{
		{Source: "profile", Query: "I'm learning Rust"},
		{Source: "code", Query: "how do I print variables in Rust?"},
	}},
	{"My name is John and my birthday is April 5th", []Classification{
		{Source: "profile", Query: "My name is John"},
		{Source: "event", Query: "my birthday is April 5th"},
	}},
	{"I graduated on May 20th 2020 and now I work as a software engineer", []Classification{
		{Source: "event", Query: "I graduated on May 20th 2020"},
		{Source: "profile", Query: "I work as a software engineer"},
	}},
	{"I prefer writing code in TypeScript over JavaScript", []Classification{
		{Source: "profile", Query: "I prefer writing code in TypeScript over JavaScript"},
	}},
	{"I ran a charity race last Saturday", []Classification{
		{Source: "event", Query: "I ran a charity race last Saturday"},
	}},
	{"I moved from Sweden 4 years ago", []Classification{
		{Source: "event", Query: "I moved from Sweden 4 years ago"},
	}},
	{"I started transitioning about 3 years ago", []Classification{
		{Source: "event", Query: "I started transitioning about 3 years ago"},
	}},
	{"I graduated from college in May 2018", []Classification{
		{Source: "event", Query: "I graduated from college in May 2018"},
	}},
	{"My 18th birthday was ten years ago when my friend gave me a bowl", []Classification{
		{Source: "event", Query: "My 18th birthday was ten years ago when my friend gave me a bowl"},
	}},
	{"I went through a tough breakup last month and now I'm focusing on myself", []Classification{
		{Source: "event", Query: "I went through a tough breakup last month"},
		{Source: "profile", Query: "I'm focusing on myself"},
	}},
	{"I usually wake up at 6 AM every day", []Classification{
		{Source: "profile", Query: "I usually wake up at 6 AM every day"},
	}},
	{"I never drink coffee after 3 PM", []Classification{
		{Source: "profile", Query: "I never drink coffee after 3 PM"},
	}},
	{"Privacy is very important to me", []Classification{
		{Source: "profile", Query: "Privacy is very important to me"},
	}},
	{"My daughter Sarah is 8 years old", []Classification{
		{Source: "profile", Query: "My daughter Sarah is 8 years old"},
	}},
	{"My best friend lives in Seattle", []Classification{
		{Source: "profile", Query: "My best friend lives in Seattle"},
	}},
	{"I'm from Tokyo but now I live in San Francisco", []Classification{
		{Source: "profile", Query: "I'm from Tokyo but now I live in San Francisco"},
	}},
	{"My email is john@example.com", []Classification{
		{Source: "profile", Query: "My email is john@example.com"},
	}},
	{"Schedule a meeting with the team next Tuesday at 10 AM", []Classification{
		{Source: "event", Query: "Schedule a meeting with the team next Tuesday at 10 AM"},
	}},
	{"Remind me to call mom tomorrow evening", []Classification{
		{Source: "event", Query: "Remind me to call mom tomorrow evening"},
	}},
	{"I visited Paris in August 2022", []Classification{
		{Source: "event", Query: "I visited Paris in August 2022"},
	}},
	{"I finished my master's degree back in 2019", []Classification{
		{Source: "event", Query: "I finished my master's degree back in 2019"},
	}},
	{"Started learning guitar 6 months ago", []Classification{
		{Source: "event", Query: "Started learning guitar 6 months ago"},
	}},
	{"I got my first car when I turned 18", []Classification{
		{Source: "event", Query: "I got my first car when I turned 18"},
	}},
	{"I was diagnosed with diabetes at age 25", []Classification{
		{Source: "event", Query: "I was diagnosed with diabetes at age 25"},
	}},
	{"I joined Google in January 2020", []Classification{
		{Source: "event", Query: "I joined Google in January 2020"},
	}},
	{"We adopted our dog Rex last summer", []Classification{
		{Source: "event", Query: "We adopted our dog Rex last summer"},
	}},
	{"Launched my startup in March 2023", []Classification{
		{Source: "event", Query: "Launched my startup in March 2023"},
	}},
	{"I'm a DevOps engineer and I usually work with Kubernetes. Can you help me debug this pod error?", []Classification{
		{Source: "profile", Query: "I'm a DevOps engineer and I usually work with Kubernetes"},
		{Source: "code", Query: "Can you help me debug this pod error?"},
	}},
	{"I got engaged last Christmas and my fiancé loves hiking", []Classification{
		{Source: "event", Query: "I got engaged last Christmas"},
		{Source: "profile", Query: "my fiancé loves hiking"},
	}},
	{"I prefer using VS Code for development. How do I set up Python debugging in it?", []Classification{
		{Source: "profile", Query: "I prefer using VS Code for development"},
		{Source: "code", Query: "How do I set up Python debugging in it?"},
	}},
	{"My son was born on June 15th 2020 and he loves dinosaurs", []Classification{
		{Source: "event", Query: "My son was born on June 15th 2020"},
		{Source: "profile", Query: "my son loves dinosaurs"},
	}},
	{"My birthday is on March 15th and our wedding anniversary is on July 22nd", []Classification{
		{Source: "event", Query: "My birthday is on March 15th and our wedding anniversary is on July 22nd"},
	}},
	{"I started my new job at Google on January 10th and my first performance review is on April 15th", []Classification{
		{Source: "event", Query: "I started my new job at Google on January 10th and my first performance review is on April 15th"},
		{Source: "profile", Query: "I work at Google"},
	}},
	{"I went to the gym yesterday and I have a doctor appointment tomorrow", []Classification{
		{Source: "event", Query: "I went to the gym yesterday and I have a doctor appointment tomorrow"},
	}},
	{"I live in New Delhi and my name is Vedant", []Classification{
		{Source: "profile", Query: "I live in New Delhi and my name is Vedant"},
	}},
	{"I live in Delhi, my name is Vedant, and my birthday is on September 15th", []Classification{
		{Source: "profile", Query: "I live in Delhi and my name is Vedant"},
		{Source: "event", Query: "my birthday is on September 15th"},
	}},
}

func packClassificationsIntoString(classifications []Classification) string {
	var lines []string
	for _, c := range classifications {
		lines = append(lines, c.Source+LLMTabSeparator+c.Query)
	}
	return strings.Join(lines, "\n")
}

var classifierSystemPromptTemplate = `You are an intelligent intent router for a personal memory assistant.
Your task is to accurately route user inputs to the correct specialized agents for MEMORY STORAGE.

CRITICAL: Your job is to identify WHAT SHOULD BE REMEMBERED about the user.

---

## Available Agents

### 1. ` + "`code`" + `
- **Purpose**: Software engineering and technical tasks (writing, debugging, explaining code)
- **Keywords**: %s
- **Route here when**: User wants help with actual coding work, debugging, or technical explanations

### 2. ` + "`profile`" + `
- **Purpose**: Store PERMANENT facts about the user (identity, preferences, traits, background) use the below keywords for help
- **Keywords**: %s
- **Route here when**: User shares static personal information that doesn't have a specific date
- **Examples**: name, job, hobbies, food preferences, personality traits, where they live

### 3. ` + "`event`" + `
- **Purpose**: Store TIME-BASED events and memories (past, present, or future)
- **Keywords**: %s
- **Route here when**: User mentions something that happened/will happen at a SPECIFIC TIME
- **Examples**: birthdays, anniversaries, "last Saturday", "3 years ago", "next month"

### 4. ` + "`image`" + `
- **Purpose**: Analyze attached images for visual information
- **Keywords**: look, see, image, photo, picture, attached
- **Route here when**: User explicitly asks to analyze the attached image or the input implies visual context

## Logic & Strategy

### 1. Look for Temporal Markers FIRST
Before classifying, scan for ANY time reference:
- Absolute: dates, years, months, days
- Relative: "ago", "last", "next", "yesterday", "tomorrow"
- Age-based: "when I was X", "at age X", "X years old"
- Ordinal: "first", "18th birthday", "second anniversary"

If temporal marker found → likely ` + "`event`" + `

### 2. Decomposition (Multi-Intent)
If input contains information for MULTIPLE agents, route to each relevant agent.
However, each agent must appear AT MOST ONCE. Consolidate all relevant information
for the same agent into a single query.
- "I'm John and my birthday is March 15th" → ` + "`profile`" + ` (name) + ` + "`event`" + ` (birthday)
- "I moved to NYC last year and now work at Google" → ` + "`event`" + ` (move) + ` + "`profile`" + ` (job)
- "I live in Delhi and my name is Vedant" → ` + "`profile`" + ` (name AND location — ONE profile line)
- "My birthday is March 15th and our anniversary is July 22nd" → ` + "`event`" + ` (both events — ONE event line)
- "I started my new job at Google on Jan 10th and my review is April 15th" → ` + "`event`" + ` (both events) + ` + "`profile`" + ` (job)

### 3. Skip Trivial Messages
Pure greetings/acknowledgments with NO factual content → empty list
- "Hi!", "Thanks!", "Great!", "Okay" → []

---

## Output Format (Strict)

One classification per line:
- Format: ` + "`SOURCE" + LLMTabSeparator + "QUERY`" + `
- ` + "`SOURCE`" + ` must be: ` + "`code`" + `, ` + "`profile`" + `, ` + "`event`" + `, or ` + "`image`" + `
- **Each SOURCE must appear AT MOST ONCE** — combine all relevant info for that agent into one QUERY
- For trivial inputs, output nothing

---

## Examples
%s

---
`

var (
	classifierOnce   sync.Once
	classifierPrompt string
)

func BuildClassifierSystemPrompt() string {
	classifierOnce.Do(func() {
		var exampleBlocks []string
		for _, ex := range classificationExamples {
			var output string
			if len(ex.Classifications) == 0 {
				output = "(empty - trivial/skip)"
			} else {
				output = packClassificationsIntoString(ex.Classifications)
			}
			exampleBlocks = append(exampleBlocks, fmt.Sprintf(
				"<example>\n<input>%s</input>\n<output>\n%s\n</output>\n</example>",
				ex.Input, output,
			))
		}
		examples := strings.Join(exampleBlocks, "\n\n")

		classifierPrompt = fmt.Sprintf(
			classifierSystemPromptTemplate,
			GetKeywordsString(CodeAgentKeywords),
			GetKeywordsString(ProfileAgentKeywords),
			GetKeywordsString(EventAgentKeywords),
			examples,
		)
	})
	return classifierPrompt
}

func PackClassificationQuery(userInput string) string {
	return fmt.Sprintf("Analyze this user input:\n\nUser Input: %s", userInput)
}
