package prompts

import (
	"fmt"
	"strings"
	"sync"
)

var profilerExamples = []struct {
	Input  string
	Output string
}{
	{
		"I work as a Senior Software Engineer at Google and have 5 years of experience in Python and TensorFlow",
		"work" + LLMTabSeparator + "company" + LLMTabSeparator + "Google\n" +
			"work" + LLMTabSeparator + "title" + LLMTabSeparator + "Senior Software Engineer\n" +
			"work" + LLMTabSeparator + "work_skills" + LLMTabSeparator + "Python (5 years), TensorFlow (5 years)",
	},
	{
		"My name is Sarah Chen, I'm 28 years old and I live in San Francisco. I speak English and Mandarin fluently.",
		"basic_info" + LLMTabSeparator + "name" + LLMTabSeparator + "Sarah Chen\n" +
			"basic_info" + LLMTabSeparator + "age" + LLMTabSeparator + "28\n" +
			"contact_info" + LLMTabSeparator + "city" + LLMTabSeparator + "San Francisco\n" +
			"contact_info" + LLMTabSeparator + "country" + LLMTabSeparator + "United States (California)\n" +
			"basic_info" + LLMTabSeparator + "language_spoken" + LLMTabSeparator + "English, Mandarin (fluent)",
	},
	{
		"I love reading science fiction novels especially Isaac Asimov. I play tennis on weekends and I'm a vegetarian who loves Italian food.",
		"interest" + LLMTabSeparator + "books" + LLMTabSeparator + "science fiction novels, especially Isaac Asimov\n" +
			"interest" + LLMTabSeparator + "sports" + LLMTabSeparator + "tennis (plays on weekends)\n" +
			"interest" + LLMTabSeparator + "foods" + LLMTabSeparator + "vegetarian, loves Italian cuisine",
	},
	{
		"I graduated from MIT with a CS degree in 2019 and I'm currently doing my Master's in Machine Learning at Stanford",
		"education" + LLMTabSeparator + "school" + LLMTabSeparator + "MIT (graduated 2019), Stanford (current, Master's)\n" +
			"education" + LLMTabSeparator + "degree" + LLMTabSeparator + "Bachelor's in Computer Science (MIT, 2019), Master's in Machine Learning (Stanford, in progress)\n" +
			"education" + LLMTabSeparator + "major" + LLMTabSeparator + "Computer Science (undergraduate), Machine Learning (graduate)",
	},
	{
		"I'm married with two children. We recently moved from New York to Austin, Texas. I enjoy painting and pottery.",
		"demographics" + LLMTabSeparator + "marital_status" + LLMTabSeparator + "married\n" +
			"demographics" + LLMTabSeparator + "number_of_children" + LLMTabSeparator + "2\n" +
			"life_event" + LLMTabSeparator + "relocation" + LLMTabSeparator + "moved from New York to Austin, Texas\n" +
			"contact_info" + LLMTabSeparator + "city" + LLMTabSeparator + "Austin\n" +
			"contact_info" + LLMTabSeparator + "country" + LLMTabSeparator + "United States (Texas)\n" +
			"interest" + LLMTabSeparator + "hobbies" + LLMTabSeparator + "painting, pottery",
	},
	{
		"My wife and I are expecting our first child next month",
		"demographics" + LLMTabSeparator + "marital_status" + LLMTabSeparator + "married\n" +
			"demographics" + LLMTabSeparator + "number_of_children" + LLMTabSeparator + " 1 (expecting first child)",
	},
	{
		"I just switched from my Android to iPhone because I prefer the ecosystem",
		"interest" + LLMTabSeparator + "hobbies" + LLMTabSeparator + "prefers Apple/iPhone ecosystem over Android",
	},
}

var profilerSystemPromptTemplate = `You are a professional psychologist.
Your responsibility is to read the user's query and extract important user profiles in a structured format.
Extract relevant facts, preferences, and attributes that help build a complete picture of the user.
You will not only extract explicitly stated information, but also infer what is implied.

## Output Format

### Think
First, think about what topics/subtopics are mentioned or implied.

### Profile
After thinking, extract facts as an ordered list:
TOPIC` + LLMTabSeparator + `SUB_TOPIC` + LLMTabSeparator + `MEMO

For example:
basic_info` + LLMTabSeparator + `name` + LLMTabSeparator + `melinda
work` + LLMTabSeparator + `title` + LLMTabSeparator + `software engineer

Each line is one fact containing:
1. TOPIC — the high-level category
2. SUB_TOPIC — the specific attribute
3. MEMO — the extracted value

Separate elements with ` + "`" + LLMTabSeparator + "`" + ` and each line with ` + "`\\n`" + `.

Final output template:
` + "```" + `
[YOUR THINKING...]
---
TOPIC` + LLMTabSeparator + `SUB_TOPIC` + LLMTabSeparator + `MEMO
...
` + "```" + `

## Few-Shot Examples
%s

## Topic Guidelines
Focus on collecting these topics and subtopics:
%s

## Rules
- Only extract topics related to the USER, not other people mentioned.
- **Infer implied facts**: If the user says "my husband", "my wife", "my partner" — infer marital_status as married. If they mention a spouse name, extract it under spouse_name.
- **Self-contained memos**: Every memo must be understandable on its own without the original query. BAD: "4 years". GOOD: "close college friends for 4 years". Always include WHO, WHAT, or context in the memo.
- If time-sensitive information is mentioned, infer the specific date when possible.
- Never use relative dates like "today" or "yesterday".
- **No duplicate topic/sub_topic pairs**: Never output the same topic::sub_topic combination more than once. If a user is relocating, extract life_event::relocation with full context, but only output contact_info::country ONCE with the NEW location.
- If nothing relevant is found, return an empty list.
- If the user input is trivial (e.g. "hi", "thanks"), return NONE.

Now perform your task.
`

var (
	profilerOnce   sync.Once
	profilerPrompt string
)

func BuildProfilerSystemPrompt() string {
	profilerOnce.Do(func() {
		var exampleBlocks []string
		for _, ex := range profilerExamples {
			exampleBlocks = append(exampleBlocks, fmt.Sprintf(
				"<example>\n<input>\n%s\n</input>\n<output>\n%s\n</output>\n</example>",
				ex.Input, ex.Output,
			))
		}
		examples := strings.Join(exampleBlocks, "\n\n")
		profilerPrompt = fmt.Sprintf(
			profilerSystemPromptTemplate,
			examples,
			FormatTopicsForPrompt(nil),
		)
	})
	return profilerPrompt
}

func PackProfilerQuery(query string) string {
	return query
}
