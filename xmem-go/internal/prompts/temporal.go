package prompts

import (
	"fmt"
	"strings"
	"sync"
)

var temporalExamples = []struct {
	Input       string
	ContextDate string
	Output      string
}{
	{
		"My birthday is on March 15th",
		"4:04 pm on 20 January, 2023",
		"DATE: 03-15\nEVENT_NAME: Birthday\nYEAR: \nDESC: User's birthday\nTIME: \nDATE_EXPRESSION: March 15th",
	},
	{
		"Our wedding anniversary is July 22nd, 2019",
		"2:32 pm on 29 January, 2023",
		"DATE: 07-22\nEVENT_NAME: Wedding Anniversary\nYEAR: 2019\nDESC: User's wedding anniversary\nTIME: \nDATE_EXPRESSION: July 22nd, 2019",
	},
	{
		"I have a dentist appointment on January 10th at 2:30 PM",
		"12:48 am on 1 February, 2023",
		"DATE: 01-10\nEVENT_NAME: Dentist Appointment\nYEAR: \nDESC: Scheduled dentist visit\nTIME: 2:30 PM\nDATE_EXPRESSION: January 10th",
	},
	{
		"My daughter's birthday is on December 25th, she was born in 2015",
		"10:43 am on 4 February, 2023",
		"DATE: 12-25\nEVENT_NAME: Daughter's Birthday\nYEAR: 2015\nDESC: Daughter's birthday celebration\nTIME: \nDATE_EXPRESSION: December 25th",
	},
	{
		"Maria received a medal from the homeless shelter the week before 9 August 2023",
		"5:44 pm on 21 July, 2023",
		"DATE: 08-02\nEVENT_NAME: Medal Received\nYEAR: 2023\nDESC: Maria received a medal from the homeless shelter\nTIME: \nDATE_EXPRESSION: the week before 9 August 2023",
	},
	{
		"John participated in a 5K charity run on the first weekend of August 2023",
		"1:25 pm on 9 July, 2023",
		"DATE: 08-05\nEVENT_NAME: 5K Charity Run\nYEAR: 2023\nDESC: John participated in a 5K charity run\nTIME: \nDATE_EXPRESSION: first weekend of August 2023",
	},
	{
		"Wanna see my moves next Fri? Can't wait!",
		"4:04 pm on 20 January, 2023",
		"DATE: 01-27\nEVENT_NAME: Dance Session\nYEAR: 2023\nDESC: Planned dance session to show moves\nTIME: \nDATE_EXPRESSION: next Friday",
	},
	{
		"The official opening night is tomorrow. I'm working hard to make everything just right.",
		"10:04 am on 19 June, 2023",
		"DATE: 06-20\nEVENT_NAME: Studio Opening Night\nYEAR: 2023\nDESC: Official opening night of the dance studio\nTIME: \nDATE_EXPRESSION: tomorrow",
	},
	{
		"I went to a fair to show off my studio yesterday, it was both stressful and great!",
		"11:24 am on 25 April, 2023",
		"DATE: 04-24\nEVENT_NAME: Fair Exhibition\nYEAR: 2023\nDESC: Attended a fair to showcase dance studio\nTIME: \nDATE_EXPRESSION: yesterday",
	},
	{
		"Started hitting the gym last week to stay on track with the venture.",
		"2:35 pm on 16 March, 2023",
		"DATE: 03-09\nEVENT_NAME: Started Gym\nYEAR: 2023\nDESC: Started going to the gym\nTIME: \nDATE_EXPRESSION: last week",
	},
	{
		"I'm getting ready for a dance comp near me next month.",
		"10:43 am on 4 February, 2023",
		"DATE: 03-04\nEVENT_NAME: Dance Competition\nYEAR: 2023\nDESC: Dance competition preparation\nTIME: \nDATE_EXPRESSION: next month",
	},
	{
		"I really like pizza",
		"4:04 pm on 20 January, 2023",
		"NO_EVENT",
	},
	{
		"I usually go running in the mornings",
		"2:32 pm on 29 January, 2023",
		"NO_EVENT",
	},
	{
		"Mom's birthday is February 14th, she loves flowers",
		"12:48 am on 1 February, 2023",
		"DATE: 02-14\nEVENT_NAME: Mom's Birthday\nYEAR: \nDESC: Mother's birthday, she loves flowers\nTIME: \nDATE_EXPRESSION: February 14th",
	},
	{
		"Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.",
		"4:04 pm on 20 January, 2023",
		"DATE: 01-19\nEVENT_NAME: Lost Job\nYEAR: 2023\nDESC: Lost job as a banker\nTIME: \nDATE_EXPRESSION: yesterday",
	},
	{
		"I have a dentist appointment on January 10th at 2:30 PM and a concert on January 15th at 8 PM",
		"12:48 am on 1 January, 2023",
		"DATE: 01-10\nEVENT_NAME: Dentist Appointment\nYEAR: \nDESC: Scheduled dentist visit\nTIME: 2:30 PM\nDATE_EXPRESSION: January 10th\n---\nDATE: 01-15\nEVENT_NAME: Concert\nYEAR: \nDESC: Concert event\nTIME: 8 PM\nDATE_EXPRESSION: January 15th",
	},
	{
		"My birthday is on March 15th and our wedding anniversary is on July 22nd",
		"4:04 pm on 20 January, 2023",
		"DATE: 03-15\nEVENT_NAME: Birthday\nYEAR: \nDESC: User's birthday\nTIME: \nDATE_EXPRESSION: March 15th\n---\nDATE: 07-22\nEVENT_NAME: Wedding Anniversary\nYEAR: \nDESC: User's wedding anniversary\nTIME: \nDATE_EXPRESSION: July 22nd",
	},
}

var temporalSystemPromptTemplate = `You are an intelligent event extraction assistant.
Your task is to extract ALL structured temporal event information from user input.

---

## Your Responsibilities

1. Extract ALL events that have a specific date or recurring date pattern
2. Identify each date in MM-DD format (month-day)
3. Extract the event name, year (if mentioned), description, and time (if mentioned)
4. **IMPORTANT**: Use the provided CONTEXT_DATE to resolve relative date expressions
5. **IMPORTANT**: If the input contains MULTIPLE events, extract EACH ONE separately

---

## Handling Relative Dates

You will be given a CONTEXT_DATE which is the date/time when the conversation occurred.
Use this to resolve relative expressions:

- "yesterday" → subtract 1 day from CONTEXT_DATE
- "tomorrow" → add 1 day to CONTEXT_DATE
- "next Friday" → find the next Friday after CONTEXT_DATE
- "last week" → subtract ~7 days from CONTEXT_DATE
- "next month" → add ~30 days to CONTEXT_DATE
- "the week before [date]" → subtract 7 days from the mentioned date
- "first weekend of [month]" → first Saturday of that month
- "last [day of week]" → the most recent occurrence of that day before CONTEXT_DATE

---

## Output Format (Strict)

For EACH event, output in this exact format:
` + "```" + `
DATE: MM-DD
EVENT_NAME: <short name of the event>
YEAR: <year, infer from context if relative date>
DESC: <brief description of the event>
TIME: <time if mentioned, otherwise leave empty>
DATE_EXPRESSION: <the original date expression from input>
` + "```" + `

### Multiple Events
If the input contains MULTIPLE events, output each event block separated by ` + "`---`" + `:
` + "```" + `
DATE: MM-DD
EVENT_NAME: <first event name>
YEAR: <year>
DESC: <description>
TIME: <time>
DATE_EXPRESSION: <original expression>
---
DATE: MM-DD
EVENT_NAME: <second event name>
YEAR: <year>
DESC: <description>
TIME: <time>
DATE_EXPRESSION: <original expression>
` + "```" + `

---

## Rules

1. **Date Format**: Always output date as MM-DD (e.g., 01-15 for January 15th)
2. **Event Name**: Keep it concise (2-5 words)
3. **Year**: Include if explicitly mentioned OR if you can infer from CONTEXT_DATE for relative dates
4. **Description**: Brief summary of what the event is about
5. **Time**: Include if mentioned (e.g., "10:00 AM", "evening")
6. **DATE_EXPRESSION**: Always include the original date expression from the input
7. **No Event**: If the input doesn't contain a datable event, output: NO_EVENT
8. **Multiple Events**: If input contains multiple events with different dates, extract EACH one separately using ` + "`---`" + ` separator
9. **Never merge events**: Each distinct event should be its own separate block

---

## Examples
%s

---
`

var (
	temporalOnce   sync.Once
	temporalPrompt string
)

func BuildTemporalSystemPrompt() string {
	temporalOnce.Do(func() {
		var exampleBlocks []string
		for _, ex := range temporalExamples {
			exampleBlocks = append(exampleBlocks, fmt.Sprintf(
				"<example>\n<context_date>%s</context_date>\n<input>%s</input>\n<output>\n%s\n</output>\n</example>",
				ex.ContextDate, ex.Input, ex.Output,
			))
		}
		examples := strings.Join(exampleBlocks, "\n\n")
		temporalPrompt = fmt.Sprintf(temporalSystemPromptTemplate, examples)
	})
	return temporalPrompt
}

func PackTemporalQuery(userInput, contextDate string) string {
	if contextDate != "" {
		return fmt.Sprintf(
			"Extract ALL temporal events from this input:\n\nCONTEXT_DATE: %s\n\nUser Input: %s\n\nIf input contains multiple events, output each one separated by ---",
			contextDate, userInput,
		)
	}
	return fmt.Sprintf(
		"Extract ALL temporal events from this input:\n\nUser Input: %s\n\nIf input contains multiple events, output each one separated by ---",
		userInput,
	)
}
