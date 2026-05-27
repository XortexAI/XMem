package prompts

import (
	"fmt"
	"strings"
	"sync"
)

var summaryExamples = []struct {
	UserQuery     string
	AgentResponse string
	Summary       string
}{
	{
		"Hi there!",
		"Hello! How can I help you today?",
		"",
	},
	{
		"I got rejected from the Google L6 role but got an offer from Meta for E6 at $450k total comp.",
		"Meta's offer is strong. The E6 level is roughly equivalent to Google L6 anyway.",
		"- User was rejected from Google L6 position\n- User received offer from Meta for E6 level role at $450k total compensation\n- Agent advised that Meta E6 is roughly equivalent to Google L6",
	},
	{
		"I'm getting this error in my FastAPI app: 'NoneType object has no attribute email' in the login endpoint.",
		"The error occurs because you're trying to access user.email before checking if the user exists. Add validation like `if user is None: raise HTTPException(404)` before accessing user attributes.",
		"- User is developing a FastAPI application with authentication\n- User encountered NoneType error when accessing user.email in login endpoint\n- Agent identified cause: accessing user.email before checking if user exists\n- Solution provided: add validation check `if user is None: raise HTTPException(404)` before accessing user attributes",
	},
	{
		"My wife Priya and I just adopted a golden retriever puppy named Buddy from the shelter.",
		"That's wonderful! Golden retrievers are amazing family dogs. How old is Buddy?",
		"- User is married, wife's name is Priya\n- User adopted a golden retriever puppy named Buddy from shelter",
	},
	{
		"I need to choose between two job offers: Startup A (equity-heavy, risky) vs BigCorp (stable, lower upside). I have a family and mortgage.",
		"Given your family and mortgage, financial stability is crucial. BigCorp offers predictability and benefits. However, if the startup's equity is significant and you have 6-12 months runway, calculated risk can pay off. What's your risk tolerance and the startup's funding stage?",
		"- User is deciding between two job offers: startup with high equity/risk vs established company with stability\n- User has family and mortgage (financial obligations)\n- Agent advised considering financial stability given obligations\n- Agent suggested factors to evaluate: risk tolerance, 6-12 month runway, startup funding stage\n- Agent noted BigCorp offers predictability and benefits, startup equity could have upside",
	},
	{
		"Should I use PostgreSQL or MongoDB for my e-commerce app?",
		"For e-commerce, PostgreSQL is generally better due to ACID compliance for transactions, complex queries for inventory/orders, and strong consistency. Use MongoDB only if you need extreme flexibility in product schemas or are handling massive write loads. What's your expected scale?",
		"- User is choosing database for e-commerce application\n- Agent recommended PostgreSQL over MongoDB for e-commerce\n- Reasons: ACID compliance for transactions, better for complex queries (inventory/orders), strong consistency\n- MongoDB suitable only for: extreme schema flexibility or massive write loads",
	},
	{
		"I collect vintage mechanical watches. Just bought a 1960s Omega Speedmaster.",
		"The Speedmaster is iconic! Is this your first Omega?",
		"- User collects vintage mechanical watches\n- User recently purchased a 1960s Omega Speedmaster",
	},
	{
		"I used to work at Tesla as a mechanical engineer but I quit last year to travel.",
		"Bold move! Where did your travels take you?",
		"- User previously worked as a mechanical engineer at Tesla\n- User quit job last year to travel",
	},
}

var summarizerSystemPromptTemplate = `You are a conversation summarization system for an AI assistant's memory.

## YOUR TASK
Summarize conversations to capture what was discussed, what was learned, and what advice/solutions were provided. This summary will be stored as compressed memory so the assistant doesn't need to keep full conversation history.

## ⚠️ ANTI-HALLUCINATION RULES (HIGHEST PRIORITY)
- **NEVER invent, assume, or infer details** that are NOT explicitly stated in the conversation.
- **Preserve ALL specific entities EXACTLY as stated**: names, dates, prices, amounts, percentages, ages, locations, companies, product names, event names, technical terms.
- If the user says "$2M seed round" → write "$2M seed round", NOT "received funding" or "raised capital".
- If the user says "March 15th" → write "March 15th", NOT "in spring" or "mid-March".
- If the user says "450k total comp" → write "$450k total compensation", NOT "high compensation".
- If the user says "3 bed, 2 bath" → write "3 bedroom, 2 bathroom", NOT "a house".
- **DO NOT generalize specific information.** Every number, date, name, and entity matters.
- **DO NOT add context or details** that aren't in the conversation. Only summarize what IS there.
- When in doubt, quote the user's exact words rather than paraphrasing.

## INPUT FORMAT
You receive:
- **User Query**: What the user said
- **Agent Response**: How the assistant responded

## WHAT TO EXTRACT

### 1. User Context & Facts
Extract meaningful information about the user:
- Personal details (name, location, relationships, family)
- Professional info (job, company, projects, career goals)
- Plans, goals, and commitments
- Health, lifestyle, preferences
- Specific circumstances or situations

### 2. Problems & Solutions
When the conversation involves problem-solving, capture:
- What problem/issue the user had
- What caused the problem (if diagnosed)
- What solution/advice was provided
- Key technical details, commands, or patterns mentioned

### 3. Advice & Recommendations
When the assistant gives advice, capture:
- What question/decision the user faced
- What options were discussed
- What recommendation was made and why
- Key factors or trade-offs mentioned

### 4. Outcomes & Decisions
- Decisions the user made or is leaning toward
- Completed actions or milestones
- Planned next steps

## WHAT NOT TO EXTRACT

Skip these types of exchanges:
- Pure greetings ("Hi", "Thanks", "Bye")
- Simple factual questions with no personal context ("What's the capital of France?")
- Hypothetical scenarios with no commitment ("If I were to...")
- Questions about others with no personal stake ("What should I get my girlfriend?")

## OUTPUT FORMAT

Return as many concise bullet points as necessary to capture:
- [Summary point with specific details]
- [Another summary point]
- [Another summary point]

### Formatting Requirements:
- Start each line with ` + "`- `" + ` (dash and space)
- **PRESERVE EXACT ENTITIES**: names, numbers, dates, prices, amounts, locations, technical terms — copy them VERBATIM
- For user facts: start with "User [verb]..."
- For problems: include both problem AND solution
- For advice: include both the question AND the recommendation
- Keep bullets concise but complete (1-2 sentences max per bullet)
- If nothing memorable exists, return empty string: ` + "`\"\"`" + `
- **NEVER add information that is not explicitly in the conversation**

### Quality Standards:

GOOD EXAMPLES:
- "User is developing a FastAPI application with authentication"
- "User encountered NoneType error accessing user.email; Agent advised adding null check before attribute access"
- "User choosing between PostgreSQL and MongoDB for e-commerce app; Agent recommended PostgreSQL for ACID compliance and transaction support"
- "User's startup raised $2M seed round led by Y Combinator; building AI tools for legal document review"

BAD EXAMPLES:
- "User has a technical issue" (too vague, missing details)
- "Agent helped the user" (what was the problem? what was the solution?)
- "User is thinking about something" (what specifically?)
- "User works somewhere" (where? what role?)

## EXAMPLES

%s

## CRITICAL REMINDERS
- **ZERO HALLUCINATION** — ONLY include facts explicitly stated in the conversation. NEVER invent or assume.
- **PRESERVE ENTITIES VERBATIM** — Dates ("March 15th"), prices ("$2M"), names ("Marcus"), locations ("Tuscany"), ages ("7 years old"), quantities ("3 bed, 2 bath") must be copied EXACTLY.
- **CAPTURE SOLUTIONS** — Don't just note the problem; include what advice/fix was provided
- **BE SPECIFIC** — Include technical terms, specific technologies, actual numbers and dates
- **BALANCE USER FACTS + CONVERSATION CONTENT** — Capture both who the user is and what was discussed
- **TECHNICAL DETAILS MATTER** — Commands, algorithms, specific approaches should be captured
- **SKIP TRIVIAL EXCHANGES** — Greetings, simple Q&A, hypotheticals -> empty string
- **QUALITY OVER QUANTITY** — Better to have 2 detailed bullets than 5 vague ones
- **WHEN UNCERTAIN** — If you're not sure about a detail, omit it rather than guess
`

var (
	summarizerOnce   sync.Once
	summarizerPrompt string
)

func BuildSummarizerSystemPrompt() string {
	summarizerOnce.Do(func() {
		var exampleBlocks []string
		for _, ex := range summaryExamples {
			output := strings.TrimSpace(ex.Summary)
			if output == "" {
				output = "(empty string - no memorable content)"
			}
			exampleBlocks = append(exampleBlocks, fmt.Sprintf(
				"<example>\n<user_query>\n%s\n</user_query>\n<agent_response>\n%s\n</agent_response>\n<summary>\n%s\n</summary>\n</example>",
				ex.UserQuery, ex.AgentResponse, output,
			))
		}
		examples := strings.Join(exampleBlocks, "\n\n")
		summarizerPrompt = fmt.Sprintf(summarizerSystemPromptTemplate, examples)
	})
	return summarizerPrompt
}

func PackSummaryQuery(userQuery, agentResponse string) string {
	return fmt.Sprintf(
		"<conversation>\n<user_query>\n%s\n</user_query>\n\n<agent_response>\n%s\n</agent_response>\n</conversation>\n\nSummarize this conversation. Include user context, problems/solutions, and key advice. Return 1 bullet point for every discrete piece of information (extract as many as needed to capture all facts) or empty string if trivial.",
		userQuery, agentResponse,
	)
}
