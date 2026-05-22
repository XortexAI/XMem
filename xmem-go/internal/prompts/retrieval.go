package prompts

import (
	"fmt"
	"strings"
	"sync"
)

var retrievalExamples = []struct {
	Query     string
	Catalog   string
	ToolCalls []struct {
		Tool string
		Args map[string]string
		Why  string
	}
}{
	{
		"What food does the user like?",
		"  - interest / foods\n  - work / company",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_profile", map[string]string{"topic": "interest"}, "Question asks about food → falls under 'interest' topic"},
		},
	},
	{
		"Where does the user work?",
		"  - interest / foods\n  - work / company\n  - work / title",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_profile", map[string]string{"topic": "work"}, "Question asks about workplace → 'work' topic returns company + title"},
		},
	},
	{
		"When is my dentist appointment?",
		"  - interest / foods",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_temporal", map[string]string{"query": "dentist appointment"}, "Date/scheduling question → search events"},
		},
	},
	{
		"When is my birthday?",
		"  - personal / name",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_temporal", map[string]string{"query": "birthday"}, "'When' question about a recurring event → temporal"},
		},
	},
	{
		"What events do I have coming up?",
		"  - work / company",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_temporal", map[string]string{"query": "upcoming events"}, "Broad events question → temporal search"},
		},
	},
	{
		"What do you know about me?",
		"  - interest / foods\n  - work / company",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_summary", map[string]string{"query": "what do you know about the user"}, "Broad question with no specific domain → summary"},
		},
	},
	{
		"What happened in our last conversation?",
		"  - personal / name",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_summary", map[string]string{"query": "last conversation"}, "General recall question → summary search"},
		},
	},
	{
		"Where do I work and when is my birthday?",
		"  - work / company\n  - work / title",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_profile", map[string]string{"topic": "work"}, "First part: workplace → 'work' topic"},
			{"search_temporal", map[string]string{"query": "birthday"}, "Second part: birthday date → temporal search"},
		},
	},
	{
		"Tell me about my hobbies and any upcoming events",
		"  - interest / hobbies\n  - personal / name",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_profile", map[string]string{"topic": "interest"}, "Hobbies fall under 'interest' topic"},
			{"search_temporal", map[string]string{"query": "upcoming events"}, "Events question → temporal"},
		},
	},
	{
		"What do I like to eat and when did I start my current job?",
		"  - interest / foods\n  - work / company",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_profile", map[string]string{"topic": "interest"}, "Food preference → 'interest' topic"},
			{"search_temporal", map[string]string{"query": "started current job"}, "When question → temporal search for job start event"},
		},
	},
	{
		"Does the user have any pets?",
		"  - work / company\n  - interest / foods",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_summary", map[string]string{"query": "pets"}, "No pet-related topic exists → try summary"},
		},
	},
	{
		"What programming languages does the user know?",
		"  - work / company",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_summary", map[string]string{"query": "programming languages"}, "No language-related topic → search summaries"},
		},
	},
	{
		"Can you show me the python script I wrote for database connection?",
		"  - work / company",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_snippet", map[string]string{"query": "python database connection script"}, "Asking for a previously written script/code → search_snippet"},
		},
	},
	{
		"What was my neo4j cypher query for deleting events?",
		"  - work / company",
		[]struct {
			Tool string
			Args map[string]string
			Why  string
		}{
			{"search_snippet", map[string]string{"query": "neo4j cypher query delete events"}, "Asking for a previously written query/code → search_snippet"},
		},
	},
}

var retrievalSystemPromptTemplate = `You are the RETRIEVAL agent in a personal semantic memory system called Xmem.

Your job is to answer questions or autocomplete sentences about the user by searching their stored
memories. You do this in two steps:

  1. Decide WHAT information you need → call the right search tool(s).
  2. Once you receive the results → compose a clear, concise answer, or provide the completion.

═══════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════════════

### 1. search_profile(topic)
   **What it searches:** Pinecone vector store (metadata filter).
   **When to use:** The question asks about a specific user attribute
   (name, job, food preference, hobby, etc.) AND a matching topic
   exists in the AVAILABLE PROFILES below.
   **How it works:** Returns ALL sub-topics under the requested topic,
   giving you full context.  For example, calling search_profile("work")
   returns company, title, and any other work-related facts.
   **You MUST use a topic value from the catalog below.**

### 2. search_temporal(query)
   **What it searches:** Neo4j graph database (semantic similarity).
   **When to use:** The question involves dates, times, "when",
   schedules, appointments, birthdays, milestones, or events.
   **How it works:** Embeds your query and compares it against stored
   event embeddings. Provide a short, descriptive query like
   "dentist appointment" or "birthday".

### 3. search_summary(query)
   **What it searches:** Pinecone vector store (semantic similarity,
   domain=summary).
   **When to use:** The question is broad/general and doesn't fit
   neatly into profile or temporal domains.
   **How it works:** Embeds your query and finds similar conversation
   summaries. Good fallback when no profile topic matches.

### 4. search_snippet(query)
   **What it searches:** Pinecone vector store (the user's private code snippets).
   **When to use:** The question asks for code, scripts, configurations, or technical solutions the user previously wrote or saved.
   **How it works:** Embeds your query and searches the user's isolated snippets namespace. It returns the raw code blocks.

═══════════════════════════════════════════════════════════════════════
AVAILABLE PROFILES (topic / sub_topic)
═══════════════════════════════════════════════════════════════════════

%s

═══════════════════════════════════════════════════════════════════════
DECISION RULES
═══════════════════════════════════════════════════════════════════════

1. **Profile first** — If a matching topic exists in the catalog,
   use search_profile with that topic. It returns ALL sub-topics
   under it, so you get full context.

2. **Temporal for dates** — Any question with "when", a date reference,
   or event-related language → search_temporal.

3. **Snippets for code** — Any question asking to retrieve a previously written script, code block, or technical configuration → search_snippet.

4. **Summary as fallback** — For broad questions like "what do you know
   about me" or when no profile topic matches → search_summary.

4. **Multi-tool is fine** — If the question spans domains, call
   multiple tools. Example: "Where do I work and when is my birthday?"
   → search_profile(topic="work") + search_temporal(query="birthday").

5. **Don't guess** — If nothing matches, call search_summary with the
   question rephrased as a short query. Never fabricate an answer
   without searching first.

═══════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════

%s

`

var (
	retrievalExamplesOnce sync.Once
	retrievalExamplesStr  string
)

func buildRetrievalExamplesBlock() string {
	retrievalExamplesOnce.Do(func() {
		var parts []string
		for _, ex := range retrievalExamples {
			var toolLines []string
			for _, tc := range ex.ToolCalls {
				argsStr := ""
				for k, v := range tc.Args {
					if argsStr != "" {
						argsStr += ", "
					}
					argsStr += fmt.Sprintf(`%s="%s"`, k, v)
				}
				toolLines = append(toolLines, fmt.Sprintf(
					"    → %s(%s)\n      Reason: %s",
					tc.Tool, argsStr, tc.Why,
				))
			}
			parts = append(parts, fmt.Sprintf(
				"  Query: \"%s\"\n  Profiles:\n%s\n  Tool calls:\n%s",
				ex.Query, ex.Catalog, strings.Join(toolLines, "\n"),
			))
		}
		retrievalExamplesStr = strings.Join(parts, "\n\n")
	})
	return retrievalExamplesStr
}

func BuildRetrievalSystemPrompt(profileCatalog string) string {
	return fmt.Sprintf(
		retrievalSystemPromptTemplate,
		profileCatalog,
		buildRetrievalExamplesBlock(),
	)
}

var answerPromptTemplate = `You are a helpful personal memory assistant. Answer or autocomplete the user's input
based on the retrieved context below.

## Retrieved Context:
%s

## User's Input:
%s

## Instructions:
1. **Autocomplete vs Question**:
   - If the User's Input looks like an incomplete sentence (e.g., "My name is ", "I work at "), your response MUST ONLY be the continuation of that sentence based on the context (e.g., "John Doe."). Do NOT repeat the prompt or answer it like a question.
   - If the User's Input is a question, answer concisely and directly using the retrieved information.
2. Use "you" when referring to the user (e.g., "Your birthday is…") if answering a question.
3. If multiple sources are relevant, combine them naturally.
4. **Partial matches count** — If the context contains information that is
   related or partially relevant to the question, share what you have.
   You may add a brief caveat like "Based on what I have…" but do NOT
   refuse to answer just because the match isn't exact.
5. **Format dates nicely** — If you see raw dates like "06-02" or "05-19",
   convert them to human-readable form: "2nd June", "19th May", etc.
   If a year is available, include it (e.g., "19th May, 2023").
6. **General / tone questions** — For broad questions about personality,
   communication style, or "what kind of person", synthesize an answer
   from all available context. Look for patterns across memories — interests,
   values, how the user talks, what they care about.
7. Do NOT fabricate facts. Only use what's in the retrieved context.
8. Only say "I don't have that information" as a LAST RESORT — when
   the context is truly empty or completely unrelated to the input.

Answer/Completion:`

func BuildAnswerPrompt(context, query string) string {
	return fmt.Sprintf(answerPromptTemplate, context, query)
}
