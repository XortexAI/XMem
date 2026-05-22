package prompts

import (
	"fmt"
	"strings"
	"sync"
)

var judgeExamples = []struct {
	Domain          string
	NewItems        string
	SimilarExisting string
	Output          string
}{
	{
		"profile",
		"1. work / company = Now at Google\n2. food / preference = Loves sushi",
		"For item: \"work / company = Now at Google\"\n  - ID: abc123 | Score: 0.91 | \"work / company = Works at Microsoft\"\nFor item: \"food / preference = Loves sushi\"\n  - (no similar records)",
		`{"operations": [{"type": "UPDATE", "content": "work / company = Now at Google", "embedding_id": "abc123", "reason": "Company changed from Microsoft to Google"}, {"type": "ADD", "content": "food / preference = Loves sushi", "embedding_id": null, "reason": "New food preference, no existing record"}], "confidence": 0.95}`,
	},
	{
		"profile",
		"1. basic_info / name = Alice",
		"For item: \"basic_info / name = Alice\"\n  - ID: prof-001 | Score: 0.99 | \"basic_info / name = Alice\"",
		`{"operations": [{"type": "NOOP", "content": "", "embedding_id": "prof-001", "reason": "Exact duplicate — name is already Alice"}], "confidence": 0.99}`,
	},
	{
		"profile",
		"1. food / diet = User is now vegetarian",
		"For item: \"food / diet = User is now vegetarian\"\n  - ID: prof-042 | Score: 0.87 | \"food / diet = User loves steak\"",
		`{"operations": [{"type": "UPDATE", "content": "food / diet = User is now vegetarian", "embedding_id": "prof-042", "reason": "Diet changed — contradicts previous steak preference"}], "confidence": 0.92}`,
	},
	{
		"profile",
		"1. basic_info / name = Bob\n2. work / role = Engineer",
		"(No similar records found — vector store is empty or search returned nothing)",
		`{"operations": [{"type": "ADD", "content": "basic_info / name = Bob", "embedding_id": null, "reason": "No existing records"}, {"type": "ADD", "content": "work / role = Engineer", "embedding_id": null, "reason": "No existing records"}], "confidence": 0.95}`,
	},
	{
		"profile",
		"1. interest / hobbies = football",
		"For item: \"interest / hobbies = football\"\n  - ID: prof-070 | Score: 0.88 | \"interest / hobbies = reading\"",
		`{"operations": [{"type": "UPDATE", "content": "interest / hobbies = reading, football", "embedding_id": "prof-070", "reason": "Hobbies is a collection — merge old (reading) with new (football) rather than overwriting"}], "confidence": 0.93}`,
	},
	{
		"profile",
		"1. interest / foods = sushi",
		"For item: \"interest / foods = sushi\"\n  - ID: prof-080 | Score: 0.85 | \"interest / foods = pizza\"",
		`{"operations": [{"type": "UPDATE", "content": "interest / foods = pizza, sushi", "embedding_id": "prof-080", "reason": "Foods is a collection — merge old (pizza) with new (sushi) rather than overwriting"}], "confidence": 0.93}`,
	},
	{
		"temporal",
		"1. 03-15 | Birthday | User's birthday",
		"For item: \"03-15 | Birthday | User's birthday\"\n  - ID: evt-001 | Score: 0.97 | \"03-15 | Birthday | User's birthday\"",
		`{"operations": [{"type": "NOOP", "content": "", "embedding_id": "evt-001", "reason": "Exact duplicate event already stored"}], "confidence": 0.99}`,
	},
	{
		"temporal",
		"1. 07-22 | Wedding Anniversary | 5th wedding anniversary celebration in Paris",
		"For item: \"07-22 | Wedding Anniversary | 5th wedding anniversary celebration in Paris\"\n  - ID: evt-010 | Score: 0.88 | \"07-22 | Wedding Anniversary | User's wedding anniversary\"",
		`{"operations": [{"type": "UPDATE", "content": "07-22 | Wedding Anniversary | 5th wedding anniversary celebration in Paris", "embedding_id": "evt-010", "reason": "Same event with richer description"}], "confidence": 0.90}`,
	},
	{
		"temporal",
		"1. 01-28 | Paris Trip | Visited Paris",
		"For item: \"01-28 | Paris Trip | Visited Paris\"\n  - (no similar records)",
		`{"operations": [{"type": "ADD", "content": "01-28 | Paris Trip | Visited Paris", "embedding_id": null, "reason": "Brand-new event, nothing similar found"}], "confidence": 0.95}`,
	},
	{
		"temporal",
		"1. 02-10 | Dentist Appointment | Rescheduled dentist visit",
		"For item: \"02-10 | Dentist Appointment | Rescheduled dentist visit\"\n  - ID: evt-020 | Score: 0.90 | \"01-10 | Dentist Appointment | Scheduled dentist visit\"",
		`{"operations": [{"type": "DELETE", "content": "", "embedding_id": "evt-020", "reason": "Date changed from 01-10 to 02-10 — old graph connection invalid"}, {"type": "ADD", "content": "02-10 | Dentist Appointment | Rescheduled dentist visit", "embedding_id": null, "reason": "New date requires new User-Date relationship in graph"}], "confidence": 0.92}`,
	},
	{
		"summary",
		"1. User works as a software engineer\n2. User adopted a cat named Luna",
		"For item: \"User works as a software engineer\"\n  - ID: sum-005 | Score: 0.94 | \"User is a software engineer\"\nFor item: \"User adopted a cat named Luna\"\n  - (no similar records)",
		`{"operations": [{"type": "NOOP", "content": "", "embedding_id": "sum-005", "reason": "Semantically identical fact already exists"}, {"type": "ADD", "content": "User adopted a cat named Luna", "embedding_id": null, "reason": "New fact about user's pet"}], "confidence": 0.93}`,
	},
	{
		"summary",
		"1. User moved from NYC to San Francisco for a new role at Google",
		"For item: \"User moved from NYC to San Francisco for a new role at Google\"\n  - ID: sum-012 | Score: 0.82 | \"User lives in NYC\"",
		`{"operations": [{"type": "UPDATE", "content": "User moved from NYC to San Francisco for a new role at Google", "embedding_id": "sum-012", "reason": "User relocated — old NYC fact is outdated"}], "confidence": 0.88}`,
	},
	{
		"summary",
		"1. User enjoys hiking on weekends\n2. User has a golden retriever named Max",
		"(No similar records found — vector store is empty or search returned nothing)",
		`{"operations": [{"type": "ADD", "content": "User enjoys hiking on weekends", "embedding_id": null, "reason": "No existing records"}, {"type": "ADD", "content": "User has a golden retriever named Max", "embedding_id": null, "reason": "No existing records"}], "confidence": 0.98}`,
	},
}

var judgeSystemPromptTemplate = `You are the JUDGE agent in a semantic memory system.

Your role is to compare NEW incoming data against EXISTING similar records
retrieved via similarity search, then decide the correct operation
for each new item.

IMPORTANT: You only DECIDE — you never perform any storage operations yourself.

---

## Operation Types

- **ADD**    — New item has no similar existing record. Store it as-is.
- **UPDATE** — New item supersedes or refines an existing record. Replace the old one.
- **DELETE** — Existing record is now invalid / contradicted. Remove it.
- **NOOP**   — New item is a duplicate of an existing record. Skip it.

---

## Domain-Specific Guidelines

### profile
Each item is a user fact in the form ` + "`topic / sub_topic = memo`" + `.
- Same topic + sub_topic but different memo → **UPDATE** (user changed a fact).
- Exact same memo → **NOOP**.
- Brand-new topic/sub_topic → **ADD**.
- Contradicting a previous fact (e.g. "vegetarian" vs "loves steak") → **UPDATE** and optionally **DELETE** the old.

**IMPORTANT — Append vs Overwrite for UPDATE:**
Some sub_topics hold a COLLECTION of values (e.g. hobbies, foods, skills,
languages, favorite_movies, music_genres).  Others hold a SINGLE value
(e.g. name, company, city, job_title).

- **Collection sub_topics (hobbies, foods, skills, etc.):**
  When a user adds a new item to a collection, the UPDATE content MUST
  MERGE the old and new values. Example:
    Existing: "interest / hobbies = reading"
    New:      "interest / hobbies = football"
    Correct UPDATE content: "interest / hobbies = reading, football"
    WRONG:   "interest / hobbies = football"  ← this LOSES "reading"

- **Singular sub_topics (name, company, city, etc.):**
  The new value simply replaces the old. Example:
    Existing: "work / company = Google"
    New:      "work / company = Meta"
    Correct UPDATE content: "work / company = Meta"

### temporal
Each item is a temporal event with ` + "`date | event_name | desc | year | time | date_expression`" + `.
Events are stored in Neo4j as ` + "`User -[HAS_EVENT]-> Date`" + ` relationships.
- Same event_name and same date → **NOOP** (duplicate).
- Same event_name, same date, but updated details (desc/time) → **UPDATE**.
- Same event_name but **different date** → **DELETE** the old relationship + **ADD** a new one (two operations for this item, since the graph connection must point to a different Date node).
- Brand-new event → **ADD**.

### summary
Each item is a bullet-point fact extracted from conversation.
- Semantically identical fact already exists → **NOOP**.
- Similar but updated/refined → **UPDATE**.
- Brand-new fact → **ADD**.

### image
Each item is a visual observation in the format ` + "`category: description`" + `.
- Semantically identical observation already exists → **NOOP**.
- Similar but updated/refined → **UPDATE**.
- Brand-new observation → **ADD**.

---

## Output Format (Strict JSON)

Return a JSON object with:
` + "```json" + `
{
    "operations": [
        {
            "type": "ADD" | "UPDATE" | "DELETE" | "NOOP",
            "content": "The exact text to store (for ADD/UPDATE)",
            "embedding_id": "ID of existing record (for UPDATE/DELETE, null for ADD)",
            "reason": "Brief explanation"
        }
    ],
    "confidence": 0.0-1.0
}
` + "```" + `

---

## Rules

1. ALWAYS return valid JSON — no markdown fences, no commentary outside the JSON.
2. Usually one operation per new item. Exception: temporal date changes produce two operations (DELETE old + ADD new).
3. For UPDATE/DELETE, ` + "`embedding_id`" + ` MUST come from the SIMILAR_EXISTING list.
4. For ADD, ` + "`embedding_id`" + ` must be null.
5. If SIMILAR_EXISTING is empty for an item, default to ADD.
6. Contradictions → UPDATE (new fact wins over old).
7. Exact duplicates → NOOP.

---

## Examples
%s

---
`

var (
	judgeOnce   sync.Once
	judgePrompt string
)

func BuildJudgeSystemPrompt() string {
	judgeOnce.Do(func() {
		var exampleBlocks []string
		for _, ex := range judgeExamples {
			exampleBlocks = append(exampleBlocks, fmt.Sprintf(
				"<example>\n<domain>%s</domain>\n<new_items>\n%s\n</new_items>\n<similar_existing>\n%s\n</similar_existing>\n<output>\n%s\n</output>\n</example>",
				ex.Domain, ex.NewItems, ex.SimilarExisting, ex.Output,
			))
		}
		examples := strings.Join(exampleBlocks, "\n\n")
		judgePrompt = fmt.Sprintf(judgeSystemPromptTemplate, examples)
	})
	return judgePrompt
}

func PackJudgeQuery(newItems []string, similarExisting []string, domain string) string {
	numberedItems := make([]string, 0, len(newItems))
	for i, item := range newItems {
		numberedItems = append(numberedItems, fmt.Sprintf("%d. %s", i+1, item))
	}
	newBlock := strings.Join(numberedItems, "\n")

	similarBlock := "(No similar records found — vector store is empty or search returned nothing)"
	if len(similarExisting) > 0 {
		similarBlock = strings.Join(similarExisting, "\n")
	}

	return fmt.Sprintf("## DOMAIN: %s\n\n## NEW_ITEMS:\n%s\n\n## SIMILAR_EXISTING:\n%s", domain, newBlock, similarBlock)
}
