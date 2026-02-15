CODE_AGENT_KEYWORDS = [
    # Programming languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "perl", "lua", "haskell",
    # Actions
    "code", "coding", "program", "script", "implement", "develop", "build",
    "refactor", "optimize", "compile", "execute", "run", "deploy",
    # Debugging
    "debug", "fix", "error", "bug", "issue", "exception", "stack trace",
    "segmentation fault", "syntax error", "runtime error", "logic error",
    "traceback", "crash", "failing", "broken",
    # Concepts
    "function", "method", "class", "object", "variable", "loop",
    "array", "list", "map", "dictionary", "set", "tuple", "struct",
    "algorithm", "data structure", "recursion", "iteration",
    # APIs & Web
    "api", "endpoint", "request", "response", "rest", "graphql",
    "sdk", "library", "framework", "package", "module", "import",
    "http", "json", "xml", "websocket",
    # DevOps
    "git", "github", "docker", "kubernetes", "ci/cd", "pipeline",
    "aws", "azure", "gcp", "cloud", "server", "database",
    # Testing
    "unit test", "test case", "mock", "testing", "pytest", "jest",
    # Requests
    "how do i code", "write a function", "explain this code",
    "why is this failing", "convert this code", "refactor this",
    "what does this do", "how does this work"
]

PROFILE_AGENT_KEYWORDS = [
    # Identity
    "my name is", "i am called", "call me", "who am i", "i'm",
    "i am a", "i work as", "my job is", "my profession",
    # Preferences
    "i like", "i love", "i prefer", "my favorite", "what do i prefer",
    "i enjoy", "i hate", "i dislike", "can't stand", "i'm into",
    # Habits
    "i usually", "i often", "i always", "i never", "i sometimes",
    "every day", "daily routine", "my habit",
    # Location & Background
    "i live in", "i am from", "my city is", "my country is",
    "i grew up in", "my hometown", "based in",
    # Contact & Demographics
    "my phone number", "my email", "my age is", "i am X years old",
    "my address", "my gender", "my nationality",
    # Relationships
    "my wife", "my husband", "my partner", "my kids", "my children",
    "my mom", "my dad", "my parents", "my family", "my friend",
    "my brother", "my sister", "my son", "my daughter",
    # Traits & Values
    "i believe", "i value", "important to me", "i care about",
    "my personality", "i'm the type of person",
    # Memory commands
    "remember this", "store this", "save this", "add to my profile",
    "keep this in mind", "note this down", "store this fact",
    "update my", "change my", "that is wrong", "actually i am",
    # Interests & Hobbies (without time)
    "my hobby", "i'm interested in", "passionate about",
    "i volunteer", "i support", "i'm learning"
]

EVENT_AGENT_KEYWORDS = [
    # Future scheduling
    "schedule", "book", "plan", "set up", "arrange", "create event",
    "remind me", "set a reminder", "alarm", "notify me", "ping me",
    "meeting", "call", "appointment", "interview", "session",
    "deadline", "standup", "demo", "reservation",
    # Relative time (future)
    "today", "tomorrow", "day after tomorrow",
    "next week", "next month", "next year", "this weekend",
    "upcoming", "soon", "later", "in a few days",
    # Days of week
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    # Time of day
    "in the morning", "in the evening", "at night", "afternoon",
    "am", "pm", "o'clock",
    # Calendar actions
    "calendar", "add to calendar", "reschedule", "cancel meeting",
    # PAST temporal expressions (CRITICAL for memory)
    "yesterday", "last week", "last month", "last year",
    "last saturday", "last sunday", "last monday", "last tuesday",
    "last wednesday", "last thursday", "last friday",
    "years ago", "months ago", "weeks ago", "days ago",
    "a year ago", "a month ago", "a week ago",
    "back in", "when i was", "used to", "in the past",
    "recently", "just", "the other day",
    # Specific years/dates
    "in 2020", "in 2021", "in 2022", "in 2023", "in 2024", "in 2025",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # Recurring/annual events
    "birthday", "anniversary", "graduation", "wedding", "funeral",
    "holiday", "christmas", "thanksgiving", "new year", "easter",
    "valentine", "halloween", "independence day",
    # Age/timeline references
    "born on", "born in", "turned", "18th birthday", "21st birthday",
    "when i turned", "at age", "years old when",
    # Life events with dates (MILESTONES)
    "moved", "started", "graduated", "married", "retired", "began",
    "finished", "completed", "launched", "opened", "closed",
    "joined", "left", "quit", "hired", "fired", "promoted",
    "divorced", "engaged", "pregnant", "gave birth", "adopted",
    "surgery", "diagnosed", "recovered", "hospitalized",
    "traveled", "visited", "went to", "came back from",
    "first time", "last time", "this was when"
]


def get_keywords_string(keywords: list) -> str:
    return ", ".join(keywords)
