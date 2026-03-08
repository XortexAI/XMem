from typing import List, Tuple

SUMMARY_EXAMPLES: List[Tuple[str, str, str]] = [
    (
        "Hi there!",
        "Hello! How can I help you today?",
        "",
    ),
    (
        "I got rejected from the Google L6 role but got an offer from Meta for E6 at $450k total comp.",
        "Meta's offer is strong. The E6 level is roughly equivalent to Google L6 anyway.",
        """
        - User was rejected from Google L6 position
        - User received offer from Meta for E6 level role at $450k total compensation
        - Agent advised that Meta E6 is roughly equivalent to Google L6
        """,
    ),
    (
        "I'm getting this error in my FastAPI app: 'NoneType object has no attribute email' in the login endpoint.",
        "The error occurs because you're trying to access user.email before checking if the user exists. Add validation like `if user is None: raise HTTPException(404)` before accessing user attributes.",
        """
        - User is developing a FastAPI application with authentication
        - User encountered NoneType error when accessing user.email in login endpoint
        - Agent identified cause: accessing user.email before checking if user exists
        - Solution provided: add validation check `if user is None: raise HTTPException(404)` before accessing user attributes
        """,
    ),
    (
        "My wife Priya and I just adopted a golden retriever puppy named Buddy from the shelter.",
        "That's wonderful! Golden retrievers are amazing family dogs. How old is Buddy?",
        """
        - User is married, wife's name is Priya
        - User adopted a golden retriever puppy named Buddy from shelter
        """,
    ),
]