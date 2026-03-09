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
        (
        "I need to choose between two job offers: Startup A (equity-heavy, risky) vs BigCorp (stable, lower upside). I have a family and mortgage.",
        "Given your family and mortgage, financial stability is crucial. BigCorp offers predictability and benefits. However, if the startup's equity is significant and you have 6-12 months runway, calculated risk can pay off. What's your risk tolerance and the startup's funding stage?",
        """
        - User is deciding between two job offers: startup with high equity/risk vs established company with stability
        - User has family and mortgage (financial obligations)
        - Agent advised considering financial stability given obligations
        - Agent suggested factors to evaluate: risk tolerance, 6-12 month runway, startup funding stage
        - Agent noted BigCorp offers predictability and benefits, startup equity could have upside
        """,
    ),
    (
        "Should I use PostgreSQL or MongoDB for my e-commerce app?",
        "For e-commerce, PostgreSQL is generally better due to ACID compliance for transactions, complex queries for inventory/orders, and strong consistency. Use MongoDB only if you need extreme flexibility in product schemas or are handling massive write loads. What's your expected scale?",
        """
        - User is choosing database for e-commerce application
        - Agent recommended PostgreSQL over MongoDB for e-commerce
        - Reasons: ACID compliance for transactions, better for complex queries (inventory/orders), strong consistency
        - MongoDB suitable only for: extreme schema flexibility or massive write loads
        """,
    ),
    (
        "I collect vintage mechanical watches. Just bought a 1960s Omega Speedmaster.",
        "The Speedmaster is iconic! Is this your first Omega?",
        """
        - User collects vintage mechanical watches
        - User recently purchased a 1960s Omega Speedmaster
        """,
    ),
    (
        "I used to work at Tesla as a mechanical engineer but I quit last year to travel.",
        "Bold move! Where did your travels take you?",
        """
        - User previously worked as a mechanical engineer at Tesla
        - User quit job last year to travel
        """,
    ),
]