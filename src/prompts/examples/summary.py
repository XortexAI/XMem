from typing import List, Tuple

SUMMARY_EXAMPLES: List[Tuple[str, str, str]] = [
    (
        "Hi there!",
        "Hello! How can I help you today?",
        "",
    ),
    (
        "Thanks so much!",
        "You're welcome! Let me know if you need anything else.",
        "",
    ),
    (
        "What's the capital of France?",
        "The capital of France is Paris.",
        "",
    ),
    (
        "My name is Sarah Chen and I just moved to Portland, Oregon last month.",
        "Welcome to Portland! It's a great city with amazing food and nature nearby.",
        """
        - User's name is Sarah Chen
        - User moved to Portland, Oregon last month
        """,
    ),
    (
        "I'm getting married next June! My fiancé's name is Marcus and we're having the wedding in Tuscany.",
        "Congratulations! A Tuscany wedding sounds beautiful.",
        """
        - User is getting married in June (next year)
        - User's fiancé is named Marcus
        - User is planning wedding in Tuscany, Italy
        """,
    ),
    (
        "I have two kids - Emma (7) and Noah (4). Emma just started second grade.",
        "That's a great age gap! How is Emma enjoying second grade?",
        """
        - User has two children: Emma (7 years old) and Noah (4 years old)
        - Emma recently started second grade
        """,
    ),
    (
        "I work as a product manager at Stripe. Been here for 3 years, mostly working on payment APIs.",
        "Stripe is doing great work! How do you like the PM role there?",
        """
        - User works as a Product Manager at Stripe
        - User has been at Stripe for 3 years
        - User's work focuses on payment APIs
        """,
    ),
    (
        "I'm thinking about leaving my consulting job to start a SaaS company focused on HR analytics.",
        "That's exciting! What gap in the HR analytics market are you trying to fill?",
        """
        - User currently works in consulting
        - User is considering starting a SaaS company focused on HR analytics
        """,
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
        "I'm learning Rust because I want to contribute to the Tokio async runtime project.",
        "Great choice! Tokio is a fantastic project to contribute to. Start with the contributor guide.",
        """
        - User is learning Rust programming language
        - User's goal is to contribute to the Tokio async runtime project
        - Agent recommended starting with the Tokio contributor guide
        """,
    ),
    (
        "I built a Chrome extension that got 10k users in the first month! It's a Notion web clipper.",
        "That's impressive growth! What's driving the adoption?",
        """
        - User built a Chrome extension (Notion web clipper)
        - User's extension reached 10,000 users in first month
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
        "How do I implement rate limiting in Redis for my API?",
        "Use the token bucket algorithm with Redis INCR and EXPIRE commands. Here's a pattern: for each request, INCR a key with the user's ID and timestamp bucket, then EXPIRE it. If the count exceeds your limit, reject the request.",
        """
        - User is building an API that requires rate limiting
        - Agent recommended token bucket algorithm using Redis INCR and EXPIRE commands
        - Implementation pattern: INCR key with user ID and timestamp bucket, set EXPIRE, reject if count exceeds limit
        """,
    ),
    (
        "My React app is re-rendering too much. I'm using useState for a large object and updating it frequently.",
        "The issue is that updating the entire object causes full re-renders. Consider using useReducer for complex state, or split the state into smaller pieces. Also, wrap child components in React.memo to prevent unnecessary re-renders.",
        """
        - User's React app has performance issues due to excessive re-rendering
        - User is using useState with a large object that updates frequently
        - Agent diagnosed: updating entire object causes full re-renders
        - Solutions provided: use useReducer for complex state, split state into smaller pieces, wrap child components in React.memo
        """,
    ),
    (
        "I've been vegan for 2 years now and recently started training for my first marathon in Chicago this October.",
        "That's ambitious! How's your training going? Are you working with a coach?",
        """
        - User has been vegan for 2 years
        - User is training for first marathon in Chicago in October (this year)
        """,
    ),
    (
        "I'm allergic to shellfish and tree nuts. Found out the hard way last year with a severe reaction.",
        "That must have been scary. Do you carry an EpiPen now?",
        """
        - User is allergic to shellfish and tree nuts
        - User had a severe allergic reaction last year
        """,
    ),
    (
        "I've been dealing with ADHD and finally got diagnosed at 32. Starting Adderall next week.",
        "Getting a diagnosis can be really validating. I hope the medication helps!",
        """
        - User has ADHD, diagnosed at age 32
        - User is starting Adderall medication next week
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
        "I've been playing classical piano for 20 years. Currently working on Rachmaninoff's Piano Concerto No. 2.",
        "That's one of the most beautiful concertos! How long have you been working on it?",
        """
        - User plays classical piano (20 years experience)
        - User is currently learning Rachmaninoff's Piano Concerto No. 2
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
        "My wife Priya and I just adopted a golden retriever puppy named Buddy from the shelter.",
        "That's wonderful! Golden retrievers are amazing family dogs. How old is Buddy?",
        """
        - User is married, wife's name is Priya
        - User adopted a golden retriever puppy named Buddy from shelter
        """,
    ),
    (
        "My mom was just diagnosed with early-stage Alzheimer's. We're moving her into assisted living next month.",
        "I'm so sorry to hear that. This must be a difficult transition for your family.",
        """
        - User's mother was recently diagnosed with early-stage Alzheimer's
        - User is helping mother move to assisted living next month
        """,
    ),
    (
        "I'm planning a 3-month sabbatical to Southeast Asia starting in March - Thailand, Vietnam, and Indonesia.",
        "That sounds amazing! Have you been to that region before?",
        """
        - User is planning a 3-month sabbatical to Southeast Asia
        - Trip starts in March and includes Thailand, Vietnam, and Indonesia
        """,
    ),
    (
        "I grew up in Mumbai but moved to the Bay Area for college (Stanford '15) and never left.",
        "Stanford's a great school! What did you study?",
        """
        - User grew up in Mumbai, India
        - User attended Stanford University, graduated in 2015
        - User has lived in the Bay Area since college
        """,
    ),
    (
        "I got into Harvard Law and Yale Law. Yale gave me a full ride but Harvard was my dream school. Leaning toward Yale for financial reasons.",
        "Both are excellent choices! The financial aspect is definitely worth considering, and Yale's a phenomenal school.",
        """
        - User was accepted to Harvard Law School and Yale Law School
        - User received full scholarship from Yale Law
        - User is leaning toward Yale due to financial considerations
        - Agent validated that financial considerations are important
        """,
    ),
    (
        "We're buying our first house in Austin! 3 bed, 2 bath in the Mueller neighborhood. Closes in 6 weeks.",
        "Congratulations! Mueller is a great neighborhood with good walkability.",
        """
        - User is buying first house in Austin, Texas
        - House is 3 bedroom, 2 bathroom in Mueller neighborhood
        - House closes in 6 weeks
        """,
    ),
    (
        "My startup just raised a $2M seed round led by Y Combinator. We're building AI tools for legal document review.",
        "Congrats on the raise! Legal AI is a hot space right now.",
        """
        - User is founder of a startup building AI tools for legal document review
        - User's startup raised $2M seed round
        - Y Combinator led the seed round
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
    (
        "If I were to learn a new language, should I pick Spanish or Mandarin?",
        "Both are great choices! Spanish is more widely spoken in the Americas, while Mandarin opens doors in Asia.",
        "",
    ),
    (
        "What should I get my girlfriend for her birthday?",
        "It depends on her interests! What does she enjoy? Does she have any hobbies?",
        "",
    ),
    (
        "I'm having trouble deploying my Docker container to AWS ECS. It keeps timing out during health checks.",
        "Health check timeouts usually mean your app isn't responding on the expected port or path. Check: 1) your container is listening on the port specified in the task definition, 2) the health check path exists and returns 200, 3) increase the health check grace period if your app has slow startup.",
        """
        - User is deploying Docker container to AWS ECS
        - User experiencing health check timeout issues
        - Agent identified potential causes: app not responding on expected port/path, health check path issues, slow startup
        - Solutions provided: verify container port matches task definition, ensure health check path returns 200, increase grace period for slow startup
        """,
    ),
]