from src.schemas.classification import Classification
from typing import List, Tuple

CLASSIFICATION_EXAMPLES: List[Tuple[str, List[Classification]]] = [
    (
        "Thank you so much!",
        []
    ),
    (
        "Hi, how are you?",
        []
    ),
    (
        "Great, thanks!",
        []
    ),
    (
        "Debug this error: TypeError: 'int' object is not iterable",
        [{"source": "code", "query": "Debug this error: TypeError: 'int' object is not iterable"}]
    ),
    (
        "Explain how the asyncio event loop works in Python",
        [{"source": "code", "query": "Explain how the asyncio event loop works in Python"}]
    ),
    (
        "Help me write a function to reverse a linked list",
        [{"source": "code", "query": "Help me write a function to reverse a linked list"}]
    ),
    (
        "I prefer dark mode in all my applications",
        [{"source": "profile", "query": "I prefer dark mode in all my applications"}]
    ),
    (
        "My name is Alice and I work at Google",
        [{"source": "profile", "query": "My name is Alice and I work at Google"}]
    ),
    (
        "I'm a vegetarian and love Italian food",
        [{"source": "profile", "query": "I'm a vegetarian and love Italian food"}]
    ),
    (
        "My birthday is on March 15th",
        [{"source": "event", "query": "My birthday is on March 15th"}]
    ),
    (
        "Our wedding anniversary is July 22nd, 2019",
        [{"source": "event", "query": "Our wedding anniversary is July 22nd, 2019"}]
    ),
    (
        "I have a dentist appointment on January 10th at 2:30 PM",
        [{"source": "event", "query": "I have a dentist appointment on January 10th at 2:30 PM"}]
    ),
    (
        "My daughter's birthday is December 25th, she was born in 2015",
        [{"source": "event", "query": "My daughter's birthday is December 25th, she was born in 2015"}]
    ),
    (
        "My name is Alice and I want to write a python script to hello world",
        [
            {"source": "profile", "query": "My name is Alice"},
            {"source": "code", "query": "I want to write a python script to hello world"}
        ]
    ),
    (
        "I'm learning Rust. How do I print variables in Rust?",
        [
            {"source": "profile", "query": "I'm learning Rust"},
            {"source": "code", "query": "how do I print variables in Rust?"}
        ]
    ),
    (
        "My name is John and my birthday is April 5th",
        [
            {"source": "profile", "query": "My name is John"},
            {"source": "event", "query": "my birthday is April 5th"}
        ]
    ),
    (
        "I graduated on May 20th 2020 and now I work as a software engineer",
        [
            {"source": "event", "query": "I graduated on May 20th 2020"},
            {"source": "profile", "query": "I work as a software engineer"}
        ]
    ),
    (
        "I prefer writing code in TypeScript over JavaScript",
        [{"source": "profile", "query": "I prefer writing code in TypeScript over JavaScript"}]
    ),
    (
        "Mom's birthday is February 14th",
        [{"source": "event", "query": "Mom's birthday is February 14th"}]
    ),
    (
        "I ran a charity race last Saturday",
        [{"source": "event", "query": "I ran a charity race last Saturday"}]
    ),
    (
        "I moved from Sweden 4 years ago",
        [{"source": "event", "query": "I moved from Sweden 4 years ago"}]
    ),
    (
        "I started transitioning about 3 years ago",
        [{"source": "event", "query": "I started transitioning about 3 years ago"}]
    ),
    (
        "I graduated from college in May 2018",
        [{"source": "event", "query": "I graduated from college in May 2018"}]
    ),
    (
        "My 18th birthday was ten years ago when my friend gave me a bowl",
        [{"source": "event", "query": "My 18th birthday was ten years ago when my friend gave me a bowl"}]
    ),
    (
        "I went through a tough breakup last month and now I'm focusing on myself",
        [
            {"source": "event", "query": "I went through a tough breakup last month"},
            {"source": "profile", "query": "I'm focusing on myself"}
        ]
    ),
    (
        "How do I set up a Docker container for my Node.js app?",
        [{"source": "code", "query": "How do I set up a Docker container for my Node.js app?"}]
    ),
    (
        "Debug this kubernetes pod crash",
        [{"source": "code", "query": "Debug this kubernetes pod crash"}]
    ),
    (
        "Write unit tests for this function using pytest",
        [{"source": "code", "query": "Write unit tests for this function using pytest"}]
    ),
    (
        "How do I mock API calls in Jest?",
        [{"source": "code", "query": "How do I mock API calls in Jest?"}]
    ),
    (
        "Explain how to make a REST API call with authentication",
        [{"source": "code", "query": "Explain how to make a REST API call with authentication"}]
    ),
    (
        "Convert this JSON response to a Python dictionary",
        [{"source": "code", "query": "Convert this JSON response to a Python dictionary"}]
    ),
    (
        "I usually wake up at 6 AM every day",
        [{"source": "profile", "query": "I usually wake up at 6 AM every day"}]
    ),
    (
        "I never drink coffee after 3 PM",
        [{"source": "profile", "query": "I never drink coffee after 3 PM"}]
    ),
    (
        "I believe in work-life balance",
        [{"source": "profile", "query": "I believe in work-life balance"}]
    ),
    (
        "Privacy is very important to me",
        [{"source": "profile", "query": "Privacy is very important to me"}]
    ),
    (
        "My daughter Sarah is 8 years old",
        [{"source": "profile", "query": "My daughter Sarah is 8 years old"}]
    ),
    (
        "My best friend lives in Seattle",
        [{"source": "profile", "query": "My best friend lives in Seattle"}]
    ),
    (
        "I'm from Tokyo but now I live in San Francisco",
        [{"source": "profile", "query": "I'm from Tokyo but now I live in San Francisco"}]
    ),
    (
        "My email is john@example.com",
        [{"source": "profile", "query": "My email is john@example.com"}]
    ),
    (
        "Schedule a meeting with the team next Tuesday at 10 AM",
        [{"source": "event", "query": "Schedule a meeting with the team next Tuesday at 10 AM"}]
    ),
    (
        "Remind me to call mom tomorrow evening",
        [{"source": "event", "query": "Remind me to call mom tomorrow evening"}]
    ),
    (
        "I have a dentist appointment this Friday at 2:30 PM",
        [{"source": "event", "query": "I have a dentist appointment this Friday at 2:30 PM"}]
    ),
    (
        "I visited Paris in August 2022",
        [{"source": "event", "query": "I visited Paris in August 2022"}]
    ),
    (
        "We got married 5 years ago",
        [{"source": "event", "query": "We got married 5 years ago"}]
    ),
    (
        "I finished my master's degree back in 2019",
        [{"source": "event", "query": "I finished my master's degree back in 2019"}]
    ),
    (
        "Started learning guitar 6 months ago",
        [{"source": "event", "query": "Started learning guitar 6 months ago"}]
    ),
    (
        "I got my first car when I turned 18",
        [{"source": "event", "query": "I got my first car when I turned 18"}]
    ),
    (
        "I was diagnosed with diabetes at age 25",
        [{"source": "event", "query": "I was diagnosed with diabetes at age 25"}]
    ),
    (
        "I joined Google in January 2020",
        [{"source": "event", "query": "I joined Google in January 2020"}]
    ),
    (
        "We adopted our dog Rex last summer",
        [{"source": "event", "query": "We adopted our dog Rex last summer"}]
    ),
    (
        "Launched my startup in March 2023",
        [{"source": "event", "query": "Launched my startup in March 2023"}]
    ),
    (
        "I'm a DevOps engineer and I usually work with Kubernetes. Can you help me debug this pod error?",
        [
            {"source": "profile", "query": "I'm a DevOps engineer and I usually work with Kubernetes"},
            {"source": "code", "query": "Can you help me debug this pod error?"}
        ]
    ),
    (
        "I got engaged last Christmas and my fiancé loves hiking",
        [
            {"source": "event", "query": "I got engaged last Christmas"},
            {"source": "profile", "query": "my fiancé loves hiking"}
        ]
    ),
    (
        "I prefer using VS Code for development. How do I set up Python debugging in it?",
        [
            {"source": "profile", "query": "I prefer using VS Code for development"},
            {"source": "code", "query": "How do I set up Python debugging in it?"}
        ]
    ),
    (
        "My son was born on June 15th 2020 and he loves dinosaurs",
        [
            {"source": "event", "query": "My son was born on June 15th 2020"},
            {"source": "profile", "query": "my son loves dinosaurs"}
        ]
    ),
]
