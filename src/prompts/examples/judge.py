from __future__ import annotations

from typing import List, Tuple

JUDGE_EXAMPLES: List[Tuple[str, str, str, str]] = [
    (
        "profile",
        '1. work / company = Now at Google\n2. food / preference = Loves sushi',
        'For item: "work / company = Now at Google"\n'
        '  - ID: abc123 | Score: 0.91 | "work / company = Works at Microsoft"\n'
        'For item: "food / preference = Loves sushi"\n'
        '  - (no similar records)',
        '{"operations": [{"type": "UPDATE", "content": "work / company = Now at Google", '
        '"embedding_id": "abc123", "reason": "Company changed from Microsoft to Google"}, '
        '{"type": "ADD", "content": "food / preference = Loves sushi", '
        '"embedding_id": null, "reason": "New food preference, no existing record"}], '
        '"confidence": 0.95}',
    ),
    (
        "profile",
        '1. basic_info / name = Alice',
        'For item: "basic_info / name = Alice"\n'
        '  - ID: prof-001 | Score: 0.99 | "basic_info / name = Alice"',
        '{"operations": [{"type": "NOOP", "content": "", '
        '"embedding_id": "prof-001", "reason": "Exact duplicate — name is already Alice"}], '
        '"confidence": 0.99}',
    ),
    (
        "profile",
        '1. food / diet = User is now vegetarian',
        'For item: "food / diet = User is now vegetarian"\n'
        '  - ID: prof-042 | Score: 0.87 | "food / diet = User loves steak"',
        '{"operations": [{"type": "UPDATE", "content": "food / diet = User is now vegetarian", '
        '"embedding_id": "prof-042", "reason": "Diet changed — contradicts previous steak preference"}], '
        '"confidence": 0.92}',
    ),
    (
        "profile",
        '1. basic_info / name = Bob\n2. work / role = Engineer',
        '(No similar records found — vector store is empty or search returned nothing)',
        '{"operations": [{"type": "ADD", "content": "basic_info / name = Bob", '
        '"embedding_id": null, "reason": "No existing records"}, '
        '{"type": "ADD", "content": "work / role = Engineer", '
        '"embedding_id": null, "reason": "No existing records"}], '
        '"confidence": 0.95}',
    ),
    (
        "profile",
        '1. interest / hobbies = football',
        'For item: "interest / hobbies = football"\n'
        '  - ID: prof-070 | Score: 0.88 | "interest / hobbies = reading"',
        '{"operations": [{"type": "UPDATE", "content": "interest / hobbies = reading, football", '
        '"embedding_id": "prof-070", "reason": "Hobbies is a collection — merge old (reading) '
        'with new (football) rather than overwriting"}], '
        '"confidence": 0.93}',
    ),
    (
        "profile",
        '1. interest / foods = sushi',
        'For item: "interest / foods = sushi"\n'
        '  - ID: prof-080 | Score: 0.85 | "interest / foods = pizza"',
        '{"operations": [{"type": "UPDATE", "content": "interest / foods = pizza, sushi", '
        '"embedding_id": "prof-080", "reason": "Foods is a collection — merge old (pizza) '
        'with new (sushi) rather than overwriting"}], '
        '"confidence": 0.93}',
    ),
    (
        "temporal",
        '1. 03-15 | Birthday | User\'s birthday',
        'For item: "03-15 | Birthday | User\'s birthday"\n'
        '  - ID: evt-001 | Score: 0.97 | "03-15 | Birthday | User\'s birthday"',
        '{"operations": [{"type": "NOOP", "content": "", '
        '"embedding_id": "evt-001", "reason": "Exact duplicate event already stored"}], '
        '"confidence": 0.99}',
    ),
    (
        "temporal",
        '1. 07-22 | Wedding Anniversary | 5th wedding anniversary celebration in Paris',
        'For item: "07-22 | Wedding Anniversary | 5th wedding anniversary celebration in Paris"\n'
        '  - ID: evt-010 | Score: 0.88 | "07-22 | Wedding Anniversary | User\'s wedding anniversary"',
        '{"operations": [{"type": "UPDATE", "content": "07-22 | Wedding Anniversary | '
        '5th wedding anniversary celebration in Paris", '
        '"embedding_id": "evt-010", "reason": "Same event with richer description"}], '
        '"confidence": 0.90}',
    ),
    (
        "temporal",
        '1. 01-28 | Paris Trip | Visited Paris',
        'For item: "01-28 | Paris Trip | Visited Paris"\n'
        '  - (no similar records)',
        '{"operations": [{"type": "ADD", "content": "01-28 | Paris Trip | Visited Paris", '
        '"embedding_id": null, "reason": "Brand-new event, nothing similar found"}], '
        '"confidence": 0.95}',
    ),
    (
        "temporal",
        '1. 02-10 | Dentist Appointment | Rescheduled dentist visit',
        'For item: "02-10 | Dentist Appointment | Rescheduled dentist visit"\n'
        '  - ID: evt-020 | Score: 0.90 | "01-10 | Dentist Appointment | Scheduled dentist visit"',
        '{"operations": [{"type": "DELETE", "content": "", '
        '"embedding_id": "evt-020", "reason": "Date changed from 01-10 to 02-10 — old graph connection invalid"}, '
        '{"type": "ADD", "content": "02-10 | Dentist Appointment | Rescheduled dentist visit", '
        '"embedding_id": null, "reason": "New date requires new User-Date relationship in graph"}], '
        '"confidence": 0.92}',
    ),
    (
        "summary",
        '1. User works as a software engineer\n2. User adopted a cat named Luna',
        'For item: "User works as a software engineer"\n'
        '  - ID: sum-005 | Score: 0.94 | "User is a software engineer"\n'
        'For item: "User adopted a cat named Luna"\n'
        '  - (no similar records)',
        '{"operations": [{"type": "NOOP", "content": "", '
        '"embedding_id": "sum-005", "reason": "Semantically identical fact already exists"}, '
        '{"type": "ADD", "content": "User adopted a cat named Luna", '
        '"embedding_id": null, "reason": "New fact about user\'s pet"}], '
        '"confidence": 0.93}',
    ),
    (
        "summary",
        '1. User moved from NYC to San Francisco for a new role at Google',
        'For item: "User moved from NYC to San Francisco for a new role at Google"\n'
        '  - ID: sum-012 | Score: 0.82 | "User lives in NYC"',
        '{"operations": [{"type": "UPDATE", '
        '"content": "User moved from NYC to San Francisco for a new role at Google", '
        '"embedding_id": "sum-012", "reason": "User relocated — old NYC fact is outdated"}], '
        '"confidence": 0.88}',
    ),
    (
        "summary",
        '1. User enjoys hiking on weekends\n2. User has a golden retriever named Max',
        '(No similar records found — vector store is empty or search returned nothing)',
        '{"operations": [{"type": "ADD", "content": "User enjoys hiking on weekends", '
        '"embedding_id": null, "reason": "No existing records"}, '
        '{"type": "ADD", "content": "User has a golden retriever named Max", '
        '"embedding_id": null, "reason": "No existing records"}], '
        '"confidence": 0.98}',
    ),
]
