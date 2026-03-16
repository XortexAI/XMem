from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union



@dataclass
class ProfileTopic:
    topic: str
    sub_topics: List[Union[str, Dict[str, str]]] = field(default_factory=list)
    description: str = ""


PROFILE_TOPICS: List[ProfileTopic] = [
    ProfileTopic(
        "basic_info",
        sub_topics=[
            "Name",
            {"name": "Age", "description": "integer"},
            "Gender",
            "birth_date",
            "nationality",
            "ethnicity",
            "language_spoken",
        ],
    ),
    ProfileTopic(
        "contact_info",
        sub_topics=["email", "phone", "city", "country"],
    ),
    ProfileTopic(
        "education",
        sub_topics=["school", "degree", "major"],
    ),
    ProfileTopic(
        "demographics",
        sub_topics=["marital_status", "spouse_name", "number_of_children", "household_income", "relationship"],
    ),
    ProfileTopic(
        "work",
        sub_topics=["company", "title", "working_industry", "previous_projects", "work_skills"],
    ),
    ProfileTopic(
        "interest",
        sub_topics=["books", "movies", "music", "foods", "sports", "hobbies", "art", "travel", "games"],
    ),
    ProfileTopic(
        "life_event",
        sub_topics=["marriage", "relocation", "retirement", "health", "achievement"],
    ),
]


def format_topics_for_prompt(topics: List[ProfileTopic] | None = None) -> str:
    topics = topics or PROFILE_TOPICS
    lines: list[str] = []
    for t in topics:
        desc = f" ({t.description})" if t.description else ""
        lines.append(f"- {t.topic}{desc}")
        for st in t.sub_topics:
            if isinstance(st, dict):
                st_desc = f"({st.get('description', '')})" if st.get("description") else ""
                lines.append(f"  - {st['name']}{st_desc}")
            else:
                lines.append(f"  - {st}")
    lines.append("...")
    return "\n".join(lines)
