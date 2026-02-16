from typing import List, Tuple

from src.config.constants import LLM_TAB_SEPARATOR

S = LLM_TAB_SEPARATOR

PROFILE_EXAMPLES: List[Tuple[str, str]] = [
    (
        "Hi, how are you?",
        "NONE",
    ),
    (
        "I work as a Senior Software Engineer at Google and have 5 years of experience in Python and TensorFlow",
        f"""\
        work{S}company{S}Google
        work{S}title{S}Senior Software Engineer
        work{S}work_skills{S}Python (5 years), TensorFlow (5 years)
        """,
    ),
    (
        "My name is Sarah Chen, I'm 28 years old and I live in San Francisco. I speak English and Mandarin fluently.",
        f"""\
        basic_info{S}name{S}Sarah Chen
        basic_info{S}age{S}28
        contact_info{S}city{S}San Francisco
        contact_info{S}country{S}United States (California)
        basic_info{S}language_spoken{S}English, Mandarin (fluent)
        """,
    ),
    (
        "I love reading science fiction novels especially Isaac Asimov. I play tennis on weekends and I'm a vegetarian who loves Italian food.",
        f"""\
        interest{S}books{S}science fiction novels, especially Isaac Asimov
        interest{S}sports{S}tennis (plays on weekends)
        interest{S}foods{S}vegetarian, loves Italian cuisine
        """,
    ),
    (
        "I graduated from MIT with a CS degree in 2019 and I'm currently doing my Master's in Machine Learning at Stanford",
        f"""\
        education{S}school{S}MIT (graduated 2019), Stanford (current, Master's)
        education{S}degree{S}Bachelor's in Computer Science (MIT, 2019), Master's in Machine Learning (Stanford, in progress)
        education{S}major{S}Computer Science (undergraduate), Machine Learning (graduate)
        """,
    ),
    (
        "I'm married with two children. We recently moved from New York to Austin, Texas. I enjoy painting and pottery.",
        f"""\
        demographics{S}marital_status{S}married
        demographics{S}number_of_children{S}2
        life_event{S}relocation{S}moved from New York to Austin, Texas
        contact_info{S}city{S}Austin
        contact_info{S}country{S}United States (Texas)
        interest{S}hobbies{S}painting, pottery
        """,
    ),
    (
        "My husband's name is Peter and we have a dog called Max",
        f"""\
        demographics{S}marital_status{S}married
        demographics{S}spouse_name{S}Peter
        interest{S}hobbies{S}has a dog named Max
        """,
    ),
    (
        "Me and my college friends have been close for 4 years now",
        f"""\
        demographics{S}relationship{S}close college friends for 4 years
        """,
    ),
    (
        "My wife and I are expecting our first child next month",
        f"""\
        demographics{S}marital_status{S}married
        demographics{S}number_of_children{S} 1 (expecting first child)
        """,
    ),
    (
        "I just switched from my Android to iPhone because I prefer the ecosystem",
        f"""\
        interest{S}hobbies{S}prefers Apple/iPhone ecosystem over Android
        """,
    ),
]
