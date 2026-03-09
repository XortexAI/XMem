from __future__ import annotations

from typing import List, Tuple

TEMPORAL_EXAMPLES: List[Tuple[str, str, str]] = [
    # Birthday example
    (
        "My birthday is on March 15th",
        "4:04 pm on 20 January, 2023",
        "DATE: 03-15\nEVENT_NAME: Birthday\nYEAR: \nDESC: User's birthday\nTIME: \nDATE_EXPRESSION: March 15th",
    ),
    # Anniversary with year
    (
        "Our wedding anniversary is July 22nd, 2019",
        "2:32 pm on 29 January, 2023",
        "DATE: 07-22\nEVENT_NAME: Wedding Anniversary\nYEAR: 2019\nDESC: User's wedding anniversary\nTIME: \nDATE_EXPRESSION: July 22nd, 2019",
    ),
    # Appointment with time
    (
        "I have a dentist appointment on January 10th at 2:30 PM",
        "12:48 am on 1 February, 2023",
        "DATE: 01-10\nEVENT_NAME: Dentist Appointment\nYEAR: \nDESC: Scheduled dentist visit\nTIME: 2:30 PM\nDATE_EXPRESSION: January 10th",
    ),
    # Holiday/Celebration
    (
        "My daughter's birthday is on December 25th, she was born in 2015",
        "10:43 am on 4 February, 2023",
        "DATE: 12-25\nEVENT_NAME: Daughter's Birthday\nYEAR: 2015\nDESC: Daughter's birthday celebration\nTIME: \nDATE_EXPRESSION: December 25th",
    ),
    # Relative date — week before
    (
        "Maria received a medal from the homeless shelter the week before 9 August 2023",
        "5:44 pm on 21 July, 2023",
        "DATE: 08-02\nEVENT_NAME: Medal Received\nYEAR: 2023\nDESC: Maria received a medal from the homeless shelter\nTIME: \nDATE_EXPRESSION: the week before 9 August 2023",
    ),
    # Relative date — first weekend of month
    (
        "John participated in a 5K charity run on the first weekend of August 2023",
        "1:25 pm on 9 July, 2023",
        "DATE: 08-05\nEVENT_NAME: 5K Charity Run\nYEAR: 2023\nDESC: John participated in a 5K charity run\nTIME: \nDATE_EXPRESSION: first weekend of August 2023",
    ),
    # Relative date — next Friday (from context)
    (
        "Wanna see my moves next Fri? Can't wait!",
        "4:04 pm on 20 January, 2023",
        "DATE: 01-27\nEVENT_NAME: Dance Session\nYEAR: 2023\nDESC: Planned dance session to show moves\nTIME: \nDATE_EXPRESSION: next Friday",
    ),
    # Relative date — tomorrow
    (
        "The official opening night is tomorrow. I'm working hard to make everything just right.",
        "10:04 am on 19 June, 2023",
        "DATE: 06-20\nEVENT_NAME: Studio Opening Night\nYEAR: 2023\nDESC: Official opening night of the dance studio\nTIME: \nDATE_EXPRESSION: tomorrow",
    ),
    # Relative date — yesterday
    (
        "I went to a fair to show off my studio yesterday, it was both stressful and great!",
        "11:24 am on 25 April, 2023",
        "DATE: 04-24\nEVENT_NAME: Fair Exhibition\nYEAR: 2023\nDESC: Attended a fair to showcase dance studio\nTIME: \nDATE_EXPRESSION: yesterday",
    ),
    # Relative date — last week
    (
        "Started hitting the gym last week to stay on track with the venture.",
        "2:35 pm on 16 March, 2023",
        "DATE: 03-09\nEVENT_NAME: Started Gym\nYEAR: 2023\nDESC: Started going to the gym\nTIME: \nDATE_EXPRESSION: last week",
    ),
    # Relative date — next month
    (
        "I'm getting ready for a dance comp near me next month.",
        "10:43 am on 4 February, 2023",
        "DATE: 03-04\nEVENT_NAME: Dance Competition\nYEAR: 2023\nDESC: Dance competition preparation\nTIME: \nDATE_EXPRESSION: next month",
    ),
    # No event case
    (
        "I really like pizza",
        "4:04 pm on 20 January, 2023",
        "NO_EVENT",
    ),
    # Ambiguous — no specific date
    (
        "I usually go running in the mornings",
        "2:32 pm on 29 January, 2023",
        "NO_EVENT",
    ),
    # Specific event with context
    (
        "Mom's birthday is February 14th, she loves flowers",
        "12:48 am on 1 February, 2023",
        "DATE: 02-14\nEVENT_NAME: Mom's Birthday\nYEAR: \nDESC: Mother's birthday, she loves flowers\nTIME: \nDATE_EXPRESSION: February 14th",
    ),
    # Lost job event
    (
        "Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.",
        "4:04 pm on 20 January, 2023",
        "DATE: 01-19\nEVENT_NAME: Lost Job\nYEAR: 2023\nDESC: Lost job as a banker\nTIME: \nDATE_EXPRESSION: yesterday",
    ),
    # Multi-event: different event types
    (
        "I have a dentist appointment on January 10th at 2:30 PM and a concert on January 15th at 8 PM",
        "12:48 am on 1 January, 2023",
        "DATE: 01-10\nEVENT_NAME: Dentist Appointment\nYEAR: \nDESC: Scheduled dentist visit\nTIME: 2:30 PM\nDATE_EXPRESSION: January 10th\n---\nDATE: 01-15\nEVENT_NAME: Concert\nYEAR: \nDESC: Concert event\nTIME: 8 PM\nDATE_EXPRESSION: January 15th",
    ),
    (
        "My birthday is on March 15th and our wedding anniversary is on July 22nd",
        "4:04 pm on 20 January, 2023",
        "DATE: 03-15\nEVENT_NAME: Birthday\nYEAR: \nDESC: User's birthday\nTIME: \nDATE_EXPRESSION: March 15th\n---\nDATE: 07-22\nEVENT_NAME: Wedding Anniversary\nYEAR: \nDESC: User's wedding anniversary\nTIME: \nDATE_EXPRESSION: July 22nd",
    ),
]
