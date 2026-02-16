import functools
import sys

# Read the file
with open('src/models/registry.py', 'r') as f:
    content = f.read()

# Replace 1: Import functools
search1 = """import importlib
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel"""

replace1 = """import functools
import importlib
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel"""

if search1 in content:
    content = content.replace(search1, replace1)
else:
    print("Search 1 not found")

# Replace 2: Add decorator
search2 = """def get_model(
    provider: Optional[Provider] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:"""

replace2 = """@functools.lru_cache(maxsize=32)
def get_model(
    provider: Optional[Provider] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:"""

if search2 in content:
    content = content.replace(search2, replace2)
else:
    print("Search 2 not found")

# Write back
with open('src/models/registry.py', 'w') as f:
    f.write(content)
