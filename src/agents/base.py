import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel

@dataclass
class BaseAgent(ABC):
    model: BaseChatModel
    name: str
    system_prompt: str = ""
    #Use field(init=False) so 'logger' is NOT required as an argument in __init__
    logger: logging.Logger = field(init=False)
    #Use __post_init__ to set up variables after __init__ is done
    def __post_init__(self):
        self.logger = logging.getLogger(f"xmem.agents.{self.name}")

    @abstractmethod
    async def arun(self, state: Dict[str, Any]) -> Any:
        ...

    def run(self, state: Dict[str, Any]) -> Any:
        import asyncio
        return asyncio.run(self.arun(state))

    def _build_messages(self, user_message: str) -> list:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _call_model(self, messages: list) -> str:
        response = await self.model.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            # Gemini thinking models may return list of dicts like
            # [{'type': 'text', 'text': '...', 'extras': {...}}]
            parts = []
            for c in content:
                if isinstance(c, dict) and "text" in c:
                    parts.append(c["text"])
                elif isinstance(c, str):
                    parts.append(c)
                else:
                    parts.append(str(c))
            content = "\n".join(parts)
        return content
