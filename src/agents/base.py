import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel


class BaseAgent(ABC):
    def __init__(self, model: BaseChatModel, name: str, system_prompt: str = ""):
        self.model = model
        self.name = name
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"xmem.agents.{name}")

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
            content = "\n".join(str(c) for c in content)
        return content
