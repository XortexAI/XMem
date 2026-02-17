"""
Image Agent — analyses images and extracts structured observations for memory.

Takes a user query (optionally with an image URL/base64) routed from the
ClassifierAgent and returns an ImageResult containing a description and
a list of structured observations.

NOTE: This agent requires a **vision-capable** model (e.g. gemini-2.5-flash,
gpt-4o, claude-3-5-sonnet).  Use ``get_vision_model()`` from ``src.models``
instead of the regular ``get_model()`` when constructing this agent.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.prompts.image import build_system_prompt, pack_image_query
from src.schemas.image import ImageAnalysis, ImageResult
from src.utils.text import parse_raw_response_to_image


class ImageAgent(BaseAgent):
    """Agent that analyses images using a vision-capable LLM.

    Unlike other agents that only send text, this agent builds **multimodal
    messages** containing both text and image content blocks so that the
    vision model can actually "see" the image.
    """

    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(
            model=model,
            name="image",
            system_prompt=build_system_prompt(),
        )

    # ------------------------------------------------------------------
    # Multimodal message builder (overrides the text-only base helper)
    # ------------------------------------------------------------------

    def _build_vision_messages(
        self,
        user_text: str,
        image_url: str = "",
    ) -> List[Any]:
        """Build a message list with multimodal content for vision models.

        LangChain vision format uses a list of content blocks inside the
        HumanMessage::

            HumanMessage(content=[
                {"type": "text",      "text": "Analyse this image."},
                {"type": "image_url", "image_url": {"url": "https://..."}},
            ])

        For base64-encoded images the URL should be a data-URI::

            "data:image/png;base64,iVBORw0KGgo..."

        Args:
            user_text: The text portion of the prompt.
            image_url: URL or base64 data-URI of the image.

        Returns:
            List of LangChain message objects ready for ``model.ainvoke()``.
        """
        messages: List[Any] = []

        # System prompt
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        # Build multimodal content blocks for the human message
        content_blocks: List[Dict[str, Any]] = []

        # Always include the text instruction
        content_blocks.append({"type": "text", "text": user_text})

        # Include image if provided
        if image_url:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        messages.append(HumanMessage(content=content_blocks))

        return messages

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def arun(self, state: Dict[str, Any]) -> ImageResult:
        """Analyse an image and extract structured observations.

        Expected state keys:
            classifier_output (str): Text query describing what to look for.
            image_url (str): URL or base64 data-URI of the image to analyse.

        Returns:
            ImageResult with description and list of observations.
        """
        query = state.get("classifier_output", "")
        image_url = state.get("image_url", "")

        if not query and not image_url:
            self.logger.debug("Empty query and no image — returning empty result.")
            return ImageResult()

        # Pack the text portion of the prompt
        user_text = pack_image_query(query, image_url="")

        # Build multimodal messages (text + image) for the vision model
        if image_url:
            messages = self._build_vision_messages(user_text, image_url=image_url)
        else:
            # No image provided — fall back to text-only messages
            messages = self._build_messages(user_text)

        # Call the vision model
        raw_content = await self._call_model(messages)

        # Parse the structured response
        parsed = parse_raw_response_to_image(raw_content)

        observations = [
            ImageAnalysis(
                category=obs["category"],
                description=obs["description"],
                confidence=obs.get("confidence"),
            )
            for obs in parsed.get("observations", [])
        ]

        result = ImageResult(
            description=parsed.get("description", ""),
            observations=observations,
        )

        # Log results
        if not result.is_empty:
            self.logger.info("=" * 50)
            self.logger.info("Image Analysis Result:")
            self.logger.info("  Description: %s", result.description)
            for idx, obs in enumerate(result.observations, 1):
                self.logger.info(
                    "  %d. [%s] %s (confidence: %s)",
                    idx,
                    obs.category,
                    obs.description,
                    obs.confidence or "N/A",
                )
            self.logger.info("Total observations: %d", len(result.observations))
            self.logger.info("=" * 50)
        else:
            self.logger.info("No observations extracted from image.")

        return result
