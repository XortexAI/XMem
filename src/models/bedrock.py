"""
Amazon Bedrock model factory (for Nova and other Bedrock-hosted models).

Uses ChatBedrockConverse from langchain-aws, which speaks the Bedrock
Converse API and supports tool-calling, streaming, etc.

AWS credentials are resolved in this order:
  1. Explicit AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
  2. ~/.aws/credentials profile
  3. IAM role (EC2/Lambda/ECS)
"""

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel

from src.config import settings


def build_bedrock_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    kwargs: dict = dict(
        model=model_name or settings.bedrock_model,
        region_name=settings.bedrock_region,
        temperature=temperature if temperature is not None else settings.temperature,
    )

    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["credentials_profile_name"] = None
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

    return ChatBedrockConverse(**kwargs)
