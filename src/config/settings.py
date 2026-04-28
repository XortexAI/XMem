from typing import Optional, List
from pydantic import Field,field_validator
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration for Xmem using pydantic-settings.
    All settings are loaded from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key"
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name"
    )
    gemini_vision_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Gemini vision model name (must support image input)"
    )
    
    claude_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic Claude API key"
    )
    claude_model: str = Field(
        default="claude-3-5-sonnet",
        description="Claude model name"
    )
    claude_vision_model: str = Field(
        default="claude-3-5-sonnet",
        description="Claude vision model name (must support image input)"
    )
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4.1-mini",
        description="OpenAI model name"
    )
    openai_vision_model: str = Field(
        default="gpt-4.1-mini",
        description="OpenAI vision model name (must support image input)"
    )

    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key"
    )
    openrouter_model: str = Field(
        default="google/gemini-2.5-flash",
        description="OpenRouter model name (e.g. google/gemini-2.5-flash, anthropic/claude-3.5-sonnet)"
    )
    openrouter_vision_model: str = Field(
        default="google/gemini-2.5-flash",
        description="OpenRouter vision model name"
    )

    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID for Bedrock"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key for Bedrock"
    )
    bedrock_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock"
    )
    bedrock_model: str = Field(
        default="us.amazon.nova-lite-v1:0",
        description="Bedrock model ID (e.g. us.amazon.nova-lite-v1:0, us.amazon.nova-pro-v1:0)"
    )
    bedrock_vision_model: str = Field(
        default="us.amazon.nova-lite-v1:0",
        description="Bedrock vision model ID (Nova Lite supports image input)"
    )

    temperature: float = Field(
        default=0.4,
        description="LLM temperature for generation"
    )
    fallback_order: List[str] = Field(
        default=["openrouter", "gemini", "claude", "openai"],
        description="Order of LLM providers to try on failure"
    )

    classifier_model: Optional[str] = Field(default=None, description="Model for classifier agent")
    profiler_model: Optional[str] = Field(default=None, description="Model for profiler agent")
    temporal_model: Optional[str] = Field(default=None, description="Model for temporal agent")
    summarizer_model: Optional[str] = Field(default=None, description="Model for summarizer agent")
    judge_model: Optional[str] = Field(default=None, description="Model for judge agent")
    retrieval_model: Optional[str] = Field(default=None, description="Model for retrieval agent")
    code_model: Optional[str] = Field(default=None, description="Model for code annotation agent")

    pinecone_api_key: str = Field(
        ...,
        description="Pinecone API key (required)"
    )
    pinecone_index_name: str = Field(
        default="xmem-index",
        description="Pinecone index name"
    )
    pinecone_namespace: str = Field(
        default="default",
        description="Pinecone namespace for organizing vectors"
    )
    pinecone_dimension: int = Field(
        default=768,
        description="Pinecone dimension for embeddings"
    )
    pinecone_metric: str = Field(
        default="cosine",
        description="Pinecone metric for embeddings"
    )
    pinecone_cloud: str = Field(
        default="aws",
        description="Pinecone cloud for embeddings"
    )
    pinecone_region: str = Field(
        default="us-east-1",
        description="Pinecone region for embeddings"
    )

    ssl_ca_bundle: Optional[str] = Field(
        default=None,
        description=(
            "Path to PEM CA bundle for TLS (SSL_CA_BUNDLE). Required behind corporate "
            "SSL inspection; Pinecone SDK ignores SSL_CERT_FILE unless this is set."
        ),
    )

    pinecone_ssl_verify: bool = Field(
        default=True,
        description=(
            "Verify TLS for Pinecone API (PINECONE_SSL_VERIFY). Set false only to debug "
            "proxy or cert issues — never in production."
        ),
    )

    embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Embedding model name (e.g. gemini-embedding-001, amazon.nova-2-multimodal-embeddings-v1:0)"
    )

    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    mongodb_database: str = Field(
        default="xmem",
        description="MongoDB database name"
    )
    
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI"
    )
    neo4j_username: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        ...,
        description="Neo4j password (required)"
    )
    neo4j_connection_timeout: float = Field(
        default=60.0,
        description="Neo4j driver TCP+handshake timeout in seconds (NEO4J_CONNECTION_TIMEOUT)",
    )

    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        description="API server port"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "https://api.xmem.in"],
        description="Allowed CORS origins"
    )
    rate_limit: int = Field(
        default=60,
        description="Rate limit (requests per minute)"
    )

    # Scanner dashboard — heuristic pre-scan estimates (tunable)
    scanner_estimate_phase1_base_seconds: float = Field(
        default=45.0,
        description="Base seconds added to Phase 1 time estimate",
    )
    scanner_estimate_phase1_seconds_per_mb: float = Field(
        default=8.0,
        description="Additional Phase 1 seconds per MB of GitHub repo size",
    )
    scanner_estimate_embedding_calls_per_mb: float = Field(
        default=15.0,
        description="Rough embedding API calls per MB for Phase 1",
    )
    scanner_estimate_avg_tokens_per_embedding: int = Field(
        default=256,
        description="Assumed average tokens billed per embedding request",
    )
    scanner_estimate_llm_tokens_per_mb: float = Field(
        default=12000.0,
        description="Rough upper-bound LLM tokens for Phase 2 per MB of repo",
    )
    scanner_estimate_embedding_cost_per_1m_tokens: Optional[float] = Field(
        default=None,
        description="USD per 1M embedding tokens (optional; enables cost estimate)",
    )
    scanner_estimate_llm_cost_per_1m_tokens: Optional[float] = Field(
        default=None,
        description="USD per 1M LLM tokens for Phase 2 (optional)",
    )
    max_request_body_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum request body size in bytes (default 10MB)"
    )
    api_keys: List[str] = Field(
        default=[],
        description="List of valid API keys for authentication (empty = no auth required)"
    )

    opik_api_key: Optional[str] = Field(
        default=None,
        description="Opik API key for observability"
    )
    opik_workspace: Optional[str] = Field(
        default="xmem",
        description="Opik workspace name"
    )
    opik_project: Optional[str] = Field(
        default="xmem-production",
        description="Opik project name"
    )

    # =============================================================================
    # Monitoring & Observability
    # =============================================================================
    environment: str = Field(
        default="production",
        description="Deployment environment: dev, staging, production"
    )
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking (leave empty to disable)"
    )
    sentry_traces_sample_rate: float = Field(
        default=1.0,
        description="Sentry performance traces sample rate (0.0 to 1.0)"
    )
    sentry_profile_sample_rate: float = Field(
        default=1.0,
        description="Sentry profiling sample rate (0.0 to 1.0)"
    )
    sentry_server_name: Optional[str] = Field(
        default=None,
        description="Sentry server name for multi-instance identification"
    )
    enable_prometheus: bool = Field(
        default=True,
        description="Enable Prometheus /metrics endpoint"
    )
    enable_analytics: bool = Field(
        default=True,
        description="Enable fire-and-forget analytics collection to MongoDB"
    )

    # =============================================================================
    # GitHub Integration (for traffic analytics on admin dashboard)
    # =============================================================================
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub personal access token for traffic API"
    )
    github_repo_owner: str = Field(
        default="XortexAI",
        description="GitHub repo owner/org"
    )
    github_repo_name: str = Field(
        default="XMem",
        description="GitHub repo name"
    )

    # =============================================================================
    # Authentication Configuration (Google OAuth + JWT)
    # =============================================================================
    google_client_id: Optional[str] = Field(
        default=None,
        description="Google OAuth client ID from Google Cloud Console"
    )
    google_client_secret: Optional[str] = Field(
        default=None,
        description="Google OAuth client secret from Google Cloud Console"
    )
    google_redirect_uri: str = Field(
        default="http://localhost:8000/auth/callback",
        description="Google OAuth redirect URI"
    )
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT token signing (change in production!)"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    jwt_expiration_days: int = Field(
        default=7,
        description="JWT token expiration time in days"
    )
    frontend_url: str = Field(
        default="http://localhost:5173",
        description="Frontend URL for redirects after auth"
    )

    @field_validator("fallback_order")
    @classmethod
    def validate_fallback_order(cls, v: List[str]) -> List[str]:
        valid_providers = {"gemini", "claude", "openai", "openrouter", "bedrock"}
        for provider in v:
            if provider not in valid_providers:
                raise ValueError(
                    f"Invalid provider '{provider}' in fallback_order. "
                    f"Must be one of: {valid_providers}"
                )
        return v
     
    def model_post_init(self, __context) -> None:
        """Validate that at least one LLM API key is provided."""
        if not any([self.gemini_api_key, self.claude_api_key, self.openai_api_key, self.openrouter_api_key, self.aws_access_key_id]):
            raise ValueError(
                "At least one LLM API key must be provided "
                "(GEMINI_API_KEY, CLAUDE_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, or AWS_ACCESS_KEY_ID)"
            )