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
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="DeepSeek API key"
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="Base URL for the DeepSeek OpenAI-compatible API"
    )
    deepseek_model: str = Field(
        default="deepseek-v4-flash",
        description="DeepSeek model name"
    )
    deepseek_vision_model: str = Field(
        default="deepseek-v4-flash",
        description="DeepSeek vision model name"
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key"
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model name"
    )
    groq_vision_model: str = Field(
        default="llama-3.2-11b-vision-preview",
        description="Groq vision-capable model name"
    )
    mimo_api_key: Optional[str] = Field(
        default=None,
        description="Xiaomi MiMo API key"
    )
    mimo_base_url: str = Field(
        default="https://api.xiaomimimo.com/v1",
        description="Base URL for the Xiaomi MiMo OpenAI-compatible API"
    )
    mimo_model: str = Field(
        default="mimo-v2.5-pro",
        description="Xiaomi MiMo model name"
    )
    mimo_vision_model: str = Field(
        default="mimo-v2.5",
        description="Xiaomi MiMo vision-capable model name"
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

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama models"
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama chat model name"
    )
    ollama_vision_model: str = Field(
        default="llava:latest",
        description="Ollama vision-capable model name"
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
    summary_judge_similarity_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Threshold score for the Judge to match summary memories"
    )
    temporal_search_similarity_threshold: float = Field(
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity threshold score for Neo4j temporal search"
    )
    llm_timeout_seconds: float = Field(
        default=45.0,
        description="Per-agent LLM call timeout in seconds",
    )
    memory_ingest_timeout_seconds: float = Field(
        default=120.0,
        description="Overall memory ingest timeout in seconds",
    )
    temporal_address: str = Field(
        default="localhost:7233",
        description="Temporal frontend address for durable v2 workflows",
    )
    temporal_namespace: str = Field(
        default="default",
        description="Temporal namespace for durable v2 workflows",
    )
    temporal_task_queue: str = Field(
        default="xmem-v2",
        description="Temporal task queue used by the XMem v2 worker",
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

    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key (required when VECTOR_STORE_PROVIDER=pinecone)"
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

    vector_store_provider: str = Field(
        default="pinecone",
        description="Vector store backend: pinecone, pgvector, chroma, or sqlite",
    )
    pgvector_url: str = Field(
        default="postgresql://xmem:xmem@localhost:5432/xmem",
        description="Postgres/pgvector connection URL",
    )
    pgvector_table: str = Field(
        default="xmem_vectors",
        description="Postgres table used by the pgvector backend",
    )
    app_store_provider: str = Field(
        default="mongo",
        description="App metadata store: mongo, postgres, or memory",
    )
    app_postgres_url: Optional[str] = Field(
        default=None,
        description="Postgres URL for app metadata. Defaults to PGVECTOR_URL.",
    )
    chroma_persist_dir: str = Field(
        default=".xmem/chroma",
        description="Local Chroma persistence directory",
    )
    sqlite_vector_path: str = Field(
        default=".xmem/xmem_vectors.db",
        description="Local SQLite vector store path",
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
    embedding_provider: str = Field(
        default="auto",
        description="Embedding provider: auto, gemini, openai, bedrock, ollama, or fastembed",
    )
    ollama_embedding_model: Optional[str] = Field(
        default=None,
        description="Ollama embedding model override, e.g. nomic-embed-text",
    )
    fastembed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model for local embeddings",
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
    neo4j_database: Optional[str] = Field(
        default=None,
        description="Neo4j database name for Query API transport (NEO4J_DATABASE)",
    )
    neo4j_transport: str = Field(
        default="auto",
        description="Neo4j transport: auto, bolt, or query_api (NEO4J_TRANSPORT)",
    )
    neo4j_ssl_verify: bool = Field(
        default=True,
        description="Verify TLS for Neo4j Query API HTTPS requests (NEO4J_SSL_VERIFY)",
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
        default="development",
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
    # Outreach (GitHub email scraper + Gmail SMTP sender)
    # =============================================================================
    gmail_app_password: Optional[str] = Field(
        default=None,
        description="Gmail App Password for sending outreach emails from xmemlabs@gmail.com"
    )
    gmail_sender_email: str = Field(
        default="xmemlabs@gmail.com",
        description="Gmail sender address for outreach"
    )
    xmem_server_url: str = Field(
        default="https://api.xmem.in",
        description="Public-facing server URL for tracking pixel/link redirects"
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
    razorpay_key_id: Optional[str] = Field(
        default=None,
        description="Razorpay public key ID used for checkout order creation"
    )
    razorpay_key_secret: Optional[str] = Field(
        default=None,
        description="Razorpay key secret used only on the API server"
    )
    razorpay_webhook_secret: Optional[str] = Field(
        default=None,
        description="Optional Razorpay webhook signing secret"
    )
    razorpay_pro_plan_id: Optional[str] = Field(
        default=None,
        description="Razorpay subscription plan ID for the Pro plan",
    )

    @field_validator("fallback_order")
    @classmethod
    def validate_fallback_order(cls, v: List[str]) -> List[str]:
        valid_providers = {"gemini", "claude", "openai", "deepseek", "groq", "mimo", "openrouter", "bedrock", "ollama"}
        for provider in v:
            if provider not in valid_providers:
                raise ValueError(
                    f"Invalid provider '{provider}' in fallback_order. "
                    f"Must be one of: {valid_providers}"
                )
        return v
     
    def model_post_init(self, __context) -> None:
        """Validate that at least one LLM API key is provided."""
        if "ollama" in [p.lower() for p in self.fallback_order]:
            return
        if not any([
            self.gemini_api_key,
            self.claude_api_key,
            self.openai_api_key,
            self.deepseek_api_key,
            self.mimo_api_key,
            self.openrouter_api_key,
            self.aws_access_key_id,
        ]):
            raise ValueError(
                "At least one LLM API key must be provided "
                "(GEMINI_API_KEY, CLAUDE_API_KEY, OPENAI_API_KEY, "
                "DEEPSEEK_API_KEY, MIMO_API_KEY, OPENROUTER_API_KEY, "
                "AWS_ACCESS_KEY_ID, or include ollama in FALLBACK_ORDER)"
            )
