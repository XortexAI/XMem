package config

import (
	"bufio"
	json "github.com/goccy/go-json"
	"errors"
	"net"
	"os"
	"strconv"
	"strings"
)

type Settings struct {
	GeminiAPIKey         string
	GeminiModel          string
	GeminiVisionModel    string
	ClaudeAPIKey         string
	ClaudeModel          string
	OpenAIAPIKey         string
	OpenAIModel          string
	OpenAIEmbeddingModel string
	OpenRouterAPIKey     string
	OpenRouterModel      string
	OllamaBaseURL        string
	OllamaModel          string
	AWSAccessKeyID       string
	AWSSecretAccessKey   string
	BedrockRegion        string
	BedrockModel         string
	Temperature          float64
	FallbackOrder        []string
	ClassifierModel      string
	ProfilerModel        string
	TemporalModel        string
	SummarizerModel      string
	JudgeModel           string
	RetrievalModel       string
	CodeModel            string
	PineconeAPIKey       string
	PineconeHost         string
	PineconeIndexName    string
	PineconeNamespace    string
	PineconeDimension    int
	PineconeMetric       string
	PineconeCloud        string
	PineconeRegion       string
	VectorStoreProvider  string
	EmbeddingProvider    string
	EmbeddingModel       string
	MongoDBURI           string
	MongoDBDatabase      string
	Neo4jURI             string
	Neo4jUsername        string
	Neo4jPassword        string
	AppStoreProvider     string
	APIHost              string
	APIPort              int
	CORSOrigins          []string
	RateLimit            int
	MaxRequestBodyBytes  int64
	APIKeys              []string
	JWTSecretKey         string
	JWTAlgorithm         string
	JWTExpirationDays    int
	EnablePrometheus     bool
	EnableAnalytics      bool
	Environment          string
	ServiceName          string
}

func Load() (Settings, error) {
	_ = loadDotEnv(".env")

	s := Settings{
		GeminiModel:         "gemini-2.5-flash",
		GeminiVisionModel:   "gemini-2.5-flash-lite",
		ClaudeModel:         "claude-3-5-sonnet",
		OpenAIModel:         "gpt-4.1-mini",
		OpenRouterModel:     "google/gemini-2.5-flash",
		OllamaBaseURL:       "http://localhost:11434",
		OllamaModel:         "llama3.1:8b",
		BedrockRegion:       "us-east-1",
		BedrockModel:        "us.amazon.nova-lite-v1:0",
		Temperature:         0.4,
		FallbackOrder:       []string{"openrouter", "gemini", "claude", "openai"},
		PineconeIndexName:   "xmem-index",
		PineconeNamespace:   "xmem-go-dev",
		PineconeDimension:   768,
		PineconeMetric:      "cosine",
		PineconeCloud:       "aws",
		PineconeRegion:      "us-east-1",
		VectorStoreProvider: "pinecone",
		EmbeddingProvider:   "openai",
		EmbeddingModel:      "text-embedding-3-small",
		MongoDBURI:          "mongodb://localhost:27017",
		MongoDBDatabase:     "xmem_go",
		Neo4jURI:            "bolt://localhost:7687",
		Neo4jUsername:       "neo4j",
		AppStoreProvider:    "mongo",
		APIHost:             "0.0.0.0",
		APIPort:             8081,
		CORSOrigins:         []string{"http://localhost:3000", "http://localhost:5173"},
		RateLimit:           60,
		MaxRequestBodyBytes: 10 * 1024 * 1024,
		JWTSecretKey:        "your-secret-key-change-in-production",
		JWTAlgorithm:        "HS256",
		JWTExpirationDays:   7,
		EnablePrometheus:    false,
		EnableAnalytics:     false,
		Environment:         "development",
		ServiceName:         "xmem-go",
	}

	s.GeminiAPIKey = env("GEMINI_API_KEY", s.GeminiAPIKey)
	s.ClaudeAPIKey = env("CLAUDE_API_KEY", s.ClaudeAPIKey)
	s.OpenAIAPIKey = env("OPENAI_API_KEY", s.OpenAIAPIKey)
	s.OpenRouterAPIKey = env("OPENROUTER_API_KEY", s.OpenRouterAPIKey)
	s.AWSAccessKeyID = env("AWS_ACCESS_KEY_ID", s.AWSAccessKeyID)
	s.AWSSecretAccessKey = env("AWS_SECRET_ACCESS_KEY", s.AWSSecretAccessKey)
	s.GeminiModel = env("GEMINI_MODEL", s.GeminiModel)
	s.GeminiVisionModel = env("GEMINI_VISION_MODEL", s.GeminiVisionModel)
	s.ClaudeModel = env("CLAUDE_MODEL", s.ClaudeModel)
	s.OpenAIModel = env("OPENAI_MODEL", s.OpenAIModel)
	s.OpenAIEmbeddingModel = env("OPENAI_EMBEDDING_MODEL", s.OpenAIEmbeddingModel)
	s.OpenRouterModel = env("OPENROUTER_MODEL", s.OpenRouterModel)
	s.OllamaBaseURL = env("OLLAMA_BASE_URL", s.OllamaBaseURL)
	s.OllamaModel = env("OLLAMA_MODEL", s.OllamaModel)
	s.BedrockRegion = env("BEDROCK_REGION", s.BedrockRegion)
	s.BedrockModel = env("BEDROCK_MODEL", s.BedrockModel)
	s.Temperature = envFloat("TEMPERATURE", s.Temperature)
	s.FallbackOrder = envList("FALLBACK_ORDER", s.FallbackOrder)
	s.ClassifierModel = env("CLASSIFIER_MODEL", s.ClassifierModel)
	s.ProfilerModel = env("PROFILER_MODEL", s.ProfilerModel)
	s.TemporalModel = env("TEMPORAL_MODEL", s.TemporalModel)
	s.SummarizerModel = env("SUMMARIZER_MODEL", s.SummarizerModel)
	s.JudgeModel = env("JUDGE_MODEL", s.JudgeModel)
	s.RetrievalModel = env("RETRIEVAL_MODEL", s.RetrievalModel)
	s.CodeModel = env("CODE_MODEL", s.CodeModel)
	s.PineconeAPIKey = env("PINECONE_API_KEY", s.PineconeAPIKey)
	s.PineconeHost = env("PINECONE_HOST", s.PineconeHost)
	s.PineconeIndexName = env("PINECONE_INDEX_NAME", s.PineconeIndexName)
	s.PineconeNamespace = env("PINECONE_NAMESPACE", s.PineconeNamespace)
	s.PineconeDimension = envInt("PINECONE_DIMENSION", s.PineconeDimension)
	s.PineconeMetric = env("PINECONE_METRIC", s.PineconeMetric)
	s.PineconeCloud = env("PINECONE_CLOUD", s.PineconeCloud)
	s.PineconeRegion = env("PINECONE_REGION", s.PineconeRegion)
	s.VectorStoreProvider = env("VECTOR_STORE_PROVIDER", s.VectorStoreProvider)
	s.EmbeddingProvider = env("EMBEDDING_PROVIDER", s.EmbeddingProvider)
	s.EmbeddingModel = env("EMBEDDING_MODEL", s.EmbeddingModel)
	if s.OpenAIEmbeddingModel == "" && strings.EqualFold(s.EmbeddingProvider, "openai") {
		s.OpenAIEmbeddingModel = s.EmbeddingModel
	}
	s.MongoDBURI = env("MONGODB_URI", s.MongoDBURI)
	s.MongoDBDatabase = env("MONGODB_DATABASE", s.MongoDBDatabase)
	s.Neo4jURI = env("NEO4J_URI", s.Neo4jURI)
	s.Neo4jUsername = env("NEO4J_USERNAME", s.Neo4jUsername)
	s.Neo4jPassword = env("NEO4J_PASSWORD", s.Neo4jPassword)
	s.AppStoreProvider = env("APP_STORE_PROVIDER", s.AppStoreProvider)
	s.APIHost = env("API_HOST", s.APIHost)
	s.APIPort = envInt("API_PORT", s.APIPort)
	s.CORSOrigins = envList("CORS_ORIGINS", s.CORSOrigins)
	s.RateLimit = envInt("RATE_LIMIT", s.RateLimit)
	s.MaxRequestBodyBytes = int64(envInt("MAX_REQUEST_BODY_BYTES", int(s.MaxRequestBodyBytes)))
	s.APIKeys = envList("API_KEYS", s.APIKeys)
	s.JWTSecretKey = env("JWT_SECRET_KEY", s.JWTSecretKey)
	s.JWTAlgorithm = env("JWT_ALGORITHM", s.JWTAlgorithm)
	s.JWTExpirationDays = envInt("JWT_EXPIRATION_DAYS", s.JWTExpirationDays)
	s.EnablePrometheus = envBool("ENABLE_PROMETHEUS", s.EnablePrometheus)
	s.EnableAnalytics = envBool("ENABLE_ANALYTICS", s.EnableAnalytics)
	s.Environment = env("ENVIRONMENT", s.Environment)

	if s.APIPort == 8000 {
		return s, errors.New("xmem-go must not default to or be configured for production Python port 8000")
	}
	if net.ParseIP(strings.TrimSpace(s.APIHost)) == nil && s.APIHost != "localhost" {
		return s, errors.New("API_HOST must be an IP address or localhost")
	}
	if s.Environment != "development" && s.Environment != "test" {
		if s.Neo4jPassword == "" {
			return s, errors.New("NEO4J_PASSWORD is required in non-development environments")
		}
		if s.PineconeAPIKey == "" {
			return s, errors.New("PINECONE_API_KEY is required in non-development environments")
		}
		if s.OpenAIAPIKey == "" {
			return s, errors.New("OPENAI_API_KEY is required in non-development environments")
		}
	}
	return s, nil
}

func (s Settings) Addr() string {
	return s.APIHost + ":" + strconv.Itoa(s.APIPort)
}

func env(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return strings.TrimSpace(v)
	}
	return fallback
}

func envInt(key string, fallback int) int {
	v := env(key, "")
	if v == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return parsed
}

func envFloat(key string, fallback float64) float64 {
	v := env(key, "")
	if v == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func envBool(key string, fallback bool) bool {
	v := strings.ToLower(env(key, ""))
	if v == "" {
		return fallback
	}
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func envList(key string, fallback []string) []string {
	v := strings.TrimSpace(env(key, ""))
	if v == "" {
		return fallback
	}
	var parsed []string
	if strings.HasPrefix(v, "[") {
		if err := json.Unmarshal([]byte(v), &parsed); err == nil {
			return cleanList(parsed)
		}
	}
	return cleanList(strings.Split(v, ","))
}

func cleanList(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func loadDotEnv(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || !strings.Contains(line, "=") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		key := strings.TrimSpace(parts[0])
		value := strings.Trim(strings.TrimSpace(parts[1]), `"'`)
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}
