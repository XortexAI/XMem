package main

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/xortexai/xmem-go/internal/agents"
	"github.com/xortexai/xmem-go/internal/config"
	"github.com/xortexai/xmem-go/internal/database"
	"github.com/xortexai/xmem-go/internal/graph"
	"github.com/xortexai/xmem-go/internal/jobs"
	"github.com/xortexai/xmem-go/internal/models"
	"github.com/xortexai/xmem-go/internal/pipelines"
	"github.com/xortexai/xmem-go/internal/storage"
	"github.com/xortexai/xmem-go/internal/weaver"
)

type runtimeDeps struct {
	ingest    *pipelines.IngestPipeline
	retrieval *pipelines.RetrievalPipeline
	keyStore  database.APIKeyStore
	jobStore  jobs.Store
}

func buildRuntime(ctx context.Context, settings config.Settings, logger *slog.Logger) (runtimeDeps, error) {
	embedder, err := buildEmbedder(settings, logger)
	if err != nil {
		return runtimeDeps{}, err
	}

	vectorStore, snippetStore, err := buildVectorStores(ctx, settings, embedder, logger)
	if err != nil {
		return runtimeDeps{}, err
	}

	temporalStore, err := buildTemporalStore(ctx, settings, logger)
	if err != nil {
		return runtimeDeps{}, err
	}

	model := models.NewRegistry(settings)
	ingest, retrieval := buildPipelines(model, vectorStore, snippetStore, temporalStore, embedder)

	keyStore, jobStore, err := buildAppStores(ctx, settings, logger)
	if err != nil {
		return runtimeDeps{}, err
	}

	return runtimeDeps{
		ingest:    ingest,
		retrieval: retrieval,
		keyStore:  keyStore,
		jobStore:  jobStore,
	}, nil
}

func buildEmbedder(settings config.Settings, logger *slog.Logger) (storage.Embedder, error) {
	fallback := storage.HashEmbedder{Dimension: settings.PineconeDimension}
	if settings.EmbeddingProvider != "openai" {
		return fallback, nil
	}

	openAIEmbedder, err := storage.NewOpenAIEmbedder(settings)
	if err == nil {
		logger.Info("using OpenAI embedder", "model", settings.OpenAIEmbeddingModel, "dimension", settings.PineconeDimension)
		return openAIEmbedder, nil
	}
	if production(settings) {
		return nil, fmt.Errorf("openai embedder initialization failed: %w", err)
	}
	logger.Warn("openai embedder unavailable, using hash embedder", "error", err)
	return fallback, nil
}

func buildVectorStores(ctx context.Context, settings config.Settings, embedder storage.Embedder, logger *slog.Logger) (storage.VectorStore, storage.VectorStore, error) {
	vectorStore := storage.VectorStore(storage.NewMemoryVectorStore())
	snippetStore := storage.VectorStore(storage.NewMemoryVectorStore())
	if settings.VectorStoreProvider != "pinecone" {
		return vectorStore, snippetStore, nil
	}

	pineconeStore, err := storage.NewPineconeVectorStore(ctx, settings, embedder, settings.PineconeNamespace)
	if err != nil {
		if production(settings) {
			return nil, nil, fmt.Errorf("pinecone vector store initialization failed: %w", err)
		}
		logger.Warn("pinecone unavailable, using memory vector store", "error", err)
	} else {
		vectorStore = pineconeStore
		logger.Info("using Pinecone vector store", "namespace", settings.PineconeNamespace)
	}

	snippetNamespace := settings.PineconeNamespace + "-snippets"
	pineconeSnippets, err := storage.NewPineconeVectorStore(ctx, settings, embedder, snippetNamespace)
	if err != nil {
		if production(settings) {
			return nil, nil, fmt.Errorf("pinecone snippet store initialization failed: %w", err)
		}
		logger.Warn("pinecone snippet store unavailable, using memory vector store", "error", err)
	} else {
		snippetStore = pineconeSnippets
		logger.Info("using Pinecone snippet vector store", "namespace", snippetNamespace)
	}

	return vectorStore, snippetStore, nil
}

func buildTemporalStore(ctx context.Context, settings config.Settings, logger *slog.Logger) (graph.TemporalStore, error) {
	fallback := graph.NewMemoryTemporalStore()
	if settings.Neo4jPassword == "" {
		return fallback, nil
	}

	neoStore, err := graph.NewNeo4jTemporalStore(ctx, settings)
	if err == nil {
		logger.Info("using Neo4j temporal store")
		return neoStore, nil
	}
	if production(settings) {
		return nil, fmt.Errorf("neo4j initialization failed: %w", err)
	}
	logger.Warn("neo4j unavailable, using memory temporal store", "error", err)
	return fallback, nil
}

func buildPipelines(model models.ChatModel, vectorStore storage.VectorStore, snippetStore storage.VectorStore, temporalStore graph.TemporalStore, embedder storage.Embedder) (*pipelines.IngestPipeline, *pipelines.RetrievalPipeline) {
	w := &weaver.Weaver{
		VectorStore:        vectorStore,
		SnippetVectorStore: snippetStore,
		Embedder:           embedder,
		TemporalStore:      temporalStore,
	}

	ingest := &pipelines.IngestPipeline{
		ModelName:  model.Name(),
		Weaver:     w,
		Classifier: agents.ClassifierAgent{Model: model},
		Profiler:   agents.ProfilerAgent{Model: model},
		Temporal:   agents.TemporalAgent{Model: model},
		Summarizer: agents.SummarizerAgent{Model: model},
		Image:      agents.ImageAgent{Model: model},
		Snippet:    agents.SnippetAgent{Model: model},
		Judge:      agents.JudgeAgent{Model: model, VectorStore: vectorStore, TopK: 3},
	}
	retrieval := &pipelines.RetrievalPipeline{
		Model:         model,
		VectorStore:   vectorStore,
		SnippetStore:  snippetStore,
		TemporalStore: temporalStore,
	}
	return ingest, retrieval
}

func buildAppStores(ctx context.Context, settings config.Settings, logger *slog.Logger) (database.APIKeyStore, jobs.Store, error) {
	keyStore := database.APIKeyStore(database.NewMemoryAPIKeyStore())
	jobStore := jobs.Store(jobs.NewMemoryStore())
	if settings.AppStoreProvider != "mongo" {
		return keyStore, jobStore, nil
	}

	mongoStore, err := database.NewMongoAPIKeyStore(ctx, settings)
	if err != nil {
		if production(settings) {
			return nil, nil, fmt.Errorf("mongodb api key store initialization failed: %w", err)
		}
		logger.Warn("mongodb unavailable, using memory API key store", "error", err)
	} else {
		keyStore = mongoStore
		logger.Info("using MongoDB API key store")
	}

	durableStore, err := database.NewMongoDurableJobStore(ctx, settings)
	if err != nil {
		if production(settings) {
			return nil, nil, fmt.Errorf("mongodb durable job store initialization failed: %w", err)
		}
		logger.Warn("mongodb durable job store unavailable, using memory job store", "error", err)
	} else {
		jobStore = durableStore
		logger.Info("using MongoDB durable job store")
	}

	return keyStore, jobStore, nil
}

func production(settings config.Settings) bool {
	return settings.Environment == "production"
}
