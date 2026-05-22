package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/xortexai/xmem-go/internal/api"
	"github.com/xortexai/xmem-go/internal/config"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	settings, err := config.Load()
	if err != nil {
		logger.Error("configuration failed", "error", err)
		os.Exit(1)
	}

	runtime, err := buildRuntime(context.Background(), settings, logger)
	if err != nil {
		logger.Error("runtime initialization failed", "error", err)
		os.Exit(1)
	}
	server := api.NewServer(settings, logger, runtime.ingest, runtime.retrieval, runtime.keyStore, runtime.jobStore)
	httpServer := &http.Server{
		Addr:              settings.Addr(),
		Handler:           server.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		logger.Info("xmem-go listening", "addr", settings.Addr(), "service", settings.ServiceName)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		logger.Error("graceful shutdown failed", "error", err)
		os.Exit(1)
	}
	logger.Info("xmem-go stopped")
}
