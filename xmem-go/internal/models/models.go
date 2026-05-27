package models

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	json "github.com/goccy/go-json"
	"github.com/xortexai/xmem-go/internal/config"
	"golang.org/x/net/http2"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type MultimodalMessage struct {
	Role    string
	Content []ContentBlock
}

type ToolCall struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type Response struct {
	Content      string
	ToolCalls    []ToolCall
	ModelName    string
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

type ChatModel interface {
	Name() string
	Generate(ctx context.Context, prompt string) (Response, error)
	GenerateWithMessages(ctx context.Context, messages []Message) (Response, error)
	GenerateVision(ctx context.Context, systemPrompt string, userText string, imageURL string) (Response, error)
	SelectTools(ctx context.Context, query string, profileCatalog []map[string]string) (Response, error)
}

func NewRegistry(settings config.Settings) (ChatModel, error) {
	for _, provider := range settings.FallbackOrder {
		provider = strings.ToLower(provider)
		switch provider {
		case "ollama":
			return NewOllamaModel(settings), nil
		case "gemini":
			if settings.GeminiAPIKey != "" {
				return NewGeminiModel(settings), nil
			}
		case "claude":
			if settings.ClaudeAPIKey != "" {
				return NewClaudeModel(settings), nil
			}
		case "openai":
			if settings.OpenAIAPIKey != "" {
				return NewOpenAICompatibleModel("openai", settings.OpenAIModel, "https://api.openai.com/v1/chat/completions", settings.OpenAIAPIKey), nil
			}
		case "openrouter":
			if settings.OpenRouterAPIKey != "" {
				return NewOpenAICompatibleModel("openrouter", settings.OpenRouterModel, "https://openrouter.ai/api/v1/chat/completions", settings.OpenRouterAPIKey), nil
			}
		}
	}
	return nil, errors.New("no LLM provider configured: set at least one API key (OPENROUTER_API_KEY, GEMINI_API_KEY, CLAUDE_API_KEY, or OPENAI_API_KEY)")
}

func MarshalJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

type HTTPModel struct {
	provider string
	model    string
	url      string
	apiKey   string
	client   *http.Client
}

func newHTTP2Client(timeout time.Duration) *http.Client {
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
		ForceAttemptHTTP2:   true,
	}
	http2.ConfigureTransport(transport)
	return &http.Client{Timeout: timeout, Transport: transport}
}

func NewOpenAICompatibleModel(provider, model, url, apiKey string) HTTPModel {
	return HTTPModel{provider: provider, model: model, url: url, apiKey: apiKey, client: newHTTP2Client(90 * time.Second)}
}

func NewGeminiModel(settings config.Settings) HTTPModel {
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + settings.GeminiModel + ":generateContent?key=" + settings.GeminiAPIKey
	return HTTPModel{provider: "gemini", model: settings.GeminiModel, url: url, apiKey: settings.GeminiAPIKey, client: newHTTP2Client(90 * time.Second)}
}

func NewClaudeModel(settings config.Settings) HTTPModel {
	return HTTPModel{provider: "claude", model: settings.ClaudeModel, url: "https://api.anthropic.com/v1/messages", apiKey: settings.ClaudeAPIKey, client: newHTTP2Client(90 * time.Second)}
}

func NewOllamaModel(settings config.Settings) HTTPModel {
	return HTTPModel{provider: "ollama", model: settings.OllamaModel, url: strings.TrimRight(settings.OllamaBaseURL, "/") + "/api/chat", client: newHTTP2Client(120 * time.Second)}
}

func (m HTTPModel) Name() string {
	return m.model
}

func (m HTTPModel) Generate(ctx context.Context, prompt string) (Response, error) {
	return m.complete(ctx, prompt, false)
}

func (m HTTPModel) GenerateWithMessages(ctx context.Context, messages []Message) (Response, error) {
	return m.completeWithMessages(ctx, messages)
}

func (m HTTPModel) GenerateVision(ctx context.Context, systemPrompt string, userText string, imageURL string) (Response, error) {
	switch m.provider {
	case "openai", "openrouter":
		contentParts := []map[string]any{
			{"type": "text", "text": userText},
		}
		if imageURL != "" {
			contentParts = append(contentParts, map[string]any{
				"type": "image_url", "image_url": map[string]string{"url": imageURL},
			})
		}
		messages := []map[string]any{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": contentParts},
		}
		body := map[string]any{"model": m.model, "messages": messages, "temperature": 0.1}
		var out struct {
			Choices []struct {
				Message struct{ Content string `json:"content"` } `json:"message"`
			} `json:"choices"`
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}
		if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
			req.Header.Set("Authorization", "Bearer "+m.apiKey)
			if m.provider == "openrouter" {
				req.Header.Set("HTTP-Referer", "http://localhost:8081")
				req.Header.Set("X-Title", "xmem-go")
			}
		}); err != nil {
			return Response{}, err
		}
		if len(out.Choices) == 0 {
			return Response{}, errors.New("empty model response")
		}
		return Response{Content: out.Choices[0].Message.Content, ModelName: m.model, InputTokens: out.Usage.PromptTokens, OutputTokens: out.Usage.CompletionTokens, TotalTokens: out.Usage.TotalTokens}, nil
	case "gemini":
		parts := []map[string]any{{"text": userText}}
		if imageURL != "" {
			parts = append(parts, map[string]any{
				"inline_data": map[string]string{"mime_type": "image/jpeg", "data": imageURL},
			})
		}
		body := map[string]any{
			"contents":         []map[string]any{{"role": "user", "parts": parts}},
			"generationConfig": map[string]any{"temperature": 0.1},
			"system_instruction": map[string]any{
				"parts": []map[string]string{{"text": systemPrompt}},
			},
		}
		var out struct {
			Candidates []struct {
				Content struct {
					Parts []struct{ Text string `json:"text"` } `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
			UsageMetadata struct {
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
				TotalTokenCount      int `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}
		if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
			return Response{}, err
		}
		if len(out.Candidates) == 0 || len(out.Candidates[0].Content.Parts) == 0 {
			return Response{}, errors.New("empty gemini response")
		}
		return Response{Content: out.Candidates[0].Content.Parts[0].Text, ModelName: m.model, InputTokens: out.UsageMetadata.PromptTokenCount, OutputTokens: out.UsageMetadata.CandidatesTokenCount, TotalTokens: out.UsageMetadata.TotalTokenCount}, nil
	case "claude":
		contentBlocks := []map[string]any{
			{"type": "text", "text": userText},
		}
		if imageURL != "" {
			contentBlocks = append(contentBlocks, map[string]any{
				"type": "image", "source": map[string]string{"type": "url", "url": imageURL},
			})
		}
		body := map[string]any{
			"model":      m.model,
			"max_tokens": 4096,
			"system":     systemPrompt,
			"messages":   []map[string]any{{"role": "user", "content": contentBlocks}},
		}
		var out struct {
			Content []struct{ Text string `json:"text"` } `json:"content"`
			Usage   struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
			req.Header.Set("x-api-key", m.apiKey)
			req.Header.Set("anthropic-version", "2023-06-01")
		}); err != nil {
			return Response{}, err
		}
		if len(out.Content) == 0 {
			return Response{}, errors.New("empty claude response")
		}
		return Response{Content: out.Content[0].Text, ModelName: m.model, InputTokens: out.Usage.InputTokens, OutputTokens: out.Usage.OutputTokens, TotalTokens: out.Usage.InputTokens + out.Usage.OutputTokens}, nil
	default:
		return m.GenerateWithMessages(ctx, []Message{{Role: "system", Content: systemPrompt}, {Role: "user", Content: userText + "\n[Image URL: " + imageURL + "]"}})
	}
}

func (m HTTPModel) completeWithMessages(ctx context.Context, messages []Message) (Response, error) {
	switch m.provider {
	case "openai", "openrouter":
		return m.completeOpenAIMessages(ctx, messages)
	case "gemini":
		return m.completeGeminiMessages(ctx, messages)
	case "claude":
		return m.completeClaudeMessages(ctx, messages)
	case "ollama":
		return m.completeOllamaMessages(ctx, messages)
	default:
		return Response{}, errors.New("unsupported provider")
	}
}

func (m HTTPModel) completeOpenAIMessages(ctx context.Context, messages []Message) (Response, error) {
	apiMessages := make([]map[string]string, 0, len(messages))
	for _, msg := range messages {
		apiMessages = append(apiMessages, map[string]string{"role": msg.Role, "content": msg.Content})
	}
	body := map[string]any{
		"model":       m.model,
		"messages":    apiMessages,
		"temperature": 0.1,
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+m.apiKey)
		if m.provider == "openrouter" {
			req.Header.Set("HTTP-Referer", "http://localhost:8081")
			req.Header.Set("X-Title", "xmem-go")
		}
	}); err != nil {
		return Response{}, err
	}
	if len(out.Choices) == 0 {
		return Response{}, errors.New("empty model response")
	}
	return Response{Content: out.Choices[0].Message.Content, ModelName: m.model, InputTokens: out.Usage.PromptTokens, OutputTokens: out.Usage.CompletionTokens, TotalTokens: out.Usage.TotalTokens}, nil
}

func (m HTTPModel) completeGeminiMessages(ctx context.Context, messages []Message) (Response, error) {
	var systemText string
	contents := []map[string]any{}
	for _, msg := range messages {
		if msg.Role == "system" {
			systemText = msg.Content
			continue
		}
		role := "user"
		if msg.Role == "assistant" || msg.Role == "model" {
			role = "model"
		}
		contents = append(contents, map[string]any{
			"role":  role,
			"parts": []map[string]string{{"text": msg.Content}},
		})
	}
	body := map[string]any{
		"contents":         contents,
		"generationConfig": map[string]any{"temperature": 0.1},
	}
	if systemText != "" {
		body["system_instruction"] = map[string]any{
			"parts": []map[string]string{{"text": systemText}},
		}
	}
	var out struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
		return Response{}, err
	}
	if len(out.Candidates) == 0 || len(out.Candidates[0].Content.Parts) == 0 {
		return Response{}, errors.New("empty gemini response")
	}
	return Response{Content: out.Candidates[0].Content.Parts[0].Text, ModelName: m.model, InputTokens: out.UsageMetadata.PromptTokenCount, OutputTokens: out.UsageMetadata.CandidatesTokenCount, TotalTokens: out.UsageMetadata.TotalTokenCount}, nil
}

func (m HTTPModel) completeClaudeMessages(ctx context.Context, messages []Message) (Response, error) {
	var systemText string
	apiMessages := []map[string]string{}
	for _, msg := range messages {
		if msg.Role == "system" {
			systemText = msg.Content
			continue
		}
		apiMessages = append(apiMessages, map[string]string{"role": msg.Role, "content": msg.Content})
	}
	body := map[string]any{
		"model":      m.model,
		"max_tokens": 4096,
		"messages":   apiMessages,
	}
	if systemText != "" {
		body["system"] = systemText
	}
	var out struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
		req.Header.Set("x-api-key", m.apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
	}); err != nil {
		return Response{}, err
	}
	if len(out.Content) == 0 {
		return Response{}, errors.New("empty claude response")
	}
	return Response{Content: out.Content[0].Text, ModelName: m.model, InputTokens: out.Usage.InputTokens, OutputTokens: out.Usage.OutputTokens, TotalTokens: out.Usage.InputTokens + out.Usage.OutputTokens}, nil
}

func (m HTTPModel) completeOllamaMessages(ctx context.Context, messages []Message) (Response, error) {
	apiMessages := make([]map[string]string, 0, len(messages))
	for _, msg := range messages {
		apiMessages = append(apiMessages, map[string]string{"role": msg.Role, "content": msg.Content})
	}
	body := map[string]any{
		"model":    m.model,
		"stream":   false,
		"messages": apiMessages,
	}
	var out struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		PromptEvalCount int `json:"prompt_eval_count"`
		EvalCount       int `json:"eval_count"`
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
		return Response{}, err
	}
	return Response{Content: out.Message.Content, ModelName: m.model, InputTokens: out.PromptEvalCount, OutputTokens: out.EvalCount, TotalTokens: out.PromptEvalCount + out.EvalCount}, nil
}

func (m HTTPModel) SelectTools(ctx context.Context, query string, catalog []map[string]string) (Response, error) {
	prompt := `Select memory retrieval tools for this query.
Return only JSON in this exact shape:
{"tool_calls":[{"name":"search_profile","args":{"topic":"work"}},{"name":"search_temporal","args":{"query":"dentist appointment"}},{"name":"search_summary","args":{"query":"..."}},{"name":"search_snippet","args":{"query":"..."}}]}
Allowed names: search_profile, search_temporal, search_summary, search_snippet.
Available profile catalog: ` + MarshalJSON(catalog) + `
Query: ` + query
	resp, err := m.complete(ctx, prompt, true)
	if err != nil {
		return Response{}, err
	}
	var parsed struct {
		ToolCalls []ToolCall `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(resp.Content)), &parsed); err != nil || len(parsed.ToolCalls) == 0 {
		return resp, nil
	}
	for i := range parsed.ToolCalls {
		if parsed.ToolCalls[i].ID == "" {
			parsed.ToolCalls[i].ID = fmt.Sprintf("call-%d", i+1)
		}
	}
	resp.ToolCalls = parsed.ToolCalls
	resp.Content = ""
	return resp, nil
}

func (m HTTPModel) complete(ctx context.Context, prompt string, jsonMode bool) (Response, error) {
	switch m.provider {
	case "openai", "openrouter":
		return m.completeOpenAI(ctx, prompt, jsonMode)
	case "gemini":
		return m.completeGemini(ctx, prompt)
	case "claude":
		return m.completeClaude(ctx, prompt)
	case "ollama":
		return m.completeOllama(ctx, prompt, jsonMode)
	default:
		return Response{}, errors.New("unsupported provider")
	}
}

func (m HTTPModel) completeOpenAI(ctx context.Context, prompt string, jsonMode bool) (Response, error) {
	body := map[string]any{
		"model": m.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"temperature": 0.1,
	}
	if jsonMode {
		body["response_format"] = map[string]string{"type": "json_object"}
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+m.apiKey)
		if m.provider == "openrouter" {
			req.Header.Set("HTTP-Referer", "http://localhost:8081")
			req.Header.Set("X-Title", "xmem-go")
		}
	}); err != nil {
		return Response{}, err
	}
	if len(out.Choices) == 0 {
		return Response{}, errors.New("empty model response")
	}
	return Response{Content: out.Choices[0].Message.Content, ModelName: m.model, InputTokens: out.Usage.PromptTokens, OutputTokens: out.Usage.CompletionTokens, TotalTokens: out.Usage.TotalTokens}, nil
}

func (m HTTPModel) completeGemini(ctx context.Context, prompt string) (Response, error) {
	body := map[string]any{
		"contents": []map[string]any{{"parts": []map[string]string{{"text": prompt}}}},
		"generationConfig": map[string]any{
			"temperature": 0.1,
		},
	}
	var out struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
		return Response{}, err
	}
	if len(out.Candidates) == 0 || len(out.Candidates[0].Content.Parts) == 0 {
		return Response{}, errors.New("empty gemini response")
	}
	return Response{Content: out.Candidates[0].Content.Parts[0].Text, ModelName: m.model, InputTokens: out.UsageMetadata.PromptTokenCount, OutputTokens: out.UsageMetadata.CandidatesTokenCount, TotalTokens: out.UsageMetadata.TotalTokenCount}, nil
}

func (m HTTPModel) completeClaude(ctx context.Context, prompt string) (Response, error) {
	body := map[string]any{
		"model":      m.model,
		"max_tokens": 1024,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	var out struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
		req.Header.Set("x-api-key", m.apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
	}); err != nil {
		return Response{}, err
	}
	if len(out.Content) == 0 {
		return Response{}, errors.New("empty claude response")
	}
	return Response{Content: out.Content[0].Text, ModelName: m.model, InputTokens: out.Usage.InputTokens, OutputTokens: out.Usage.OutputTokens, TotalTokens: out.Usage.InputTokens + out.Usage.OutputTokens}, nil
}

func (m HTTPModel) completeOllama(ctx context.Context, prompt string, jsonMode bool) (Response, error) {
	body := map[string]any{
		"model":  m.model,
		"stream": false,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	if jsonMode {
		body["format"] = "json"
	}
	var out struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		PromptEvalCount int `json:"prompt_eval_count"`
		EvalCount       int `json:"eval_count"`
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
		return Response{}, err
	}
	return Response{Content: out.Message.Content, ModelName: m.model, InputTokens: out.PromptEvalCount, OutputTokens: out.EvalCount, TotalTokens: out.PromptEvalCount + out.EvalCount}, nil
}

func (m HTTPModel) doJSON(ctx context.Context, method, url string, body any, out any, decorate func(*http.Request)) error {
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if decorate != nil {
		decorate(req)
	}
	resp, err := m.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s model request failed: %s: %s", m.provider, resp.Status, string(b))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func extractJSONObject(text string) string {
	text = strings.TrimSpace(text)
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start >= 0 && end >= start {
		return text[start : end+1]
	}
	return text
}
