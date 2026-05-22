package models

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/xortexai/xmem-go/internal/config"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ToolCall struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type Response struct {
	Content   string
	ToolCalls []ToolCall
	ModelName string
}

type ChatModel interface {
	Name() string
	Generate(ctx context.Context, prompt string) (Response, error)
	GenerateWithMessages(ctx context.Context, messages []Message) (Response, error)
	SelectTools(ctx context.Context, query string, profileCatalog []map[string]string) (Response, error)
}

type LocalModel struct {
	name string
}

func NewLocalModel(name string) LocalModel {
	if name == "" {
		name = "local-rule-model"
	}
	return LocalModel{name: name}
}

func NewRegistry(settings config.Settings) ChatModel {
	for _, provider := range settings.FallbackOrder {
		provider = strings.ToLower(provider)
		switch provider {
		case "ollama":
			return NewOllamaModel(settings)
		case "gemini":
			if settings.GeminiAPIKey != "" {
				return NewGeminiModel(settings)
			}
		case "claude":
			if settings.ClaudeAPIKey != "" {
				return NewClaudeModel(settings)
			}
		case "openai":
			if settings.OpenAIAPIKey != "" {
				return NewOpenAICompatibleModel("openai", settings.OpenAIModel, "https://api.openai.com/v1/chat/completions", settings.OpenAIAPIKey)
			}
		case "openrouter":
			if settings.OpenRouterAPIKey != "" {
				return NewOpenAICompatibleModel("openrouter", settings.OpenRouterModel, "https://openrouter.ai/api/v1/chat/completions", settings.OpenRouterAPIKey)
			}
		case "bedrock":
			if settings.AWSAccessKeyID != "" {
				return NewLocalModel(settings.BedrockModel)
			}
		}
	}
	return NewLocalModel("local-rule-model")
}

func (m LocalModel) Name() string {
	return m.name
}

func (m LocalModel) Generate(_ context.Context, prompt string) (Response, error) {
	answer := strings.TrimSpace(prompt)
	if strings.Contains(prompt, "CONTEXT:") {
		answer = synthesizeFromPrompt(prompt)
	}
	return Response{Content: answer, ModelName: m.name}, nil
}

func (m LocalModel) GenerateWithMessages(_ context.Context, messages []Message) (Response, error) {
	combined := ""
	for _, msg := range messages {
		combined += msg.Content + "\n"
	}
	return Response{Content: strings.TrimSpace(combined), ModelName: m.name}, nil
}

func (m LocalModel) SelectTools(_ context.Context, query string, catalog []map[string]string) (Response, error) {
	lowered := strings.ToLower(query)
	calls := []ToolCall{}
	id := 1
	add := func(name string, args map[string]any) {
		calls = append(calls, ToolCall{ID: "call-" + string(rune('0'+id)), Name: name, Args: args})
		id++
	}

	for _, item := range catalog {
		topic := item["topic"]
		if topic != "" && strings.Contains(lowered, strings.ToLower(topic)) {
			add("search_profile", map[string]any{"topic": topic})
			break
		}
	}
	if len(calls) == 0 && containsAny(lowered, "name", "work", "job", "company", "hobby", "food", "like", "prefer", "profile") {
		topic := "personal"
		if containsAny(lowered, "work", "job", "company") {
			topic = "work"
		} else if containsAny(lowered, "food", "eat", "prefer") {
			topic = "food"
		} else if containsAny(lowered, "hobby", "like", "enjoy") {
			topic = "interest"
		}
		add("search_profile", map[string]any{"topic": topic})
	}
	if containsAny(lowered, "when", "date", "schedule", "appointment", "birthday", "tomorrow", "today", "event") {
		add("search_temporal", map[string]any{"query": query})
	}
	if containsAny(lowered, "code", "script", "function", "snippet") {
		add("search_snippet", map[string]any{"query": query})
	}
	if len(calls) == 0 || containsAny(lowered, "remember", "conversation", "summary", "context", "what") {
		add("search_summary", map[string]any{"query": query})
	}
	return Response{ToolCalls: calls, ModelName: m.name}, nil
}

func synthesizeFromPrompt(prompt string) string {
	context := after(prompt, "CONTEXT:")
	if idx := strings.Index(context, "QUERY:"); idx >= 0 {
		context = context[:idx]
	}
	lines := []string{}
	for _, line := range strings.Split(context, "\n") {
		line = strings.TrimSpace(line)
		if line != "" && line != "No results found." {
			lines = append(lines, line)
		}
	}
	if len(lines) == 0 {
		return "I could not find any stored memories that answer that."
	}
	return "Based on stored memories, " + strings.Trim(strings.Join(lines, " "), ". ") + "."
}

func after(text, marker string) string {
	idx := strings.Index(text, marker)
	if idx < 0 {
		return ""
	}
	return strings.TrimSpace(text[idx+len(marker):])
}

func containsAny(text string, words ...string) bool {
	for _, word := range words {
		if strings.Contains(text, word) {
			return true
		}
	}
	return false
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
	local    LocalModel
}

func NewOpenAICompatibleModel(provider, model, url, apiKey string) HTTPModel {
	return HTTPModel{provider: provider, model: model, url: url, apiKey: apiKey, client: &http.Client{Timeout: 90 * time.Second}, local: NewLocalModel(model)}
}

func NewGeminiModel(settings config.Settings) HTTPModel {
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + settings.GeminiModel + ":generateContent?key=" + settings.GeminiAPIKey
	return HTTPModel{provider: "gemini", model: settings.GeminiModel, url: url, apiKey: settings.GeminiAPIKey, client: &http.Client{Timeout: 90 * time.Second}, local: NewLocalModel(settings.GeminiModel)}
}

func NewClaudeModel(settings config.Settings) HTTPModel {
	return HTTPModel{provider: "claude", model: settings.ClaudeModel, url: "https://api.anthropic.com/v1/messages", apiKey: settings.ClaudeAPIKey, client: &http.Client{Timeout: 90 * time.Second}, local: NewLocalModel(settings.ClaudeModel)}
}

func NewOllamaModel(settings config.Settings) HTTPModel {
	return HTTPModel{provider: "ollama", model: settings.OllamaModel, url: strings.TrimRight(settings.OllamaBaseURL, "/") + "/api/chat", client: &http.Client{Timeout: 120 * time.Second}, local: NewLocalModel(settings.OllamaModel)}
}

func (m HTTPModel) Name() string {
	return m.model
}

func (m HTTPModel) Generate(ctx context.Context, prompt string) (Response, error) {
	content, err := m.complete(ctx, prompt, false)
	if err != nil {
		return m.local.Generate(ctx, prompt)
	}
	return Response{Content: content, ModelName: m.model}, nil
}

func (m HTTPModel) GenerateWithMessages(ctx context.Context, messages []Message) (Response, error) {
	content, err := m.completeWithMessages(ctx, messages)
	if err != nil {
		combined := ""
		for _, msg := range messages {
			combined += strings.ToUpper(msg.Role) + ":\n" + msg.Content + "\n\n"
		}
		return m.Generate(ctx, strings.TrimSpace(combined))
	}
	return Response{Content: content, ModelName: m.model}, nil
}

func (m HTTPModel) completeWithMessages(ctx context.Context, messages []Message) (string, error) {
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
		return "", errors.New("unsupported provider")
	}
}

func (m HTTPModel) completeOpenAIMessages(ctx context.Context, messages []Message) (string, error) {
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
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+m.apiKey)
		if m.provider == "openrouter" {
			req.Header.Set("HTTP-Referer", "http://localhost:8081")
			req.Header.Set("X-Title", "xmem-go")
		}
	}); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", errors.New("empty model response")
	}
	return out.Choices[0].Message.Content, nil
}

func (m HTTPModel) completeGeminiMessages(ctx context.Context, messages []Message) (string, error) {
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
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
		return "", err
	}
	if len(out.Candidates) == 0 || len(out.Candidates[0].Content.Parts) == 0 {
		return "", errors.New("empty gemini response")
	}
	return out.Candidates[0].Content.Parts[0].Text, nil
}

func (m HTTPModel) completeClaudeMessages(ctx context.Context, messages []Message) (string, error) {
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
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
		req.Header.Set("x-api-key", m.apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
	}); err != nil {
		return "", err
	}
	if len(out.Content) == 0 {
		return "", errors.New("empty claude response")
	}
	return out.Content[0].Text, nil
}

func (m HTTPModel) completeOllamaMessages(ctx context.Context, messages []Message) (string, error) {
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
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
		return "", err
	}
	return out.Message.Content, nil
}

func (m HTTPModel) SelectTools(ctx context.Context, query string, catalog []map[string]string) (Response, error) {
	prompt := `Select memory retrieval tools for this query.
Return only JSON in this exact shape:
{"tool_calls":[{"name":"search_profile","args":{"topic":"work"}},{"name":"search_temporal","args":{"query":"dentist appointment"}},{"name":"search_summary","args":{"query":"..."}},{"name":"search_snippet","args":{"query":"..."}}]}
Allowed names: search_profile, search_temporal, search_summary, search_snippet.
Available profile catalog: ` + MarshalJSON(catalog) + `
Query: ` + query
	content, err := m.complete(ctx, prompt, true)
	if err != nil {
		return m.local.SelectTools(ctx, query, catalog)
	}
	var parsed struct {
		ToolCalls []ToolCall `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(content)), &parsed); err != nil || len(parsed.ToolCalls) == 0 {
		return m.local.SelectTools(ctx, query, catalog)
	}
	for i := range parsed.ToolCalls {
		if parsed.ToolCalls[i].ID == "" {
			parsed.ToolCalls[i].ID = fmt.Sprintf("call-%d", i+1)
		}
	}
	return Response{ToolCalls: parsed.ToolCalls, ModelName: m.model}, nil
}

func (m HTTPModel) complete(ctx context.Context, prompt string, jsonMode bool) (string, error) {
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
		return "", errors.New("unsupported provider")
	}
}

func (m HTTPModel) completeOpenAI(ctx context.Context, prompt string, jsonMode bool) (string, error) {
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
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer "+m.apiKey)
		if m.provider == "openrouter" {
			req.Header.Set("HTTP-Referer", "http://localhost:8081")
			req.Header.Set("X-Title", "xmem-go")
		}
	}); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", errors.New("empty model response")
	}
	return out.Choices[0].Message.Content, nil
}

func (m HTTPModel) completeGemini(ctx context.Context, prompt string) (string, error) {
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
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
		return "", err
	}
	if len(out.Candidates) == 0 || len(out.Candidates[0].Content.Parts) == 0 {
		return "", errors.New("empty gemini response")
	}
	return out.Candidates[0].Content.Parts[0].Text, nil
}

func (m HTTPModel) completeClaude(ctx context.Context, prompt string) (string, error) {
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
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, func(req *http.Request) {
		req.Header.Set("x-api-key", m.apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
	}); err != nil {
		return "", err
	}
	if len(out.Content) == 0 {
		return "", errors.New("empty claude response")
	}
	return out.Content[0].Text, nil
}

func (m HTTPModel) completeOllama(ctx context.Context, prompt string, jsonMode bool) (string, error) {
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
	}
	if err := m.doJSON(ctx, http.MethodPost, m.url, body, &out, nil); err != nil {
		return "", err
	}
	return out.Message.Content, nil
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
