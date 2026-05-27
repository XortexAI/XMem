package agents

import (
	"context"
	"strings"
	"testing"

	"github.com/xortexai/xmem-go/internal/models"
)

type stubModel struct {
	content string
}

func (m stubModel) Name() string { return "stub" }

func (m stubModel) Generate(context.Context, string) (models.Response, error) {
	return models.Response{Content: m.content}, nil
}

func (m stubModel) GenerateWithMessages(context.Context, []models.Message) (models.Response, error) {
	return models.Response{Content: m.content}, nil
}

func (m stubModel) GenerateVision(_ context.Context, _ string, _ string, _ string) (models.Response, error) {
	return models.Response{Content: m.content}, nil
}

func (m stubModel) SelectTools(context.Context, string, []map[string]string) (models.Response, error) {
	return models.Response{Content: m.content}, nil
}

func TestTemporalDateValidationMatchesPythonAgent(t *testing.T) {
	valid := []string{"01-31", "02-29", "04-30", "12-31"}
	for _, date := range valid {
		if !isValidDate(date) {
			t.Fatalf("expected %s to be valid", date)
		}
	}

	invalid := []string{"", "1-01", "01-1", "00-10", "13-01", "02-30", "04-31"}
	for _, date := range invalid {
		if isValidDate(date) {
			t.Fatalf("expected %s to be invalid", date)
		}
	}
}

func TestSnippetAgentParsesArrayTags(t *testing.T) {
	agent := SnippetAgent{Model: stubModel{content: `{"snippets":[{"content":"Binary search handles empty arrays","code_snippet":"return -1","language":"cpp","snippet_type":"algorithm","tags":["dsa","binary-search"]}]}`}}

	items := agent.Run(context.Background(), "save this snippet")
	if len(items) != 1 {
		t.Fatalf("items length = %d", len(items))
	}
	if !strings.Contains(items[0], "dsa,binary-search") {
		t.Fatalf("expected comma-joined tags, got %q", items[0])
	}
}

func TestCodeAgentParsesAnnotations(t *testing.T) {
	agent := CodeAgent{Model: stubModel{content: `{"annotations":[{"target_symbol":"PaymentProcessor.process","target_file":"src/payments.go","content":"Add an idempotency key before retrying charges.","annotation_type":"fix","severity":"high","repo":"payments"}]}`}}

	items := agent.Run(context.Background(), "remember this fix")
	if len(items) != 1 {
		t.Fatalf("items length = %d", len(items))
	}
	if got := items[0]; !strings.Contains(got, "fix | PaymentProcessor.process | src/payments.go | payments | high | Add an idempotency key") {
		t.Fatalf("unexpected annotation content: %q", got)
	}
}

func TestImageAgentFormatsPythonPipelineItems(t *testing.T) {
	agent := ImageAgent{Model: stubModel{content: "DESCRIPTION: Whiteboard with API architecture\n\nOBSERVATIONS:\n- [technical] Shows service-to-database flow (confidence: high)"}}

	items := agent.Run(context.Background(), "analyze architecture", "https://example.com/image.png")
	if len(items) != 2 {
		t.Fatalf("items length = %d", len(items))
	}
	if items[0] != "[Image] Whiteboard with API architecture" {
		t.Fatalf("unexpected description item: %q", items[0])
	}
	if items[1] != "[Image/technical] Shows service-to-database flow (high)" {
		t.Fatalf("unexpected observation item: %q", items[1])
	}
}
