package pipelines

import (
	"context"
	"strings"
	"sync"

	"github.com/xortexai/xmem-go/internal/agents"
	"github.com/xortexai/xmem-go/internal/contracts"
	"github.com/xortexai/xmem-go/internal/weaver"
)

type IngestPipeline struct {
	ModelName  string
	Weaver     *weaver.Weaver
	Classifier agents.ClassifierAgent
	Profiler   agents.ProfilerAgent
	Temporal   agents.TemporalAgent
	Summarizer agents.SummarizerAgent
	Image      agents.ImageAgent
	Code       agents.CodeAgent
	Snippet    agents.SnippetAgent
	Judge      agents.JudgeAgent
}

type IngestState struct {
	Classification []agents.Classification
	ProfileJudge   weaver.JudgeResult
	TemporalJudge  weaver.JudgeResult
	SummaryJudge   weaver.JudgeResult
	ImageJudge     weaver.JudgeResult
	SnippetJudge   weaver.JudgeResult
	ProfileWeaver  weaver.WeaverResult
	TemporalWeaver weaver.WeaverResult
	SummaryWeaver  weaver.WeaverResult
	ImageWeaver    weaver.WeaverResult
	SnippetWeaver  weaver.WeaverResult
}

func (p *IngestPipeline) Run(ctx context.Context, req contracts.IngestRequest, userID string) (contracts.IngestResponse, error) {
	if req.AgentResponse == "" {
		req.AgentResponse = "Acknowledged."
	}
	cfg := GetEffortConfig(req.EffortLevel)
	var state IngestState
	var err error
	if cfg.Level == EffortHigh && EstimateTokens(req.UserQuery) > cfg.ChunkThresholdTokens {
		chunks := ChunkText(req.UserQuery, cfg.ChunkSizeTokens, cfg.OverlapTokens)
		for idx, chunk := range chunks {
			chunkReq := req
			chunkReq.UserQuery = chunk
			if idx > 0 {
				chunkReq.ImageURL = ""
			}
			state, err = p.invoke(ctx, chunkReq, userID)
			if err != nil {
				return contracts.IngestResponse{}, err
			}
		}
	} else {
		state, err = p.invoke(ctx, req, userID)
		if err != nil {
			return contracts.IngestResponse{}, err
		}
	}
	return p.toResponse(state), nil
}

func (p *IngestPipeline) invoke(ctx context.Context, req contracts.IngestRequest, userID string) (IngestState, error) {
	state := IngestState{}
	state.Classification = p.Classifier.Run(ctx, req.UserQuery, req.ImageURL)

	hasProfile, hasTemporal, hasCode, hasImage := false, false, false, false
	profileQueries, temporalQueries, codeQueries, imageQueries := []string{}, []string{}, []string{}, []string{}
	for _, c := range state.Classification {
		switch c.Source {
		case "profile":
			hasProfile = true
			profileQueries = append(profileQueries, c.Query)
		case "event":
			hasTemporal = true
			temporalQueries = append(temporalQueries, c.Query)
		case "code":
			hasCode = true
			codeQueries = append(codeQueries, c.Query)
		case "image":
			hasImage = true
			imageQueries = append(imageQueries, c.Query)
		}
	}
	if req.ImageURL != "" && len(imageQueries) == 0 {
		hasImage = true
		imageQueries = append(imageQueries, "Analyze this image for memory-relevant details.")
	}
	isTrivial := len(splitWords(req.UserQuery)) < 4 && !hasProfile && !hasTemporal && !hasCode && !hasImage

	var wg sync.WaitGroup
	var mu sync.Mutex
	run := func(fn func() IngestState) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			partial := fn()
			mu.Lock()
			mergeState(&state, partial)
			mu.Unlock()
		}()
	}

	if !isTrivial {
		run(func() IngestState {
			items := p.Summarizer.Run(ctx, req.UserQuery, req.AgentResponse)
			judge := p.Judge.JudgeItems(ctx, weaver.DomainSummary, items, userID, 0.8)
			return IngestState{SummaryJudge: judge, SummaryWeaver: p.Weaver.Execute(ctx, judge, weaver.DomainSummary, userID)}
		})
	}
	if hasProfile {
		run(func() IngestState {
			facts := p.Profiler.Run(ctx, strings.Join(profileQueries, " "))
			judge := p.Judge.JudgeProfile(ctx, facts, userID)
			return IngestState{ProfileJudge: judge, ProfileWeaver: p.Weaver.Execute(ctx, judge, weaver.DomainProfile, userID)}
		})
	}
	if hasTemporal {
		run(func() IngestState {
			events := p.Temporal.Run(ctx, strings.Join(temporalQueries, " "), req.SessionDatetime)
			judge := p.Judge.JudgeTemporal(ctx, events, userID)
			return IngestState{TemporalJudge: judge, TemporalWeaver: p.Weaver.Execute(ctx, judge, weaver.DomainTemporal, userID)}
		})
	}
	if hasImage {
		run(func() IngestState {
			items := p.Image.Run(ctx, strings.Join(imageQueries, " "), req.ImageURL)
			judge := p.Judge.JudgeItems(ctx, weaver.DomainSummary, items, userID, 0.8)
			return IngestState{ImageJudge: judge, ImageWeaver: p.Weaver.Execute(ctx, judge, weaver.DomainSummary, userID)}
		})
	}
	if hasCode {
		run(func() IngestState {
			items := p.Snippet.Run(ctx, strings.Join(codeQueries, " "))
			judge := p.Judge.JudgeItems(ctx, weaver.DomainSnippet, items, userID, 0.8)
			return IngestState{SnippetJudge: judge, SnippetWeaver: p.Weaver.Execute(ctx, judge, weaver.DomainSnippet, userID)}
		})
	}
	wg.Wait()
	return state, nil
}

func (p *IngestPipeline) toResponse(state IngestState) contracts.IngestResponse {
	classifications := make([]any, 0, len(state.Classification))
	for _, c := range state.Classification {
		classifications = append(classifications, map[string]any{"source": c.Source, "query": c.Query})
	}
	return contracts.IngestResponse{
		Model:          p.ModelName,
		Classification: classifications,
		Profile:        domainResult(state.ProfileJudge, state.ProfileWeaver),
		Temporal:       domainResult(state.TemporalJudge, state.TemporalWeaver),
		Summary:        domainResult(state.SummaryJudge, state.SummaryWeaver),
		Image:          domainResult(state.ImageJudge, state.ImageWeaver),
	}
}

func domainResult(judge weaver.JudgeResult, wr weaver.WeaverResult) *contracts.DomainResult {
	if len(judge.Operations) == 0 {
		return nil
	}
	ops := make([]contracts.OperationDetail, 0, len(judge.Operations))
	for _, op := range judge.Operations {
		ops = append(ops, contracts.OperationDetail{Type: string(op.Type), Content: op.Content, Reason: op.Reason})
	}
	return &contracts.DomainResult{
		Confidence: judge.Confidence,
		Operations: ops,
		Weaver:     &contracts.WeaverSummary{Succeeded: wr.Succeeded(), Skipped: wr.Skipped(), Failed: wr.Failed()},
	}
}

func mergeState(dst *IngestState, src IngestState) {
	if len(src.ProfileJudge.Operations) > 0 {
		dst.ProfileJudge = src.ProfileJudge
		dst.ProfileWeaver = src.ProfileWeaver
	}
	if len(src.TemporalJudge.Operations) > 0 {
		dst.TemporalJudge = src.TemporalJudge
		dst.TemporalWeaver = src.TemporalWeaver
	}
	if len(src.SummaryJudge.Operations) > 0 {
		dst.SummaryJudge = src.SummaryJudge
		dst.SummaryWeaver = src.SummaryWeaver
	}
	if len(src.ImageJudge.Operations) > 0 {
		dst.ImageJudge = src.ImageJudge
		dst.ImageWeaver = src.ImageWeaver
	}
	if len(src.SnippetJudge.Operations) > 0 {
		dst.SnippetJudge = src.SnippetJudge
		dst.SnippetWeaver = src.SnippetWeaver
	}
}

func splitWords(text string) []string {
	out := []string{}
	for _, w := range stringsFields(text) {
		if w != "" {
			out = append(out, w)
		}
	}
	return out
}

func stringsFields(text string) []string {
	var fields []string
	start := -1
	for i, r := range text {
		if r == ' ' || r == '\n' || r == '\t' {
			if start >= 0 {
				fields = append(fields, text[start:i])
				start = -1
			}
			continue
		}
		if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		fields = append(fields, text[start:])
	}
	return fields
}
