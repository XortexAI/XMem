package pipelines

import "strings"

type EffortLevel string

const (
	EffortLow  EffortLevel = "low"
	EffortHigh EffortLevel = "high"
)

type EffortConfig struct {
	Level                EffortLevel
	ChunkThresholdTokens int
	ChunkSizeTokens      int
	OverlapTokens        int
}

func GetEffortConfig(level string) EffortConfig {
	if strings.ToLower(level) == string(EffortHigh) {
		return EffortConfig{Level: EffortHigh, ChunkThresholdTokens: 200, ChunkSizeTokens: 200, OverlapTokens: 15}
	}
	return EffortConfig{Level: EffortLow, ChunkThresholdTokens: 999999, ChunkSizeTokens: 999999, OverlapTokens: 0}
}

func EstimateTokens(text string) int {
	n := len(text) / 4
	if n < 1 {
		return 1
	}
	return n
}

func ChunkText(text string, chunkSizeTokens int, overlapTokens int) []string {
	charsPerChunk := chunkSizeTokens * 4
	overlapChars := overlapTokens * 4
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if len(text) <= charsPerChunk {
		return []string{strings.TrimSpace(text)}
	}

	chunks := []string{}
	start := 0
	prevTail := ""
	for start < len(text) {
		end := start + charsPerChunk
		if end > len(text) {
			end = len(text)
		}
		window := text[start:end]
		searchFrom := len(window) / 3
		lastPeriod := strings.LastIndex(window[searchFrom:], ". ")
		advance := len(window)
		segment := window
		if lastPeriod >= 0 {
			lastPeriod += searchFrom
			segment = window[:lastPeriod+1]
			advance = lastPeriod + 1
		}
		segment = strings.TrimSpace(segment)
		if segment != "" {
			full := segment
			if prevTail != "" {
				full = strings.TrimSpace(prevTail) + " " + segment
			}
			chunks = append(chunks, strings.TrimSpace(full))
			tailLen := overlapChars
			if tailLen > len(segment) {
				tailLen = len(segment)
			}
			if tailLen > 0 {
				prevTail = segment[len(segment)-tailLen:]
			} else {
				prevTail = ""
			}
		}
		if advance <= 0 {
			advance = 1
		}
		start += advance
	}
	return chunks
}
