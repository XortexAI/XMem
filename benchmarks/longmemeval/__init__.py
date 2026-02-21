"""
LongMemEval Benchmark for XMEM.

Benchmarks XMEM's long-term memory on the LongMemEval dataset (ICLR 2025).
Evaluates five core abilities:
  - Information Extraction
  - Multi-Session Reasoning
  - Knowledge Updates
  - Temporal Reasoning
  - Abstention

Contains:
  - add.py: Ingests LongMemEval chat sessions into XMEM memory
  - search.py: Answers 500 questions using stored memories
  - evaluate.py: Evaluates using LongMemEval's GPT-4o evaluator + LLM judge
"""
