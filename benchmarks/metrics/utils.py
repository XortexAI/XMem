"""
Text Metrics Utilities for LoCoMo Benchmark.

Calculates:
- F1 Score (Token precision/recall)
- BLEU Score (N-gram similarity)

These match the metrics used by Memobase, Mem0, and other memory systems.
"""
import re
import string
from collections import Counter
from typing import Dict, List


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    
    - Convert to lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    if not text:
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    
    return text.strip()


def get_tokens(text: str) -> List[str]:
    """Get tokens from normalized text."""
    normalized = normalize_answer(text)
    if not normalized:
        return []
    return normalized.split()


def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate F1 score between prediction and ground truth.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        prediction: Generated answer
        ground_truth: Expected answer
        
    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    common = sum((pred_counter & gt_counter).values())
    
    if common == 0:
        return 0.0
    
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def calculate_bleu_scores(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate BLEU scores (1-gram to 4-gram).
    
    Returns BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    We primarily use BLEU-1 (B1) as shown in the LoCoMo benchmark table.
    
    Args:
        prediction: Generated answer
        ground_truth: Expected answer
        
    Returns:
        Dictionary with bleu1, bleu2, bleu3, bleu4 scores
    """
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)
    
    if not pred_tokens or not gt_tokens:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    
    scores = {}
    
    for n in range(1, 5):
        pred_ngrams = []
        gt_ngrams = []
        
        for i in range(len(pred_tokens) - n + 1):
            pred_ngrams.append(tuple(pred_tokens[i:i+n]))
        
        for i in range(len(gt_tokens) - n + 1):
            gt_ngrams.append(tuple(gt_tokens[i:i+n]))
        
        if not pred_ngrams or not gt_ngrams:
            scores[f"bleu{n}"] = 0.0
            continue
        
        pred_counter = Counter(pred_ngrams)
        gt_counter = Counter(gt_ngrams)
        
        matches = sum((pred_counter & gt_counter).values())
        precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
        
        scores[f"bleu{n}"] = precision
    
    return scores


def calculate_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate all metrics for a prediction.
    
    Args:
        prediction: Generated answer
        ground_truth: Expected answer
        
    Returns:
        Dictionary with f1, bleu1, bleu2, bleu3, bleu4 scores
    """
    f1 = calculate_f1_score(prediction, ground_truth)
    bleu = calculate_bleu_scores(prediction, ground_truth)
    
    return {
        "f1": f1,
        **bleu
    }


def aggregate_scores(scores_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate a list of score dictionaries into mean values.
    
    Args:
        scores_list: List of score dictionaries
        
    Returns:
        Dictionary with mean scores
    """
    if not scores_list:
        return {"f1": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    
    keys = scores_list[0].keys()
    result = {}
    
    for key in keys:
        values = [s.get(key, 0.0) for s in scores_list]
        result[key] = sum(values) / len(values)
    
    return result


def calculate_bleu1(prediction: str, ground_truth: str) -> float:
    """Calculate just BLEU-1 score."""
    return calculate_bleu_scores(prediction, ground_truth)["bleu1"]
