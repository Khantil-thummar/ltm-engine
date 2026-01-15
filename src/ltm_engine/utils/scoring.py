"""Scoring utilities for hybrid memory retrieval."""

import math
from datetime import datetime, timezone
from typing import Any


def calculate_recency_score(
    created_at: datetime,
    half_life_days: float = 30.0,
    reference_time: datetime | None = None,
) -> float:
    """
    Calculate recency score using exponential decay.
    
    Uses the formula: score = e^(-λt) where λ = ln(2) / half_life
    
    Args:
        created_at: When the memory was created
        half_life_days: Days until score drops to 50%
        reference_time: Reference time for calculation (defaults to now)
        
    Returns:
        Recency score between 0.0 and 1.0
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    # Ensure both are timezone-aware
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    # Calculate age in days
    age_seconds = (reference_time - created_at).total_seconds()
    age_days = age_seconds / (24 * 60 * 60)

    # Prevent negative age
    if age_days < 0:
        return 1.0

    # Calculate decay rate (lambda)
    decay_rate = math.log(2) / half_life_days

    # Calculate score using exponential decay
    score = math.exp(-decay_rate * age_days)

    return max(0.0, min(1.0, score))


def calculate_frequency_score(
    access_count: int,
    max_expected_accesses: int = 100,
) -> float:
    """
    Calculate frequency score based on access count.
    
    Uses logarithmic scaling to prevent frequent items from dominating.
    
    Args:
        access_count: Number of times the memory was accessed
        max_expected_accesses: Expected maximum for normalization
        
    Returns:
        Frequency score between 0.0 and 1.0
    """
    if access_count <= 0:
        return 0.0

    # Use log scaling: log(1 + count) / log(1 + max)
    score = math.log(1 + access_count) / math.log(1 + max_expected_accesses)

    return max(0.0, min(1.0, score))


def calculate_combined_score(
    semantic_score: float,
    recency_score: float,
    frequency_score: float,
    confidence_score: float,
    weight_semantic: float = 0.4,
    weight_recency: float = 0.25,
    weight_frequency: float = 0.15,
    weight_confidence: float = 0.2,
) -> float:
    """
    Calculate combined relevance score using weighted sum.
    
    Args:
        semantic_score: Semantic similarity score (0-1)
        recency_score: Recency decay score (0-1)
        frequency_score: Access frequency score (0-1)
        confidence_score: Confidence score (0-1)
        weight_*: Weights for each component (should sum to 1.0)
        
    Returns:
        Combined score between 0.0 and 1.0
    """
    # Normalize weights if they don't sum to 1
    total_weight = weight_semantic + weight_recency + weight_frequency + weight_confidence
    if abs(total_weight - 1.0) > 0.001:
        weight_semantic /= total_weight
        weight_recency /= total_weight
        weight_frequency /= total_weight
        weight_confidence /= total_weight

    combined = (
        weight_semantic * semantic_score
        + weight_recency * recency_score
        + weight_frequency * frequency_score
        + weight_confidence * confidence_score
    )

    return max(0.0, min(1.0, combined))


def normalize_scores(
    scores: list[float],
    min_score: float | None = None,
    max_score: float | None = None,
) -> list[float]:
    """
    Normalize a list of scores to 0-1 range.
    
    Args:
        scores: List of raw scores
        min_score: Minimum value for normalization (auto-detect if None)
        max_score: Maximum value for normalization (auto-detect if None)
        
    Returns:
        List of normalized scores
    """
    if not scores:
        return []

    if min_score is None:
        min_score = min(scores)
    if max_score is None:
        max_score = max(scores)

    score_range = max_score - min_score
    if score_range == 0:
        return [1.0] * len(scores)

    return [(s - min_score) / score_range for s in scores]


def calculate_importance_decay(
    current_importance: float,
    last_accessed: datetime | None,
    half_life_days: float = 30.0,
    reference_time: datetime | None = None,
) -> float:
    """
    Calculate decayed importance score.
    
    Args:
        current_importance: Current importance score
        last_accessed: Last access time (uses created_at if None)
        half_life_days: Days until importance drops to 50%
        reference_time: Reference time (defaults to now)
        
    Returns:
        Decayed importance score
    """
    if last_accessed is None:
        return current_importance

    recency = calculate_recency_score(
        last_accessed,
        half_life_days=half_life_days,
        reference_time=reference_time,
    )

    # Importance decays with recency
    return current_importance * recency


def calibrate_confidence(
    current_confidence: float,
    is_correct: bool,
    learning_rate: float = 0.1,
) -> float:
    """
    Calibrate confidence based on feedback (Bayesian-style update).
    
    Args:
        current_confidence: Current confidence score
        is_correct: Whether the memory was found to be correct
        learning_rate: How much to adjust confidence
        
    Returns:
        Updated confidence score
    """
    if is_correct:
        # Increase confidence, with diminishing returns
        adjustment = learning_rate * (1.0 - current_confidence)
        new_confidence = current_confidence + adjustment
    else:
        # Decrease confidence
        adjustment = learning_rate * current_confidence
        new_confidence = current_confidence - adjustment

    return max(0.0, min(1.0, new_confidence))
