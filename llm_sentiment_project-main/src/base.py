from dataclasses import dataclass
from typing import List, Optional, Tuple

"""
base.py — Core interfaces and data structures for the ABSA project.

Defines:
- AspectSentiment: Data class for aspect-level sentiment analysis pairs.
- ABSAAnalyzer: Abstract base class that defines the unified API for all ABSA analyzers.
"""


@dataclass
class AspectSentiment:
    """
    Represents a single aspect-sentiment prediction.

    Attributes:
        aspect: The feature or entity being discussed (e.g., "pizza", "service").
        sentiment: The sentiment label toward the aspect ('positive', 'negative', or 'neutral').
        confidence: Confidence score between 0.0 and 1.0.
        text_span: Optional character index range (start, end) in the original text.
    """
    aspect: str
    sentiment: str  # 'positive' | 'negative' | 'neutral'
    confidence: float
    text_span: Optional[Tuple[int, int]]  # (start, end) or None


class ABSAAnalyzer:
    """Base interface for all ABSA implementations."""

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze input text and return aspect-sentiment predictions.

        Args:
            text: Input text to analyze (e.g., a product or restaurant review).

        Returns:
            A list of AspectSentiment objects representing each detected aspect and its sentiment.
        """
        raise NotImplementedError
