"""
Lexicon-Based Aspect-Based Sentiment Analysis (ABSA) Implementation
Uses spaCy for linguistic processing and VADER for sentiment analysis
with domain-aware customizations and sophisticated dependency parsing.
"""

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
import warnings

from src.base import AspectSentiment, ABSAAnalyzer
from src.utils import (
    DOMAIN_SENTIMENT_OVERRIDES, PHRASE_SENTIMENT_OVERRIDES,
    extract_aspects_linguistic, apply_sentiment_modifiers,
    calculate_confidence, choose_sentiment_label, detect_domain,
    validate_aspect_sentiment
)


@dataclass
class LexiconABSASettings:
    """Configuration settings for lexicon-based ABSA analyzer."""
    spacy_model: str = "en_core_web_sm"
    min_opinion_strength: float = 0.02
    positive_threshold: float = 0.05
    negative_threshold: float = -0.05
    enable_domain_aware: bool = True


class LexiconABSA(ABSAAnalyzer):
    """Advanced lexicon-based ABSA analyzer with multi-domain support."""

    def __init__(self, settings: Optional[LexiconABSASettings] = None):
        self.settings = settings or LexiconABSASettings()
        self.vader = SentimentIntensityAnalyzer()
        self.nlp = self._load_spacy_model()
        self._processed_pairs: Set[Tuple[int, int]] = set()

    def _load_spacy_model(self):
        """Load spaCy model with fallback handling."""
        try:
            return spacy.load(self.settings.spacy_model)
        except OSError:
            warnings.warn(
                f"spaCy model '{self.settings.spacy_model}' not found. "
                "Falling back to blank English model. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp

    def _get_opinion_score(self, word: str, domain: str = "general") -> float:
        """Get sentiment score with domain-aware overrides and phrase handling."""
        word_lower = word.lower()

        # Check phrase overrides first (highest priority)
        if word_lower in PHRASE_SENTIMENT_OVERRIDES:
            return PHRASE_SENTIMENT_OVERRIDES[word_lower]

        # Check domain-specific overrides
        if (self.settings.enable_domain_aware and
                domain in DOMAIN_SENTIMENT_OVERRIDES and
                word_lower in DOMAIN_SENTIMENT_OVERRIDES[domain]):
            return DOMAIN_SENTIMENT_OVERRIDES[domain][word_lower]

        return self.vader.polarity_scores(word)["compound"]

    def _find_aspect_opinion_pairs(self, doc, aspects: Dict[int, Tuple]) -> List[Tuple[int, int]]:
        """Find aspect-opinion pairs using sophisticated dependency patterns."""
        pairs = []
        self._processed_pairs.clear()
        aspect_indices = set(aspects.keys())

        # Pattern configuration
        conjunctions = {"but", "yet", "however"}
        aspect_dependencies = {"nsubj", "nsubjpass"}

        for token in doc:
            # PATTERN 1: Direct adjectival modifier - "friendly staff"
            if token.dep_ == "amod" and token.head.i in aspect_indices and token.pos_ == "ADJ":
                self._add_pair(pairs, (token.head.i, token.i))

            # PATTERN 2: Adjectival complement - "staff was friendly"
            elif token.dep_ == "acomp" and token.pos_ == "ADJ":
                self._handle_acomp_pattern(token, aspect_indices, pairs)

            # PATTERN 3: Object of opinion verb - "love this place"
            elif (token.dep_ == "dobj" and token.i in aspect_indices and
                  token.head.lemma_ in {"love", "like", "hate", "dislike", "enjoy", "recommend", "prefer"}):
                self._add_pair(pairs, (token.i, token.head.i))

            # PATTERN 4: Prepositional object with adjectival head - "great for families"
            elif token.dep_ == "pobj" and token.i in aspect_indices:
                self._handle_pobj_pattern(token, aspect_indices, pairs)

            # PATTERN 5: Coordinated Clause Aggregation - "time was acceptable but not great"
            if token.dep_ == "cc" and token.lower_ in conjunctions:
                self._handle_coordinated_clause(token, aspect_indices, pairs, aspect_dependencies)

        return pairs

    def _add_pair(self, pairs: List[Tuple[int, int]], pair: Tuple[int, int]):
        """Helper to add pair if not already processed."""
        if pair not in self._processed_pairs:
            pairs.append(pair)
            self._processed_pairs.add(pair)

    def _handle_acomp_pattern(self, token, aspect_indices, pairs):
        """Handle adjectival complement pattern."""
        for child in token.head.children:
            if child.dep_ in {"nsubj", "nsubjpass"} and child.i in aspect_indices:
                self._add_pair(pairs, (child.i, token.i))

    def _handle_pobj_pattern(self, token, aspect_indices, pairs):
        """Handle prepositional object pattern."""
        for ancestor in token.ancestors:
            if ancestor.pos_ == "ADJ" and ancestor.i not in aspect_indices:
                self._add_pair(pairs, (token.i, ancestor.i))
                break

    def _handle_coordinated_clause(self, token, aspect_indices, pairs, aspect_dependencies):
        """Handle coordinated clause pattern."""
        second_clause_head = next((c for c in token.head.children
                                   if c.dep_ == "conj" and c.i > token.i), None)

        if second_clause_head and second_clause_head.pos_ in {"ADJ", "NOUN", "VERB"}:
            first_clause_subj = next((c for c in token.head.children
                                      if c.dep_ in aspect_dependencies and c.i < token.i), None)

            if first_clause_subj and first_clause_subj.i in aspect_indices:
                self._add_pair(pairs, (first_clause_subj.i, second_clause_head.i))

    def _aggregate_aspect_sentiments(self, doc, aspects, pairs) -> Dict[str, Dict]:
        """Aggregate sentiment scores for each aspect, handling multiple opinions."""
        aspect_aggregates: Dict[str, Dict] = {}
        domain = detect_domain(doc.text) if self.settings.enable_domain_aware else "general"

        for aspect_idx, opinion_idx in pairs:
            aspect_text, span = aspects[aspect_idx]
            opinion_token = doc[opinion_idx]

            # Get and validate sentiment score
            raw_score = self._get_opinion_score(opinion_token.text, domain)
            if abs(raw_score) < self.settings.min_opinion_strength and raw_score != 0.0:
                continue

            # Apply linguistic modifiers and aggregate
            final_score = apply_sentiment_modifiers(raw_score, opinion_token, doc)
            self._add_to_aggregate(aspect_aggregates, aspect_text, span, final_score)

        return aspect_aggregates

    def _add_to_aggregate(self, aggregates: Dict, aspect_text: str, span: Tuple, score: float):
        """Add sentiment score to aspect aggregate."""
        if aspect_text not in aggregates:
            aggregates[aspect_text] = {'scores': [], 'span': span, 'opinion_count': 0}

        aggregates[aspect_text]['scores'].append(score)
        aggregates[aspect_text]['opinion_count'] += 1

    def analyze(self, text: str) -> List[AspectSentiment]:
        """Analyze text and extract aspect-sentiment pairs."""
        if not text or not text.strip():
            return []

        # Process text and extract aspects
        doc = self.nlp(text)
        aspects = extract_aspects_linguistic(doc)

        # Find pairs and aggregate sentiments
        pairs = self._find_aspect_opinion_pairs(doc, aspects)
        aspect_aggregates = self._aggregate_aspect_sentiments(doc, aspects, pairs)

        # Create final results
        results = []
        for aspect_text, aggregate in aspect_aggregates.items():
            if not aggregate['scores']:
                continue

            # Calculate average sentiment and confidence
            avg_score = sum(aggregate['scores']) / len(aggregate['scores'])
            sentiment = choose_sentiment_label(avg_score, self.settings.positive_threshold,
                                               self.settings.negative_threshold)
            confidence = calculate_confidence(avg_score, aggregate['opinion_count'])

            # Create and validate AspectSentiment object
            aspect_sentiment = AspectSentiment(
                aspect=aspect_text,
                sentiment=sentiment,
                confidence=round(confidence, 2),
                text_span=aggregate['span']
            )

            if validate_aspect_sentiment(aspect_text, sentiment, confidence):
                results.append(aspect_sentiment)

        return results


def create_lexicon_analyzer() -> LexiconABSA:
    """Create a pre-configured lexicon analyzer."""
    return LexiconABSA(LexiconABSASettings())


# Example usage and testing
if __name__ == "__main__":
    analyzer = create_lexicon_analyzer()

    test_texts = [
        "The food was delicious but the service was slow.",
        "Not bad for the price.",
        "Gym workouts are intense but rewarding.",
        "I love this cinema despite the hot room.",
        "The decor was okay but the lighting was mediocre.",
        "The portion size was average.",
        "The wait time was acceptable but not great. The food was delicious.",
        "The room was fine, nothing special",
        "The service was okay, nothing to complain about."
    ]

    for text in test_texts:
        print(f"\n{text}")
        results = analyzer.analyze(text)

        for result in results:
            print(f"  → {result.aspect:<20} | {result.sentiment:<8} | conf: {result.confidence:.2f}")