"""
Utility functions for Aspect-Based Sentiment Analysis (ABSA)
Provides linguistic constants, aspect extraction, and sentiment utilities
"""
from typing import Tuple, Set, Dict
import spacy
from spacy.tokens import Doc
import json
import random

def load_jsonl(file_path):
    """Load a JSON Lines file and return a list of dicts."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data.append(json.loads(line))
    return data

def sample_reviews(file_path, sample_size=10, seed=42):
    """Return a random sample of reviews for testing."""
    random.seed(seed)
    data = load_jsonl(file_path)
    return random.sample(data, min(sample_size, len(data)))

def save_json(data, file_path, indent=2):
    """Save a list of dicts or any JSON-serializable object to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)



# ------------------------------------------------------------
# LEXICAL CONSTANTS
# ------------------------------------------------------------

NEGATIONS: Set[str] = {
    "no", "not", "n't", "never", "none", "nothing", "neither", "nowhere",
    "hardly", "scarcely", "barely", "cannot", "can't", "doesn't", "isn't",
    "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "couldn't", "don't"
}

INTENSIFIERS: Dict[str, float] = {
    "very": 1.5, "extremely": 1.8, "really": 1.4, "super": 1.5,
    "amazingly": 1.6, "incredibly": 1.7, "too": 1.3,
    "so": 1.3, "quite": 1.2, "highly": 1.4, "totally": 1.5, "deeply": 1.4,
    "especially": 1.3, "utterly": 1.6, "perfectly": 1.5, "absolutely": 1.7
}

DIMINISHERS: Dict[str, float] = {
    "somewhat": 0.8, "slightly": 0.7, "a bit": 0.8, "little": 0.8,
    "kind of": 0.8, "sort of": 0.8, "barely": 0.6, "hardly": 0.6,
    "fairly": 0.85, "rather": 0.85, "moderately": 0.8, "relatively": 0.85
}

GENERIC_EXCLUDE: Set[str] = {
    "thing", "something", "lot", "review", "friend", "day", "weekend",
    "option", "star", "rating", "reason", "way", "people",
    "everyone", "anyone", "everything", "anything", "bit", "kind",
    "visit", "hour", "minute", "month", "year", "location", "spot",
    "area", "moment", "part", "point", "case", "fact", "level"
}

DOMAIN_SENTIMENT_OVERRIDES: Dict[str, Dict[str, float]] = {
    'gym': {
        'intense': 0.5, 'hard': 0.4, 'challenging': 0.6, 'sweaty': 0.3,
        'pushing': 0.5, 'burn': 0.4, 'strong': 0.6, "tiring": 0.2
    },
    'salon': {
        'cute': 0.5, 'cheap': 0.3, 'clean': 0.6, 'manicure': 0.2,
        'pedicure': 0.2, 'polish': 0.1, 'design': 0.3, "casual": 0.2,
    },
    'restaurant': {
        'fresh': 0.6, 'healthy': 0.5, 'busy': -0.3, 'packed': -0.4,
        'crowded': -0.5, 'bland': -0.6, 'spicy': 0.3, 'tender': 0.5,
        "delicious": 0.6, "cold": -0.5, "okay": 0.0,
        "helpful": 0.5, "slow": -0.4,"responsive": 0.4, "unresponsive": -0.5
    },
    'cinema': {
        'hot': -0.5, 'cold': -0.5, 'loud': -0.4, 'quiet': 0.3,
        'comfortable': 0.5, 'crowded': -0.4,
    },
    'retail': {
        'affordable': 0.5, 'pricey': -0.4, 'expensive': -0.5,
        'budget': 0.3, 'quality': 0.4,   "worth it": 0.6, "not worth it": -0.6,
    },
     'general': {
    "mediocre": -0.3,
    "average": 0.0,
    "fine": 0.0,
    "acceptable": 0.0,
    "okay": 0.0,
    "adequate": 0.0,
    "standard": 0.0
    },
    "other": {
        "beautiful": 0.5,
        "extravagant": 0.4,
        "basic": 0.0
    }
}

PHRASE_SENTIMENT_OVERRIDES: Dict[str, float] = {
    "not bad": 0.4, "no problem": 0.3, "nothing special": -0.3,
    "worked with us": 0.5, "super friendly": 0.7, "never disappoints": 0.6,
    "highly recommend": 0.8, "not good": -0.4, "far from good": -0.6,
    "could be better": -0.3, "loved it": 0.8, "hated it": -0.8, "but not great": -0.4
}


# ------------------------------------------------------------
# ASPECT EXTRACTION FUNCTIONS
# ------------------------------------------------------------

def extract_aspects_linguistic(doc) -> Dict[int, Tuple[str, Tuple[int, int]]]:
    """
    Extract aspects from text using spaCy's linguistic features.
    """
    aspects = {}
    processed_tokens = set()

    # Fallback for models without dependency parsing
    if not doc.has_annotation("DEP"):
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                lemma = token.lemma_.lower()
                if (lemma not in GENERIC_EXCLUDE and
                        len(token.text) > 2 and
                        not token.is_stop):
                    aspects[token.i] = (token.text.lower(),
                                        (token.idx, token.idx + len(token.text)))
        return aspects

    # Process noun chunks as primary aspect sources
    for chunk in doc.noun_chunks:
        head = chunk.root

        # Skip pronouns and generic nouns
        if (head.pos_ == "PRON" or
                head.lemma_.lower() in GENERIC_EXCLUDE or
                any(t.lemma_.lower() in GENERIC_EXCLUDE for t in chunk)):
            continue

        # Build compound aspect phrase
        compound_parts = []
        tokens_in_aspect = []

        # Include adjectives and compounds before the head
        for token in chunk:
            if token.dep_ in {"compound", "amod"} or token == head:
                if token.lemma_.lower() not in GENERIC_EXCLUDE:
                    compound_parts.append(token.text.lower())
                    tokens_in_aspect.append(token)
                    processed_tokens.add(token.i)

        if compound_parts and len(' '.join(compound_parts)) > 2:
            aspect_text = ' '.join(compound_parts)
            aspects[head.i] = (aspect_text, (chunk.start_char, chunk.end_char))

    # Add standalone meaningful nouns not in chunks
    for token in doc:
        if (token.i not in processed_tokens and
                token.pos_ in {"NOUN", "PROPN"} and
                token.lemma_.lower() not in GENERIC_EXCLUDE and
                len(token.text) > 2 and not token.is_stop):
            aspects[token.i] = (token.text.lower(),
                                (token.idx, token.idx + len(token.text)))

    return aspects


# ------------------------------------------------------------
# SENTIMENT UTILITY FUNCTIONS
# ------------------------------------------------------------

def apply_sentiment_modifiers(score: float, token: spacy.tokens.Token, doc: spacy.tokens.Doc) -> float:
    """
    Apply negation, intensifiers, and diminishers to sentiment score.
    """
    modified_score = score
    negated = False

    # Check for intensifiers/diminishers
    for child in token.children:
        if child.dep_ == "advmod":
            child_text = child.text.lower()
            if child_text in INTENSIFIERS:
                modified_score *= INTENSIFIERS[child_text]
            elif child_text in DIMINISHERS:
                modified_score *= DIMINISHERS[child_text]

    # Check for negation
    if any(t.dep_ == "neg" for t in token.children):
        negated = True

    if not negated and token.head is not None and token.head.pos_ == "VERB":
        if any(t.dep_ == "neg" for t in token.head.children):
            negated = True

    if not negated and token.text.lower() in NEGATIONS and token.dep_ != "neg":
        negated = True

    # Valence-aware negation handling
    if negated:
        if modified_score < 0:  # "not bad" -> positive
            modified_score = -modified_score * 0.8
        else:  # "not good" -> negative
            modified_score = -modified_score * 0.6

    return max(-1.0, min(1.0, modified_score))

def calculate_confidence(score: float, opinion_count: int = 1) -> float:
    """
    Calculate confidence using custom logic for neutral scores and opinion count.
    """
    base_confidence = min(abs(score), 1.0)
    final_confidence = base_confidence

    # Boost confidence for multiple agreeing opinions
    if opinion_count > 1:
        final_confidence = min(final_confidence + 0.1, 1.0)

    # Override for explicit neutral (0.0)
    if score == 0.0:
        return 0.3

    return final_confidence


def choose_sentiment_label(score: float,
                           pos_threshold: float = 0.05,
                           neg_threshold: float = -0.05) -> str:
    """
    Map sentiment score to discrete label using VADER's default thresholds.
    """
    if score > pos_threshold:
        return "positive"
    elif score < neg_threshold:
        return "negative"
    else:
        return "neutral"


# ------------------------------------------------------------
# DOMAIN DETECTION FUNCTION
# ------------------------------------------------------------

def detect_domain(text: str) -> str:
    """
    Detect domain based on keyword presence in text.
    """
    text_lower = text.lower()

    domain_keywords = {
        'restaurant': {'restaurant', 'food', 'menu', 'dish', 'meal', 'dinner', 'lunch', 'breakfast', 'eat'},
        'gym': {'gym', 'workout', 'exercise', 'fitness', 'trainer', 'cardio', 'weights'},
        'salon': {'salon', 'nail', 'manicure', 'pedicure', 'haircut', 'stylist', 'spa'},
        'cinema': {'cinema', 'movie', 'film', 'theater', 'screen', 'projection'},
        'retail': {'shop', 'store', 'buy', 'purchase', 'price', 'sale', 'retail'}
    }

    for domain, keywords in domain_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return domain

    return 'general'

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

def validate_aspect_sentiment(aspect: str, sentiment: str, confidence: float) -> bool:
    """
    Validate that an AspectSentiment object meets quality criteria.
    """
    if not aspect or len(aspect.strip()) < 2:
        return False
    if sentiment not in {"positive", "negative", "neutral"}:
        return False
    if not (0.0 <= confidence <= 1.0):
        return False
    return True