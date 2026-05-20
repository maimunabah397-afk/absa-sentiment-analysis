import ollama
import json
import re
import time
from typing import List
from src.base import ABSAAnalyzer, AspectSentiment

DEFAULT_SYSTEM_PROMPT = """Act as an Aspect-Based Sentiment Analysis Expert with over 20 years of experience. 

1. First, list all explicit aspects mentioned
2. For each aspect, find opinion words
3. Apply negation rules
4. Classify sentiment as positive, negative, or neutral.


IMPORTANT RULES:
1. Handle negations carefully (e.g., "NOT good" = negative, "not bad" = positive)
2. When you see "but/however", different aspects may have different sentiments
3. Extract ALL aspects mentioned, even if multiple in one sentence
4. Only extract aspects that are explicitly named in the text
5. Sentiment must be: positive, negative, or neutral

Return valid JSON array format. Examples:

Input: "The pizza was delicious but service was slow."
Output: [{"aspect": "pizza", "sentiment": "positive", "confidence": 0.95}, {"aspect": "service", "sentiment": "negative", "confidence": 0.8}]

Input: "Great ambiance and friendly staff."
Output: [{"aspect": "ambiance", "sentiment": "positive", "confidence": 0.90}, {"aspect": "staff", "sentiment": "positive", "confidence": 0.9}]

Input: "The food was NOT good and the coffee was cold."
Output: [{"aspect": "food", "sentiment": "negative", "confidence": 0.9}, {"aspect": "coffee", "sentiment": "negative", "confidence": 0.85}]
"""

class LLMABSA(ABSAAnalyzer):
    """LLM-based ABSA using Ollama."""

    def __init__(self, model_name: str = "llama3", max_retries: int = 3):
        self.model_name = model_name
        self.max_retries = max_retries
        self._verify_ollama()

    def _verify_ollama(self):
        try:
            ollama.list()
        except Exception:
            raise RuntimeError("Ollama not running. Install and start Ollama first.")

    def _extract_json(self, text: str) -> List[dict]:
        """Extract JSON array from the LLM response."""
        json_pattern = r'\[[\s\S]*?\]'
        matches = re.findall(json_pattern, text)

        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue
        return []

    def _build_prompt(self, text: str) -> str:
        """Build prompt with just system instructions and input text."""
        return f"{DEFAULT_SYSTEM_PROMPT}\n\nInput: \"{text}\"\nOutput:"

    def _find_text_span(self, text: str, aspect: str) -> tuple | None:
        """Find the most likely span for the aspect in text."""
        aspect_clean = aspect.strip().lower()
        if len(aspect_clean) < 2:
            return None

        text_lower = text.lower()
        start = text_lower.find(aspect_clean)
        if start != -1:
            return start, start + len(aspect_clean)

        # Try partial matches for multi-word aspects
        for i in range(len(aspect_clean.split()) - 1, 0, -1):
            phrase = ' '.join(aspect_clean.split()[:i])
            start = text_lower.find(phrase)
            if start != -1:
                return start, start + len(phrase)

        return None

    def analyze(self, text: str) -> List[AspectSentiment]:
        if not text or not text.strip():
            return []

        text = text[:5000]  # truncate long text
        prompt = self._build_prompt(text)

        for attempt in range(self.max_retries):
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={'temperature': 0.1}
                )

                parsed_items = self._extract_json(response['response'])
                results = []

                for item in parsed_items:
                    if (isinstance(item, dict) and
                            'aspect' in item and
                            item.get('sentiment') in ['positive', 'negative', 'neutral']):
                        confidence = float(item.get('confidence', 0.8))
                        results.append(AspectSentiment(
                            aspect=str(item['aspect']).strip(),
                            sentiment=item['sentiment'],
                            confidence=confidence,
                            text_span=self._find_text_span(text, str(item['aspect']))                        ))

                return results

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(min(1.0 * (2 ** attempt), 5.0))
                else:
                    print(f"LLM analysis failed: {e}")

        return []
if __name__ == "__main__":
    analyzer = LLMABSA(model_name="llama3")

    test_text = "The food was amazing but the service was slow."
    results = analyzer.analyze(test_text)

    print(f"Input: {test_text}")
    for r in results:
        print(f"  → Aspect: {r.aspect}, Sentiment: {r.sentiment}, Confidence: {r.confidence}")

    # Basic sanity check
    assert any(r.aspect == "food" and r.sentiment == "positive" for r in results)
    assert any(r.aspect == "service" and r.sentiment == "negative" for r in results)
