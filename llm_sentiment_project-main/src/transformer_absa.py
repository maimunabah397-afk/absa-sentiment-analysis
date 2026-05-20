from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from src.base import ABSAAnalyzer, AspectSentiment

class TransformerABSA(ABSAAnalyzer):
    """
    Pre-trained Transformer-based ABSA Model.
    Automatic aspect detection using spaCy.
    """
    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1", device: Optional[int] = None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.device >= 0:
            self.model.to(device)

        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

        # Initialize spaCy for aspect detection
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def detect_aspects(self, text: str) -> List[str]:
        """
        Extract candidate aspects from text using spaCy noun chunks.
        Filters out pronouns and very short/common words.
        """
        doc = self.spacy_nlp(text)
        aspects = set()
        stop_pos = {"PRON"}  # exclude pronouns
        stop_words = {"i", "we", "they", "us", "this", "that", "it", "he", "she"}

        for np in doc.noun_chunks:
            words = np.text.strip().split()
            if len(words) <= 4 and all(t.pos_ not in stop_pos and t.text.lower() not in stop_words for t in np):
                aspects.add(np.text.strip().lower())

        return list(aspects) if aspects else ["overall"]

    def analyze(self, text: str, aspects: Optional[List[str]] = None) -> List[AspectSentiment]:
        """
        Analyze text and return aspect-sentiment pairs.
        If `aspects` is None, detect automatically.
        """
        results: List[AspectSentiment] = []

        if aspects is None:
            aspects = self.detect_aspects(text)

        for aspect in aspects:
            # Model expects both text and aspect as input
            encoded = self.tokenizer(
                text,
                aspect,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            if self.device >= 0:
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                label_id = torch.argmax(probs).item()

            sentiment = self.label_map[label_id]
            confidence = float(probs[label_id])
            span = (text.lower().find(aspect.lower()), text.lower().find(aspect.lower()) + len(aspect))

            results.append(AspectSentiment(
                aspect=aspect,
                sentiment=sentiment,
                confidence=confidence,
                text_span=span
            ))

        return results


if __name__ == "__main__":
    analyzer = TransformerABSA()

    # Example sentence with 5 aspects: camera, battery, screen, software, price
    sample_text = (
        "The Pizza is almost always cold. "
        "The camera quality is amazing, the battery does not last long. "
        "The screen is bright. "
        "The software is smooth, but the price is a bit high."
    )

    output = analyzer.analyze(sample_text)

    print("\nDetected aspects and sentiment:")
    for r in output:
        print(f"➜ {r.aspect:<20} | {r.sentiment:<8} | conf={r.confidence:.2f}")

