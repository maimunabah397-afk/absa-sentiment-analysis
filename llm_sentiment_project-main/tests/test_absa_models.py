import pandas as pd
import pytest


from src.lexicon_absa import LexiconABSA
from src.llm_absa import LLMABSA
from src.transformer_absa import TransformerABSA

@pytest.mark.integration  # optional tag since it uses real data
def test_absa_models_with_real_data():
    transformer = TransformerABSA()
    lexicon = LexiconABSA()
    ollama =  LLMABSA()

    df = pd.read_csv('../data/yelp_sample.csv')
    sample_reviews = df['text'].dropna().head(1)

    for text in sample_reviews:
        # Transformer ABSA
        transformer_results = transformer.analyze(text)
        # Lexicon ABSA
        lexicon_results = lexicon.analyze(text)

        ollama_results = ollama.analyze(text)

        # Assertions
        assert isinstance(transformer_results, list)
        assert isinstance(lexicon_results, list)
        if transformer_results:
            for r in transformer_results:
                assert hasattr(r, "aspect")
                assert hasattr(r, "sentiment")
                assert 0.0 <= r.confidence <= 1.0
        if lexicon_results:
            for r in lexicon_results:
                assert hasattr(r, "aspect")
                assert hasattr(r, "sentiment")
                assert 0.0 <= r.confidence <= 1.0
        if ollama_results:
            for r in ollama_results:
                assert hasattr(r, "aspect")
                assert hasattr(r, "sentiment")
                assert 0.0 <= r.confidence <= 1.0


if __name__ == "__main__":
    print("Running ABSA unit test on real dataset...")
    test_absa_models_with_real_data()
    print("All tests passed successfully!")
