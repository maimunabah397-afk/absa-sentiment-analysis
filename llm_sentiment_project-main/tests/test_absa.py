import pandas as pd
from src.transformer_absa import TransformerABSA
from src.lexicon_absa import LexiconABSA

## This class isn't corrrect they didn't ask,to test ont he CSV they asked for unit test, so basically testing functionality I THINk
def test_absa_models_dataset():
    df = pd.read_csv('../data/yelp_sample.csv')
    sample_reviews = df['text'].dropna().head(10)

    transformer_analyzer = TransformerABSA()
    lexicon_analyzer = LexiconABSA()

    for i, text in enumerate(sample_reviews, 1):
        print(f"\n=== Review {i} ===")
        print(f"Text: {text}\n")

        # Transformer ABSA
        print("TransformerABSA results:")
        transformer_results = transformer_analyzer.analyze(text)
        if not transformer_results:
            print("  No aspects detected.")
        else:
            for r in transformer_results:
                print(f"  ➜ {r.aspect:<20} | {r.sentiment:<8} | conf={r.confidence:.2f}")

        # Lexicon ABSA
        print("\nLexiconABSA results:")
        lexicon_results = lexicon_analyzer.analyze(text)
        if not lexicon_results:
            print("  No aspects detected.")
        else:
            for r in lexicon_results:
                print(f"  ➜ {r.aspect:<20} | {r.sentiment:<8} | conf={r.confidence:.2f}")

        print("\n" + "="*60)


if __name__ == "__main__":
    test_absa_models_dataset()
