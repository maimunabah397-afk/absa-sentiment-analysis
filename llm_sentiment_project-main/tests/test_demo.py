"""
run_demo.py
------------
Read the *yelp_sample.csv* (single column `text`) and run Lexicon-ABSA on the first 5 rows.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Sequence
from src.lexicon_absa_smth  import LexiconABSA, AspectSentiment


# ----------------------------------------------------------------------
# Pretty-print helper
# ----------------------------------------------------------------------
def format_result(text: str, results: Sequence[AspectSentiment]) -> str:
    if not results:
        return ""
    lines = [f"TEXT: {text}"]
    for r in results:
        lines.append(f"  → {r.aspect:<20} | {r.sentiment:<8} | conf: {r.confidence:.2f}")
    return "\n".join(lines)


def load_reviews_from_csv(
    path: Path,
    text_column: str = "text",
    limit: int | None = None,
) -> List[str]:
    """
    Return a list of review strings from a CSV that contains at least one column
    called *text_column* (default = 'text').
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(f, dialect=dialect)

        reviews = []
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            txt = row.get(text_column, "").strip()
            if txt:
                reviews.append(txt)
        return reviews


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(
    csv_path: Path,
    text_column: str = "text",
    limit: int = 5,
) -> None:
    analyzer = LexiconABSA()
    print("=== LEXICON-ABSA – FIRST 5 REVIEWS FROM CSV ===\n")

    reviews = load_reviews_from_csv(csv_path, text_column=text_column, limit=limit)

    for idx, txt in enumerate(reviews, start=1):
        results = analyzer.analyze(txt)
        print(format_result(txt, results))
        print()


if __name__ == "__main__":

    CSV_FILE = Path(__file__).parent / "../data" / "yelp_sample.csv"

    # --------------------------------------------------------------
    # 2. Name of the column that holds the review text
    # --------------------------------------------------------------
    TEXT_COL = "text"

    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

    main(CSV_FILE, text_column=TEXT_COL, limit=5)