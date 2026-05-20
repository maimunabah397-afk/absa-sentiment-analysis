# Project overview and setup instructions 

## Introduction
This project performs Aspect-Based Sentiment Analysis (ABSA) on Yelp reviews to extract specific aspects (like "food", "service") 
and their associated sentiments. We compare three different implementation approaches on a curated sample of **50 reviews**
extracted from the full Yelp dataset using `src/make_sample.py` to enable efficient analysis and comparison.

### Implementations
1. **Lexicon-based:** spaCy (noun chunks, dependency parsing) + VADER (sentiment scoring)
2. **Transformer-based:** DeBERTa-v3 (`yangheng/deberta-v3-base-absa-v1.1`) via HuggingFace
3. **LLM-based:** Llama3 via Ollama (local inference)

## Data Summary
- Total reviews: 50
- Average review length: 95.4 words
- Average sentences per review: 8.1
- Sentence complexity: ~11.8 words/sentence

#### Sentiment Distribution

- Positive: 43 (86%)
- Negative: 6 (12%)
- Neutral: 1 (2%)

#### Aspect Insights

- Top aspect candidates: place, time, food, service, night
- Multi-word aspects detected: 4 (e.g., "this place", "the food")
- Reviews with multi-aspect sentences: 26 (52%)

#### Negations and Contrastive Structures

- **Negations:** 27 reviews (54%) contain negations, averaging 0.68 per review
- **Contrastive structures:** 16 reviews (32%) contain "but/however", with higher complexity (10.8 vs 8.1 sentences on average)

These patterns indicate that many reviews express mixed or nuanced sentiments.

## Project File Structure

```
llm_sentiment_project/
├── .venv/                     # virtual environment
├── data/                       # datasets and generated samples
│
├── doc/                        # documentation files (report.pdf)
│
├── notebooks/
│   ├── comparison.ipynb         # Model comparison   (runtime: takes ~15 mins in a best-case-scenario)
│   ├── exploration.ipynb        # Data exploration 
│   
│
├── src/                        # core source code
│   ├── __init__.py
│   ├── base.py                 # Base classes and interfaces 
│   ├── lexicon_absa.py         # Implementation 1
│   ├── transformer_absa.py     # Implementation 2
│   ├── llm_absa.py             # Implementation 3
│   ├── make_sample.py          # Makes a sample of the data to use for analysis
│   └── utils.py
│
├── tests/                      # unit tests
│   ├── test_absa.py
│  
│
├── README.md                   # project documentation
└── requirement.txt              # dependency list

```

# Dependencies and installation guide 

## Installation
```bash
# 1. Clone the repo
git clone https://gitlab.com/data_61/llm_sentiment_project.git
cd llm_sentiment_project

# 2. Create & activate a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirement.txt
python -m spacy download en_core_web_sm

# 4. Install Ollama (see https://ollama.com or project Description pdf)
ollama pull llama3

# Verify Installation
ollama --version 


## Usage examples for each implementation

Each implementation can be tested individually:

```bash
python -m src.lexicon_absa
python -m src.transformer_absa
python -m src.llm_absa  # Requires Ollama running locally
```

**Note:** Before running `llm_absa.py`, ensure Ollama is running on your PC locally. Open the Ollama application.


---

```markdown

## API Documentation

Each implementation follows the `ABSAAnalyzer` interface defined in `src/base.py`:

```python
class ABSAAnalyzer:
    def analyze(self, text: str, aspects: Optional[List[str]] = None) -> List[AspectSentiment]:
        """Analyze text and return aspect-sentiment pairs."""
        pass
```

```
See inline docstrings in:
- `src/lexicon_absa.py`
- `src/transformer_absa.py`
- `src/llm_absa.py`
```


# References
- [Mistral vs Llama 3 Comparison](https://www.openxcell.com/blog/mistral-vs-llama-3/)
- [Python `__init__.py` Best Practices](https://labex.io/tutorials/python-how-to-properly-set-up-an-init-py-file-in-a-python-package-398237)
- [Python `__init__.py` Guide](https://coderslegacy.com/python-init-py-best-practices/)
- [Sentiment Analysis with Python](https://huggingface.co/blog/sentiment-analysis-python)
- [ABSA on Kaggle](https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis)
- [Lexicon Preprocessing](https://ijrti.org/papers/IJRTI2312113.pdf#:~:text=This%20paper%20explains%20how%20to%20perform%20lexicon-based%20sentiment,in%20the%20list%20or%20consider%20using%20weighted%20dictionaries.)

## Design decisions and rationale

### 1. LexiconABSA

**Aspect Extraction**
- Uses **spaCy** with **noun chunks** and **dependency parsing**.
- Supports **1–4 word phrases** (e.g., `"this place"`, `"the food"`, `"my wife"`).
- Applies **strict filtering** to remove pronouns/determiners, keeping only substantive terms.
- **Strength**: Captures explicit aspects reliably (`food`, `service`, `staff`).
- **Weakness**: Misses implicit aspects (e.g., `"the place was dead"` → no noun chunk for `"atmosphere"`).

**Opinion Extraction**
- **Aspect → Opinion** mapping (chosen for simplicity and completeness).
- Uses spaCy’s dependency tree to link aspects to opinion modifiers.

**Negation Handling**
- Traverses dependency paths:
  - Direct: `"delicious pizza"` (`amod`)
  - Predicate: `"pizza is delicious"` (`nsubj → copula → acomp`)
  - Chained: `"not very good"` (`neg → advmod → amod`)

> **Why dependency parsing?**  
> Word proximity fails in complex sentences:  
> `"The pizza, which my friend ordered, was cold"` — `"cold"` is 7 tokens away.  
> Dependency parsing handles **average sentence length (11.8 words)** and **contrastive structures (32% of reviews)**.

**Confidence Threshold**
- Based on [VADER scoring guidelines](https://vadersentiment.readthedocs.io/en/latest/pages/about_the_scoring.html).
- Initial threshold `0.05` caused excessive **neutral** labels due to VADER’s low magnitude scores.
- **Adjusted** to better reflect sentiment intensity.

**Key Limitations**
- VADER’s lexicon lacks domain-specific terms → neutral scores.
- No implicit aspect detection.
- Fails on layered negations (`"I wouldn't say it wasn't bad"`).
- No sarcasm detection.
- spaCy parser errors on malformed input.

**Potential Improvements (Not Implemented)**
- Extend VADER with domain lexicon: `{"cold": -0.5, "smooth": 0.5}`
- Apply **aspect-aware boosting** (e.g., `"high"` negative only for `"price"`).

---

### 2. TransformerABSA

**Aspect Extraction**
- Single-pass **spaCy noun chunk** extraction over the full review (vs. per-sentence in Lexicon).
- Lighter filtering for broader coverage.

**Confidence Scoring**
- **Softmax** over model logits → calibrated probabilities summing to 1.0.

**Negation Handling**
- **None explicit** — relies on **pre-trained DeBERTa** representations.
- Rationale: Model trained on massive corpora learns negation patterns (`"not good"`, `"never again"`).

**Text Truncation**
- Uses tokenizer’s built-in truncation for long reviews.

**Strengths**
- Handles **complex negation**, **long-range dependencies**, and **contrastive sentences**.
- No manual rules needed.

**Limitations**
- Depends on noun chunk quality.
- Softmax can be **overconfident** (high confidence ≠ true certainty).

---

### 3. LLMABSA (Llama-3)

**Why Llama-3?**  
> *“Mistral is fast. Llama-3 is smart.”*  
> — This project prioritizes **reasoning** over speed.  
> Llama-3 excels in **complex negation (54% of reviews)** and **contrastive logic (32%)**.

**Core Mechanism**
- Uses **semantic understanding**, not pattern matching.
- Generates **structured JSON output** with aspects, sentiment, and confidence.

**Prompt Engineering**
```text
Act as an Aspect-Based Sentiment Analysis Expert...
- Handle negations carefully
- When you see 'but/however'...
- Extract ALL aspects mentioned
- Only extract EXPLICITLY mentioned aspects
```

**Summary**
- VADER is word-list based → every word in the phrase affects the score
- DeBERTa is attention-based → learns to focus on relevant words and ignore noise
- Lexicon is fast and SpaCy does an okay job in aspect detection but VADER has too small dictionary context
- Transformers perform quite well, it's confidence scores are quite high and might not reflect reality
- LLM performs best in accuracy, overall it's too slow and not ideal for fast-paced environments

