# Aspect-Based Sentiment Analysis (ABSA) Project

## Project Overview

This project performs Aspect-Based Sentiment Analysis (ABSA) on 50 Yelp reviews to extract specific aspects (such as food and service) and their associated sentiments.

It compares three different approaches: lexicon-based NLP, transformer-based models, and LLM-based analysis to evaluate their performance on real-world text data.

The goal is to understand the trade-offs between traditional NLP, deep learning, and LLM-based methods in sentiment analysis tasks.

## Objective

The objective of this project is to:
- Extract aspects from customer reviews
- Classify sentiment linked to each aspect
- Compare different NLP approaches
- Analyze how each method handles negation and complex sentences

## Implementations

Lexicon-based approach:
Uses spaCy for noun chunk extraction and dependency parsing combined with VADER for sentiment scoring. This approach is fast but limited in handling context and implicit meaning.

Transformer-based approach:
Uses a pretrained DeBERTa-v3 ABSA model from HuggingFace. This method handles context, negation, and long-range dependencies better than lexicon-based methods and outputs probability-based confidence scores.

LLM-based approach:
Uses Llama 3 via Ollama with prompt engineering to perform aspect-based sentiment analysis. This method relies on semantic reasoning, handles complex sentences well, and provides the most accurate results but is slower and computationally expensive.

## Dataset Summary

- Total reviews: 50
- Average review length: 95.4 words
- Average sentences per review: 8.1
- Sentiment distribution:
  - Positive: 86%
  - Negative: 12%
  - Neutral: 2%

## Key Insights

- 54% of reviews contain negations
- 32% contain contrastive structures such as "but" or "however"
- Most common aspects: food, service, place, time
- 52% of reviews contain multiple aspects
- Many reviews express mixed or nuanced sentiment

## Project Structure

llm_sentiment_project/
├── data/
├── doc/
├── notebooks/
│   ├── comparison.ipynb
│   ├── exploration.ipynb
├── src/
│   ├── base.py
│   ├── lexicon_absa.py
│   ├── transformer_absa.py
│   ├── llm_absa.py
│   ├── make_sample.py
│   ├── utils.py
├── tests/
├── README.md
└── requirement.txt

## Installation

Clone the repository:

git clone <repo-url>
cd llm_sentiment_project

Create virtual environment:

python -m venv .venv
source .venv/bin/activate   (Windows: .venv\Scripts\activate)

Install dependencies:

pip install -r requirement.txt
python -m spacy download en_core_web_sm

Install LLM model:

ollama pull llama3

## Usage

Run each implementation separately:

python -m src.lexicon_absa
python -m src.transformer_absa
python -m src.llm_absa

Note: The LLM implementation requires Ollama to be running locally.

## Key Findings

Lexicon-based methods are fast but limited in understanding context and implicit meaning.

Transformer-based models perform better in handling context, negation, and longer dependencies.

LLM-based approaches provide the best reasoning capability and accuracy but are slower and require more resources.

## Conclusion

This project demonstrates the trade-offs between three approaches to sentiment analysis: speed (lexicon-based), balance (transformers), and accuracy (LLMs). Each method has strengths depending on the use case.
