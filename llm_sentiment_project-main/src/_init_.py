"""
src package — Core implementations for Aspect-Based Sentiment Analysis (ABSA).

This package provides:
- Base interfaces and data structures
- Lexicon-based, Transformer-based, and LLM-based ABSA analyzers
"""

# Base Classes and Interfaces
from src.base import (ABSAAnalyzer, AspectSentiment)
from src.lexicon_absa import LexiconABSA
from src.llm_absa import LLMABSA
from src.transformer_absa import TransformerABSA

__version__ = "1.0.0"
__author__ = "Edobor Aisosa and Maimuna Bah"

__all__ = [
    "ABSAAnalyzer",
    "AspectSentiment",
    "LexiconABSA",
    "TransformerABSA",
    "LLMABSA",
]
