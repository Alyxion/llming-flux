"""Pluggable query expansion for FTS5 search.

Subclass ``QueryExpander`` and override ``expand()`` to add domain-specific
translation dictionaries, synonym tables, or stemming.
"""

import re
from typing import Callable

# Default stopwords (English + German basics)
DEFAULT_STOPWORDS: set[str] = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "or", "and", "but", "if", "so", "than", "too", "very", "just",
    # German
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einem", "einen",
    "und", "oder", "aber", "wenn", "weil", "dass", "ist", "sind", "war", "waren",
    "wird", "werden", "wurde", "wurden", "hat", "haben", "hatte", "hatten",
    "kann", "können", "muss", "müssen", "soll", "sollen",
    "bei", "mit", "von", "zu", "zur", "zum", "für", "auf", "aus", "als", "an",
    "in", "im", "um", "über", "unter", "nach", "vor", "zwischen", "durch",
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "was", "wie", "wo", "wer", "wann", "warum", "welche", "welcher", "welches",
    "nicht", "kein", "keine", "auch", "noch", "schon", "nur", "sehr",
    "gibt", "diesen", "dieser", "diese", "diesem", "pro", "per",
}


def clean_word(word: str) -> str:
    """Remove punctuation, keep hyphens in the middle, lowercase."""
    cleaned = re.sub(r"[^\w\-]", "", word)
    return cleaned.strip("-").lower()


def build_fts_query(terms: list[str]) -> str:
    """Build an FTS5 query string from expanded terms using OR.

    Single words get prefix matching (``term*``), multi-word phrases
    and hyphenated terms get exact quoting (``"term"``).
    """
    if not terms:
        return ""
    escaped = []
    for term in terms:
        if " " in term or "-" in term:
            escaped.append(f'"{term}"')
        else:
            escaped.append(f"{term}*")
    return " OR ".join(escaped)


class QueryExpander:
    """Base query expander — tokenises and removes stopwords.

    Override ``expand()`` in subclasses to add bilingual dictionaries,
    synonym expansion, stemming, etc.
    """

    def __init__(self, stopwords: set[str] | None = None):
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS

    def expand(self, query: str) -> list[str]:
        """Return a list of search terms for FTS5.

        The default implementation splits on whitespace, cleans, and
        filters stopwords.  Subclasses should call ``super().expand()``
        and then add translations/synonyms.
        """
        words = []
        for w in query.lower().split():
            cleaned = clean_word(w)
            if len(cleaned) >= 2 and cleaned not in self.stopwords:
                words.append(cleaned)
        return sorted(set(words), key=len, reverse=True)

    def build_fts(self, query: str) -> str:
        """Convenience: expand + build FTS5 query in one call."""
        return build_fts_query(self.expand(query))
