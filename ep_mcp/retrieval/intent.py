"""Intent classification for query-adaptive retrieval routing.

Classifies incoming queries into intent types and returns recommended
retrieval parameter adjustments. Pure heuristic — no LLM dependency.

Intent types:
- ENTITY  : "what is X", "define X", "explain X" → BM25-heavy (exact term matching)
- HOW     : "how to X", "how do I X", "steps to X" → vector-heavy (semantic)
- WHY     : "why X", "reason for X", "purpose of X" → vector-heavy (conceptual)
- WHEN    : "when X", "which version", "release of X" → BM25-heavy + type hints
- GENERAL : fallback → use configured defaults
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryIntent(str, Enum):
    ENTITY = "ENTITY"
    HOW = "HOW"
    WHY = "WHY"
    WHEN = "WHEN"
    GENERAL = "GENERAL"


@dataclass
class IntentResult:
    """Classification result with recommended retrieval adjustments."""

    intent: QueryIntent

    # Recommended weight overrides (None = use configured default)
    vector_weight: float | None = None
    text_weight: float | None = None

    # Optional type hint to pass as a soft boost signal (not a hard filter)
    suggested_type: str | None = None

    def __repr__(self) -> str:
        return (
            f"IntentResult(intent={self.intent.value}, "
            f"vector_weight={self.vector_weight}, text_weight={self.text_weight})"
        )


# ---------------------------------------------------------------------------
# Pattern tables — order matters; first match wins
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, QueryIntent)
_PATTERNS: list[tuple[re.Pattern[str], QueryIntent]] = [
    # WHEN — temporal / versioning queries
    (re.compile(
        r"\b(when (was|is|did|does|will|should)|which version|"
        r"what version|release (of|date|notes?)|added in|since version|"
        r"changelog|history of|deprecated in|available (since|in))\b",
        re.IGNORECASE,
    ), QueryIntent.WHEN),

    # HOW — procedural / instructional queries
    (re.compile(
        r"\b(how (do|does|can|should|to|would)|steps? (to|for)|"
        r"guide (to|for)|instructions? (for|to)|configure|set up|setup|"
        r"enable|disable|install|deploy|create|build|implement|integrate)\b",
        re.IGNORECASE,
    ), QueryIntent.HOW),

    # WHY — reasoning / conceptual queries
    (re.compile(
        r"\b(why (is|are|does|do|would|should|can't|cannot|doesn't|don't)|"
        r"reason (for|why|behind)|purpose of|benefit of|advantage of|"
        r"what('s| is) the (point|reason|purpose|benefit|advantage)|"
        r"difference between|compared? (to|with)|vs\.?)\b",
        re.IGNORECASE,
    ), QueryIntent.WHY),

    # ENTITY — definitional / reference queries
    (re.compile(
        r"\b(what (is|are|does|do)|define|definition of|explain|"
        r"describe|tell me about|overview of|introduction to|"
        r"meaning of|concept of|what('s| is) a[n]?)\b",
        re.IGNORECASE,
    ), QueryIntent.ENTITY),
]


class IntentClassifier:
    """Classifies a query string into a QueryIntent with retrieval adjustments.

    Weights are tuned for EP pack retrieval:
    - ENTITY: BM25 boost helps match exact entity names in pack content
    - HOW/WHY: Vector boost helps match conceptual / procedural similarity
    - WHEN: BM25 boost helps find version numbers, dates, keywords
    - GENERAL: configured defaults (typically 0.7 vector / 0.3 BM25)

    All weights sum to 1.0.
    """

    # Weight table: intent → (vector_weight, text_weight)
    _WEIGHTS: dict[QueryIntent, tuple[float, float]] = {
        QueryIntent.ENTITY: (0.45, 0.55),
        QueryIntent.HOW:    (0.80, 0.20),
        QueryIntent.WHY:    (0.80, 0.20),
        QueryIntent.WHEN:   (0.40, 0.60),
        QueryIntent.GENERAL: (None, None),  # use configured defaults
    }

    def classify(self, query: str) -> IntentResult:
        """Classify a query and return an IntentResult with weight recommendations.

        Args:
            query: Raw query string from the search request.

        Returns:
            IntentResult with intent type and optional weight overrides.
        """
        query = query.strip()
        if not query:
            return IntentResult(intent=QueryIntent.GENERAL)

        matched_intent = QueryIntent.GENERAL
        for pattern, intent in _PATTERNS:
            if pattern.search(query):
                matched_intent = intent
                break

        vector_w, text_w = self._WEIGHTS[matched_intent]

        return IntentResult(
            intent=matched_intent,
            vector_weight=vector_w,
            text_weight=text_w,
        )
