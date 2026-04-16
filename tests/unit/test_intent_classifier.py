"""Unit tests for IntentClassifier."""

import pytest
from ep_mcp.retrieval.intent import IntentClassifier, QueryIntent


@pytest.fixture
def clf():
    return IntentClassifier()


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class TestIntentClassification:
    def test_entity_what_is(self, clf):
        r = clf.classify("what is territory hierarchy")
        assert r.intent == QueryIntent.ENTITY

    def test_entity_define(self, clf):
        r = clf.classify("define alignment score")
        assert r.intent == QueryIntent.ENTITY

    def test_entity_explain(self, clf):
        r = clf.classify("explain how territory groups work")
        assert r.intent == QueryIntent.ENTITY

    def test_how_steps(self, clf):
        r = clf.classify("steps to create a territory plan")
        assert r.intent == QueryIntent.HOW

    def test_how_how_do(self, clf):
        r = clf.classify("how do I set up a territory group")
        assert r.intent == QueryIntent.HOW

    def test_how_configure(self, clf):
        r = clf.classify("configure rep locator embed")
        assert r.intent == QueryIntent.HOW

    def test_why_reason(self, clf):
        r = clf.classify("why does alignment matter for sales")
        assert r.intent == QueryIntent.WHY

    def test_why_difference(self, clf):
        r = clf.classify("difference between territory types")
        assert r.intent == QueryIntent.WHY

    def test_when_version(self, clf):
        r = clf.classify("when was auto-build added")
        assert r.intent == QueryIntent.WHEN

    def test_when_which_version(self, clf):
        r = clf.classify("which version introduced territory locking")
        assert r.intent == QueryIntent.WHEN

    def test_general_bare_noun(self, clf):
        r = clf.classify("territory overlap rules")
        assert r.intent == QueryIntent.GENERAL

    def test_general_empty(self, clf):
        r = clf.classify("")
        assert r.intent == QueryIntent.GENERAL

    def test_general_whitespace(self, clf):
        r = clf.classify("   ")
        assert r.intent == QueryIntent.GENERAL


# ---------------------------------------------------------------------------
# Weight recommendations
# ---------------------------------------------------------------------------

class TestWeightRecommendations:
    def test_entity_weights(self, clf):
        r = clf.classify("what is a territory")
        assert r.vector_weight == 0.45
        assert r.text_weight == 0.55

    def test_how_weights(self, clf):
        r = clf.classify("how do I build a territory hierarchy")
        assert r.vector_weight == 0.80
        assert r.text_weight == 0.20

    def test_why_weights(self, clf):
        r = clf.classify("why would I use nested territories")
        assert r.vector_weight == 0.80
        assert r.text_weight == 0.20

    def test_when_weights(self, clf):
        r = clf.classify("when was this feature released")
        assert r.vector_weight == 0.40
        assert r.text_weight == 0.60

    def test_general_no_override(self, clf):
        r = clf.classify("sales territory alignment")
        assert r.vector_weight is None
        assert r.text_weight is None

    def test_weights_sum_to_one(self, clf):
        queries = [
            "what is a territory",
            "how do I configure this",
            "why does this happen",
            "which version added this",
        ]
        for q in queries:
            r = clf.classify(q)
            if r.vector_weight is not None:
                total = round(r.vector_weight + r.text_weight, 6)
                assert total == 1.0, f"Weights don't sum to 1.0 for '{q}': {total}"


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

class TestCaseInsensitivity:
    def test_upper(self, clf):
        assert clf.classify("WHAT IS TERRITORY").intent == QueryIntent.ENTITY

    def test_mixed(self, clf):
        assert clf.classify("How Do I Configure This").intent == QueryIntent.HOW
