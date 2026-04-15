"""Tests for adaptive threshold filtering."""

import pytest

from ep_mcp.retrieval.scorer import apply_adaptive_threshold


class TestApplyAdaptiveThreshold:
    """Tests for apply_adaptive_threshold."""

    def test_empty_scores(self):
        assert apply_adaptive_threshold({}) == {}

    def test_below_activation_floor(self):
        """All results discarded when best score is below activation floor."""
        scores = {1: 0.10, 2: 0.08, 3: 0.05}
        result = apply_adaptive_threshold(scores, activation_floor=0.15)
        assert result == {}

    def test_at_activation_floor(self):
        """Best score exactly at activation floor passes."""
        scores = {1: 0.15, 2: 0.10}
        result = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.05
        )
        assert 1 in result
        # 0.15 * 0.55 = 0.0825 → effective_min = max(0.0825, 0.05) = 0.0825
        # 0.10 >= 0.0825 → kept
        assert 2 in result

    def test_ratio_cutoff_filters_weak_results(self):
        """Results below the ratio cutoff are removed."""
        scores = {1: 0.80, 2: 0.50, 3: 0.30, 4: 0.10}
        result = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        # 0.80 * 0.55 = 0.44 → effective_min = max(0.44, 0.10) = 0.44
        assert 1 in result  # 0.80 >= 0.44
        assert 2 in result  # 0.50 >= 0.44
        assert 3 not in result  # 0.30 < 0.44
        assert 4 not in result  # 0.10 < 0.44

    def test_absolute_floor_overrides_low_ratio(self):
        """Absolute floor kicks in when ratio would go too low."""
        scores = {1: 0.18, 2: 0.12, 3: 0.05}
        result = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        # 0.18 * 0.55 = 0.099 → effective_min = max(0.099, 0.10) = 0.10
        assert 1 in result  # 0.18 >= 0.10
        assert 2 in result  # 0.12 >= 0.10
        assert 3 not in result  # 0.05 < 0.10

    def test_single_result_above_floor(self):
        """A single strong result is returned."""
        scores = {1: 0.70}
        result = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        assert result == {1: 0.70}

    def test_single_result_below_floor(self):
        """A single weak result is rejected."""
        scores = {1: 0.05}
        result = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        assert result == {}

    def test_tight_cluster_all_kept(self):
        """Tightly clustered scores should all survive."""
        scores = {1: 0.65, 2: 0.62, 3: 0.58, 4: 0.55}
        result = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        # 0.65 * 0.55 = 0.3575 → all above that
        assert len(result) == 4

    def test_wide_spread_keeps_top(self):
        """Wide score spread filters aggressively."""
        scores = {1: 0.90, 2: 0.60, 3: 0.40, 4: 0.20, 5: 0.05}
        result = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        # 0.90 * 0.55 = 0.495 → effective_min = 0.495
        assert set(result.keys()) == {1, 2}  # only 0.90 and 0.60

    def test_preserves_scores(self):
        """Returned scores are the original values, not modified."""
        scores = {1: 0.75, 2: 0.50}
        result = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        assert result[1] == 0.75
        assert result[2] == 0.50

    def test_defaults(self):
        """Default parameters work sensibly."""
        scores = {1: 0.60, 2: 0.40, 3: 0.20, 4: 0.05}
        result = apply_adaptive_threshold(scores)
        # defaults: activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        # 0.60 * 0.55 = 0.33 → effective_min = 0.33
        assert 1 in result
        assert 2 in result
        assert 3 not in result
        assert 4 not in result

    def test_backward_compat_vs_flat(self):
        """Adaptive is strictly better than flat 0.35 for small packs with high scores."""
        scores = {1: 0.80, 2: 0.70, 3: 0.50, 4: 0.30}
        # Old flat: min_score=0.35 keeps {1, 2, 3}
        flat = {cid: s for cid, s in scores.items() if s >= 0.35}
        assert set(flat.keys()) == {1, 2, 3}

        # Adaptive: 0.80 * 0.55 = 0.44 → keeps {1, 2, 3}
        adaptive = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        assert set(adaptive.keys()) == {1, 2, 3}

    def test_adaptive_helps_large_pack_compressed_scores(self):
        """For large packs where scores compress, adaptive keeps more than flat 0.35."""
        # Simulated large-pack scenario: all scores lower due to IDF dilution
        scores = {1: 0.40, 2: 0.32, 3: 0.28, 4: 0.23, 5: 0.15}
        # Old flat: min_score=0.35 keeps only {1}
        flat = {cid: s for cid, s in scores.items() if s >= 0.35}
        assert set(flat.keys()) == {1}

        # Adaptive: 0.40 * 0.55 = 0.22 → keeps {1, 2, 3, 4}
        adaptive = apply_adaptive_threshold(
            scores, activation_floor=0.15, score_ratio=0.55, absolute_floor=0.10
        )
        assert set(adaptive.keys()) == {1, 2, 3, 4}
