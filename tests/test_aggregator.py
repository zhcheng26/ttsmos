# tests/test_aggregator.py
import pytest
from pipeline.aggregator import ResultAggregator


WEIGHTS = {"mos": 0.3, "sim": 0.3, "cer": 0.3, "prosody": 0.1}


def test_weighted_score_perfect():
    """所有指标满分时，weighted_score 应为 5.0"""
    agg = ResultAggregator(weights=WEIGHTS)
    score = agg.compute_weighted_score(
        mos_score=5.0, sim_score=1.0, cer=0.0, prosody_score=1.0
    )
    assert score == pytest.approx(5.0)


def test_weighted_score_zero_cer_penalty():
    """CER=1.0 时该维度贡献应为 0"""
    agg = ResultAggregator(weights=WEIGHTS)
    score = agg.compute_weighted_score(
        mos_score=5.0, sim_score=1.0, cer=1.0, prosody_score=1.0
    )
    assert score < 5.0


def test_aggregate_row():
    """aggregate_row 返回含 weighted_score 的 dict"""
    agg = ResultAggregator(weights=WEIGHTS)
    row = {"mos_score": 3.5, "sim_score": 0.8, "cer": 0.05, "prosody_score": 0.7}
    result = agg.aggregate_row(row)
    assert "weighted_score" in result
    assert 1.0 <= result["weighted_score"] <= 5.0
