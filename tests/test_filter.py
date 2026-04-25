# tests/test_filter.py
import pytest
from pipeline.filter import BadSampleFilter

THRESHOLDS = {
    "mos": 3.0,
    "sim": 0.7,
    "cer": 0.15,
    "prosody": 0.5,
    "weighted": 2.5,
}


def make_row(mos=4.0, sim=0.8, cer=0.05, prosody=0.7, weighted=4.0):
    return {
        "mos_score": mos,
        "sim_score": sim,
        "cer": cer,
        "prosody_score": prosody,
        "weighted_score": weighted,
    }


def test_good_sample_not_flagged():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row()
    result = f.filter_row(row)
    assert result["is_bad"] is False
    assert result["bad_reason"] == ""


def test_rule_a_mos_triggers():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(mos=2.5)
    result = f.filter_row(row)
    assert result["is_bad"] is True
    assert "mos" in result["bad_reason"]


def test_rule_a_sim_triggers():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(sim=0.5)
    result = f.filter_row(row)
    assert result["is_bad"] is True
    assert "sim" in result["bad_reason"]


def test_rule_a_cer_triggers():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(cer=0.3)
    result = f.filter_row(row)
    assert result["is_bad"] is True
    assert "cer" in result["bad_reason"]


def test_rule_b_weighted_triggers():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(weighted=2.0)
    result = f.filter_row(row)
    assert result["is_bad"] is True
    assert "weighted" in result["bad_reason"]


def test_multiple_reasons_recorded():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(mos=2.0, sim=0.5)
    result = f.filter_row(row)
    assert "mos" in result["bad_reason"]
    assert "sim" in result["bad_reason"]
