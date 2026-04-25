# tests/test_reporters.py
import pytest
import pandas as pd
from pathlib import Path
from reporter.csv_reporter import CSVReporter


SAMPLE_ROWS = [
    {"file": "sample_001", "system": "sysA", "mos_score": 3.8, "sim_score": 0.82,
     "cer": 0.05, "prosody_score": 0.75, "weighted_score": 3.6,
     "is_bad": False, "bad_reason": ""},
    {"file": "sample_002", "system": "sysA", "mos_score": 2.1, "sim_score": 0.61,
     "cer": 0.22, "prosody_score": 0.48, "weighted_score": 2.1,
     "is_bad": True, "bad_reason": "mos,sim,cer"},
    {"file": "sample_001", "system": "sysB", "mos_score": 4.0, "sim_score": 0.85,
     "cer": 0.03, "prosody_score": 0.80, "weighted_score": 4.1,
     "is_bad": False, "bad_reason": ""},
]


def test_csv_reporter_creates_results(tmp_path):
    reporter = CSVReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    assert (tmp_path / "results.csv").exists()
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 3
    assert "mos_score" in df.columns


def test_csv_reporter_creates_bad_samples(tmp_path):
    reporter = CSVReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    assert (tmp_path / "bad_samples.csv").exists()
    df = pd.read_csv(tmp_path / "bad_samples.csv")
    assert len(df) == 1
    assert df.iloc[0]["file"] == "sample_002"


def test_csv_reporter_column_order(tmp_path):
    reporter = CSVReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    df = pd.read_csv(tmp_path / "results.csv")
    expected_cols = ["file", "system", "mos_score", "sim_score", "cer",
                     "prosody_score", "weighted_score", "is_bad", "bad_reason"]
    assert list(df.columns) == expected_cols


def test_html_reporter_creates_file(tmp_path):
    from reporter.html_reporter import HTMLReporter
    reporter = HTMLReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    html_path = tmp_path / "report.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "sysA" in content
    assert "sysB" in content


def test_html_reporter_contains_bad_samples(tmp_path):
    from reporter.html_reporter import HTMLReporter
    reporter = HTMLReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    content = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "sample_002" in content
