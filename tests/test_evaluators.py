# tests/test_evaluators.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock


def make_wav(path, sr=16000, duration=2.0, freq=440):
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)
    return path


# ── MOSEvaluator ────────────────────────────────────────────────────────────

def test_mos_evaluator_returns_score(tmp_path):
    """MOSEvaluator.evaluate_batch 应返回含 mos_score 的 dict 列表"""
    wav = make_wav(tmp_path / "test.wav")

    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[3.5, 4.0, 3.8]])]
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    with patch("evaluators.mos_evaluator.ort.InferenceSession", return_value=mock_session):
        from evaluators.mos_evaluator import MOSEvaluator
        ev = MOSEvaluator(model_path="fake.onnx", device="cpu")
        results = ev.evaluate_batch([wav])

    assert len(results) == 1
    assert "mos_score" in results[0]
    assert 1.0 <= results[0]["mos_score"] <= 5.0


def test_mos_evaluator_batch(tmp_path):
    """批量评测应返回与输入等长的列表"""
    wavs = [make_wav(tmp_path / f"t{i}.wav", freq=440 + i * 50) for i in range(4)]

    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[3.5, 4.0, 3.8]])]
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    with patch("evaluators.mos_evaluator.ort.InferenceSession", return_value=mock_session):
        from evaluators.mos_evaluator import MOSEvaluator
        ev = MOSEvaluator(model_path="fake.onnx", device="cpu")
        results = ev.evaluate_batch(wavs)

    assert len(results) == 4


# ── SimilarityEvaluator ─────────────────────────────────────────────────────

def test_similarity_evaluator_returns_score(tmp_path):
    """SimilarityEvaluator 应返回 0-1 之间的 sim_score"""
    wav = make_wav(tmp_path / "syn.wav")
    ref1 = make_wav(tmp_path / "ref1.wav", freq=300)
    ref2 = make_wav(tmp_path / "ref2.wav", freq=320)

    import torch
    fake_embedding = torch.randn(1, 192)

    mock_classifier = MagicMock()
    mock_classifier.encode_batch.return_value = fake_embedding

    with patch("evaluators.similarity_evaluator.EncoderClassifier.from_hparams",
               return_value=mock_classifier):
        from evaluators.similarity_evaluator import SimilarityEvaluator
        ev = SimilarityEvaluator(device="cpu")
        ev.build_ref_embedding([ref1, ref2])
        results = ev.evaluate_batch([wav])

    assert len(results) == 1
    assert "sim_score" in results[0]
    assert 0.0 <= results[0]["sim_score"] <= 1.0


# ── ASREvaluator ─────────────────────────────────────────────────────────────

def test_asr_evaluator_returns_cer(tmp_path):
    """ASREvaluator 应返回 cer（0-1 之间的浮点）"""
    wav = make_wav(tmp_path / "asr_test.wav")
    target_text = "你好世界"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "你好世界"}

    with patch("evaluators.asr_evaluator.whisper.load_model", return_value=mock_model):
        from evaluators.asr_evaluator import ASREvaluator
        ev = ASREvaluator(model_name="tiny", device="cpu")
        results = ev.evaluate_batch([wav], target_text=target_text)

    assert len(results) == 1
    assert "cer" in results[0]
    assert results[0]["cer"] == pytest.approx(0.0)


def test_asr_evaluator_nonzero_cer(tmp_path):
    """转写错误时 CER 应大于 0"""
    wav = make_wav(tmp_path / "asr_test2.wav")
    target_text = "你好世界"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "错误文本内容"}

    with patch("evaluators.asr_evaluator.whisper.load_model", return_value=mock_model):
        from evaluators.asr_evaluator import ASREvaluator
        ev = ASREvaluator(model_name="tiny", device="cpu")
        results = ev.evaluate_batch([wav], target_text=target_text)

    assert results[0]["cer"] > 0.0


# ── ProsodyEvaluator ─────────────────────────────────────────────────────────

def test_prosody_evaluator_returns_score(tmp_path):
    """ProsodyEvaluator 应返回 0-1 之间的 prosody_score"""
    ref1 = make_wav(tmp_path / "pref1.wav", freq=200, duration=2.0)
    ref2 = make_wav(tmp_path / "pref2.wav", freq=220, duration=2.0)
    syn = make_wav(tmp_path / "psyn.wav", freq=210, duration=2.0)

    from evaluators.prosody_evaluator import ProsodyEvaluator
    ev = ProsodyEvaluator(device="cpu")
    ev.build_ref_stats([ref1, ref2])
    results = ev.evaluate_batch([syn])

    assert len(results) == 1
    assert "prosody_score" in results[0]
    assert 0.0 <= results[0]["prosody_score"] <= 1.0
