# evaluators/prosody_evaluator.py
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count

from evaluators.base import BaseEvaluator

SAMPLE_RATE = 16000


def _extract_prosody_features(path: str):
    """提取音频的韵律特征（在子进程中运行）。返回 (f0_mean, voiced_ratio)。"""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE,
    )
    if voiced_flag is not None and voiced_flag.any():
        voiced = f0[voiced_flag]
    else:
        voiced = np.array([0.0])
    f0_mean = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
    voiced_ratio = float(voiced_flag.sum() / len(voiced_flag)) if voiced_flag is not None else 0.0
    return f0_mean, voiced_ratio


class ProsodyEvaluator(BaseEvaluator):
    """使用 librosa 提取 F0 和语速特征，计算合成音频与 ref 分布的相似度。"""

    def __init__(self, device: str = "cuda", n_workers: Optional[int] = None):
        super().__init__(device)
        self.n_workers = n_workers or max(1, cpu_count() // 2)
        self.ref_f0_mean: float = 0.0
        self.ref_voiced_ratio: float = 0.0
        self._has_ref: bool = False

    def build_ref_stats(self, ref_paths: List[Path]):
        """预计算 ref 音频的平均 F0 均值和语速。ref_paths 为空时跳过，prosody_score 将返回 None。"""
        if not ref_paths:
            self._has_ref = False
            return
        self._has_ref = True
        path_strs = [str(p) for p in ref_paths]
        with Pool(processes=self.n_workers) as pool:
            feats = pool.map(_extract_prosody_features, path_strs)
        f0_means = [f[0] for f in feats if f[0] > 0]
        voiced_ratios = [f[1] for f in feats]
        self.ref_f0_mean = float(np.mean(f0_means)) if f0_means else 150.0
        self.ref_voiced_ratio = float(np.mean(voiced_ratios))

    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        # 无参考音频时返回 None，由聚合器跳过该维度
        if not self._has_ref:
            return [{"prosody_score": None} for _ in audio_paths]

        path_strs = [str(p) for p in audio_paths]
        with Pool(processes=self.n_workers) as pool:
            feats = pool.map(_extract_prosody_features, path_strs)

        results = []
        for (f0_mean, voiced_ratio) in feats:
            if self.ref_f0_mean > 0 and f0_mean > 0:
                f0_diff = abs(f0_mean - self.ref_f0_mean) / self.ref_f0_mean
                f0_score = float(max(0.0, 1.0 - f0_diff))
            else:
                f0_score = 0.5
            rate_diff = abs(voiced_ratio - self.ref_voiced_ratio)
            rate_score = float(max(0.0, 1.0 - rate_diff))
            prosody_score = (f0_score + rate_score) / 2.0
            results.append({
                "prosody_score": round(prosody_score, 4),
                "f0_mean": round(f0_mean, 2),
                "voiced_ratio": round(voiced_ratio, 4),
            })
        return results
