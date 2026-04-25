# evaluators/similarity_evaluator.py
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Dict, Any

try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier
from evaluators.base import BaseEvaluator

SAMPLE_RATE = 16000


class SimilarityEvaluator(BaseEvaluator):
    """使用 ECAPA-TDNN 计算合成音频与 ref 均值 embedding 的 cosine 相似度。"""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )
        self.ref_embedding: torch.Tensor = None

    def _load_audio(self, path: Path) -> torch.Tensor:
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        return torch.tensor(audio, dtype=torch.float32)  # (T,)

    def _get_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (T,) → embedding: (1, D)"""
        audio = audio.unsqueeze(0).to(self.device)  # (1, T)
        with torch.no_grad():
            emb = self.classifier.encode_batch(audio)  # (1, 1, D) or (1, D)
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        return F.normalize(emb, dim=-1)

    def build_ref_embedding(self, ref_paths: List[Path]):
        """预计算 ref 均值 embedding，在评测前必须调用一次。"""
        embeddings = []
        for p in ref_paths:
            audio = self._load_audio(p)
            embeddings.append(self._get_embedding(audio))
        self.ref_embedding = torch.stack(embeddings).mean(dim=0)  # (1, D)
        self.ref_embedding = F.normalize(self.ref_embedding, dim=-1)

    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        if self.ref_embedding is None:
            raise RuntimeError("请先调用 build_ref_embedding() 构建参考 embedding")
        results = []
        for path in audio_paths:
            audio = self._load_audio(path)
            emb = self._get_embedding(audio)
            sim = F.cosine_similarity(emb, self.ref_embedding, dim=-1).item()
            sim = float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))  # [-1,1] → [0,1]
            results.append({"sim_score": sim})
        return results
