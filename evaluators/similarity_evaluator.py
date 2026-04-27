# evaluators/similarity_evaluator.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Dict, Any

# 强制离线，防止网络请求
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier
from evaluators.base import BaseEvaluator

SAMPLE_RATE = 16000

# 项目内模型路径（相对于项目根目录）
_PROJECT_ROOT = Path(__file__).parent.parent
_ECAPA_DIR = _PROJECT_ROOT / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"


class SimilarityEvaluator(BaseEvaluator):
    """使用 ECAPA-TDNN 计算合成音频与 ref 均值 embedding 的 cosine 相似度。"""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.classifier = EncoderClassifier.from_hparams(
            source=str(_ECAPA_DIR),
            savedir=str(_ECAPA_DIR),
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
        """预计算 ref 均值 embedding。ref_paths 为空时跳过，sim_score 将返回 None。"""
        if not ref_paths:
            self.ref_embedding = None
            return
        embeddings = []
        for p in ref_paths:
            audio = self._load_audio(p)
            embeddings.append(self._get_embedding(audio))
        self.ref_embedding = torch.stack(embeddings).mean(dim=0)  # (1, D)
        self.ref_embedding = F.normalize(self.ref_embedding, dim=-1)

    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        # 无参考音频时返回 None，由聚合器跳过该维度
        if self.ref_embedding is None:
            return [{"sim_score": None} for _ in audio_paths]
        results = []
        for path in audio_paths:
            audio = self._load_audio(path)
            emb = self._get_embedding(audio)
            sim = F.cosine_similarity(emb, self.ref_embedding, dim=-1).item()
            sim = float(max(0.0, min(5.0, ((sim + 1.0) / 2.0) * 5.0)))  # [-1,1] → [0,5]
            results.append({"sim_score": sim})
        return results
