# evaluators/mos_evaluator.py
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Any

from evaluators.base import BaseEvaluator

SAMPLE_RATE = 16000
CLIP_LEN = 9  # DNSMOS P.835 要求 9 秒输入


class MOSEvaluator(BaseEvaluator):
    """使用 DNSMOS P.835 ONNX 模型评测音质，输出 OVRL MOS 分数。"""

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(device)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if str(self.device) != "cpu"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _load_and_pad(self, path: Path) -> np.ndarray:
        """加载音频，重采样到 16kHz，裁剪/补零到 9 秒，归一化。"""
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        target_len = CLIP_LEN * SAMPLE_RATE
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        # 归一化
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        return audio.astype(np.float32)

    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for path in audio_paths:
            audio = self._load_and_pad(path)
            inp = audio[np.newaxis, :]  # (1, 144000)
            out = self.session.run(None, {self.input_name: inp})
            # out[0] shape: (1, 3) -> [SIG, BAK, OVRL]
            ovrl = float(out[0][0][2])
            ovrl = float(np.clip(ovrl, 1.0, 5.0))
            results.append({"mos_score": ovrl})
        return results
