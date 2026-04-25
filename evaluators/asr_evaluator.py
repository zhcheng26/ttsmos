# evaluators/asr_evaluator.py
import whisper
from jiwer import cer as compute_cer
from pathlib import Path
from typing import List, Dict, Any

from evaluators.base import BaseEvaluator


class ASREvaluator(BaseEvaluator):
    """使用 Whisper 转写音频，与目标文本对比计算 CER（字错率）。"""

    def __init__(self, model_name: str = "medium", device: str = "cuda"):
        super().__init__(device)
        self.model = whisper.load_model(model_name, device=str(self.device))

    def evaluate_batch(
        self, audio_paths: List[Path], target_text: str = "", **kwargs
    ) -> List[Dict[str, Any]]:
        results = []
        for path in audio_paths:
            result = self.model.transcribe(str(path), language="zh")
            hypothesis = result["text"].strip()
            reference = target_text.strip()
            if not reference:
                cer_score = 0.0
            else:
                cer_score = float(compute_cer(reference, hypothesis))
                cer_score = min(cer_score, 1.0)
            results.append({"cer": cer_score, "transcription": hypothesis})
        return results
