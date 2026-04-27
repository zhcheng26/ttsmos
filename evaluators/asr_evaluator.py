# evaluators/asr_evaluator.py
import os
import whisper
from jiwer import cer as compute_cer
from pathlib import Path
from typing import List, Dict, Any

from evaluators.base import BaseEvaluator

_PROJECT_ROOT = Path(__file__).parent.parent
_WHISPER_DIR = _PROJECT_ROOT / "models" / "whisper"

# 将项目内 whisper 模型目录注入到 whisper 的搜索路径
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / "models" / "_xdg_cache"))


class ASREvaluator(BaseEvaluator):
    """使用 Whisper 转写音频，与目标文本对比计算 CER（字错率）。"""

    def __init__(self, model_name: str = "medium", device: str = "cuda"):
        super().__init__(device)
        model_file = _WHISPER_DIR / f"{model_name}.pt"
        if model_file.exists():
            self.model = whisper.load_model(str(model_file), device=str(self.device))
        else:
            raise FileNotFoundError(
                f"Whisper 模型文件不存在: {model_file}\n"
                f"请将 {model_name}.pt 放到 models/whisper/ 目录下。"
            )

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
