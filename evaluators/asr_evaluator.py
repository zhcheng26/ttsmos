# evaluators/asr_evaluator.py
import os
import re
import whisper
from jiwer import cer as compute_cer
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from evaluators.base import BaseEvaluator

_PROJECT_ROOT = Path(__file__).parent.parent
_WHISPER_DIR = _PROJECT_ROOT / "models" / "whisper"

os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / "models" / "_xdg_cache"))

_PUNCT = re.compile(r"[\s\u3000\uff0c\u3002\uff01\uff1f\u201c\u201d\u2018\u2019，。！？""''、；：]+")


def _normalize(text: str) -> str:
    """去除空白与常见标点后比较，提升句准鲁棒性。"""
    return _PUNCT.sub("", text.strip())


class ASREvaluator(BaseEvaluator):
    """使用 Whisper 转写音频，计算 CER（字错率）和句准（Sentence Accuracy）。

    target_text 支持三种形式：
    - None 或 ""：跳过 CER/sent_acc，结果返回 None
    - str：所有音频使用同一参考文本
    - List[str | None]：与 audio_paths 等长，每条音频对应各自参考文本
    """

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
        self,
        audio_paths: List[Path],
        target_text: Union[None, str, List[Optional[str]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, path in enumerate(audio_paths):
            # 确定当前文件的参考文本
            if target_text is None:
                ref = None
            elif isinstance(target_text, list):
                ref = target_text[i]
                if ref is not None:
                    ref = ref.strip() or None
            else:
                ref = target_text.strip() or None

            result = self.model.transcribe(str(path), language="zh")
            hyp = result["text"].strip()

            if ref is None:
                results.append({"cer": None, "sent_acc": None, "transcription": hyp})
            else:
                cer_score = min(float(compute_cer(ref, hyp)), 1.0)
                sent_acc = 1.0 if _normalize(hyp) == _normalize(ref) else 0.0
                results.append({
                    "cer": cer_score,
                    "sent_acc": sent_acc,
                    "transcription": hyp,
                })
        return results
