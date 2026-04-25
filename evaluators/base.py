# evaluators/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import torch


class BaseEvaluator(ABC):
    """所有 Evaluator 的基类，定义统一接口。"""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        """
        对一批音频文件进行评测。
        Returns: 与 audio_paths 等长的结果列表，每个元素是含评分字段的 dict。
        """
        ...

    def evaluate(self, audio_path: Path, **kwargs) -> Dict[str, Any]:
        """单条评测，调用 evaluate_batch 实现。"""
        return self.evaluate_batch([audio_path], **kwargs)[0]
