# pipeline/aggregator.py
from typing import Dict, Any


class ResultAggregator:
    """
    根据权重计算加权综合分。
    公式: weighted_score = w_mos*(mos/5) + w_sim*sim + w_cer*(1-cer) + w_prosody*prosody
    归一化到 [0, 5] 保持与 MOS 量纲一致。
    """

    def __init__(self, weights: Dict[str, float]):
        self.w = weights
        total = sum(weights.values())
        self.w_norm = {k: v / total for k, v in weights.items()}

    def compute_weighted_score(
        self,
        mos_score: float,
        sim_score: float,
        cer: float,
        prosody_score: float,
    ) -> float:
        """
        各维度先归一化到 [0,1]（MOS 除以 5），再加权，最后乘以 5 恢复到 [0,5] 量纲。
        """
        norm = {
            "mos": mos_score / 5.0,
            "sim": sim_score,
            "cer": 1.0 - min(cer, 1.0),
            "prosody": prosody_score,
        }
        score_01 = sum(self.w_norm[k] * norm[k] for k in norm)
        return round(score_01 * 5.0, 4)

    def aggregate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """接受含 mos_score/sim_score/cer/prosody_score 的 dict，追加 weighted_score。"""
        row = dict(row)
        row["weighted_score"] = self.compute_weighted_score(
            mos_score=row.get("mos_score", 0.0),
            sim_score=row.get("sim_score", 0.0),
            cer=row.get("cer", 0.0),
            prosody_score=row.get("prosody_score", 0.0),
        )
        return row
