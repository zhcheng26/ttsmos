# pipeline/aggregator.py
from typing import Dict, Any, Optional


class ResultAggregator:
    """
    根据权重计算加权综合分。
    公式: weighted_score = w_mos*(mos/5) + w_sim*(sim/5) + w_cer*(1-cer) + w_prosody*prosody
    归一化到 [0, 5] 保持与 MOS 量纲一致。

    若某维度值为 None，则从权重中剔除该维度后重新归一化。
    """

    def __init__(self, weights: Dict[str, float]):
        self.w = weights

    def compute_weighted_score(
        self,
        mos_score: Optional[float],
        sim_score: Optional[float],
        cer: Optional[float],
        prosody_score: Optional[float],
    ) -> Optional[float]:
        candidates = {}
        if mos_score is not None:
            candidates["mos"] = (mos_score / 5.0, self.w.get("mos", 0.3))
        if cer is not None:
            candidates["cer"] = (1.0 - min(cer, 1.0), self.w.get("cer", 0.3))
        if sim_score is not None:
            candidates["sim"] = (sim_score / 5.0, self.w.get("sim", 0.3))  # [0,5] → [0,1]
        if prosody_score is not None:
            candidates["prosody"] = (prosody_score, self.w.get("prosody", 0.1))

        if not candidates:
            return None
        total_w = sum(w for _, w in candidates.values())
        score_01 = sum(v * w for v, w in candidates.values()) / total_w
        return round(score_01 * 5.0, 4)

    def aggregate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """接受含 mos_score/sim_score/cer/prosody_score 的 dict，追加 weighted_score。"""
        row = dict(row)
        row["weighted_score"] = self.compute_weighted_score(
            mos_score=row.get("mos_score"),
            sim_score=row.get("sim_score"),
            cer=row.get("cer"),
            prosody_score=row.get("prosody_score"),
        )
        return row
