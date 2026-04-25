# pipeline/filter.py
from typing import Dict, Any, List


class BadSampleFilter:
    """
    双重规则过滤：
    - 规则A：任意单项维度不达标
    - 规则B：加权综合分不达标
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.th = thresholds

    def filter_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        reasons: List[str] = []

        # 规则A
        if row.get("mos_score", 5.0) < self.th["mos"]:
            reasons.append("mos")
        if row.get("sim_score", 1.0) < self.th["sim"]:
            reasons.append("sim")
        if row.get("cer", 0.0) > self.th["cer"]:
            reasons.append("cer")
        if row.get("prosody_score", 1.0) < self.th["prosody"]:
            reasons.append("prosody")

        # 规则B
        if row.get("weighted_score", 5.0) < self.th["weighted"]:
            reasons.append("weighted")

        row["is_bad"] = len(reasons) > 0
        row["bad_reason"] = ",".join(reasons)
        return row

    def filter_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.filter_row(r) for r in rows]
