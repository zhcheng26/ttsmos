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

        # 规则A（值为 None 表示该维度不可用，跳过）
        mos = row.get("mos_score")
        if mos is not None and mos < self.th["mos"]:
            reasons.append("mos")
        sim = row.get("sim_score")
        if sim is not None and sim < self.th["sim"]:
            reasons.append("sim")
        cer = row.get("cer")
        if cer is not None and cer > self.th["cer"]:
            reasons.append("cer")
        prosody = row.get("prosody_score")
        if prosody is not None and prosody < self.th["prosody"]:
            reasons.append("prosody")

        # 规则B
        weighted = row.get("weighted_score")
        if weighted is not None and weighted < self.th["weighted"]:
            reasons.append("weighted")

        row["is_bad"] = len(reasons) > 0
        row["bad_reason"] = ",".join(reasons)
        return row

    def filter_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.filter_row(r) for r in rows]
