# reporter/csv_reporter.py
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

COLUMN_ORDER = [
    "file", "system", "mos_score", "sim_score", "cer", "sent_acc",
    "prosody_score", "weighted_score", "is_bad", "bad_reason",
]


class CSVReporter:
    """将评测结果写入 results.csv 和 bad_samples.csv。"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rows: List[Dict[str, Any]]):
        df = pd.DataFrame(rows)
        for col in COLUMN_ORDER:
            if col not in df.columns:
                df[col] = None
        df = df[COLUMN_ORDER]
        # 保留文件名前导零，强制为字符串
        df["file"] = df["file"].astype(str)

        df.to_csv(self.output_dir / "results.csv", index=False)

        bad_df = df[df["is_bad"] == True]
        bad_df.to_csv(self.output_dir / "bad_samples.csv", index=False)
