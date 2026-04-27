# reporter/html_reporter.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from pathlib import Path
from typing import List, Dict, Any, Optional

DIMS = ["mos_score", "sim_score", "cer", "sent_acc", "prosody_score", "weighted_score"]
DIM_LABELS = ["MOS", "相似度(0-5)", "CER", "句准", "韵律", "综合分"]


def _fmt(val, fmt=".3f") -> str:
    """将数值格式化为字符串；None / NaN 显示为 N/A。"""
    if val is None:
        return "N/A"
    try:
        if pd.isna(val):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return format(val, fmt)


def _safe_mean(series: pd.Series) -> Optional[float]:
    """忽略 None/NaN 后求均值；全为空时返回 None。"""
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return float(valid.mean()) if len(valid) > 0 else None


class HTMLReporter:
    """生成包含图表和差样本列表的自包含 plotly HTML 报告。"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rows: List[Dict[str, Any]]):
        df = pd.DataFrame(rows)
        html_parts = [
            "<html><head><meta charset='utf-8'>",
            "<title>TTS 音质评测报告</title></head><body>",
            "<h1>TTS 系统音质评测报告</h1>",
        ]

        html_parts.append(self._summary_table(df))
        html_parts.append(self._radar_chart(df))
        html_parts.append(self._histogram_charts(df))
        html_parts.append(self._bad_samples_table(df))

        html_parts.append("</body></html>")
        out_path = self.output_dir / "report.html"
        out_path.write_text("\n".join(html_parts), encoding="utf-8")

    def _summary_table(self, df: pd.DataFrame) -> str:
        systems = df["system"].unique()
        rows_html = ""
        for sys in sorted(systems):
            sub = df[df["system"] == sys]
            row_vals = "".join(
                f"<td>{_fmt(_safe_mean(sub[d]))}</td>"
                for d in DIMS if d in sub.columns
            )
            rows_html += f"<tr><td><b>{sys}</b></td>{row_vals}</tr>"
        header = "".join(f"<th>{l}</th>" for l in DIM_LABELS)
        return (
            f"<h2>系统均值对比</h2>"
            f"<table border='1' cellpadding='5'>"
            f"<tr><th>系统</th>{header}</tr>{rows_html}</table>"
        )

    def _radar_chart(self, df: pd.DataFrame) -> str:
        categories = ["MOS/5", "相似度/5", "1-CER", "韵律", "综合分/5"]
        fig = go.Figure()
        for sys in sorted(df["system"].unique()):
            sub = df[df["system"] == sys]

            def safe(col, transform=lambda x: x):
                m = _safe_mean(sub[col]) if col in sub.columns else None
                return transform(m) if m is not None else 0.0

            values = [
                safe("mos_score", lambda x: x / 5.0),
                safe("sim_score", lambda x: x / 5.0),
                safe("cer", lambda x: 1.0 - x),
                safe("prosody_score"),
                safe("weighted_score", lambda x: x / 5.0),
            ]
            values.append(values[0])  # 闭合雷达图
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=sys,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="系统雷达图对比",
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    def _histogram_charts(self, df: pd.DataFrame) -> str:
        parts = ["<h2>各维度分数分布</h2>"]
        for dim, label in zip(DIMS, DIM_LABELS):
            if dim not in df.columns:
                continue
            col = pd.to_numeric(df[dim], errors="coerce")
            if col.dropna().empty:
                parts.append(f"<p>{label}：无数据（未提供参考音频）</p>")
                continue
            plot_df = df.copy()
            plot_df[dim] = col
            fig = px.histogram(
                plot_df.dropna(subset=[dim]),
                x=dim, color="system", barmode="overlay",
                title=f"{label} 分布", nbins=20,
            )
            parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
        return "\n".join(parts)

    def _bad_samples_table(self, df: pd.DataFrame) -> str:
        bad = df[df["is_bad"] == True]
        if bad.empty:
            return "<h2>差样本列表</h2><p>无差样本，全部通过。</p>"
        rows_html = ""
        for _, row in bad.iterrows():
            highlight = "background-color:#ffe0e0;"
            rows_html += (
                f"<tr style='{highlight}'>"
                f"<td>{row['file']}</td><td>{row['system']}</td>"
                f"<td>{_fmt(row.get('mos_score'))}</td>"
                f"<td>{_fmt(row.get('sim_score'))}</td>"
                f"<td>{_fmt(row.get('cer'))}</td>"
                f"<td>{_fmt(row.get('sent_acc'))}</td>"
                f"<td>{_fmt(row.get('prosody_score'))}</td>"
                f"<td>{_fmt(row.get('weighted_score'))}</td>"
                f"<td>{row.get('bad_reason', '')}</td>"
                f"</tr>"
            )
        return (
            "<h2>差样本列表（需人工复核）</h2>"
            "<table border='1' cellpadding='5'>"
            "<tr><th>文件</th><th>系统</th><th>MOS</th><th>相似度(0-5)</th>"
            "<th>CER</th><th>句准</th><th>韵律</th><th>综合分</th><th>触发原因</th></tr>"
            f"{rows_html}</table>"
        )
