#!/usr/bin/env python3
# app.py — TTS 音质评测 Gradio Web UI
"""
启动: python app.py
"""
import os

# 禁止所有 HuggingFace / Transformers 网络请求，强制使用本地缓存
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import yaml
import gradio as gr
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, Dict, List

from pipeline.loader import AudioPairLoader
from pipeline.aggregator import ResultAggregator
from pipeline.filter import BadSampleFilter
from evaluators.mos_evaluator import MOSEvaluator
from evaluators.similarity_evaluator import SimilarityEvaluator
from evaluators.asr_evaluator import ASREvaluator
from evaluators.prosody_evaluator import ProsodyEvaluator
from reporter.csv_reporter import CSVReporter
from reporter.html_reporter import HTMLReporter


# ── 全局模型缓存（避免重复加载）──────────────────────────────────────────────
_models = {}


def load_models(cfg: dict):
    global _models
    device = cfg.get("device", "cpu")
    key = (cfg["dnsmos"]["model_path"], cfg["asr"]["model"], device)
    if key not in _models:
        _models[key] = {
            "mos": MOSEvaluator(model_path=cfg["dnsmos"]["model_path"], device=device),
            "sim": SimilarityEvaluator(device=device),
            "asr": ASREvaluator(model_name=cfg["asr"]["model"], device=device),
            "prosody": ProsodyEvaluator(device=device),
        }
    return _models[key]


def resolve_target_text(
    raw: str, file_ids: List[str]
) -> Union[None, str, Dict[str, Optional[str]]]:
    """解析目标文本输入，返回三种形式之一：
    - None：留空，跳过 CER/句准
    - str：所有文件使用同一文本
    - dict[file_id, str|None]：目录模式，按文件名逐一匹配 .txt
    """
    t = (raw or "").strip()
    if not t:
        return None

    p = Path(t)
    if p.is_dir():
        text_map: Dict[str, Optional[str]] = {}
        for fid in file_ids:
            txt_file = p / f"{fid}.txt"
            text_map[fid] = txt_file.read_text(encoding="utf-8").strip() if txt_file.exists() else None
        return text_map

    return t  # 单一文本字符串


def get_batch_texts(
    text_src: Union[None, str, Dict[str, Optional[str]]],
    batch_ids: List[str],
) -> Union[None, str, List[Optional[str]]]:
    """将文本来源转换为当前 batch 所需格式。"""
    if text_src is None or isinstance(text_src, str):
        return text_src
    return [text_src.get(fid) for fid in batch_ids]


# ── 核心评测逻辑 ─────────────────────────────────────────────────────────────

def run_evaluation(
    ref_dir, sysA_dir, sysB_dir,
    target_text_input,
    config_path,
    output_dir,
    progress=gr.Progress(),
):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        progress(0.05, desc="加载音频配对...")
        loader = AudioPairLoader(ref_dir=ref_dir or None, sysA_dir=sysA_dir, sysB_dir=sysB_dir)
        pairs = loader.get_pairs()
        ref_paths = loader.get_ref_paths()
        file_ids = [p["file_id"] for p in pairs]

        if not pairs:
            return None, None, "❌ 未找到匹配的音频对，请检查目录和文件名。", None

        text_src = resolve_target_text(target_text_input, file_ids)
        if text_src is None:
            text_desc = "未提供目标文本，跳过 CER/句准"
        elif isinstance(text_src, dict):
            n_matched = sum(1 for v in text_src.values() if v)
            text_desc = f"目录模式：{n_matched}/{len(file_ids)} 个文件匹配到文本"
        else:
            text_desc = f"统一文本：{text_src[:20]}..."

        progress(0.10, desc=f"找到 {len(pairs)} 对音频，初始化模型...")
        evs = load_models(cfg)

        ref_desc = "构建参考 embedding 和韵律统计..." if ref_paths else "未提供参考音频，跳过 sim/prosody..."
        progress(0.20, desc=ref_desc)
        evs["sim"].build_ref_embedding(ref_paths)
        evs["prosody"].build_ref_stats(ref_paths)

        aggregator = ResultAggregator(weights=cfg["weights"])
        sample_filter = BadSampleFilter(thresholds=cfg["thresholds"])
        batch_size = cfg.get("batch_size", 8)

        all_rows = []
        systems = [("sysA", [p["sysA_path"] for p in pairs]),
                   ("sysB", [p["sysB_path"] for p in pairs])]

        for sys_idx, (sys_name, audio_paths) in enumerate(systems):
            base_progress = 0.30 + sys_idx * 0.30
            for batch_start in range(0, len(audio_paths), batch_size):
                batch_paths = audio_paths[batch_start: batch_start + batch_size]
                batch_ids = file_ids[batch_start: batch_start + batch_size]
                frac = batch_start / len(audio_paths)
                progress(
                    base_progress + frac * 0.28,
                    desc=f"评测 {sys_name} ({batch_start+len(batch_paths)}/{len(audio_paths)})...",
                )

                batch_texts = get_batch_texts(text_src, batch_ids)
                with ThreadPoolExecutor(max_workers=3) as ex:
                    fut_mos = ex.submit(evs["mos"].evaluate_batch, batch_paths)
                    fut_sim = ex.submit(evs["sim"].evaluate_batch, batch_paths)
                    fut_asr = ex.submit(evs["asr"].evaluate_batch, batch_paths,
                                        target_text=batch_texts)
                mos_r = fut_mos.result()
                sim_r = fut_sim.result()
                asr_r = fut_asr.result()
                pro_r = evs["prosody"].evaluate_batch(batch_paths)

                for i, fid in enumerate(batch_ids):
                    row = {
                        "file": fid, "system": sys_name,
                        "mos_score": mos_r[i]["mos_score"],
                        "sim_score": sim_r[i]["sim_score"],
                        "cer": asr_r[i]["cer"],
                        "sent_acc": asr_r[i]["sent_acc"],
                        "prosody_score": pro_r[i]["prosody_score"],
                    }
                    row = aggregator.aggregate_row(row)
                    row = sample_filter.filter_row(row)
                    all_rows.append(row)

        progress(0.85, desc="生成报告...")
        out = Path(output_dir)
        CSVReporter(out).write(all_rows)
        HTMLReporter(out).write(all_rows)

        df = pd.DataFrame(all_rows)
        display_cols = ["file", "system", "mos_score", "sim_score",
                        "cer", "sent_acc", "prosody_score", "weighted_score", "is_bad", "bad_reason"]
        df_display = df[[c for c in display_cols if c in df.columns]]

        score_cols = [c for c in ["mos_score", "sim_score", "cer", "sent_acc",
                                   "prosody_score", "weighted_score"] if c in df.columns]
        summary = df.groupby("system")[score_cols].mean().round(4).reset_index()

        bad_count = int(df["is_bad"].sum())
        html_path = out / "report.html"
        progress(1.0, desc="完成！")

        status = (
            f"✅ 评测完成！共 {len(all_rows)} 条样本，差样本 {bad_count} 条。\n"
            f"文本模式：{text_desc}\n"
            f"报告已保存至: {html_path}"
        )
        return df_display, summary, status, str(html_path)

    except Exception as e:
        import traceback
        return None, None, f"❌ 错误: {e}\n{traceback.format_exc()}", None


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="TTS 音质评测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🎙️ TTS 音质评测系统
多维度对比两个 TTS 系统的音频质量：MOS、说话人相似度、CER、句准、韵律
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入配置")
                ref_dir = gr.Textbox(
                    label="原声目录 (ref_dir)【可选】",
                    placeholder="留空则跳过 sim/prosody 维度",
                    value="",
                )
                sysA_dir = gr.Textbox(
                    label="系统A 目录 (sysA_dir)",
                    placeholder="demo/data/sysA",
                    value="demo/data/sysA",
                )
                sysB_dir = gr.Textbox(
                    label="系统B 目录 (sysB_dir)",
                    placeholder="demo/data/sysB",
                    value="demo/data/sysB",
                )
                target_text = gr.Textbox(
                    label="目标文本【可选】",
                    placeholder="留空跳过CER/句准 | 输入文本统一使用 | 输入目录路径按文件名逐一匹配.txt",
                    value="",
                    lines=3,
                )
                config_path = gr.Textbox(
                    label="配置文件路径",
                    value="config.yaml",
                )
                output_dir = gr.Textbox(
                    label="输出目录",
                    value="results/",
                )
                run_btn = gr.Button("▶ 开始评测", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 评测结果")
                status_box = gr.Textbox(
                    label="状态",
                    interactive=False,
                    lines=4,
                )
                with gr.Tabs():
                    with gr.Tab("📊 全量结果"):
                        results_table = gr.DataFrame(
                            label="逐样本评测结果",
                            interactive=False,
                        )
                    with gr.Tab("📈 系统对比"):
                        summary_table = gr.DataFrame(
                            label="系统均值对比",
                            interactive=False,
                        )
                    with gr.Tab("📄 HTML 报告"):
                        report_path_box = gr.Textbox(
                            label="报告文件路径",
                            interactive=False,
                        )
                        gr.Markdown(
                            "报告生成后，用浏览器打开上方路径查看完整交互式图表。"
                        )

        run_btn.click(
            fn=run_evaluation,
            inputs=[ref_dir, sysA_dir, sysB_dir,
                    target_text, config_path, output_dir],
            outputs=[results_table, summary_table, status_box, report_path_box],
        )

        gr.Markdown("""
---
**维度说明**
| 字段 | 含义 | 越好 |
|------|------|------|
| mos_score | DNSMOS 音质分 (1-5) | 越高越好，需 > 4.2 |
| sim_score | 说话人相似度 (0-5) | 越高越好，需 > 4.2 |
| cer | 字错率 (0-1) | 越低越好 |
| sent_acc | 句准 (0或1) | 1为完全正确 |
| prosody_score | 韵律相似度 (0-1) | 越高越好 |
| weighted_score | 综合加权分 (0-5) | 越高越好 |
| is_bad | 是否需人工复核 | False 为优 |

**目标文本三种模式**
| 输入方式 | 效果 |
|----------|------|
| 留空 | 跳过 CER 和句准，结果显示 N/A |
| 直接输入文本 | 所有音频使用同一参考文本 |
| 输入目录路径 | 按音频文件名匹配同名 .txt 文件（如 sample_001.txt）|
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
