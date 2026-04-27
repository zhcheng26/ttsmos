#!/usr/bin/env python3
# run_eval.py
"""
TTS 音质评测主入口。

用法:
    python run_eval.py \
        --ref ref_dir/ --sysA sysA_dir/ --sysB sysB_dir/ \
        --text target_text.txt --config config.yaml --output results/
"""
import os
import argparse
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from tqdm import tqdm

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pipeline.loader import AudioPairLoader
from pipeline.aggregator import ResultAggregator
from pipeline.filter import BadSampleFilter
from evaluators.mos_evaluator import MOSEvaluator
from evaluators.similarity_evaluator import SimilarityEvaluator
from evaluators.asr_evaluator import ASREvaluator
from evaluators.prosody_evaluator import ProsodyEvaluator
from reporter.csv_reporter import CSVReporter
from reporter.html_reporter import HTMLReporter


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_batch_texts(
    text_src: Union[None, str, Dict[str, Optional[str]]],
    batch_ids: List[str],
) -> Union[None, str, List[Optional[str]]]:
    if text_src is None or isinstance(text_src, str):
        return text_src
    return [text_src.get(fid) for fid in batch_ids]


def load_text_dir(text_dir: str, file_ids: List[str]) -> Dict[str, Optional[str]]:
    p = Path(text_dir)
    return {
        fid: (p / f"{fid}.txt").read_text(encoding="utf-8").strip()
        if (p / f"{fid}.txt").exists() else None
        for fid in file_ids
    }


def evaluate_system(
    system_name: str,
    audio_paths,
    text_src: Union[None, str, Dict[str, Optional[str]]],
    mos_ev: MOSEvaluator,
    sim_ev: SimilarityEvaluator,
    asr_ev: ASREvaluator,
    prosody_ev: ProsodyEvaluator,
    batch_size: int,
    aggregator: ResultAggregator,
    sample_filter: BadSampleFilter,
    file_ids,
) -> list:
    """对一个系统的所有音频进行评测，返回结果行列表。"""
    all_rows = []

    for batch_start in tqdm(
        range(0, len(audio_paths), batch_size),
        desc=f"Evaluating {system_name}",
    ):
        batch_paths = audio_paths[batch_start: batch_start + batch_size]
        batch_ids = file_ids[batch_start: batch_start + batch_size]
        batch_texts = resolve_batch_texts(text_src, batch_ids)

        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_mos = executor.submit(mos_ev.evaluate_batch, batch_paths)
            fut_sim = executor.submit(sim_ev.evaluate_batch, batch_paths)
            fut_asr = executor.submit(asr_ev.evaluate_batch, batch_paths,
                                      target_text=batch_texts)

        mos_results = fut_mos.result()
        sim_results = fut_sim.result()
        asr_results = fut_asr.result()
        prosody_results = prosody_ev.evaluate_batch(batch_paths)

        for i, fid in enumerate(batch_ids):
            row = {
                "file": fid,
                "system": system_name,
                "mos_score": mos_results[i]["mos_score"],
                "sim_score": sim_results[i]["sim_score"],
                "cer": asr_results[i]["cer"],
                "sent_acc": asr_results[i]["sent_acc"],
                "prosody_score": prosody_results[i]["prosody_score"],
            }
            row = aggregator.aggregate_row(row)
            row = sample_filter.filter_row(row)
            all_rows.append(row)

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="TTS 多维度音质评测")
    parser.add_argument("--ref", default="", help="原声音频目录（可选；不提供时跳过 sim/prosody 维度）")
    parser.add_argument("--sysA", required=True, help="系统A 音频目录")
    parser.add_argument("--sysB", required=True, help="系统B 音频目录")
    parser.add_argument("--text", default="",
                        help="目标文本（可选）：留空跳过CER/句准；文本文件路径；或目录路径（按文件名匹配同名.txt）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--output", default="results/", help="输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.get("device", "cuda")
    batch_size = cfg.get("batch_size", 16)
    # 解析目标文本
    raw_text = (args.text or "").strip()
    if not raw_text:
        target_text = None
        print("    未提供目标文本，跳过 CER/句准")
    else:
        p_text = Path(raw_text)
        if p_text.is_dir():
            target_text = raw_text   # 目录路径，后续在 evaluate_system 中按文件名解析
            print(f"    目标文本目录: {raw_text}")
        elif p_text.is_file():
            target_text = p_text.read_text(encoding="utf-8").strip()
            print(f"    目标文本（统一）: {target_text[:40]}...")
        else:
            target_text = raw_text   # 直接当作文本字符串
            print(f"    目标文本（统一）: {target_text[:40]}")

    print("[1/6] 加载音频配对...")
    loader = AudioPairLoader(
        ref_dir=args.ref or None,
        sysA_dir=args.sysA,
        sysB_dir=args.sysB,
    )
    pairs = loader.get_pairs()
    ref_paths = loader.get_ref_paths()
    ref_info = f"{len(ref_paths)} 个" if ref_paths else "未提供（跳过 sim/prosody）"
    print(f"    配对数: {len(pairs)}, ref音频: {ref_info}")

    if not pairs:
        print("错误: 未找到匹配的音频对，请检查目录和文件名。")
        return

    print("[2/6] 初始化模型...")
    mos_ev = MOSEvaluator(
        model_path=cfg["dnsmos"]["model_path"], device=device
    )
    sim_ev = SimilarityEvaluator(device=device)
    asr_ev = ASREvaluator(
        model_name=cfg["asr"]["model"], device=device
    )
    prosody_ev = ProsodyEvaluator(device=device)

    print("[3/6] 构建参考 embedding 和韵律统计...")
    sim_ev.build_ref_embedding(ref_paths)
    prosody_ev.build_ref_stats(ref_paths)

    aggregator = ResultAggregator(weights=cfg["weights"])
    sample_filter = BadSampleFilter(thresholds=cfg["thresholds"])

    sysA_paths = [p["sysA_path"] for p in pairs]
    sysB_paths = [p["sysB_path"] for p in pairs]
    file_ids = [p["file_id"] for p in pairs]

    # 若 target_text 为目录路径，转成 dict；否则直接传 str 或 None
    if isinstance(target_text, str) and target_text and Path(target_text).is_dir():
        text_src = load_text_dir(target_text, file_ids)
        n = sum(1 for v in text_src.values() if v)
        print(f"    目录模式：{n}/{len(file_ids)} 个文件匹配到文本")
    else:
        text_src = target_text  # str | None

    print("[4/6] 评测系统A...")
    rows_A = evaluate_system(
        "sysA", sysA_paths, text_src,
        mos_ev, sim_ev, asr_ev, prosody_ev,
        batch_size, aggregator, sample_filter, file_ids,
    )

    print("[5/6] 评测系统B...")
    rows_B = evaluate_system(
        "sysB", sysB_paths, text_src,
        mos_ev, sim_ev, asr_ev, prosody_ev,
        batch_size, aggregator, sample_filter, file_ids,
    )

    all_rows = rows_A + rows_B

    print("[6/6] 生成报告...")
    output_dir = Path(args.output)
    CSVReporter(output_dir).write(all_rows)
    HTMLReporter(output_dir).write(all_rows)

    bad_count = sum(1 for r in all_rows if r["is_bad"])
    print(f"\n完成！结果保存至 {output_dir}")
    print(f"  总样本数: {len(all_rows)}  |  差样本数: {bad_count}")
    print(f"  results.csv / bad_samples.csv / report.html")


if __name__ == "__main__":
    main()
