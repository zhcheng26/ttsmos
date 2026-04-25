#!/usr/bin/env python3
# run_eval.py
"""
TTS 音质评测主入口。

用法:
    python run_eval.py \
        --ref ref_dir/ --sysA sysA_dir/ --sysB sysB_dir/ \
        --text target_text.txt --config config.yaml --output results/
"""
import argparse
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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


def evaluate_system(
    system_name: str,
    audio_paths,
    target_text: str,
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

        # MOS、相似度、ASR 并发推理；Prosody 用自身多进程
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_mos = executor.submit(mos_ev.evaluate_batch, batch_paths)
            fut_sim = executor.submit(sim_ev.evaluate_batch, batch_paths)
            fut_asr = executor.submit(
                asr_ev.evaluate_batch, batch_paths, target_text=target_text
            )

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
                "prosody_score": prosody_results[i]["prosody_score"],
            }
            row = aggregator.aggregate_row(row)
            row = sample_filter.filter_row(row)
            all_rows.append(row)

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="TTS 多维度音质评测")
    parser.add_argument("--ref", required=True, help="原声音频目录")
    parser.add_argument("--sysA", required=True, help="系统A 音频目录")
    parser.add_argument("--sysB", required=True, help="系统B 音频目录")
    parser.add_argument("--text", required=True, help="目标文本文件路径")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--output", default="results/", help="输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.get("device", "cuda")
    batch_size = cfg.get("batch_size", 16)
    target_text = Path(args.text).read_text(encoding="utf-8").strip()

    print("[1/6] 加载音频配对...")
    loader = AudioPairLoader(
        ref_dir=args.ref,
        sysA_dir=args.sysA,
        sysB_dir=args.sysB,
    )
    pairs = loader.get_pairs()
    ref_paths = loader.get_ref_paths()
    print(f"    配对数: {len(pairs)}, ref音频数: {len(ref_paths)}")

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

    print("[4/6] 评测系统A...")
    rows_A = evaluate_system(
        "sysA", sysA_paths, target_text,
        mos_ev, sim_ev, asr_ev, prosody_ev,
        batch_size, aggregator, sample_filter, file_ids,
    )

    print("[5/6] 评测系统B...")
    rows_B = evaluate_system(
        "sysB", sysB_paths, target_text,
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
