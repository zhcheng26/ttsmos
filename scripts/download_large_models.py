"""
下载大体积 Whisper 模型（large-v2 / large-v3）到 models/whisper/
这些模型超过 GitHub LFS 2GB 限制，不纳入版本控制，需手动下载。

用法:
    python scripts/download_large_models.py             # 下载 large-v3（默认）
    python scripts/download_large_models.py --model large-v2
    python scripts/download_large_models.py --model large-v3
"""
import argparse
from pathlib import Path
import whisper

WHISPER_DIR = Path(__file__).parent.parent / "models" / "whisper"


def main():
    parser = argparse.ArgumentParser(description="下载大体积 Whisper 模型")
    parser.add_argument(
        "--model",
        choices=["large-v2", "large-v3"],
        default="large-v3",
        help="要下载的模型名称（默认: large-v3）",
    )
    args = parser.parse_args()

    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    target = WHISPER_DIR / f"{args.model}.pt"

    if target.exists():
        print(f"模型已存在: {target} ({target.stat().st_size / 1e9:.1f}G)，跳过下载。")
        return

    print(f"开始下载 {args.model} 到 {WHISPER_DIR} ...")
    whisper.load_model(args.model, download_root=str(WHISPER_DIR))
    print(f"下载完成: {target}")


if __name__ == "__main__":
    main()
