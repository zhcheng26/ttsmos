"""生成演示用合成音频（用正弦波模拟不同质量的TTS输出）"""
import numpy as np
import soundfile as sf
from pathlib import Path


SR = 16000

def make_speech_like(path, freq=150, duration=4.0, noise_level=0.0, formants=None):
    """生成带谐波+噪声的语音模拟信号"""
    t = np.linspace(0, duration, int(SR * duration))
    # 基频 + 谐波
    audio = np.sin(2 * np.pi * freq * t)
    for h in [2, 3, 4, 5]:
        audio += (1/h) * np.sin(2 * np.pi * freq * h * t)
    # 加噪声
    if noise_level > 0:
        audio += noise_level * np.random.randn(len(t))
    # 归一化
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.7
    sf.write(str(path), audio.astype(np.float32), SR)


def generate(base_dir="demo/data"):
    base = Path(base_dir)
    ref_dir  = base / "ref"
    sysA_dir = base / "sysA"
    sysB_dir = base / "sysB"
    for d in [ref_dir, sysA_dir, sysB_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ref: 原声，女声音域约180Hz，干净
    for i in range(5):
        make_speech_like(ref_dir / f"ref_{i:03d}.wav", freq=180+i*5, noise_level=0.01)

    # sysA: 系统A，质量较好，音高接近原声
    samples = [
        ("sample_001", 175, 0.02),
        ("sample_002", 182, 0.03),
        ("sample_003", 178, 0.02),
        ("sample_004", 170, 0.05),
        ("sample_005", 185, 0.02),
        ("sample_006", 172, 0.08),  # 这条会质量差一些
        ("sample_007", 180, 0.02),
        ("sample_008", 176, 0.03),
    ]
    for name, freq, noise in samples:
        make_speech_like(sysA_dir / f"{name}.wav", freq=freq, noise_level=noise)

    # sysB: 系统B，部分样本音高偏移+噪声更大
    samplesB = [
        ("sample_001", 210, 0.03),  # 音高偏高
        ("sample_002", 190, 0.02),
        ("sample_003", 160, 0.15),  # 噪声大
        ("sample_004", 200, 0.04),
        ("sample_005", 195, 0.03),
        ("sample_006", 155, 0.20),  # 噪声很大，差样本
        ("sample_007", 205, 0.04),
        ("sample_008", 192, 0.03),
    ]
    for name, freq, noise in samplesB:
        make_speech_like(sysB_dir / f"{name}.wav", freq=freq, noise_level=noise)

    # 目标文本
    (base / "target_text.txt").write_text(
        "今天天气不错，我们一起去公园散步吧。", encoding="utf-8"
    )

    print(f"Demo audio generated in {base}/")
    print(f"  ref:  {len(list(ref_dir.glob('*.wav')))} files")
    print(f"  sysA: {len(list(sysA_dir.glob('*.wav')))} files")
    print(f"  sysB: {len(list(sysB_dir.glob('*.wav')))} files")


if __name__ == "__main__":
    generate()
