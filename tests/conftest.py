# tests/conftest.py
import pytest
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path


@pytest.fixture
def tmp_audio_dirs(tmp_path):
    """创建临时目录，各放3条合成音频，ref放2条"""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    ref_dir = tmp_path / "ref"
    sysA_dir = tmp_path / "sysA"
    sysB_dir = tmp_path / "sysB"
    ref_dir.mkdir()
    sysA_dir.mkdir()
    sysB_dir.mkdir()

    for i in range(2):
        audio = (0.3 * np.sin(2 * np.pi * (300 + i * 50) * t)).astype(np.float32)
        sf.write(str(ref_dir / f"ref_{i:03d}.wav"), audio, sr)

    for i in range(3):
        audio = (0.3 * np.sin(2 * np.pi * (440 + i * 100) * t)).astype(np.float32)
        sf.write(str(sysA_dir / f"sample_{i:03d}.wav"), audio, sr)
        sf.write(str(sysB_dir / f"sample_{i:03d}.wav"), audio * 0.8, sr)

    return {"ref": ref_dir, "sysA": sysA_dir, "sysB": sysB_dir}
