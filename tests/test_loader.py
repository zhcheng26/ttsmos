# tests/test_loader.py
import pytest
from pipeline.loader import AudioPairLoader


def test_pair_count(tmp_audio_dirs):
    loader = AudioPairLoader(
        ref_dir=tmp_audio_dirs["ref"],
        sysA_dir=tmp_audio_dirs["sysA"],
        sysB_dir=tmp_audio_dirs["sysB"],
    )
    pairs = loader.get_pairs()
    assert len(pairs) == 3


def test_pair_keys(tmp_audio_dirs):
    loader = AudioPairLoader(
        ref_dir=tmp_audio_dirs["ref"],
        sysA_dir=tmp_audio_dirs["sysA"],
        sysB_dir=tmp_audio_dirs["sysB"],
    )
    pair = loader.get_pairs()[0]
    assert "file_id" in pair
    assert "sysA_path" in pair
    assert "sysB_path" in pair


def test_ref_pool(tmp_audio_dirs):
    loader = AudioPairLoader(
        ref_dir=tmp_audio_dirs["ref"],
        sysA_dir=tmp_audio_dirs["sysA"],
        sysB_dir=tmp_audio_dirs["sysB"],
    )
    ref_paths = loader.get_ref_paths()
    assert len(ref_paths) == 2
    assert all(p.suffix == ".wav" for p in ref_paths)


def test_unmatched_files_skipped(tmp_audio_dirs):
    """sysB 比 sysA 多一个文件，多余的应被跳过"""
    import soundfile as sf
    import numpy as np
    extra = tmp_audio_dirs["sysB"] / "extra_999.wav"
    sf.write(str(extra), np.zeros(16000, dtype=np.float32), 16000)
    loader = AudioPairLoader(
        ref_dir=tmp_audio_dirs["ref"],
        sysA_dir=tmp_audio_dirs["sysA"],
        sysB_dir=tmp_audio_dirs["sysB"],
    )
    assert len(loader.get_pairs()) == 3
