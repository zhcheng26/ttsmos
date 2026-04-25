# pipeline/loader.py
from pathlib import Path
from typing import List, Dict


class AudioPairLoader:
    """按文件名（去扩展名）匹配 sysA/sysB 音频对，加载 ref 目录路径池。"""

    AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}

    def __init__(self, ref_dir, sysA_dir, sysB_dir):
        self.ref_dir = Path(ref_dir)
        self.sysA_dir = Path(sysA_dir)
        self.sysB_dir = Path(sysB_dir)
        self._pairs: List[Dict] = []
        self._ref_paths: List[Path] = []
        self._load()

    def _audio_files(self, directory: Path) -> Dict[str, Path]:
        return {
            p.stem: p
            for p in sorted(directory.iterdir())
            if p.suffix.lower() in self.AUDIO_EXTS
        }

    def _load(self):
        sysA_files = self._audio_files(self.sysA_dir)
        sysB_files = self._audio_files(self.sysB_dir)
        common = sorted(set(sysA_files.keys()) & set(sysB_files.keys()))
        self._pairs = [
            {
                "file_id": fid,
                "sysA_path": sysA_files[fid],
                "sysB_path": sysB_files[fid],
            }
            for fid in common
        ]
        self._ref_paths = sorted(
            p for p in self.ref_dir.iterdir()
            if p.suffix.lower() in self.AUDIO_EXTS
        )

    def get_pairs(self) -> List[Dict]:
        return self._pairs

    def get_ref_paths(self) -> List[Path]:
        return self._ref_paths
