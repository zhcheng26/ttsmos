# Speech Quality Evaluation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个多维度 TTS 音质评测 Pipeline，对比两个 TTS 系统的 MOS、说话人相似度、CER、韵律分数，自动筛出差样本并输出 CSV + HTML 报告。

**Architecture:** AudioPairLoader 加载并配对音频 → 四个独立 Evaluator 并行 GPU 推理 → ResultAggregator 计算加权分 → BadSampleFilter 双重过滤 → Reporter 输出 CSV + plotly HTML。

**Tech Stack:** Python 3.8+, PyTorch, onnxruntime-gpu, openai-whisper, speechbrain (ECAPA-TDNN), librosa, pandas, plotly, pyyaml, tqdm, soundfile, jiwer

---

## File Map

| 文件 | 职责 |
|------|------|
| `config.yaml` | 所有阈值、权重、路径、batch_size 配置 |
| `requirements.txt` | 依赖列表 |
| `pipeline/loader.py` | AudioPairLoader：按文件名配对音频，加载 ref 池 |
| `pipeline/aggregator.py` | ResultAggregator：加权综合分计算 |
| `pipeline/filter.py` | BadSampleFilter：规则A+B 双重过滤 |
| `evaluators/base.py` | BaseEvaluator 抽象类 |
| `evaluators/mos_evaluator.py` | DNSMOS P.835（ONNX）→ mos_score |
| `evaluators/similarity_evaluator.py` | ECAPA-TDNN cosine sim → sim_score |
| `evaluators/asr_evaluator.py` | Whisper 转写 → CER |
| `evaluators/prosody_evaluator.py` | librosa F0/语速 → prosody_score |
| `reporter/csv_reporter.py` | 写 results.csv 和 bad_samples.csv |
| `reporter/html_reporter.py` | 用 plotly 生成自包含 HTML 报告 |
| `run_eval.py` | 主入口，命令行参数，串联全流程 |
| `tests/conftest.py` | 测试用合成音频 fixture |
| `tests/test_loader.py` | loader 单元测试 |
| `tests/test_aggregator.py` | aggregator 单元测试 |
| `tests/test_filter.py` | filter 单元测试 |
| `tests/test_evaluators.py` | evaluator 接口测试（mock 推理） |
| `tests/test_reporters.py` | reporter 输出格式测试 |

---

## Task 1: 项目脚手架（config + requirements + 目录结构）

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `evaluators/__init__.py`
- Create: `pipeline/__init__.py`
- Create: `reporter/__init__.py`
- Create: `tests/__init__.py`
- Create: `models/` (存放 DNSMOS ONNX 文件，git-ignored)

- [ ] **Step 1: 创建 config.yaml**

```yaml
# config.yaml
device: cuda          # cuda 或 cpu
batch_size: 16

weights:
  mos: 0.3
  sim: 0.3
  cer: 0.3
  prosody: 0.1

thresholds:
  mos: 3.0            # MOS 低于此值触发规则A
  sim: 0.7            # 相似度低于此值触发规则A
  cer: 0.15           # CER 高于此值触发规则A（15%）
  prosody: 0.5        # 韵律分低于此值触发规则A
  weighted: 2.5       # 综合分低于此值触发规则B

asr:
  model: medium       # whisper 模型: tiny/base/small/medium/large-v3

dnsmos:
  model_path: models/sig_bak_ovr.onnx   # DNSMOS P.835 ONNX 文件路径

paths:
  ref: ref_dir/
  sysA: sysA_dir/
  sysB: sysB_dir/
  target_text: target_text.txt
  output: results/
```

- [ ] **Step 2: 创建 requirements.txt**

```
torch>=2.0.0
torchaudio>=2.0.0
onnxruntime-gpu>=1.16.0
openai-whisper>=20231117
speechbrain>=0.5.15
librosa>=0.10.0
pandas>=2.0.0
plotly>=5.15.0
pyyaml>=6.0
tqdm>=4.65.0
soundfile>=0.12.0
jiwer>=3.0.0
numpy>=1.24.0
scipy>=1.10.0
```

- [ ] **Step 3: 创建空 __init__.py 和目录**

```bash
mkdir -p evaluators pipeline reporter tests models
touch evaluators/__init__.py pipeline/__init__.py reporter/__init__.py tests/__init__.py
echo "models/" >> .gitignore
echo "results/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

- [ ] **Step 4: 下载 DNSMOS ONNX 模型**

从 Microsoft DNS-Challenge 仓库下载 `sig_bak_ovr.onnx`：
```bash
# 从 https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS
# 手动下载 sig_bak_ovr.onnx 放到 models/ 目录
# 或使用 git lfs / wget（需自行获取 raw 链接）
ls models/sig_bak_ovr.onnx   # 确认文件存在
```

- [ ] **Step 5: 安装依赖**

```bash
pip install -r requirements.txt
```

- [ ] **Step 6: Commit**

```bash
git init
git add config.yaml requirements.txt evaluators/__init__.py pipeline/__init__.py reporter/__init__.py tests/__init__.py .gitignore
git commit -m "chore: project scaffold, config and requirements"
```

---

## Task 2: AudioPairLoader（pipeline/loader.py）

**Files:**
- Create: `pipeline/loader.py`
- Create: `tests/conftest.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: 写 tests/conftest.py（合成音频 fixture）**

```python
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
```

- [ ] **Step 2: 写 tests/test_loader.py（先写失败测试）**

```python
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
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
pytest tests/test_loader.py -v
# 预期: ImportError: No module named 'pipeline.loader'
```

- [ ] **Step 4: 实现 pipeline/loader.py**

```python
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
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/test_loader.py -v
# 预期: 4 passed
```

- [ ] **Step 6: Commit**

```bash
git add pipeline/loader.py tests/conftest.py tests/test_loader.py
git commit -m "feat: AudioPairLoader with file-name matching and ref pool"
```

---

## Task 3: BaseEvaluator 抽象类（evaluators/base.py）

**Files:**
- Create: `evaluators/base.py`

- [ ] **Step 1: 实现 evaluators/base.py**

```python
# evaluators/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import torch


class BaseEvaluator(ABC):
    """所有 Evaluator 的基类，定义统一接口。"""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        """
        对一批音频文件进行评测。
        Returns: 与 audio_paths 等长的结果列表，每个元素是含评分字段的 dict。
        """
        ...

    def evaluate(self, audio_path: Path, **kwargs) -> Dict[str, Any]:
        """单条评测，调用 evaluate_batch 实现。"""
        return self.evaluate_batch([audio_path], **kwargs)[0]
```

- [ ] **Step 2: Commit**

```bash
git add evaluators/base.py
git commit -m "feat: BaseEvaluator abstract interface"
```

---

## Task 4: MOSEvaluator（evaluators/mos_evaluator.py）

**Files:**
- Create: `evaluators/mos_evaluator.py`
- Modify: `tests/test_evaluators.py`

DNSMOS P.835 使用 ONNX 模型对音频打分，输出 SIG（信号质量）、BAK（背景噪声）、OVRL（整体 MOS）三个维度，我们取 OVRL 作为 `mos_score`。

模型输入：16kHz 单声道音频，长度 9 秒（截断或补零），归一化到 [-1,1]，shape `(1, 9*16000)`。

- [ ] **Step 1: 写 MOSEvaluator 测试（先写）**

```python
# tests/test_evaluators.py
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock


def make_wav(path, sr=16000, duration=2.0, freq=440):
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)
    return path


def test_mos_evaluator_returns_score(tmp_path):
    """MOSEvaluator.evaluate_batch 应返回含 mos_score 的 dict 列表"""
    wav = make_wav(tmp_path / "test.wav")

    # mock onnxruntime 避免依赖真实模型文件
    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[3.5, 4.0, 3.8]])]  # SIG, BAK, OVRL

    with patch("evaluators.mos_evaluator.ort.InferenceSession", return_value=mock_session):
        from evaluators.mos_evaluator import MOSEvaluator
        ev = MOSEvaluator(model_path="fake.onnx", device="cpu")
        results = ev.evaluate_batch([wav])

    assert len(results) == 1
    assert "mos_score" in results[0]
    assert 1.0 <= results[0]["mos_score"] <= 5.0


def test_mos_evaluator_batch(tmp_path):
    """批量评测应返回与输入等长的列表"""
    wavs = [make_wav(tmp_path / f"t{i}.wav", freq=440 + i * 50) for i in range(4)]

    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[3.5, 4.0, 3.8]] * 4)]

    with patch("evaluators.mos_evaluator.ort.InferenceSession", return_value=mock_session):
        from evaluators.mos_evaluator import MOSEvaluator
        ev = MOSEvaluator(model_path="fake.onnx", device="cpu")
        results = ev.evaluate_batch(wavs)

    assert len(results) == 4
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_evaluators.py::test_mos_evaluator_returns_score -v
# 预期: ImportError
```

- [ ] **Step 3: 实现 evaluators/mos_evaluator.py**

```python
# evaluators/mos_evaluator.py
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Any

from evaluators.base import BaseEvaluator

SAMPLE_RATE = 16000
CLIP_LEN = 9  # DNSMOS P.835 要求 9 秒输入


class MOSEvaluator(BaseEvaluator):
    """使用 DNSMOS P.835 ONNX 模型评测音质，输出 OVRL MOS 分数。"""

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(device)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if str(self.device) != "cpu"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _load_and_pad(self, path: Path) -> np.ndarray:
        """加载音频，重采样到 16kHz，裁剪/补零到 9 秒，归一化。"""
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        target_len = CLIP_LEN * SAMPLE_RATE
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        # 归一化
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        return audio.astype(np.float32)

    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for path in audio_paths:
            audio = self._load_and_pad(path)
            inp = audio[np.newaxis, :]  # (1, 144000)
            out = self.session.run(None, {self.input_name: inp})
            # out[0] shape: (1, 3) -> [SIG, BAK, OVRL]
            ovrl = float(out[0][0][2])
            ovrl = float(np.clip(ovrl, 1.0, 5.0))
            results.append({"mos_score": ovrl})
        return results
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_evaluators.py::test_mos_evaluator_returns_score tests/test_evaluators.py::test_mos_evaluator_batch -v
# 预期: 2 passed
```

- [ ] **Step 5: Commit**

```bash
git add evaluators/mos_evaluator.py tests/test_evaluators.py
git commit -m "feat: MOSEvaluator using DNSMOS P.835 ONNX"
```

---

## Task 5: SimilarityEvaluator（evaluators/similarity_evaluator.py）

**Files:**
- Create: `evaluators/similarity_evaluator.py`
- Modify: `tests/test_evaluators.py`

使用 speechbrain ECAPA-TDNN 提取 speaker embedding，计算合成音频与 ref 均值 embedding 的 cosine 相似度。

- [ ] **Step 1: 写测试（追加到 tests/test_evaluators.py）**

```python
# 追加到 tests/test_evaluators.py

def test_similarity_evaluator_returns_score(tmp_path):
    """SimilarityEvaluator 应返回 0-1 之间的 sim_score"""
    wav = make_wav(tmp_path / "syn.wav")
    ref1 = make_wav(tmp_path / "ref1.wav", freq=300)
    ref2 = make_wav(tmp_path / "ref2.wav", freq=320)

    import torch
    fake_embedding = torch.randn(1, 192)

    mock_classifier = MagicMock()
    mock_classifier.encode_batch.return_value = fake_embedding

    with patch("evaluators.similarity_evaluator.EncoderClassifier.from_hparams",
               return_value=mock_classifier):
        from evaluators.similarity_evaluator import SimilarityEvaluator
        ev = SimilarityEvaluator(device="cpu")
        ev.build_ref_embedding([ref1, ref2])
        results = ev.evaluate_batch([wav])

    assert len(results) == 1
    assert "sim_score" in results[0]
    assert 0.0 <= results[0]["sim_score"] <= 1.0
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_evaluators.py::test_similarity_evaluator_returns_score -v
# 预期: ImportError
```

- [ ] **Step 3: 实现 evaluators/similarity_evaluator.py**

```python
# evaluators/similarity_evaluator.py
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from typing import List, Dict, Any

from speechbrain.pretrained import EncoderClassifier
from evaluators.base import BaseEvaluator

SAMPLE_RATE = 16000


class SimilarityEvaluator(BaseEvaluator):
    """使用 ECAPA-TDNN 计算合成音频与 ref 均值 embedding 的 cosine 相似度。"""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )
        self.ref_embedding: torch.Tensor = None

    def _load_audio(self, path: Path) -> torch.Tensor:
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        return waveform.squeeze(0)  # (T,)

    def _get_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (T,) → embedding: (1, D)"""
        audio = audio.unsqueeze(0).to(self.device)  # (1, T)
        with torch.no_grad():
            emb = self.classifier.encode_batch(audio)  # (1, 1, D) or (1, D)
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        return F.normalize(emb, dim=-1)

    def build_ref_embedding(self, ref_paths: List[Path]):
        """预计算 ref 均值 embedding，在评测前必须调用一次。"""
        embeddings = []
        for p in ref_paths:
            audio = self._load_audio(p)
            embeddings.append(self._get_embedding(audio))
        self.ref_embedding = torch.stack(embeddings).mean(dim=0)  # (1, D)
        self.ref_embedding = F.normalize(self.ref_embedding, dim=-1)

    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        if self.ref_embedding is None:
            raise RuntimeError("请先调用 build_ref_embedding() 构建参考 embedding")
        results = []
        for path in audio_paths:
            audio = self._load_audio(path)
            emb = self._get_embedding(audio)
            sim = F.cosine_similarity(emb, self.ref_embedding, dim=-1).item()
            sim = float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))  # [-1,1] → [0,1]
            results.append({"sim_score": sim})
        return results
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_evaluators.py::test_similarity_evaluator_returns_score -v
# 预期: 1 passed
```

- [ ] **Step 5: Commit**

```bash
git add evaluators/similarity_evaluator.py tests/test_evaluators.py
git commit -m "feat: SimilarityEvaluator using ECAPA-TDNN cosine similarity"
```

---

## Task 6: ASREvaluator（evaluators/asr_evaluator.py）

**Files:**
- Create: `evaluators/asr_evaluator.py`
- Modify: `tests/test_evaluators.py`

使用 Whisper 转写合成音频，与目标文本对比计算 CER（字错率）。使用 `jiwer` 库计算。

- [ ] **Step 1: 写测试（追加到 tests/test_evaluators.py）**

```python
# 追加到 tests/test_evaluators.py

def test_asr_evaluator_returns_cer(tmp_path):
    """ASREvaluator 应返回 cer（0-1 之间的浮点）"""
    wav = make_wav(tmp_path / "asr_test.wav")
    target_text = "你好世界"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "你好世界"}

    with patch("evaluators.asr_evaluator.whisper.load_model", return_value=mock_model):
        from evaluators.asr_evaluator import ASREvaluator
        ev = ASREvaluator(model_name="tiny", device="cpu")
        results = ev.evaluate_batch([wav], target_text=target_text)

    assert len(results) == 1
    assert "cer" in results[0]
    assert results[0]["cer"] == pytest.approx(0.0)   # 完全匹配 CER=0


def test_asr_evaluator_nonzero_cer(tmp_path):
    """转写错误时 CER 应大于 0"""
    wav = make_wav(tmp_path / "asr_test2.wav")
    target_text = "你好世界"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "错误文本内容"}

    with patch("evaluators.asr_evaluator.whisper.load_model", return_value=mock_model):
        from evaluators.asr_evaluator import ASREvaluator
        ev = ASREvaluator(model_name="tiny", device="cpu")
        results = ev.evaluate_batch([wav], target_text=target_text)

    assert results[0]["cer"] > 0.0
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_evaluators.py::test_asr_evaluator_returns_cer -v
# 预期: ImportError
```

- [ ] **Step 3: 实现 evaluators/asr_evaluator.py**

```python
# evaluators/asr_evaluator.py
import whisper
from jiwer import cer as compute_cer
from pathlib import Path
from typing import List, Dict, Any

from evaluators.base import BaseEvaluator


class ASREvaluator(BaseEvaluator):
    """使用 Whisper 转写音频，与目标文本对比计算 CER（字错率）。"""

    def __init__(self, model_name: str = "medium", device: str = "cuda"):
        super().__init__(device)
        self.model = whisper.load_model(model_name, device=str(self.device))

    def evaluate_batch(
        self, audio_paths: List[Path], target_text: str = "", **kwargs
    ) -> List[Dict[str, Any]]:
        results = []
        for path in audio_paths:
            result = self.model.transcribe(str(path), language="zh")
            hypothesis = result["text"].strip()
            reference = target_text.strip()
            if not reference:
                # 没有参考文本时 CER 无法计算，返回 0
                cer_score = 0.0
            else:
                cer_score = float(compute_cer(reference, hypothesis))
                cer_score = min(cer_score, 1.0)  # 上限截断至 1.0
            results.append({"cer": cer_score, "transcription": hypothesis})
        return results
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_evaluators.py::test_asr_evaluator_returns_cer tests/test_evaluators.py::test_asr_evaluator_nonzero_cer -v
# 预期: 2 passed
```

- [ ] **Step 5: Commit**

```bash
git add evaluators/asr_evaluator.py tests/test_evaluators.py
git commit -m "feat: ASREvaluator using Whisper + jiwer CER"
```

---

## Task 7: ProsodyEvaluator（evaluators/prosody_evaluator.py）

**Files:**
- Create: `evaluators/prosody_evaluator.py`
- Modify: `tests/test_evaluators.py`

提取 F0 均值、语速（voiced 帧比例），与 ref 音频的统计分布对比，归一化到 0-1 的 prosody_score。
`prosody_score = (f0_score + rate_score) / 2`

- [ ] **Step 1: 写测试（追加到 tests/test_evaluators.py）**

```python
# 追加到 tests/test_evaluators.py

def test_prosody_evaluator_returns_score(tmp_path):
    """ProsodyEvaluator 应返回 0-1 之间的 prosody_score"""
    ref1 = make_wav(tmp_path / "pref1.wav", freq=200, duration=2.0)
    ref2 = make_wav(tmp_path / "pref2.wav", freq=220, duration=2.0)
    syn = make_wav(tmp_path / "psyn.wav", freq=210, duration=2.0)

    from evaluators.prosody_evaluator import ProsodyEvaluator
    ev = ProsodyEvaluator(device="cpu")
    ev.build_ref_stats([ref1, ref2])
    results = ev.evaluate_batch([syn])

    assert len(results) == 1
    assert "prosody_score" in results[0]
    assert 0.0 <= results[0]["prosody_score"] <= 1.0
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_evaluators.py::test_prosody_evaluator_returns_score -v
# 预期: ImportError
```

- [ ] **Step 3: 实现 evaluators/prosody_evaluator.py**

```python
# evaluators/prosody_evaluator.py
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count

from evaluators.base import BaseEvaluator

SAMPLE_RATE = 16000


def _extract_prosody_features(path: str):
    """提取音频的韵律特征（在子进程中运行）。返回 (f0_mean, voiced_ratio)。"""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE,
    )
    voiced = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
    f0_mean = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
    voiced_ratio = float(voiced_flag.sum() / len(voiced_flag)) if voiced_flag is not None else 0.0
    return f0_mean, voiced_ratio


class ProsodyEvaluator(BaseEvaluator):
    """使用 librosa 提取 F0 和语速特征，计算合成音频与 ref 分布的相似度。"""

    def __init__(self, device: str = "cuda", n_workers: Optional[int] = None):
        super().__init__(device)
        self.n_workers = n_workers or max(1, cpu_count() // 2)
        self.ref_f0_mean: float = 0.0
        self.ref_voiced_ratio: float = 0.0

    def build_ref_stats(self, ref_paths: List[Path]):
        """预计算 ref 音频的平均 F0 均值和语速，在评测前必须调用一次。"""
        path_strs = [str(p) for p in ref_paths]
        with Pool(processes=self.n_workers) as pool:
            feats = pool.map(_extract_prosody_features, path_strs)
        f0_means = [f[0] for f in feats if f[0] > 0]
        voiced_ratios = [f[1] for f in feats]
        self.ref_f0_mean = float(np.mean(f0_means)) if f0_means else 150.0
        self.ref_voiced_ratio = float(np.mean(voiced_ratios))

    def _score_single(self, path: Path) -> Dict[str, Any]:
        f0_mean, voiced_ratio = _extract_prosody_features(str(path))
        # F0 相似度：归一化差值，ref_f0_mean 为基准
        if self.ref_f0_mean > 0 and f0_mean > 0:
            f0_diff = abs(f0_mean - self.ref_f0_mean) / self.ref_f0_mean
            f0_score = float(max(0.0, 1.0 - f0_diff))
        else:
            f0_score = 0.5

        # 语速相似度
        rate_diff = abs(voiced_ratio - self.ref_voiced_ratio)
        rate_score = float(max(0.0, 1.0 - rate_diff))

        prosody_score = (f0_score + rate_score) / 2.0
        return {
            "prosody_score": round(prosody_score, 4),
            "f0_mean": round(f0_mean, 2),
            "voiced_ratio": round(voiced_ratio, 4),
        }

    def evaluate_batch(self, audio_paths: List[Path], **kwargs) -> List[Dict[str, Any]]:
        path_strs = [str(p) for p in audio_paths]
        with Pool(processes=self.n_workers) as pool:
            feats = pool.map(_extract_prosody_features, path_strs)

        results = []
        for (f0_mean, voiced_ratio) in feats:
            if self.ref_f0_mean > 0 and f0_mean > 0:
                f0_diff = abs(f0_mean - self.ref_f0_mean) / self.ref_f0_mean
                f0_score = float(max(0.0, 1.0 - f0_diff))
            else:
                f0_score = 0.5
            rate_diff = abs(voiced_ratio - self.ref_voiced_ratio)
            rate_score = float(max(0.0, 1.0 - rate_diff))
            prosody_score = (f0_score + rate_score) / 2.0
            results.append({
                "prosody_score": round(prosody_score, 4),
                "f0_mean": round(f0_mean, 2),
                "voiced_ratio": round(voiced_ratio, 4),
            })
        return results
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_evaluators.py::test_prosody_evaluator_returns_score -v
# 预期: 1 passed
```

- [ ] **Step 5: Commit**

```bash
git add evaluators/prosody_evaluator.py tests/test_evaluators.py
git commit -m "feat: ProsodyEvaluator using librosa F0 and voiced ratio"
```

---

## Task 8: ResultAggregator（pipeline/aggregator.py）

**Files:**
- Create: `pipeline/aggregator.py`
- Create: `tests/test_aggregator.py`

- [ ] **Step 1: 写 tests/test_aggregator.py（先写）**

```python
# tests/test_aggregator.py
import pytest
from pipeline.aggregator import ResultAggregator


WEIGHTS = {"mos": 0.3, "sim": 0.3, "cer": 0.3, "prosody": 0.1}


def test_weighted_score_perfect():
    """所有指标满分时，weighted_score 应为 5.0"""
    agg = ResultAggregator(weights=WEIGHTS)
    score = agg.compute_weighted_score(
        mos_score=5.0, sim_score=1.0, cer=0.0, prosody_score=1.0
    )
    # weighted = 0.3*5 + 0.3*5*(1/1) + 0.3*5*(1-0) + 0.1*5
    # 实际: 0.3*5 + 0.3*1*5 + 0.3*(1-0)*5 + 0.1*1*5 = 5.0
    assert score == pytest.approx(5.0)


def test_weighted_score_zero_cer_penalty():
    """CER=1.0 时该维度贡献应为 0"""
    agg = ResultAggregator(weights=WEIGHTS)
    score = agg.compute_weighted_score(
        mos_score=5.0, sim_score=1.0, cer=1.0, prosody_score=1.0
    )
    # cer 贡献 = 0.3 * (1-1.0) * 5 = 0
    assert score < 5.0


def test_aggregate_row():
    """aggregate_row 返回含 weighted_score 的 dict"""
    agg = ResultAggregator(weights=WEIGHTS)
    row = {"mos_score": 3.5, "sim_score": 0.8, "cer": 0.05, "prosody_score": 0.7}
    result = agg.aggregate_row(row)
    assert "weighted_score" in result
    assert 1.0 <= result["weighted_score"] <= 5.0
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_aggregator.py -v
# 预期: ImportError
```

- [ ] **Step 3: 实现 pipeline/aggregator.py**

```python
# pipeline/aggregator.py
from typing import Dict, Any


class ResultAggregator:
    """
    根据权重计算加权综合分。
    公式: weighted_score = w_mos*(mos/5) + w_sim*sim + w_cer*(1-cer) + w_prosody*prosody
    归一化到 [0, 5] 保持与 MOS 量纲一致。
    """

    def __init__(self, weights: Dict[str, float]):
        self.w = weights
        total = sum(weights.values())
        # 归一化权重，确保加权后范围合理
        self.w_norm = {k: v / total for k, v in weights.items()}

    def compute_weighted_score(
        self,
        mos_score: float,
        sim_score: float,
        cer: float,
        prosody_score: float,
    ) -> float:
        """
        各维度先归一化到 [0,1]（MOS 除以 5），再加权，最后乘以 5 恢复到 [0,5] 量纲。
        """
        norm = {
            "mos": mos_score / 5.0,
            "sim": sim_score,
            "cer": 1.0 - min(cer, 1.0),
            "prosody": prosody_score,
        }
        score_01 = sum(self.w_norm[k] * norm[k] for k in norm)
        return round(score_01 * 5.0, 4)

    def aggregate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """接受含 mos_score/sim_score/cer/prosody_score 的 dict，追加 weighted_score。"""
        row = dict(row)
        row["weighted_score"] = self.compute_weighted_score(
            mos_score=row.get("mos_score", 0.0),
            sim_score=row.get("sim_score", 0.0),
            cer=row.get("cer", 0.0),
            prosody_score=row.get("prosody_score", 0.0),
        )
        return row
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_aggregator.py -v
# 预期: 3 passed
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/aggregator.py tests/test_aggregator.py
git commit -m "feat: ResultAggregator weighted score (normalized to [0,5])"
```

---

## Task 9: BadSampleFilter（pipeline/filter.py）

**Files:**
- Create: `pipeline/filter.py`
- Create: `tests/test_filter.py`

- [ ] **Step 1: 写 tests/test_filter.py（先写）**

```python
# tests/test_filter.py
import pytest
from pipeline.filter import BadSampleFilter

THRESHOLDS = {
    "mos": 3.0,
    "sim": 0.7,
    "cer": 0.15,
    "prosody": 0.5,
    "weighted": 2.5,
}


def make_row(mos=4.0, sim=0.8, cer=0.05, prosody=0.7, weighted=4.0):
    return {
        "mos_score": mos,
        "sim_score": sim,
        "cer": cer,
        "prosody_score": prosody,
        "weighted_score": weighted,
    }


def test_good_sample_not_flagged():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row()
    result = f.filter_row(row)
    assert result["is_bad"] is False
    assert result["bad_reason"] == ""


def test_rule_a_mos_triggers():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(mos=2.5)  # mos < 3.0
    result = f.filter_row(row)
    assert result["is_bad"] is True
    assert "mos" in result["bad_reason"]


def test_rule_a_sim_triggers():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(sim=0.5)  # sim < 0.7
    result = f.filter_row(row)
    assert result["is_bad"] is True
    assert "sim" in result["bad_reason"]


def test_rule_a_cer_triggers():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(cer=0.3)  # cer > 0.15
    result = f.filter_row(row)
    assert result["is_bad"] is True
    assert "cer" in result["bad_reason"]


def test_rule_b_weighted_triggers():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(weighted=2.0)  # weighted < 2.5
    result = f.filter_row(row)
    assert result["is_bad"] is True
    assert "weighted" in result["bad_reason"]


def test_multiple_reasons_recorded():
    f = BadSampleFilter(thresholds=THRESHOLDS)
    row = make_row(mos=2.0, sim=0.5)
    result = f.filter_row(row)
    assert "mos" in result["bad_reason"]
    assert "sim" in result["bad_reason"]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_filter.py -v
# 预期: ImportError
```

- [ ] **Step 3: 实现 pipeline/filter.py**

```python
# pipeline/filter.py
from typing import Dict, Any, List


class BadSampleFilter:
    """
    双重规则过滤：
    - 规则A：任意单项维度不达标
    - 规则B：加权综合分不达标
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.th = thresholds

    def filter_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        reasons: List[str] = []

        # 规则A
        if row.get("mos_score", 5.0) < self.th["mos"]:
            reasons.append("mos")
        if row.get("sim_score", 1.0) < self.th["sim"]:
            reasons.append("sim")
        if row.get("cer", 0.0) > self.th["cer"]:
            reasons.append("cer")
        if row.get("prosody_score", 1.0) < self.th["prosody"]:
            reasons.append("prosody")

        # 规则B
        if row.get("weighted_score", 5.0) < self.th["weighted"]:
            reasons.append("weighted")

        row["is_bad"] = len(reasons) > 0
        row["bad_reason"] = ",".join(reasons)
        return row

    def filter_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.filter_row(r) for r in rows]
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_filter.py -v
# 预期: 6 passed
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/filter.py tests/test_filter.py
git commit -m "feat: BadSampleFilter with rule A (per-dim) and rule B (weighted)"
```

---

## Task 10: CSVReporter（reporter/csv_reporter.py）

**Files:**
- Create: `reporter/csv_reporter.py`
- Create: `tests/test_reporters.py`

- [ ] **Step 1: 写 tests/test_reporters.py（先写）**

```python
# tests/test_reporters.py
import pytest
import pandas as pd
from pathlib import Path
from reporter.csv_reporter import CSVReporter


SAMPLE_ROWS = [
    {"file": "001", "system": "sysA", "mos_score": 3.8, "sim_score": 0.82,
     "cer": 0.05, "prosody_score": 0.75, "weighted_score": 3.6,
     "is_bad": False, "bad_reason": ""},
    {"file": "002", "system": "sysA", "mos_score": 2.1, "sim_score": 0.61,
     "cer": 0.22, "prosody_score": 0.48, "weighted_score": 2.1,
     "is_bad": True, "bad_reason": "mos,sim,cer"},
    {"file": "001", "system": "sysB", "mos_score": 4.0, "sim_score": 0.85,
     "cer": 0.03, "prosody_score": 0.80, "weighted_score": 4.1,
     "is_bad": False, "bad_reason": ""},
]


def test_csv_reporter_creates_results(tmp_path):
    reporter = CSVReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    assert (tmp_path / "results.csv").exists()
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 3
    assert "mos_score" in df.columns


def test_csv_reporter_creates_bad_samples(tmp_path):
    reporter = CSVReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    assert (tmp_path / "bad_samples.csv").exists()
    df = pd.read_csv(tmp_path / "bad_samples.csv")
    assert len(df) == 1
    assert df.iloc[0]["file"] == "002"


def test_csv_reporter_column_order(tmp_path):
    reporter = CSVReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    df = pd.read_csv(tmp_path / "results.csv")
    expected_cols = ["file", "system", "mos_score", "sim_score", "cer",
                     "prosody_score", "weighted_score", "is_bad", "bad_reason"]
    assert list(df.columns) == expected_cols
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_reporters.py::test_csv_reporter_creates_results -v
# 预期: ImportError
```

- [ ] **Step 3: 实现 reporter/csv_reporter.py**

```python
# reporter/csv_reporter.py
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

COLUMN_ORDER = [
    "file", "system", "mos_score", "sim_score", "cer",
    "prosody_score", "weighted_score", "is_bad", "bad_reason",
]


class CSVReporter:
    """将评测结果写入 results.csv 和 bad_samples.csv。"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rows: List[Dict[str, Any]]):
        df = pd.DataFrame(rows)
        # 保证列顺序，缺失列填 None
        for col in COLUMN_ORDER:
            if col not in df.columns:
                df[col] = None
        df = df[COLUMN_ORDER]

        df.to_csv(self.output_dir / "results.csv", index=False)

        bad_df = df[df["is_bad"] == True]
        bad_df.to_csv(self.output_dir / "bad_samples.csv", index=False)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_reporters.py::test_csv_reporter_creates_results tests/test_reporters.py::test_csv_reporter_creates_bad_samples tests/test_reporters.py::test_csv_reporter_column_order -v
# 预期: 3 passed
```

- [ ] **Step 5: Commit**

```bash
git add reporter/csv_reporter.py tests/test_reporters.py
git commit -m "feat: CSVReporter writes results.csv and bad_samples.csv"
```

---

## Task 11: HTMLReporter（reporter/html_reporter.py）

**Files:**
- Create: `reporter/html_reporter.py`
- Modify: `tests/test_reporters.py`

生成包含：①系统均值对比表 ②雷达图 ③各维度分布柱状图 ④差样本列表的自包含 HTML。

- [ ] **Step 1: 追加测试到 tests/test_reporters.py**

```python
# 追加到 tests/test_reporters.py

def test_html_reporter_creates_file(tmp_path):
    from reporter.html_reporter import HTMLReporter
    reporter = HTMLReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    html_path = tmp_path / "report.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "sysA" in content
    assert "sysB" in content


def test_html_reporter_contains_bad_samples(tmp_path):
    from reporter.html_reporter import HTMLReporter
    reporter = HTMLReporter(output_dir=tmp_path)
    reporter.write(SAMPLE_ROWS)
    content = (tmp_path / "report.html").read_text(encoding="utf-8")
    # 差样本的 file_id 应出现在 HTML 中
    assert "002" in content
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_reporters.py::test_html_reporter_creates_file -v
# 预期: ImportError
```

- [ ] **Step 3: 实现 reporter/html_reporter.py**

```python
# reporter/html_reporter.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
from typing import List, Dict, Any

DIMS = ["mos_score", "sim_score", "cer", "prosody_score", "weighted_score"]
DIM_LABELS = ["MOS", "相似度", "CER", "韵律", "综合分"]


class HTMLReporter:
    """生成包含图表和差样本列表的自包含 plotly HTML 报告。"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rows: List[Dict[str, Any]]):
        df = pd.DataFrame(rows)
        html_parts = ["<html><head><meta charset='utf-8'>",
                      "<title>TTS 音质评测报告</title></head><body>",
                      "<h1>TTS 系统音质评测报告</h1>"]

        # 1. 系统均值对比表
        html_parts.append(self._summary_table(df))

        # 2. 雷达图
        html_parts.append(self._radar_chart(df))

        # 3. 各维度分布直方图
        html_parts.append(self._histogram_charts(df))

        # 4. 差样本列表
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
                f"<td>{sub[d].mean():.3f}</td>" for d in DIMS
            )
            rows_html += f"<tr><td><b>{sys}</b></td>{row_vals}</tr>"
        header = "".join(f"<th>{l}</th>" for l in DIM_LABELS)
        return (f"<h2>系统均值对比</h2>"
                f"<table border='1' cellpadding='5'>"
                f"<tr><th>系统</th>{header}</tr>{rows_html}</table>")

    def _radar_chart(self, df: pd.DataFrame) -> str:
        categories = ["MOS/5", "相似度", "1-CER", "韵律", "综合分/5"]
        fig = go.Figure()
        for sys in sorted(df["system"].unique()):
            sub = df[df["system"] == sys]
            values = [
                sub["mos_score"].mean() / 5.0,
                sub["sim_score"].mean(),
                1.0 - sub["cer"].mean(),
                sub["prosody_score"].mean(),
                sub["weighted_score"].mean() / 5.0,
            ]
            values.append(values[0])  # 闭合
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
            fig = px.histogram(df, x=dim, color="system", barmode="overlay",
                               title=f"{label} 分布", nbins=20)
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
                f"<td>{row.get('mos_score', ''):.3f}</td>"
                f"<td>{row.get('sim_score', ''):.3f}</td>"
                f"<td>{row.get('cer', ''):.3f}</td>"
                f"<td>{row.get('prosody_score', ''):.3f}</td>"
                f"<td>{row.get('weighted_score', ''):.3f}</td>"
                f"<td>{row.get('bad_reason', '')}</td>"
                f"</tr>"
            )
        return (
            "<h2>差样本列表（需人工复核）</h2>"
            "<table border='1' cellpadding='5'>"
            "<tr><th>文件</th><th>系统</th><th>MOS</th><th>相似度</th>"
            "<th>CER</th><th>韵律</th><th>综合分</th><th>触发原因</th></tr>"
            f"{rows_html}</table>"
        )
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_reporters.py -v
# 预期: 5 passed
```

- [ ] **Step 5: Commit**

```bash
git add reporter/html_reporter.py tests/test_reporters.py
git commit -m "feat: HTMLReporter with plotly radar/histogram charts and bad sample table"
```

---

## Task 12: 主入口 run_eval.py（串联全流程）

**Files:**
- Create: `run_eval.py`

- [ ] **Step 1: 实现 run_eval.py**

```python
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    # 按 batch 推理，各 evaluator 并发执行
    for batch_start in tqdm(
        range(0, len(audio_paths), batch_size),
        desc=f"Evaluating {system_name}",
    ):
        batch_paths = audio_paths[batch_start: batch_start + batch_size]
        batch_ids = file_ids[batch_start: batch_start + batch_size]

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
```

- [ ] **Step 2: 全量测试**

```bash
pytest tests/ -v
# 预期: 所有测试通过
```

- [ ] **Step 3: Commit**

```bash
git add run_eval.py
git commit -m "feat: run_eval.py main entry integrating full evaluation pipeline"
```

---

## Task 13: 端到端冒烟测试（可选，使用真实模型）

仅在真实模型和数据就位后执行。

- [ ] **Step 1: 确认 DNSMOS 模型文件存在**

```bash
ls -lh models/sig_bak_ovr.onnx
```

- [ ] **Step 2: 准备少量测试音频（各3条）放到测试目录**

```bash
ls ref_dir/ sysA_dir/ sysB_dir/
```

- [ ] **Step 3: 创建目标文本文件**

```bash
echo "你好，这是一段测试文本，用于验证语音合成质量。" > target_text.txt
```

- [ ] **Step 4: 运行端到端评测**

```bash
python run_eval.py \
  --ref ref_dir/ \
  --sysA sysA_dir/ \
  --sysB sysB_dir/ \
  --text target_text.txt \
  --config config.yaml \
  --output results/
```

- [ ] **Step 5: 检查输出**

```bash
ls results/
# 预期: results.csv  bad_samples.csv  report.html
head results/results.csv
```

- [ ] **Step 6: 最终 Commit**

```bash
git add .
git commit -m "chore: final integration verified, evaluation pipeline complete"
```
