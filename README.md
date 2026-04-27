# TTS 音质评测系统

多维度对比两个 TTS 系统的音频质量，支持 Web UI 和命令行两种运行方式，完全离线运行。

---

## 目录结构

```
.
├── app.py                  # Gradio Web UI 入口
├── run_eval.py             # 命令行评测入口
├── config.yaml             # 评测配置文件
├── requirements.txt        # Python 依赖
├── models/                 # 所有模型文件（离线，无需网络）
│   ├── sig_bak_ovr.onnx                        # DNSMOS P.835 模型
│   ├── whisper/
│   │   └── tiny.pt                             # Whisper ASR 模型
│   └── speechbrain/
│       └── spkrec-ecapa-voxceleb/              # ECAPA-TDNN 说话人相似度模型
├── demo/
│   ├── gen_demo_audio.py   # 生成演示音频
│   └── data/               # 演示音频（gen_demo_audio.py 生成后出现）
│       ├── ref/            # 原声音频
│       ├── sysA/           # 系统A 音频
│       ├── sysB/           # 系统B 音频
│       └── target_text.txt # 目标文本
├── pipeline/               # 数据加载 / 聚合 / 过滤
├── evaluators/             # 各维度评测器
└── reporter/               # CSV / HTML 报告生成
```

---

## 环境安装

```bash
pip install -r requirements.txt
```

> 所有模型文件已放在 `models/` 目录，运行时不访问网络。

---

## 快速开始

### 1. 生成演示音频（可选）

```bash
python demo/gen_demo_audio.py
```

生成后在 `demo/data/` 下产生 `ref/`、`sysA/`、`sysB/` 目录及 `target_text.txt`。

### 2. Web UI 方式

```bash
python app.py
```

浏览器访问 `http://localhost:7860`，在界面中填写目录路径后点击"开始评测"。

### 3. 命令行方式

```bash
python run_eval.py \
    --ref  demo/data/ref \
    --sysA demo/data/sysA \
    --sysB demo/data/sysB \
    --text demo/data/target_text.txt \
    --config config.yaml \
    --output results/
```

**无参考音频时**（只评测 MOS + CER，跳过说话人相似度和韵律）：

```bash
python run_eval.py \
    --sysA demo/data/sysA \
    --sysB demo/data/sysB \
    --text demo/data/target_text.txt \
    --config config.yaml \
    --output results/
```

> `--ref` 为可选参数。Web UI 中"原声目录"留空效果相同。

---

## 音频文件要求

| 项目 | 要求 |
|------|------|
| 格式 | `.wav` / `.flac` / `.mp3` / `.ogg` |
| sysA 与 sysB 文件名 | 必须一一对应（相同文件名，不含扩展名） |
| ref 目录 | **可选**。提供时计算说话人相似度和韵律；不提供时这两个维度输出 `None`，加权分仅由 MOS + CER 决定 |
| 采样率 | 任意（内部自动重采样至 16 kHz） |

---

## 配置说明（config.yaml）

```yaml
device: cpu           # 推理设备：cpu 或 cuda

batch_size: 16        # 每批处理的音频数量

weights:              # 各维度加权系数（归一化后使用）
  mos: 0.3
  sim: 0.3
  cer: 0.3
  prosody: 0.1

thresholds:           # 差样本判定阈值
  mos: 4.2            # MOS 低于此值 → 标记为差样本（1-5 分制）
  sim: 4.2            # 说话人相似度低于此值 → 标记为差样本（0-5 分制）
  cer: 0.15           # 字错率高于 15% → 标记为差样本
  prosody: 0.5        # 韵律分低于此值 → 标记为差样本
  weighted: 2.5       # 综合加权分低于此值 → 标记为差样本

asr:
  model: tiny         # Whisper 模型规格，对应 models/whisper/<model>.pt
                      # 可选：tiny / base / small / medium / large-v3

dnsmos:
  model_path: models/sig_bak_ovr.onnx   # DNSMOS ONNX 文件路径
```

---

## 评测维度说明

| 字段 | 模型 | 含义 | 分值范围 | 越好 |
|------|------|------|----------|------|
| `mos_score` | DNSMOS P.835 | 音质主观感受分 | 1 ~ 5 | 越高越好，需 > 4.2 |
| `sim_score` | ECAPA-TDNN | 与原声说话人的相似度 | 0 ~ 5 | 越高越好，需 > 4.2 |
| `cer` | Whisper | 字错率（与目标文本对比） | 0 ~ 1 | 越低越好 |
| `prosody_score` | librosa | 基频均值 + 语速与原声的相似度 | 0 ~ 1 | 越高越好 |
| `weighted_score` | — | 各维度加权综合分 | 0 ~ 5 | 越高越好 |
| `is_bad` | — | 是否标记为差样本 | True/False | False 为优 |
| `bad_reason` | — | 触发差样本的维度列表 | 字符串 | 空为优 |

**加权综合分计算公式：**

```
weighted_score = (w_mos*(mos/5) + w_sim*sim + w_cer*(1-cer) + w_prosody*prosody) * 5
```

**差样本判定规则：**
- 规则 A：任意单项维度不达标（低于 thresholds 对应阈值）
- 规则 B：加权综合分低于 `thresholds.weighted`

满足任一规则即标记 `is_bad=True`。

---

## 输出文件

评测完成后在 `results/`（或 `--output` 指定目录）生成：

| 文件 | 内容 |
|------|------|
| `results.csv` | 全量逐样本评测结果 |
| `bad_samples.csv` | 仅差样本列表 |
| `report.html` | 交互式图表报告（系统对比 + 各维度分布） |

---

## 切换 Whisper 模型

如需更高 ASR 精度，将对应模型文件放到 `models/whisper/` 后修改 `config.yaml`：

```yaml
asr:
  model: medium   # 对应 models/whisper/medium.pt
```

可用规格（精度从低到高）：`tiny` → `base` → `small` → `medium` → `large-v3`

---

## 运行测试

```bash
pytest tests/ -v
```
