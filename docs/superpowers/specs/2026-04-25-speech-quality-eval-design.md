# 语音质量评价系统设计文档

**日期：** 2026-04-25  
**项目：** xtts_pingjia — TTS 系统多维度音质评测  

---

## 一、背景与目标

对两个 TTS 系统（系统A、系统B）合成的相同固定文本音频进行多维度质量评测，以原声音频作为参考（用于说话人相似度），最终输出：
1. 全量评测结果（CSV）
2. 可视化对比报告（HTML）
3. 质量差的样本列表，供人工复核（bad_samples.csv + HTML 内嵌）

---

## 二、输入数据结构

```
ref_dir/     # 原声音频（参考 speaker，文本不固定）
sysA_dir/    # 系统A 合成音频（固定文本）
sysB_dir/    # 系统B 合成音频（固定文本）
target_text.txt  # 固定目标文本（用于 ASR CER 计算）
```

**配对逻辑：** sysA 和 sysB 按文件名（去扩展名）一一对应；ref_dir 音频统一作为 speaker 相似度参考池，对每条合成音频取全部 ref 的 embedding 均值进行比较。

---

## 三、整体架构与数据流

```
输入目录
    ↓
AudioPairLoader          按文件名配对 sysA/sysB，加载 ref 池
    ↓
EvaluatorPipeline        并行 GPU 批量推理（各 Evaluator 独立）
  ├── MOSEvaluator       DNSMOS P.835 → mos_score
  ├── SimilarityEvaluator ECAPA-TDNN cosine sim → sim_score
  ├── ASREvaluator       Whisper 转写 → CER/WER
  └── ProsodyEvaluator   librosa → pitch_diff, rate_diff → prosody_score
    ↓
ResultAggregator         加权综合分 weighted_score
    ↓
BadSampleFilter          双重规则过滤 → is_bad, bad_reason
    ↓
Reporter
  ├── results.csv
  ├── bad_samples.csv
  └── report.html
```

---

## 四、各评测维度

| 维度 | 模型/方法 | 输入 | 输出 |
|------|-----------|------|------|
| MOS 音质 | DNSMOS P.835（ONNX，GPU provider） | 单条合成音频 | mos_score（1-5） |
| 说话人相似度 | ECAPA-TDNN / WeSpeaker，cosine similarity | 合成音频 + ref 均值 embedding | sim_score（0-1） |
| 可懂度 CER | Whisper（medium/large-v3）+ 目标文本对比 | 合成音频 + target_text | cer（%） |
| 韵律 | librosa 提取 F0 均值/方差、语速、停顿比例 | 合成音频（对比 ref 分布） | prosody_score（0-1） |

**GPU 并行策略：**
- Whisper / ECAPA 支持 batch inference，batch_size 可配置
- DNSMOS 使用 ONNX Runtime GPU provider
- Prosody（librosa）用 `multiprocessing.Pool` CPU 并行
- 各 Evaluator 之间用 `ThreadPoolExecutor` 并发执行

---

## 五、加权综合分

```
weighted_score = w1×mos + w2×sim + w3×(1-cer) + w4×prosody_score
```

默认权重（可在 config.yaml 调整）：

| 维度 | 默认权重 |
|------|----------|
| mos | 0.3 |
| sim | 0.3 |
| cer | 0.3 |
| prosody | 0.1 |

---

## 六、Bad Sample 过滤规则

**规则A（任意单项不达标）：**
```python
rule_a = (mos < 3.0) | (sim < 0.7) | (cer > 0.15) | (prosody_score < 0.5)
```

**规则B（综合分不达标）：**
```python
rule_b = (weighted_score < 2.5)
```

```python
is_bad = rule_a | rule_b
```

`bad_reason` 字段记录具体触发维度（如 `"mos,sim"`），便于人工定位问题。

所有阈值通过 `config.yaml` 配置，无需修改代码。

---

## 七、输出格式

### results.csv

| 字段 | 说明 |
|------|------|
| file | 文件名（去扩展名） |
| system | sysA / sysB |
| mos | MOS 分数 |
| sim | 说话人相似度 |
| cer | 字错率（%） |
| prosody | 韵律分数 |
| weighted | 加权综合分 |
| is_bad | True/False |
| bad_reason | 触发维度列表 |

### report.html（plotly 自包含）

1. 系统对比汇总表：A vs B 各维度均值
2. 雷达图：两系统各维度综合对比
3. 柱状图：各维度分数分布直方图
4. 差样本列表：文件名 + 各维度分数 + 触发原因（高亮）

---

## 八、项目结构

```
10.xtts_pingjia/
├── run_eval.py
├── config.yaml
├── evaluators/
│   ├── __init__.py
│   ├── base.py
│   ├── mos_evaluator.py
│   ├── similarity_evaluator.py
│   ├── asr_evaluator.py
│   └── prosody_evaluator.py
├── pipeline/
│   ├── loader.py
│   ├── aggregator.py
│   └── filter.py
├── reporter/
│   ├── csv_reporter.py
│   └── html_reporter.py
└── requirements.txt
```

---

## 九、运行方式

```bash
python run_eval.py \
  --ref ref_dir/ \
  --sysA sysA_dir/ \
  --sysB sysB_dir/ \
  --text target_text.txt \
  --config config.yaml \
  --output results/
```

---

## 十、config.yaml 结构

```yaml
device: cuda
batch_size: 16

weights:
  mos: 0.3
  sim: 0.3
  cer: 0.3
  prosody: 0.1

thresholds:
  mos: 3.0
  sim: 0.7
  cer: 0.15
  prosody: 0.5
  weighted: 2.5

asr:
  model: medium   # whisper 模型大小

paths:
  ref: ref_dir/
  sysA: sysA_dir/
  sysB: sysB_dir/
  target_text: target_text.txt
  output: results/
```

---

## 十一、依赖

```
torch
torchaudio
onnxruntime-gpu
openai-whisper
wespeaker / speechbrain   # ECAPA-TDNN
librosa
pandas
plotly
jinja2
pyyaml
tqdm
```
