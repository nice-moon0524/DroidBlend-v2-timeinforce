# DroidBlend 中文复现指南

## 1. 复现目标

DroidBlend 的目标不是完整复现 DroidSpeak 或 CacheBlend，而是验证二者的组合思路：

1. 先使用 DroidSpeak 已 profiling 出来的连续关键层组，让 MistralLite 对这些层进行 receiver 侧重算。
2. 再使用 CacheBlend 的 high-KV-deviation token 思想，从第一层 sender/receiver KV 差异中选出差异最大的 token。
3. 对非 DroidSpeak 全层重算的层，把这些高差异 token 的 receiver KV 回填到混合 cache 中。

最终比较：

- `full_prefill`：receiver 对完整 prompt 正常 prefill。
- `full_kv_reuse`：receiver 直接复用 sender 全部 KV。
- `droidspeak`：只做连续关键层组重算。
- `cacheblend_reference`：只做 token 级 reference KV 回填。
- `droidblend_reference`：先 DroidSpeak 层级重算，再做 CacheBlend token 级回填。

## 2. 不属于本项目的内容

本项目不做以下实验：

- 不重新做 DroidSpeak 的单层逐层敏感性测试。
- 不重新枚举 DroidSpeak 的 Pareto frontier。
- 不复现 DroidSpeak 的在线 serving、throughput、网络流水线实验。
- 不复现 CacheBlend 的 RAG 多 chunk 系统实验。
- 不声称当前 reference 实现具有真实 sparse token prefill 的延迟收益。

## 3. 项目结构和脚本输入输出

### `core/`

- `model_loading.py`
  - 输入：模型路径、设备、数据记录。
  - 输出：sender/receiver 模型、tokenizer、统一 prompt token。
  - 默认 sender 路径是 `/opt/hhy/models/Mistral-7B-v0.1`，receiver 路径是 `/opt/hhy/models/mistrallite`。

- `partial_prefill.py`
  - 输入：sender KV、sender E cache、DroidSpeak 的 `recompute_layers`、prompt token。
  - 输出：DroidSpeak 混合 cache 和第一个 decode token 的 logits。
  - 用途：实现层维度的连续关键层重算。

- `droidblend_prefill.py`
  - 输入：sender KV、sender E cache、DroidSpeak `recompute_layers`、`important_fraction` 或 `top_k`。
  - 输出：DroidBlend 混合 cache、token 选择统计、next-token logits。
  - 用途：在 DroidSpeak 混合 cache 上继续做 token 级 KV 修正。
  - 重要说明：当前是 correctness/reference 版本，会用 receiver full prefill 得到参考 KV。

### `kv_cache/`

- `layer_kv_extractor.py`
  - 输入：sender 模型和 prompt token。
  - 输出：每层 KV cache 与每层入口 E cache。

- `cache_utils.py`、`layer_kv_injector.py`
  - 输入：需要写入的 layer KV。
  - 输出：支持稀疏层填充的 HuggingFace DynamicCache。

### `experiments/`

- `run_droidblend_experiment.py`
  - 输入：
    - `--sender`
    - `--receiver`
    - `--profiling-results`
    - `--data-dir`
    - `--important-fraction`
    - `--max-samples`
  - 输出：
    - `results/droidblend_quality_latency.json`
    - `results/droidblend_token_selection.json`
  - 用途：主实验，比较五种方法的 F1 和 prefill latency。

### `scripts/`

- `run_all.sh`
  - 输入：环境变量和可选命令行参数。
  - 输出：主实验 JSON 和图表。
  - 用途：服务器上一键完成 DroidBlend 主流程。

- `plot_results.py`
  - 输入：
    - `results/droidblend_quality_latency.json`
    - `results/droidblend_token_selection.json`
  - 输出：
    - `results/figures/droidblend_quality_latency.pdf`
    - `results/figures/droidblend_token_selection.pdf`
  - 用途：把实验结果画成可读图表。

### `data/processed/`

复用 DroidSpeak-new 标准化后的 QA 数据：

- `hotpotqa_train.jsonl`
- `hotpotqa_test.jsonl`
- `2wikimqa_test.jsonl`
- `multifieldqa_en_test.jsonl`

每条数据包含：

- `id`
- `context`
- `question`
- `answer`

prompt 统一构造为：

```text
{context}

Question: {question}
Answer:
```

## 4. 本地能完成的工作

本地不加载 Mistral-7B/MistralLite 大模型。可以完成：

```bash
python -m pytest -q
```

这一步用于验证：

- tiny Mistral 上的 partial prefill 能跑通。
- DroidBlend token selection 能选出高 KV deviation token。
- mixed cache 能构造。
- F1 metric 正常。

如果本地 Python 缺 `pytest`、`torch` 或 `transformers`，这是本地环境问题，不代表项目代码不完整。

## 5. 服务器复现流程

### 5.1 先跑 DroidSpeak-new

这一步的目的不是运行 DroidBlend，而是生成 DroidBlend 依赖的层选择结果。

```bash
cd /path/to/DroidSpeak-new
export DROID_SPEAK_SENDER='/opt/hhy/models/Mistral-7B-v0.1'
export DROID_SPEAK_RECEIVER='/opt/hhy/models/mistrallite'
export DROID_SPEAK_DEVICE='cuda'
bash scripts/run_all.sh
```

关键输出：

```text
profiling_results.json
```

DroidBlend 只读取其中的：

```json
{
  "optimal": {
    "layers": [start, end]
  }
}
```

### 5.2 DroidBlend smoke test

这一步只跑少量样本，用来检查模型路径、数据路径、profiling 文件和融合 cache 流程。

```bash
cd /path/to/DroidBlend
python -m pip install -r requirements.txt
export DROIDBLEND_PROFILE='../DroidSpeak-new/profiling_results.json'
export DROIDBLEND_MAX_SAMPLES=2
bash scripts/run_all.sh
```

这一步会产生临时但可检查的：

- `results/droidblend_quality_latency.json`
- `results/droidblend_token_selection.json`
- `results/figures/*.pdf`

这些不是最终实验结果，只用于确认流程能跑通。

### 5.3 DroidBlend 全量实验

这一步才是正式输出。

```bash
unset DROIDBLEND_MAX_SAMPLES
bash scripts/run_all.sh
```

正式结果包括：

- `results/droidblend_quality_latency.json`
  - 各数据集上五种方法的 F1 和 prefill latency。
- `results/droidblend_token_selection.json`
  - 每条样本被选中的 HKVD token 数量、比例和索引。
- `results/figures/droidblend_quality_latency.pdf`
  - 质量/延迟散点图。
- `results/figures/droidblend_token_selection.pdf`
  - token 选择比例图。

## 6. 常用参数

### 改 token 选择比例

```bash
export DROIDBLEND_IMPORTANT_FRACTION=0.15
bash scripts/run_all.sh
```

用途：选择第一层 KV deviation 最大的 15% token。

### 改成固定 top-k

```bash
export DROIDBLEND_TOP_K=128
bash scripts/run_all.sh
```

用途：每条 prompt 选择最多 128 个高差异 token。设置 `DROIDBLEND_TOP_K` 后优先使用 top-k。

### 单独运行主实验

```bash
python -m experiments.run_droidblend_experiment \
  --sender /opt/hhy/models/Mistral-7B-v0.1 \
  --receiver /opt/hhy/models/mistrallite \
  --profiling-results ../DroidSpeak-new/profiling_results.json \
  --data-dir data/processed \
  --important-fraction 0.2
```

用途：跳过一键脚本，直接运行主实验，适合排错。

## 7. 如何判断项目可以开始服务器复现

需要满足：

- `data/processed/*.jsonl` 存在。
- DroidSpeak-new 已生成 `profiling_results.json`。
- sender 路径存在：`/opt/hhy/models/Mistral-7B-v0.1`。
- receiver 路径存在：`/opt/hhy/models/mistrallite`。
- smoke test 能生成两个 JSON 和两个 PDF。

如果 `profiling_results.json` 不存在，先回到 DroidSpeak-new 跑 profiling；不要在 DroidBlend 中重新做 Pareto。
