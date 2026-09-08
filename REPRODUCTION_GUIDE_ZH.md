# DroidBlend-v2-timeinforce 中文交接与复现指南

## 1. 当前实验目标

本项目的目标是寻找 DroidSpeak + CacheBlend 融合场景中的质量/时延平衡点：在 receiver F1 不下降的前提下，尽量减少 full-token 重算层段和 sparse-token 重算比例。

现在的关键设计是：

- `selection_layer=0` 固定。第 0 层 receiver 必须 full-token 重算，用来计算 `diff_k = ||K_receiver - K_sender||^2` 并选择 HKVD token。
- `recompute_layers=[start,end]` 可变。它是一段任意连续的 full-token receiver 重算层段，不要求从第 0 层开始。
- `token_recompute_ratio=r` 可变。除第 0 层和 full recompute span 外，其他层只对 HKVD token 做 receiver sparse 重算并回填 KV。

这样就避免了把 `selection_layer=30` 这类需要 receiver 中高层全量 KV 的设置放进对比，也避免了把 full recompute span 锁死为 `[0,K)`。

## 2. 当前实验流程

正式执行时只跑一个主流程：

```bash
bash scripts/run_all.sh
```

这个主流程会在同一批 HotpotQA 50 样本上一次性产出所有方法结果，便于后续统一比较 F1、latency 和重算工作量。

### 2.1 基础对照结果

这些方法用于确定质量上限和 KV 复用下限：

- `receiver_full`：receiver 对完整 prompt 正常 prefill，作为 F1 对齐基线和质量上限参考。
- `full_kv_reuse`：receiver 直接复用 sender 全层 KV，不做 DroidSpeak 层重算，也不做 token 修正，用于观察朴素跨模型 KV 复用的质量损失。

### 2.2 DroidSpeak Pareto 对照结果

方法名格式：

```text
droidspeak_layers_xx_yy
```

脚本读取 `../DroidSpeak-new/profiling_results.json` 中的 `pareto_frontier`，把其中每一个连续层段 `[start,end]` 都作为 DroidSpeak baseline 跑一次。这里不在 DroidBlend 项目里重新做 DroidSpeak profiling。

执行逻辑：

- `[start,end]` 层段：receiver full-token 重算；
- 其他层：复用 sender KV；
- `sender_e_cache[start]` 作为进入该层段的隐藏状态。

### 2.3 DroidBlend 主 sweep 结果

方法名格式：

```text
hybrid_layers_xx_yy_ratio_r
```

当前默认搜索空间：

```bash
export DROIDBLEND_RECOMPUTE_SPANS=pareto
export DROIDBLEND_TOKEN_RATIOS=0.10,0.20,0.30,0.40,0.50
```

含义是：对 DroidSpeak Pareto 中的每个连续层段 `[start,end]`，分别叠加 10%、20%、30%、40%、50% 的 token sparse 重算比例。

DroidBlend 的执行逻辑：

1. `selection_layer=0` 固定，receiver 第 0 层 full-token 重算。
2. 用 receiver/sender 第 0 层 K 差异计算 HKVD 分数。
3. 按 `token_recompute_ratio=r` 选择分数最高的一组 token，后续层固定使用这一组选中 token。
4. `[start,end]` 连续层段执行 full-token receiver 重算。
5. 不在 `{0} ∪ [start,end]` 的其他层，只对选中 token 执行 sparse receiver 重算并回填 KV。
6. 用混合 KV cache replay 最后一个 prompt token，得到 next-token logits，再在计时之外 greedy decode 得到预测答案。

### 2.4 token-only 消融结果

方法名格式：

```text
token_only_ratio_r
```

这个方法只把第 0 层作为 full-token 重算层，其余层只做 selected-token sparse 重算，不额外加入 DroidSpeak 连续层段。它是 CacheBlend-style token 修正对照。

该方法的单个比例由下面这个环境变量控制：

```bash
export DROIDBLEND_TOKEN_RATIO=0.15
```

注意：`DROIDBLEND_TOKEN_RATIO` 只控制 `token_only_ratio_r`；`DROIDBLEND_TOKEN_RATIOS` 控制 DroidBlend 主 sweep 的多个比例网格。

## 3. 默认数据集协议

本项目默认对齐 DroidSpeak Pareto profiler 的数据设置：

- 数据目录：`data/processed`
- 数据文件：`hotpotqa_train.jsonl`
- 样本数：50 条
- 结果中的 dataset 名称：`hotpotqa_50`

DroidSpeak-new 的 Pareto profiler 默认参数也是 `--dataset data/processed/hotpotqa_train.jsonl --max-samples 50`。因此当前 DroidBlend-v2-timeinforce 默认只在这 50 条 HotpotQA 样本上跑 baseline、DroidSpeak Pareto baseline 和 DroidBlend sweep。

## 4. 服务器运行流程

### 4.1 准备 DroidSpeak-new profiling

```bash
cd /path/to/DroidSpeak-new
conda activate kvcache
export DROID_SPEAK_SENDER=/opt/hhy/models/Mistral-7B-v0.1
export DROID_SPEAK_RECEIVER=/opt/hhy/models/mistrallite
export DROID_SPEAK_DEVICE=cuda
bash scripts/run_all.sh
```

需要生成：

```text
/path/to/DroidSpeak-new/profiling_results.json
```

### 4.2 DroidBlend 离线检查

```bash
cd /path/to/DroidBlend-v2-timeinforce
conda activate kvcache
python -m experiments.run_hybrid_experiment \
  --offline-only \
  --mode all \
  --profiling-results ../DroidSpeak-new/profiling_results.json \
  --data-dir data/processed \
  --output-dir results/hybrid_offline_check
```

预期会打印 `sweep_spans`，例如：

```json
[[12,13], [10,19], [8,21], [0,31]]
```

### 4.3 小样本 smoke test

```bash
export DROIDBLEND_SENDER=/opt/hhy/models/Mistral-7B-v0.1
export DROIDBLEND_RECEIVER=/opt/hhy/models/mistrallite
export DROIDBLEND_PROFILE=../DroidSpeak-new/profiling_results.json
export DROIDBLEND_DATA_DIR=data/processed
export DROIDBLEND_RECOMPUTE_SPANS=pareto
export DROIDBLEND_TOKEN_RATIOS=0.10,0.20,0.30,0.40,0.50
export DROIDBLEND_MAX_SAMPLES=2
bash scripts/run_all.sh
```

### 4.4 推荐分批实验

不建议第一次直接用 `run_all.sh` 把所有 token ratio 一次跑完。推荐先跑一部分：

```bash
unset DROIDBLEND_MAX_SAMPLES
export DROIDBLEND_BATCH_TOKEN_RATIOS=0.10,0.20
export DROIDBLEND_BATCH_OUTPUT_ROOT=results/hybrid_by_ratio
bash scripts/run_token_ratio_batches.sh
```

确认 10% 和 20% 两个比例都正常产出后，晚上可以继续跑更多比例：

```bash
export DROIDBLEND_BATCH_TOKEN_RATIOS=0.10,0.20,0.30,0.40,0.50
export DROIDBLEND_BATCH_OUTPUT_ROOT=results/hybrid_by_ratio
export DROIDBLEND_BATCH_RESUME=1
bash scripts/run_token_ratio_batches.sh
```

`DROIDBLEND_BATCH_RESUME=1` 会跳过已经有 `summary.json` 的比例目录，所以白天跑过的 10%/20% 不会重复跑。每个比例都会保存到独立目录，例如：

```text
results/hybrid_by_ratio/ratio_0p10/
results/hybrid_by_ratio/ratio_0p20/
results/hybrid_by_ratio/ratio_0p30/
```

每个比例跑完都会立即生成该比例自己的 `summary.json`、`summary.csv`、`per_example.jsonl`、`selected_config.json` 和 `figures/`，不会等所有比例全部跑完才保存。

## 5. 显存策略

双 GPU 推荐：

```bash
export DROIDBLEND_SENDER_DEVICE=cuda:0
export DROIDBLEND_RECEIVER_DEVICE=cuda:1
bash scripts/run_all.sh
```

单 GPU 流程验证可把 sender 放 CPU，但正式 latency 不建议这样报：

```bash
export DROIDBLEND_SENDER_DEVICE=cpu
export DROIDBLEND_RECEIVER_DEVICE=cuda:0
export DROIDBLEND_MAX_SAMPLES=1
bash scripts/run_all.sh
```

缩小搜索空间：

```bash
export DROIDBLEND_RECOMPUTE_SPANS=8-21,10-19,12-17
export DROIDBLEND_BATCH_TOKEN_RATIOS=0.10,0.20,0.30
bash scripts/run_token_ratio_batches.sh
```

## 6. 输出文件

默认输出目录：`results/hybrid/`

| 文件 | 内容 |
|---|---|
| `summary.json` | 单个输出目录内所有 method 的 F1、latency、参数元数据；分批运行时每个 token ratio 目录各有一份。 |
| `summary.csv` | 扁平结果表，适合论文图表和 Excel。 |
| `per_example.jsonl` | 每条样本各方法预测、时延、prompt tokens、selected token 数量；每条样本写入后会立即 flush。 |
| `selected_config.json` | 默认以 `receiver_full` 为 F1 下限，选择重算工作量最低的 DroidBlend 配置。 |
| `run_config.json` | 本次命令行参数记录。 |
| `figures/quality_latency_*.pdf/png` | 质量/时延散点图。 |
| `figures/hybrid_f1_heatmap_*.pdf/png` | recompute span × token ratio 的 F1 heatmap。 |

### 6.1 指标含义和 latency 口径

本项目会输出以下主要指标：

- `f1`：预测答案与参考答案之间的 QA token-level F1，越高越好。
- `latency_s`：单条样本、单个方法的一次计时时延，单位秒。
- `prefill_latency_s_mean`：某个方法在 HotpotQA 50 条样本上的平均 `latency_s`。
- `prefill_latency_s_p50`：中位数时延，反映典型样本速度。
- `prefill_latency_s_p95`：95 分位时延，反映长尾慢样本。
- `recompute_layers=[start,end]`：连续 full-token receiver 重算层段。
- `recompute_layer_count`：层段本身的层数，计算为 `end-start+1`。
- `full_recompute_layer_count`：实际 full-token 重算层数。DroidBlend 固定第 0 层用于 token selection，所以当 span 不包含第 0 层时，实际 full 层数是 `1 + recompute_layer_count`。
- `token_recompute_ratio`：非 full-token 重算层中，被 sparse receiver 重算并回填 KV 的 token 比例。
- `selected_count` / `candidate_count`：实际选中 token 数 / 候选 token 数。
- `recompute_work_ratio`：估算重算工作量，公式是 `(full_recompute_layer_count + sparse_layer_count * token_recompute_ratio) / total_layers`。

Latency 口径需要特别说明：`latency_s` 的计时从 `_timed()` 开始执行某个方法的 prefill/cache 构建函数开始，到该方法完成 prompt cache 并得到 next-token logits 为止。它包含该方法内部的 receiver full/sparse layer 计算，以及 `finalize_prompt_logits` 中最后一个 prompt token 的 logits 准备；不包含 `_timed()` 之后用于计算 F1 的 greedy decode，不包含每条样本前置的 sender KV/hidden cache 提取时间，也不包含真实多机部署时的 KV 网络传输时间。
## 7. 推荐汇报方式

主表建议列：

- Dataset
- Method
- F1
- Latency mean / p50 / p95
- Recompute span `[start,end]`
- Token recompute ratio
- Selected token fraction
- Recompute work ratio

主图建议：

1. 质量/时延散点图：比较 `receiver_full`、`full_kv_reuse`、DroidSpeak Pareto、DroidBlend candidates。
2. Span/token heatmap：横轴为 `[start,end]`，纵轴为 token ratio，颜色为 F1。
3. `selected_config.json` 表：展示全局最优是否满足所有数据集 F1 不下降。

## 8. 本地已验证内容

在本机 `kvcache` 环境确认：

```text
torch 2.6.0+cu124
transformers 4.57.1
cuda_available True
```

已通过：

```bash
python -m compileall core experiments scripts evaluation kv_cache data tests
python -m experiments.run_hybrid_experiment --offline-only --mode all --profiling-results ..\DroidSpeak-new\profiling_results.json --data-dir data\processed --output-dir results\hybrid_offline_check
```

也通过了 tiny Mistral smoke：`recompute_layers=(2,3)` 能正常执行，说明 full recompute span 不再被强制绑定到第 0 层。

本机 `kvcache` 环境暂未安装 pytest，因此完整 pytest 没跑。安装后可运行：

```bash
python -m pytest -q
```

## 9. 已知边界

- 这是 HuggingFace reference 级实验代码，不是 fused sparse prefill kernel。
- 主路径不读取 receiver full-prefix KV oracle；`receiver_full` 只作为质量/时延基线单独运行。
- `selection_layer=0` 固定，不和 `selection_layer=30` 等需要 receiver 中高层全量 KV 的设置比较。
- latency 不包含真实多机网络传输；跨机器部署时需单独测 KV 传输时间。







