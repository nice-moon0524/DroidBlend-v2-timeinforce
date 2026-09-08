> Current anchor experiment interface: hybrid methods use `--selection-anchor-layer 0,start`, `--selection-score-mode k|kv`, `--token-recompute-ratio`, `--recompute-layers`, and `--span-mode contiguous`. They use receiver K/V projections only for token selection and never depend on receiver full KV. Use `python -m experiments.run_hybrid_experiment --validate-baseline-lock` before server runs. The older `selection_layer=0 full recompute` description below is historical.
# DroidBlend-v2-timeinforce

本项目用于构建一个可放到服务器运行的 DroidSpeak + CacheBlend 融合实验。当前设计已经区分两件事：

- `selection_layer=0` 固定：第 0 层 receiver 全量重算，用来计算 sender/receiver K deviation 并选择 HKVD token。
- `recompute_layers=[start,end]` 可变：这是一段任意连续的 full-token receiver 重算层段，不要求从第 0 层开始，默认从 DroidSpeak-new 的 Pareto frontier 读取。

## 核心实验定义

- `receiver_full`：receiver 对完整 prompt 正常 prefill，是质量上限和 F1 对齐基线。
- `full_kv_reuse`：receiver 直接使用 sender 全量 KV，不做修正，用于观察朴素跨模型 KV 复用的质量损失。
- `droidspeak_layers_xx_yy`：DroidSpeak-new Pareto frontier 中的每个层组，作为 DroidSpeak 基线，不固定 8-21。
- `token_only_ratio_r`：只固定第 0 层全量重算并选 token，其余层只对选中 token 做 sparse 重算。
- `hybrid_layers_xx_yy_ratio_r`：DroidBlend 主实验。第 0 层用于 token selection；`[xx,yy]` 这一段连续层 full-token 重算；其他层只重算选中 token。

## 参数中文含义

| 参数 | 中文含义 | 是否搜索 |
|---|---|---|
| `selection_layer` | token 检查层，固定第 0 层 | 否 |
| `recompute_layers` / `recompute_spans` | 任意连续全量重算层段 `[start,end]` | 是 |
| `token_recompute_ratio` / `r` | 非全量重算层中继续重算并回填 KV 的 token 比例 | 是 |
| `recompute_work_ratio` | 估算重算工作量，`(full_layers + sparse_layers*r)/L` | 汇报指标 |

## 默认数据集

默认与 DroidSpeak Pareto profiler 对齐，只跑 `data/processed/hotpotqa_train.jsonl` 的 50 条 HotpotQA 样本，在结果中记为 `hotpotqa_50`。如需临时小样本 smoke test，使用 `DROIDBLEND_MAX_SAMPLES` 截取前 N 条。
## 有效入口

服务器主入口：

```bash
bash scripts/run_all.sh
```

它会依次执行：

1. `python -m experiments.run_hybrid_experiment` 生成 `summary.json`、`summary.csv`、`per_example.jsonl`；
2. `python -m scripts.select_hybrid_config` 生成 `selected_config.json`；
3. `python -m scripts.plot_hybrid_results` 生成质量/时延散点图和 span/token heatmap。

默认 sweep 使用：

```bash
export DROIDBLEND_RECOMPUTE_SPANS=pareto
export DROIDBLEND_TOKEN_RATIOS=0.10,0.20,0.30,0.40,0.50
```

也可以手动指定层段：

```bash
export DROIDBLEND_RECOMPUTE_SPANS=8-21,10-19,12-17
bash scripts/run_all.sh
```

## 服务器快速开始

```bash
cd /path/to/DroidBlend-v2-timeinforce
conda activate kvcache
python -m compileall core experiments scripts evaluation kv_cache data

export DROIDBLEND_SENDER=/opt/hhy/models/Mistral-7B-v0.1
export DROIDBLEND_RECEIVER=/opt/hhy/models/mistrallite
export DROIDBLEND_PROFILE=../DroidSpeak-new/profiling_results.json
export DROIDBLEND_DATA_DIR=data/processed
export DROIDBLEND_MAX_SAMPLES=2
bash scripts/run_all.sh
```

确认 smoke test 通过后，建议按 token ratio 分批运行，先跑一部分：

```bash
unset DROIDBLEND_MAX_SAMPLES
export DROIDBLEND_BATCH_TOKEN_RATIOS=0.10,0.20
export DROIDBLEND_BATCH_OUTPUT_ROOT=results/hybrid_by_ratio
bash scripts/run_token_ratio_batches.sh
```

晚上可继续跑更多比例，已有 `summary.json` 的比例目录会被跳过：

```bash
export DROIDBLEND_BATCH_TOKEN_RATIOS=0.10,0.20,0.30,0.40,0.50
export DROIDBLEND_BATCH_RESUME=1
bash scripts/run_token_ratio_batches.sh
```

## 指标口径

- `f1`：预测答案与参考答案的 QA token-level F1，越高越好。
- `prefill_latency_s_mean/p50/p95`：每个方法的端到端测量时延统计，单位秒。计时从进入该方法的 prefill/cache 构建开始，到该方法完成 prompt cache 并得到 next-token logits 为止；不包含随后用于计算 F1 的 greedy decode，不包含前置的 sender KV/hidden cache 提取，也不包含真实跨机器 KV 网络传输。
- `recompute_layers`：full-token receiver 重算的连续层段 `[start,end]`。
- `full_recompute_layer_count`：实际 full-token 重算层数。第 0 层固定用于 token selection，因此不在 span 内时也会计入。
- `token_recompute_ratio`：非 full-token 重算层中 sparse 重算的 token 比例。
- `selected_count/candidate_count`：被选中重算的 token 数与可选 token 数。
- `recompute_work_ratio`：估算重算工作量，`(full_layers + sparse_layers * token_recompute_ratio) / total_layers`，越低表示理论重算越少。
## 主要输出

- `results/hybrid/summary.json`：HotpotQA 50 上各方法的 F1、mean/p50/p95 latency、参数元数据。分批运行时，每个比例会保存到 `results/hybrid_by_ratio/ratio_x/summary.json`。
- `results/hybrid/summary.csv`：便于画表的扁平表格。
- `results/hybrid/per_example.jsonl`：每条样本的预测、时延和 token 选择统计；现在每条样本写入后会立即 flush。
- `results/hybrid/selected_config.json`：按“不允许 F1 下降，重算工作量最低”选出的全局和逐数据集配置。
- `results/hybrid/figures/*.pdf|png`：汇报用质量/时延图与 recompute-span/token-ratio heatmap。

更完整的交接说明见 `REPRODUCTION_GUIDE_ZH.md`。






