# Token Anchor K/KV 四组消融实验

本实验是独立的新入口，不修改、不替换也不调用以下旧实验入口：

- `core/sparse_droidblend.py`
- `experiments/run_hybrid_experiment.py`
- `scripts/run_all.sh`

实现位于：

- `core/token_anchor_sparse_prefill.py`
- `experiments/run_token_anchor_ablation.py`

## 实验目的

在完全相同的 HotpotQA 前 50 条样本上，对 DroidSpeak lock 中的 16 个 Pareto contiguous spans，比较以下四种 token 选择策略：

| 组名 | anchor 的 receiver 计算 | token score |
|---|---|---|
| `layer0-k` | 全有效 token 的 receiver layer 0 K 投影 | 仅 K 相对差异 |
| `layer0-kv` | 全有效 token 的 receiver layer 0 K/V 投影 | K 与 V 相对差异均值 |
| `start-k` | 全有效 token 的 receiver span 起始层 K 投影 | 仅 K 相对差异 |
| `start-kv` | 全有效 token 的 receiver span 起始层 K/V 投影 | K 与 V 相对差异均值 |

每种策略固定选择差异最大的 20% 有效 prefill token。每个 Pareto span 仍按 DroidSpeak 对所有有效 token 重算；在 span 之外，只对这 20% token 执行 receiver 稀疏重算并回填 KV。

本实验不调用 receiver full prefill，不读取 receiver full KV 或 full hidden state。`receiver_full_f1` 仅从锁定 baseline JSON 读取，用于报告 F1 差值。

## Score 定义

对 token `t`，每个 head 的 K 相对差异为：

```text
score_k(t) = mean_h( ||K_receiver[h,t] - K_sender[h,t]||_2
                     / (||K_sender[h,t]||_2 + 1e-12) )
```

`score_v(t)` 使用相同公式，将 K 改为 V。K+V 模式为：

```text
score_kv(t) = 0.5 * score_k(t) + 0.5 * score_v(t)
```

只对 attention mask 中有效 token 排序，不包含 padding。`selected_count = ceil(0.20 * candidate_count)`，同分时按 token index 升序，写出的 `selected_indices` 始终升序。

注意：K-only 不计算 V 的**选择分数**。当 anchor 本身是 DroidSpeak dense span 的第一层时，该层后续执行 attention 仍需要 V，这是模型层计算必需成本，不是 K+V scorer 的额外成本。

## 时延口径

`prefill_latency_s` 从进入独立 hybrid prefill 开始计时，包含：

1. receiver anchor K 或 K/V 投影；
2. token score 与 top-20% 选择；
3. DroidSpeak dense span 的 receiver 计算；
4. span 外 selected token 的稀疏 receiver 重算与 KV merge；
5. 得到首个 decode logit 所需的 final prompt token 计算。

不包含 sender KV/E cache 提取、greedy decode、真实网络传输，也不包含 receiver full prefill。

## 本地离线检查

以下命令不加载真实大模型，不运行 F1 或 latency 实验：

```powershell
Set-Location D:\pythonProject\DroidBlend-v2-timeinforce
python -m pytest tests/test_token_anchor_sparse_prefill.py -q
python -m experiments.run_token_anchor_ablation --offline-validate
```

第一条验证 K/KV 排序、padding 排除、比例取整与稳定 tie-break。第二条验证 baseline lock 恰有 16 个唯一 span、数据文件存在、ratio 固定为 0.20、四组名称合法。

## 服务器正式执行

在服务器进入项目目录后执行。该命令会加载 sender 和 receiver 模型，实际计算 F1 与 prefill latency，因此不要在本地执行。

```bash
cd /path/to/DroidBlend-v2-timeinforce
conda activate kvcache

python -m experiments.run_token_anchor_ablation \
  --sender /opt/hhy/models/Mistral-7B-v0.1 \
  --receiver /opt/hhy/models/mistrallite \
  --data-dir data/processed \
  --baseline-lock docs/droidspeak_pareto_baseline_lock.json \
  --max-samples 50 \
  --token-recompute-ratio 0.20 \
  --groups layer0-k,layer0-kv,start-k,start-kv \
  --output-dir results/token_anchor_ablation/hotpotqa_50_ratio_0p20
```

该命令执行 `16 spans * 4 groups * 50 samples = 3200` 次 hybrid prefill，输出所有四组的 F1 与 prefill latency。四组在每个 span 上共享同一条数据、同一 prompt 截断设置和同一 token ratio。

如需先在服务器验证模型路径、显存和执行环境，可只运行一个 span-group 组合的代码路径目前不提供缩减 span 参数，因此建议仍使用独立的离线检查；正式实验必须保持完整 16 span 和 4 group 矩阵，避免数据不可比。

## 输出文件

正式命令默认写入 `results/token_anchor_ablation/hotpotqa_50_ratio_0p20/`：

| 文件 | 内容 |
|---|---|
| `summary.json` | 每个 span/group 方法的 F1、mean/p50/p95 prefill latency、累计 token 统计、相对 DroidSpeak 与 receiver_full 的 F1 差值 |
| `summary.csv` | `summary.json` 的扁平表，适合后续分析和绘图 |
| `per_example.jsonl` | 每条样本、每个方法的预测、latency、selected/candidate indices、K score，以及 K+V 模式下的 V score |
| `run_config.json` | 本次实际 CLI 参数，用于精确复现 |

方法名格式为：

```text
token_anchor_{layer0|start}_score_{k|kv}_layers_{start}-{end}_ratio_0p20
```

例如 `token_anchor_start_score_kv_layers_12-13_ratio_0p20` 表示 span `[12,13]`、anchor 为 12、K+V 评分。

## 当前阶段不做的事

- 不修改或重跑旧 `run_hybrid_experiment.py` 实验。
- 不扫描 token ratio；本轮固定 0.20。
- 不做 disjoint span。
- 不生成最终候选或绘图；后续分析将读取这次的 CSV/JSONL 比较 K 与 K+V 选择的 token 重合度和效果。
