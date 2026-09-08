# 实验流程与结果解释

## 1. 模型路径与前置检查

当前正式配置为 `configs/experiment.filtered16k.yaml`：

```yaml
sender_path: /opt/hhy/models/Llama-3.1-8B
receiver_path: /opt/hhy/models/Llama-3.1-8B-Instruct
tokenizer_path: /opt/hhy/models/Llama-3.1-8B-Instruct
```

模型来自 ModelScope 不影响实验；必须以下载后的文件为准。首先运行：

```bash
python scripts/compare_llama31_models.py \
  --output outputs/llama31_8b_base_to_instruct/model_compatibility.json
python -m droidblend.cli --config configs/experiment.filtered16k.yaml audit-inputs --split calibration
```

前者检查完整词表映射、多个文本探针的实际 token IDs，以及 safetensors 中全部参数名/形状；
后者检查模型配置与 DroidBlend 运行所需的结构字段。两者都通过才可以采集缓存。

对本次 Llama Base→Instruct 模型对，`eos_token_id` 与 chat template 可能不同。这是可接受的：
程序固定用 B tokenizer 生成输入并只由 B 解码、判断终止符；报告中 `same_special_token_ids`
可以为 `false`，但 `same_shared_input_special_token_ids` 和 `all_probes_equal` 必须为 `true`。

## 2. 数据格式与划分

校准集、保留集分别是 JSONL；每行格式如下：

```json
{"id":"q001","prompt":"上下文与问题，末尾停在答案之前","answers":["标准答案"]}
```

两份数据必须没有重复 id，也不能有完全相同的 prompt。程序会强制检查。
`prompt` 不得包含标准答案。若使用旧的本地 HotpotQA 数据，可用 `prepare-hotpotqa` 转换，
再人工或用固定随机种子拆分为 calibration/evaluation。

建议初始划分：50 个样本时先用 30 calibration / 20 evaluation；正式报告应采用论文相近的
更大规模划分，并保存具体样本 id 列表。

## 3. 校准实验

```bash
python -m droidblend.cli --config configs/experiment.filtered16k.yaml baseline --split calibration
python -m droidblend.cli --config configs/experiment.filtered16k.yaml capture-sender --split calibration
python -m droidblend.cli --config configs/experiment.filtered16k.yaml sensitivity
python -m droidblend.cli --config configs/experiment.filtered16k.yaml profile
python -m droidblend.cli --config configs/experiment.filtered16k.yaml select-pareto
python -m droidblend.cli --config configs/experiment.filtered16k.yaml report
```

其中 `baseline` 必须最先执行：它在一套由 B tokenizer 生成的完全相同 input IDs 上，分别
测量 A、B 的原生 QA-F1、EM 和前缀预填充时间，并写入
`outputs/.../baseline/native_pair_calibration.json`。只有
`droid_direction_gate_passed=true`（B 的 QA-F1 不低于 A）才继续固定 A→B 方向。

第一次真实运行应先设置：

```yaml
profiling:
  max_profiles_per_run: 3
  measure_latency_repetitions: 2
  warmup_repetitions: 1
```

仅用于确认服务器上的 Transformers、Flash/SDPA、BF16 与真实 checkpoint 可以跑通。之后将
`max_profiles_per_run` 恢复为 `null`，并使用足够的重复次数完成全部连续层组 profiling。

敏感层图只是诊断：单层替换造成大质量下降的层值得关注，但最终只能由连续层组 profiling
决定 `[s,e)`，因为相邻层的误差存在相互作用。

## 4. 保留集评测与三组基线

```bash
python -m droidblend.cli --config configs/experiment.filtered16k.yaml capture-sender --split evaluation
python -m droidblend.cli --config configs/experiment.filtered16k.yaml evaluate
```

对每个保留样本，结果包含：

1. **B 原生**：B 从 token 完整预填充，是质量与缓存误差的参考；
2. **直接 A-KV 复用**：B 接收所有 A KV，但不重算前缀层，量化朴素跨模型复用的损失；
3. **Droid 混合**：A 的 KV 和 `E_A[s]` 传给 B，B 仅重算选定 `[s,e)`；
4. **缓存相对 L2**：`||KV_hybrid-KV_B_native|| / ||KV_B_native||`，仅作诊断，不是选层标准。

质量主指标是 QA-F1，同时报告 EM。选层规则是 calibration 上相对 B 原生质量下降不超过
`quality_tolerance_relative`（默认 5%）时，选择混合预填充最快的点。

`hybrid_prefill_ms` 包含 B 侧缓存装配、位置/掩码准备和选定 B block 的执行，不包含真实
网络传输；网络开销应依据 `transfer_bytes` 在多智能体部署时单独测量。

## 5. 工件与复现记录

每次 CLI 命令会更新 `outputs/<实验名>/run_manifest.json`，其中保存解析后的配置、随机种子、
Python/torch/transformers 版本及可见 GPU。报告论文图表时必须同时保留该文件、输入 JSONL
的版本和最终 `selected.json`。

## 6. 后续 Droid + CacheBlend 实验

先验证 A→B 的 Droid Pareto 点，再构造两个独立片段 `P1`、`P2`：各自先从 A 迁移到 B，
然后仅在 B 端执行 CacheBlend 融合。必须报告四组对照：

| 组别 | 模型差异 | P1→P2 缺少跨片段注意力 | 目的 |
|---|---:|---:|---|
| B 原生 `P1+P2` | 无 | 无 | 质量上界 |
| 仅 Droid | 有 | 有 | 隔离跨模型缓存迁移误差 |
| 仅 CacheBlend（B） | 无 | 已修正 | 隔离上下文拼接误差 |
| Droid 后接 CacheBlend | 有 | 已修正 | 提出的组合方法 |

组合方法中的 CacheBlend token 选择器看到的差异，可能同时来自 A→B 迁移和 P1/P2 上下文融合；
因此不能把它直接当作原始 CacheBlend 的 token 误差，必须以上述消融实验拆开解释。
