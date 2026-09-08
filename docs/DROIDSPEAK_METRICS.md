# DroidSpeak 性能测评口径

本文档对应 `scripts/plot_droidspeak_metrics.py`。图表只使用实验产生的 JSON，绝不把尚未测得的分布式系统指标伪装为结果。

| 图 | 对应论文证据 | 本项目数据来源 | 回答的问题 |
| --- | --- | --- | --- |
| `01_sender_receiver_baseline.png` | 论文 §3.1、Figure 2(a) 的接收端任务优势前提 | `baseline/native_pair_calibration.json` | B 是否应作为最终回答的 Receiver？ |
| `02_layer_sensitivity.png` | 论文 Figure 4 | `sensitivity/single_layer_kv_swap.json` | 哪些层直接复用 A 的 K/V 会严重损害 B 的质量？ |
| `03_quality_prefill_pareto.png` | 论文 Figure 10 | `profiling/points.json` 与 `selected.json` | 重算多少连续层，才能在质量与 prefill 延迟间取得折中？ |
| `04_heldout_quality_and_prefill.png` | 论文 §5.2 的最终对比 | `evaluation/layers_*.json` | 在未参与选层的 20 条样本上，DroidSpeak 是否比直接复用质量更高，并比 B 完整 prefill 更快？ |

## 指标定义

- **QA-F1**：对生成答案和标准答案进行小写化、去标点、忽略 `a/an/the` 后，按 token overlap 计算 Precision/Recall 的调和平均；多参考答案取最高值，再对样本平均。
- **关键层**：单层替换实验中，`(F1_native_B - F1_swap_layer) / F1_native_B > 10%` 的层。10% 是与论文 Figure 4 一致的标记阈值，不是选最终层组的质量预算。
- **质量预算**：当前配置的 `quality_tolerance_relative: 0.05`，意为候选层组相对 B-native 的 QA-F1 损失不得超过 5%。
- **本地 hybrid prefill 延迟**：从 CPU 缓存物化到 GPU、构造混合缓存、运行 B 的选定连续层组的端到端单机时间。代码在计时边界进行 CUDA synchronize，避免只记录异步 kernel 发射时间。

## 与论文系统测评的边界

论文还报告了跨节点 KV/E-cache 传输下的 TTFT、TBT、E2E 与 QPS（Figure 11、12、15）。当前工程是单 GPU Hugging Face 复现，尚未接入 vLLM、多副本请求到达过程或真实网络传输，因此只能报告算法级的质量、层敏感度和本地 prefill 时间。后续搭建两节点/多副本服务后，才能把真实网络 transfer 与排队时间加入 TTFT/QPS 图中。
