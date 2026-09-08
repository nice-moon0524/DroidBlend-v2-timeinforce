# DroidSpeak 复现保真审计

下表把“复现 DroidSpeak”的主张对应到项目实现和已有证据，同时明确尚未在真实服务器模型上
证明的部分。

| 论文组件 | DroidBlend 实现 | 当前证据 |
|---|---|---|
| 同构 A/B 前提 | `compat.py` 比较模型类型、架构、深度、维度、KV head、RoPE 和 tokenizer 指纹；`compare_llama31_models.py` 进一步比较完整词表与实际 token IDs。 | 合成端到端测试；真实 pair 仍需运行两项检查。 |
| A 端缓存持久化 | `capture.py` 保存每层 `KV_A[l]` 与所需的 `E_A[s]`。 | 合成测试真实写入并读回工件。 |
| 层敏感性诊断 | `sensitivity.py` 以 B 原生缓存为底，只替换一层为 A KV 后由 B 解码。 | 合成测试产生 JSON 和柱状图。 |
| 连续部分重算 | `runtime.partial_prefill` 从 `E_A[s]` 起，清空组内缓存，真实执行 B 的 `[s,e)`，组外注入 A KV。 | 运行时 smoke test 与合成端到端测试。 |
| B 端解码 | `append_suffix_and_generate` 用混合缓存处理末尾 prompt token 并由 B 继续生成。 | 运行时和端到端测试。 |
| 连续层组选择 | `schedules.py`、`profile.py`、`pareto.py`、`selection.py` 遍历合法连续层组并按质量预算选最快点。 | 合成测试完成 profiling 与选择。 |
| 质量/时延/复用对照 | `evaluate.py` 输出 B 原生、直接 A-KV 复用、Droid 混合；`profile.py` 输出预填充时间、缓存 L2、传输字节。 | 合成测试生成评测 JSON。 |
| 保留集约束 | `data.require_disjoint` 拒绝两集合重复 id 和 prompt。 | 单元测试。 |
| 结果解释 | `reports.py` 绘制敏感层柱状图与 speedup-质量损失散点图。 | 合成测试产生两张 PNG。 |

## 不应作出的主张

- 自动测试使用极小的随机 Llama，只验证状态流、CLI 和工件接口；它不能证明真实任务的质量恢复
  或加速比。
- 当前项目测量 B 侧构建时间和传输字节，尚未测量真实 agent 间网络传输端到端时延。
- 执行适配器面向标准 Llama/Mistral decoder；换用其他架构必须实现并验证新适配器，不能仅放宽
  配置检查。

## 真实实验验收门槛

以下条件全部满足后，才可将结果称为 A=`Llama-3.1-8B` 到
B=`Llama-3.1-8B-Instruct` 的 DroidSpeak 实证复现：

1. `scripts/compare_llama31_models.py` 和 `validate-pair` 均无硬不一致；
2. calibration 与 evaluation 的 A 缓存均采集完成；
3. calibration 上完成敏感层分析和完整配置的连续层组 profiling；
4. 选出的点满足事先声明的质量预算；
5. 保留集上同时保存 B 原生、直接复用和 Droid 混合结果、运行清单及两张图。
