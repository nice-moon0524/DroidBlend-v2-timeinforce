# 设计说明：DroidSpeak 的独立复现

## 范围与符号

当前项目只复现 DroidSpeak，不实现 CacheBlend 的多片段融合。模型 A 是发送端，模型 B
是接收端。两者必须具有完全相同的 decoder 架构、层数、hidden size、attention/KV head、
head dimension、RoPE 配置和 tokenizer 映射；权重可以因 SFT 或 LoRA 合并而不同。

对一个独立前缀 `x`，A 产生：

- `KV_A[l]`：第 `l` 层的完整 K/V 缓存；
- `E_A[s]`：进入第 `s` 个 decoder layer 之前的隐藏激活。

`E_A[s]` 是让 B 从层 `s` 开始真实执行的交接激活。它既不是 decoder 输出，也不能替代
K/V 缓存。

## 一个候选连续层组

对候选组 `G=[s,e)`，B 的混合缓存按下列规则构建：

```text
对每一层 l：
  l 不在 G：直接写入 KV_A[l]
  l 在 G 内：以 E_A[s] 为输入，完整执行 B 的真实第 l 层
             将新得到的 KV_B^hybrid[l] 写入缓存
```

`G` 内的每层从空缓存开始，防止 B 新键值被错误地附加到旧 A 键值之后。B 对整个独立前缀
重算这些层；随后由 B 正常处理 prompt 末尾预留的 token 并继续解码。故近似只存在于
`G` 外传来的 A 状态。

## 离线与在线流程

1. 比对 A/B 的模型配置和 tokenizer；实验中用 B tokenizer 得到一套相同 input IDs。
2. 对 calibration 数据枚举所有合法连续层组，得到所需的起点集合 `s`。
3. A 独立预填充每个前缀，保存所有 `KV_A` 和所需 `E_A[s]`，此阶段不运行 B。
4. 做单层 KV 交换敏感性测试：保持 B 原生 KV，只将一层替换成 A 的 KV，测量质量下降。
5. 对每个连续层组运行 B 的混合构建，记录质量、预填充时间和诊断指标。
6. 在 calibration 上选择满足质量预算、预填充最快的 Pareto 点。
7. 只在保留集上报告 B 原生、直接 A-KV 复用和 Droid 混合三者结果。

缓存 manifest 会保存模型指纹、tokenized prefix 长度和输入哈希。若重新 tokenizer 后的
prefix IDs 不相同，程序会拒绝复用缓存。

## 有意保留的边界

- 执行适配器面向标准 Llama/Mistral decoder 结构；不支持的注意力结构会显式失败。
- 项目测量 B 侧的缓存构建时间和传输字节数；真实多智能体网络传输时间必须在部署后另测。
- CacheBlend 是第二阶段：只有 A→B 缓存迁移本身经验证后，才应把它与片段融合组合。
