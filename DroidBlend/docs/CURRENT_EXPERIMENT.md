# 当前正式实验：LongBench HotpotQA 无截断 16K

正式配置为 `configs/experiment.filtered16k.yaml`。模型 A 为
`Llama-3.1-8B`，模型 B 为 `Llama-3.1-8B-Instruct`；两者共享由 B 的
tokenizer 和 chat template 渲染的一套 input_ids。

## 数据来源与筛选

数据来自 Hugging Face `zai-org/LongBench` 的 `data.zip:data/hotpotqa.jsonl`。
源文件共有 200 条。每条样本均以完整的 receiver-chat prompt 编码，且不使用任何
截断策略。数据构建阶段采用 16,000-token 阈值；服务器端实际编码中仅一条样本达到
16,394 tokens，因此运行上限设为 16,400，仍保留完整上下文。用固定随机种子
`20260804` 选择 50 条，
前 30 条为 calibration，后 20 条为 evaluation。

发送到服务器的目录只有：

```text
data/longbench_hotpotqa_16k/
  calibration.jsonl
  evaluation.jsonl
  selection_manifest.json
```

`selection_manifest.json` 记录源压缩包 SHA-256、筛选 tokenizer 的词表摘要和
chat template 摘要。构建时筛选 tokenizer 的词表摘要为：

```text
eb58f0e94d29822ac1c2055c2f72e0cdb162119e0db3bc23729e6d2bf4174edf
```

这是先前服务器模型兼容性报告中 B tokenizer 的相同摘要。部署后仍必须运行
`audit-inputs`；其输出中的 `tokenizer_vocab_digest`、`chat_template_sha256` 和
`all_within_limit` 是实际模型端的最终核验。

## 运行顺序

```bash
python -m droidblend.cli --config configs/experiment.filtered16k.yaml audit-inputs --split calibration
python -m droidblend.cli --config configs/experiment.filtered16k.yaml audit-inputs --split evaluation

CUDA_VISIBLE_DEVICES=0 python -m droidblend.cli \
  --config configs/experiment.filtered16k.yaml baseline --split calibration
```

只有新的 baseline 质量合理后，才运行 sender cache capture 和 sensitivity。
