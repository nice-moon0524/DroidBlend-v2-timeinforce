# DroidBlend: DroidSpeak + CacheBlend for MistralLite

本项目用于验证一个融合式跨模型 KV cache 复用方法：

- **DroidSpeak 维度**：读取 DroidSpeak-new 已经完成的 profiling 结果，使用其中的连续关键层组，让 receiver 对这些层做全 token 重算。
- **CacheBlend 维度**：在非关键层上，根据第一层 sender/receiver KV deviation 选择高差异 token，只把这些 token 的 receiver KV 回填进混合 cache。

当前模型对固定为：

| 角色 | 模型 | 服务器默认路径 |
|---|---|---|
| sender | Mistral-7B-v0.1 | `/opt/hhy/models/Mistral-7B-v0.1` |
| receiver | MistralLite | `/opt/hhy/models/mistrallite` |

本项目不重新做 DroidSpeak 的单层敏感性实验，也不重新枚举 Pareto frontier。正式运行时直接读取
`../DroidSpeak-new/profiling_results.json` 中的 `optimal.layers`。

## 当前实现边界

`core/droidblend_prefill.py` 是 **correctness/reference 版本**。因为 HuggingFace Transformers
没有通用的任意 token sparse prefill kernel，代码会先运行 receiver full prefill 得到参考 KV，
再选择 HKVD token 并回填。这可以验证融合逻辑和质量趋势，但不能声称真实 sparse token
重算的延迟收益。

## 项目结构

- `core/`
  - `model_loading.py`：模型、tokenizer、prompt 构造。
  - `partial_prefill.py`：DroidSpeak 连续层重算。
  - `droidblend_prefill.py`：DroidSpeak 层重算 + CacheBlend token 修正。
- `kv_cache/`
  - `layer_kv_extractor.py`：抽取 sender KV cache 和 E cache。
  - `layer_kv_injector.py`、`cache_utils.py`：构造可部分填充的 DynamicCache。
- `experiments/`
  - `run_droidblend_experiment.py`：主实验，比较 `full_prefill`、`full_kv_reuse`、`droidspeak`、`cacheblend_reference`、`droidblend_reference`。
- `evaluation/`
  - `metrics.py`：SQuAD-style token F1。
- `scripts/`
  - `run_all.sh` / `run_all.ps1`：服务器一键流程。
  - `plot_results.py`：生成 DroidBlend 质量/延迟图和 token 选择图。
- `data/processed/`
  - 复用 DroidSpeak-new 已整理好的 QA JSONL 数据。
- `tests/`
  - tiny Mistral 本地 smoke tests。

## 服务器运行

先确保 DroidSpeak-new 已经生成 profiling：

```bash
cd /path/to/DroidSpeak-new
bash scripts/run_all.sh
```

然后运行 DroidBlend：

```bash
cd /path/to/DroidBlend
python -m pip install -r requirements.txt
export DROIDBLEND_PROFILE='../DroidSpeak-new/profiling_results.json'
export DROIDBLEND_MAX_SAMPLES=2
bash scripts/run_all.sh
```

确认 smoke test 能跑通后执行全量：

```bash
unset DROIDBLEND_MAX_SAMPLES
bash scripts/run_all.sh
```

主要输出：

- `results/droidblend_quality_latency.json`
- `results/droidblend_token_selection.json`
- `results/figures/droidblend_quality_latency.pdf`
- `results/figures/droidblend_token_selection.pdf`

更完整的中文说明见 `REPRODUCTION_GUIDE_ZH.md`。

## 历史代码说明

旧 `src/droidblend/`、`configs/`、`outputs/` 和部分旧文档来自之前的 Llama-3.1 尝试稿，
本轮没有删除它们。当前 Mistral/MistralLite 融合实验的有效入口以本 README、
`REPRODUCTION_GUIDE_ZH.md` 和根目录下的 `core/kv_cache/experiments/scripts/tests` 为准。
