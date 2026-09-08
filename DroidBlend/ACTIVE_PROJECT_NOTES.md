# 当前有效项目说明

本轮 DroidBlend 的有效入口是根目录下这组 DroidSpeak-new 风格代码：

- `core/`
- `kv_cache/`
- `experiments/run_droidblend_experiment.py`
- `evaluation/`
- `scripts/run_all.sh`
- `scripts/run_all.ps1`
- `scripts/plot_results.py`
- `tests/test_droidblend_prefill.py`

旧项目中已有的 `src/droidblend/`、`configs/`、`outputs/` 以及 `docs/` 下若干旧文档，是之前
Llama-3.1 版本 DroidSpeak 尝试稿的历史材料。本轮没有删除它们，是为了保留参考价值；但它们
不是 Mistral-7B/MistralLite DroidBlend 服务器复现实验的执行入口。

当前服务器实验请以 `README.md` 和 `REPRODUCTION_GUIDE_ZH.md` 为准。
