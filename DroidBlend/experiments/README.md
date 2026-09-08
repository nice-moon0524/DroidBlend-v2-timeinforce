# experiments 说明

当前 DroidBlend 主实验入口是：

- `run_droidblend_experiment.py`

它会读取外部 DroidSpeak profiling 结果，并比较：

- `full_prefill`
- `full_kv_reuse`
- `droidspeak`
- `cacheblend_reference`
- `droidblend_reference`

`run_quality_experiment.py`、`run_layer_sensitivity.py`、`run_cross_dataset.py` 是从
DroidSpeak-new 复制来的参考脚本。本轮 DroidBlend 服务器流程不会调用它们，也不会重新做
单层敏感性测试或 Pareto/cross-dataset profiling。
