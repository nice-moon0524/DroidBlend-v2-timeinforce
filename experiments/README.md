# experiments 说明

当前主实验入口是：

- `run_hybrid_experiment.py`

它读取 `../DroidSpeak-new/profiling_results.json` 中的 DroidSpeak Pareto frontier，并在同一套数据上比较：

- `receiver_full`
- `full_kv_reuse`
- `droidspeak_layers_xx_yy`
- `token_only_ratio_r`
- `hybrid_layers_xx_yy_ratio_r`

主搜索变量是两个：

- `recompute_layers=[start,end]`：任意连续 full-token receiver 重算层段，默认来自 DroidSpeak Pareto frontier。
- `token_recompute_ratio=r`：非全量重算层中 selected token 的重算比例。

`selection_layer` 固定为第 0 层，因为本实验不假设 receiver 已经拥有中高层全量 KV。

`run_droidblend_experiment.py` 是旧 reference/oracle 入口，会使用 receiver full KV 做 token 回填对照；`run_quality_experiment.py`、`run_layer_sensitivity.py`、`run_cross_dataset.py` 是 DroidSpeak-new 迁移来的历史脚本。本轮服务器实验请优先使用 `run_hybrid_experiment.py`。
