# 当前有效项目说明

本轮 DroidBlend-v2-timeinforce 的有效入口是：

- `experiments/run_hybrid_experiment.py`
- `scripts/run_all.sh`
- `scripts/select_hybrid_config.py`
- `scripts/plot_hybrid_results.py`
- `core/sparse_droidblend.py`

实验固定 `selection_layer=0`，搜索 `recompute_layers=[start,end]` 与 `token_recompute_ratio` 两个变量。`recompute_layers` 是任意连续层段，不要求从第 0 层开始；默认候选来自 DroidSpeak-new 的 `profiling_results.json` Pareto frontier。

旧 `src/droidblend/`、`configs/`、`outputs/` 和 reference/oracle 脚本保留作历史参考，不是当前服务器主流程。
