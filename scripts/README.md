# scripts 说明

当前主入口是：

默认数据集对齐 DroidSpeak Pareto profiler：`data/processed/hotpotqa_train.jsonl` 的 50 条 HotpotQA 样本。

- `run_all.sh`：服务器 bash 一键运行，默认执行 baseline + DroidSpeak Pareto 基线 + DroidBlend span/token sweep，并自动选点和绘图。
- `run_baselines.sh`：只运行 receiver_full、full_kv_reuse、DroidSpeak Pareto 和 token-only 基线。
- `run_sweep.sh`：只运行 DroidBlend `recompute_layers` / `token_recompute_ratio` 搜索。
- `run_token_ratio_batches.sh`：按 token ratio 分批运行；每个比例一个输出目录，跑完一个比例就保存完整 `summary.json/csv/per_example/figures`。
- `select_hybrid_config.py`：从 `summary.json` 中按 F1 不下降和重算最少原则挑选配置。
- `plot_hybrid_results.py`：从 `summary.json` 生成质量/时延图和 recompute-span/token-ratio heatmap。

关键环境变量：

- `DROIDBLEND_TOKEN_RATIO=0.15`：指定 `token_only_ratio_r` 这个对照实验的单个 token 重算比例。

- `DROIDBLEND_RECOMPUTE_SPANS=pareto`：默认从 DroidSpeak-new Pareto frontier 读取连续层段。
- `DROIDBLEND_RECOMPUTE_SPANS=8-21,10-19`：手动指定连续层段。
- `DROIDBLEND_TOKEN_RATIOS=0.10,0.20,0.30,0.40,0.50`：指定单次 `run_all.sh` 内 DroidBlend sweep 的 token 重算比例网格。
- `DROIDBLEND_BATCH_TOKEN_RATIOS=0.10,0.20`：指定 `run_token_ratio_batches.sh` 分批运行的比例列表，默认先跑 10% 和 20%。
- `DROIDBLEND_BATCH_OUTPUT_ROOT=results/hybrid_by_ratio`：分批运行的根输出目录。
- `DROIDBLEND_BATCH_RESUME=1`：如果某个比例目录已有 `summary.json`，默认跳过，方便断点续跑。

旧的 `plot_results.py`、`plot_droidspeak_metrics.py`、`render_*` 和数据准备脚本保留为历史参考或辅助工具，不是本轮 DroidBlend-v2-timeinforce 的主入口。


