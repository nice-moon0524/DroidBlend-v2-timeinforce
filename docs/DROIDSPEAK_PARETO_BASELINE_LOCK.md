# DroidSpeak Pareto Baseline Lock

## 目的

本文件完成“敏感层 + 部分 token 重算实验路线”的第一个目标：锁定 DroidSpeak Pareto 敏感层结果，作为后续 DroidBlend token selection anchor、token scorer 和 token ratio sweep 的固定层段基线。

后续第一轮实验不再重新搜索 DroidSpeak 层段，而是固定使用 `../DroidSpeak-new/profiling_results.json` 中的 Pareto frontier。机器可读版本见 `docs/droidspeak_pareto_baseline_lock.json`。

## 数据来源

- Profile 来源：`../DroidSpeak-new/profiling_results.json`
- Baseline 结果来源：`results/hybrid_by_ratio/ratio_0p10/summary.json`
- Dataset：`hotpotqa_50`
- `receiver_full` F1：`0.8003809523809522`
- 样本数：`50`

`profiling_results.json` 里的 `profile_*` 指标来自 DroidSpeak-new profiling；`baseline_*` 指标来自当前 DroidBlend-v2-timeinforce 中同一批 HotpotQA 50 上重新跑出的 `droidspeak_layers_xx_yy` baseline。后续筛选 hybrid 配置时，第一道质量门槛使用同层段 DroidSpeak baseline：

```text
hybrid_layers_xx_yy_ratio_r F1 >= droidspeak_layers_xx_yy F1
```

如果要筛选“完全不低于 receiver full”的高质量配置，再额外使用：

```text
hybrid F1 >= receiver_full F1
```

## 固定 Pareto 层段

| Span | Method | Layers | DroidSpeak F1 | p50 latency (s) | mean latency (s) | p95 latency (s) |
|---|---|---:|---:|---:|---:|---:|
| 12-13 | `droidspeak_layers_12_13` | 2 | 0.2919 | 0.0278 | 0.0288 | 0.0294 |
| 14-17 | `droidspeak_layers_14_17` | 4 | 0.4816 | 0.0293 | 0.0299 | 0.0322 |
| 12-17 | `droidspeak_layers_12_17` | 6 | 0.6388 | 0.0310 | 0.0321 | 0.0353 |
| 12-19 | `droidspeak_layers_12_19` | 8 | 0.6609 | 0.0332 | 0.0345 | 0.0399 |
| 10-19 | `droidspeak_layers_10_19` | 10 | 0.7537 | 0.0349 | 0.0369 | 0.0440 |
| 10-21 | `droidspeak_layers_10_21` | 12 | 0.7557 | 0.0364 | 0.0392 | 0.0477 |
| 8-21 | `droidspeak_layers_08_21` | 14 | 0.7728 | 0.0383 | 0.0418 | 0.0517 |
| 10-25 | `droidspeak_layers_10_25` | 16 | 0.7937 | 0.0406 | 0.0443 | 0.0563 |
| 10-27 | `droidspeak_layers_10_27` | 18 | 0.7870 | 0.0422 | 0.0469 | 0.0607 |
| 6-25 | `droidspeak_layers_06_25` | 20 | 0.7937 | 0.0443 | 0.0498 | 0.0669 |
| 10-31 | `droidspeak_layers_10_31` | 22 | 0.7950 | 0.0469 | 0.0526 | 0.0716 |
| 2-25 | `droidspeak_layers_02_25` | 24 | 0.8004 | 0.0489 | 0.0554 | 0.0771 |
| 6-31 | `droidspeak_layers_06_31` | 26 | 0.8004 | 0.0515 | 0.0581 | 0.0820 |
| 4-31 | `droidspeak_layers_04_31` | 28 | 0.8204 | 0.0533 | 0.0608 | 0.0871 |
| 2-31 | `droidspeak_layers_02_31` | 30 | 0.8270 | 0.0551 | 0.0642 | 0.0929 |
| 0-31 | `droidspeak_layers_00_31` | 32 | 0.8004 | 0.0548 | 0.0641 | 0.0958 |

## 后续实验约束

第一轮 token-anchor A/B 实验固定这些层段，只改变 token selection anchor：

- `selection_anchor_layer=0`
- `selection_anchor_layer=recompute_layers[0]`

第一轮 scorer 保持：

- `selection_score_mode=k`

token ratio sweep 使用：

```text
0.10,0.20,0.30,0.40,0.50
```

非连续敏感层组合暂不进入第一轮主搜索。
