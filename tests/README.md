# tests 说明

当前 pytest 配置只收集 Mistral/MistralLite DroidBlend 相关测试和 DroidSpeak 基础 smoke tests。

重点测试：

- `test_droidblend_prefill.py`
- `test_partial_prefill_smoke.py`
- `test_experiment_methods_smoke.py`
- `test_cache_utils.py`
- `test_metrics.py`
- `test_model_loading.py`

其他测试文件来自旧 Llama-3.1 尝试稿，保留作历史参考，不作为当前目标的验收范围。
