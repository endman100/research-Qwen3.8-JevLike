# 資料與方法來源

核對日期：2026-09-21。資料來源與方法參考分開列示，不移用他人效能數字。

## 實際資料

- Repository：[endman100/skill-router-nexus](https://github.com/endman100/skill-router-nexus)。
- 本次補查的base commit：[b3523bec95b03f0f0999bfbe07d2004120ce2a15](https://github.com/endman100/skill-router-nexus/tree/b3523bec95b03f0f0999bfbe07d2004120ce2a15)。
- **實驗使用未提交的工作目錄快照**：`Writing-Craft` 描述比base commit新增劇本範圍。[實際差異](../data/router_worktree.patch)。不可直接把base commit宣稱為精確資料版本。
- 原始router快照文字與本次工作目錄一致；[taxonomy.json](../data/taxonomy.json)直接複製原實驗檔、SHA-256與来源見 [provenance.json](../data/provenance.json)。
- [task.txt](../data/task.txt)是一則自建需求，[system.txt](../data/system.txt)是分類規則。不是Jev測試集，也不是下列HF repo的presets。

## 方法與軟體參考

| 來源 | 用途／區別 |
|---|---|
| [Jev／TypeSafe官方介紹](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 型別化決策與平行輸出的概念。Jev的RLCD為Reinforcement Learning for Calibrated Decisions；此實驗沒有其訓練或校準。 |
| [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) | Parallel Constrained Decoding／共享前綴概念參考；未使用其presets或程式作為本實驗引擎。 |
| [shreyansh26/Qwen-2.5-1B-RLCD](https://huggingface.co/shreyansh26/Qwen-2.5-1B-RLCD) | Transformers/PyTorch的batched、tree方法；與此vLLM多請求batching不同。 |
| [Unsloth模型固定版本](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/tree/f0b7c9e722f5565102fff8481c99e4d86ae099c7) | 實際模型；NVFP4＋FP8混合量化。 |
| [vLLM](https://github.com/vllm-project/vllm) | 執行版本0.29.0。 |
| [Structured outputs](https://docs.vllm.ai/en/v0.29.0/features/structured_outputs/) | JSON Schema與choice。 |
| [Prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | 相同token前綴重用；不是壓縮context。 |
| [Metrics](https://docs.vllm.ai/en/latest/design/metrics/) | TTFT、prefill、decode事件區間。 |
| [Hybrid cache](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) | 混合attention／recurrent狀態的快取配置。 |
| [Batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) | 不應預設跨batch結果完全相同；本實驗沒有啟用。 |

## 模型revision與依賴證據

模型revision從實驗前的本地Hugging Face下載metadata恢復為 `f0b7c9e722f5565102fff8481c99e4d86ae099c7`，並向HF API核對存在。
小型檔案的Git blob SHA-1與metadata etag一致；本次未重新雜湊22GB權重，不能將metadata當成完整權重位元驗證。
[檔案級證據](../configs/model_provenance.json)。
完整歷史pip lock没有保留；依賴清單與本次環境補查分開看，不把當前環境假稱當時freeze。
不包含模型權重、憑證、私人絕對路徑或整套skills；外部軟體與模型遵守各自授權。
