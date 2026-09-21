# Qwen3.8 Jev-like：71 類二元分類

## 前提

本實驗起點是 HF [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) 的 **Parallel Constrained Decoding**，另參考其 [Transformers 版本](https://huggingface.co/shreyansh26/Qwen-2.5-1B-RLCD)：
對有限選項欄位共用前綴、平行判斷，減少逐 token 生成完整 JSON 的工作。71 類各自回答 `true/false`，可同時成立，適合用來檢驗這個思路。

## 實驗目標

**嘗試將上述 HF repo 的方法思路套用到 Qwen3.8-27B-NVFP4＋vLLM，確認相較直接 Structured Output 是否更快、快多少。**
固定同一模型、任務與 71 類定義，各組測 10 次；比較取得完整 71-key bool dict 的總耗時與答案差異，並以分段計時分析瓶頸。

本次以 vLLM prefix caching＋批次請求實作其核心思路，未直接移植原 repo 的引擎或 Tree Parallel，亦未進行 Jev／RLCD 訓練；不預設能重現原作者的加速倍率。

## 結果

RTX 5090 32GB、Windows 11／WSL2、vLLM 0.29.0。**同一個任務，各組 10 次**，不是 10 種任務。

| 方法 | Context／client workers／server sequences | 計時條件 | 中位數 |
|---|---|---|---:|
| A：JSON Schema | 32K／1／4 | 熱前綴 | 46.589 s |
| B：二元分類 | 32K／4／4 | 熱前綴 | 4.398 s |
| B：四路分段 | 32K／4／4 | 冷前綴＋分類＋組裝 | 5.040 s |
| B：71 路分段 | 6K／71／71 | 冷前綴＋分類＋組裝 | **3.465 s** |

A/B 熱前綴快 **10.59×**，每輪 **6 類答案不同**。71 路比四路分段快 **1.45×**，另有 **3 類翻轉**。
後兩組在不同時段測量、同時改動 context 與並行度，不能單獨歸因於某個參數。
沒有人工標註；一致率不等於正確率，機率未校準。

<details>
<summary>全部 10 次總耗時（秒）</summary>

| 次數 | A 熱前綴 | B 四路熱前綴 | B 四路冷分段 | B 71 路冷分段 |
|---:|---:|---:|---:|---:|
| 1 | 46.212 | 4.640 | 5.311 | 3.457 |
| 2 | 47.421 | 3.979 | 4.991 | 3.454 |
| 3 | 46.492 | 4.285 | 5.109 | 3.516 |
| 4 | 47.272 | 4.613 | 4.810 | 3.462 |
| 5 | 45.242 | 4.736 | 5.158 | 3.544 |
| 6 | 47.027 | 4.122 | 5.058 | 3.508 |
| 7 | 46.685 | 3.995 | 4.767 | 3.460 |
| 8 | 45.767 | 4.511 | 5.402 | 3.455 |
| 9 | 45.273 | 4.580 | 4.797 | 3.488 |
| 10 | 46.801 | 4.043 | 5.022 | 3.468 |

</details>

## 方法與理論值

A 逐 token 寫 JSON；B 重用共同前綴，每類只生成 **1 token**（`true=1802`／`false=3721`），再組成 dict。
兩者均關閉 thinking、temperature=0。B 將兩候選 logprobs 正規化得到 P(true)，以 **0.5** 判定。

分段測試每輪使用新 cache salt：冷前綴 → 71 類 → 組裝，暖機不計。前綴請求多產生一個丟棄 token，
並非純 GPU prefill。總時間包含 HTTP、排隊與中途讀取指標，不包含載入模型及存檔。
最新平均：**0.704 s 前綴＋2.771 s 分類＋0.000649 s 組裝**，另約 0.005 s 指標開銷。

**完全平行的理想情境：** `T = T_prefix + max(T₁…T₇₁) + T_assembly`。
假設 71 類都能維持舊四路平均 0.2406 s、完全重疊且無資源競爭：
`0.7042 + 0.2406 + 0.000649 ≈ 0.9455 s`，含量測開銷約 **0.95 s**。
這以平均代替未知的最慢分支，**不是實測或硬體下限**。71 個 HTTP 並行也不等於一次 GPU forward。

## 重現

僅保留四個檔案：[程式](benchmark.py)、[固定輸入與設定](experiment.json)、[40 次數據與各類結果](results.json)、本頁。
輸入約 5K tokens，6K 可完整容納。固定 NVFP4／FP8 KV、Triton、eager、batch token budget=2048；
模型 revision 已固定。需要相容 NVIDIA GPU／驅動、Linux 或 WSL2、Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
pip install vllm==0.29.0 torch==2.13.0 httpx==0.28.1 jsonschema==4.25.1
python benchmark.py --serve 6k71
```

另一終端啟用相同環境，執行：

```bash
python benchmark.py --profile 6k71 --out runs/6k71.json
python benchmark.py --verify  # 離線核對收錄結果，不跑 GPU
```

四路分段：先停止舊服務，再 `--serve 32k4`，測試用 `--profile 32k4`。
A/B：同一 32k4 服務，測試加 `--protocol paired`。預設各 10 次，每次指定新的 `--out`。
程式檢查版本、分詞與前綴，不截斷或自動重試推論。已有權重可用環境變數 `MODEL_PATH` 指向相同版本。

本程式為精簡重構，未重新跑 GPU；完整歷史依賴 lock 未保留，不保證位元級重現。
[原始程式與證據封存](https://github.com/endman100/research-Qwen3.8-JevLike/tree/6e2747a078d9e8d00861d218c49bbb6c140423d2)保留在 Git 歷史，原始數據未變更。

## 來源

**資料：** [skill-router-nexus](https://github.com/endman100/skill-router-nexus) 的 71 類快照＋一則自建需求，均在 `experiment.json`。
快照包含未提交的 Writing-Craft 描述修改，以收錄內容為準，不是 Jev 的測試集。

**方法參考：** [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [MLX 版本](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) · [Transformers 版本](https://huggingface.co/shreyansh26/Qwen-2.5-1B-RLCD)。
**實際執行：** [Unsloth 模型](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) · [vLLM](https://github.com/vllm-project/vllm)。
