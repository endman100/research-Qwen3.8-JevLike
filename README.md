# research-Qwen3.8-JevLike

**用同一個 Qwen3.8-27B-NVFP4，把完整 JSON 生成改成 71 個獨立 `true/false` 判斷。**
RTX 5090 32GB／vLLM 0.29.0；一個固定 Skill Router 任務，各配置重複 10 次。
這是 Jev-like 推論實驗，**不是 Jev、RLCD 訓練或 Tree Parallel 的復現**。

## 結果

| 方法 | Context／client workers／server sequences | 計時範圍 | 中位耗時 |
|---|---|---|---:|
| A：完整 JSON Schema | 32K／1／4 | 熱前綴，完整 71-key JSON | **46.589 s** |
| B：71 個二元判斷 | 32K／4／4 | 熱前綴，完整 71-key dict | **4.398 s** |
| B：四路分段 | 32K／4／4 | 冷前綴＋71 類＋組裝 | **5.040 s** |
| B：高並行分段 | 6K／71／71 | 冷前綴＋71 類＋組裝 | **3.465 s** |

A/B 熱前綴：**10.59× 更快，但每輪 6 類答案不同**。
高並行 vs 四路分段：**1.45× 更快，但每輪 3 類翻轉**；這是歷史組間比較，不能單獨歸因於 context。

**結論：省掉大量 JSON token 可降低延遲，尚非同品質／無損加速。**
沒有人工標註；一致率不是正確率，模型機率未校準。
[全部 10 次數據與差異](docs/RESULTS.md) · [原始回覆封存](results/records.zip)

## 方法

```text
A：需求＋71 類定義 → JSON Schema 逐 token 生成 → 完整 bool dict
B：共享前綴快取 → 71 個 true/false 首 token 判斷 → 程式組裝相同 dict
```

B 每類 `max_tokens=1`、`enable_thinking=false`。`true/false` 各一個 token（1802／3721）；
從兩候選 logprobs 計算 `P(true)`，以 **0.5** 判定 bool。保留所有 71 keys，包括 `false`。
71 個 `P(true)` 不必加總為 1。71 個並行 HTTP 也不代表 71 個 GPU 分支一次完成。

最新平均：前綴 **0.704s**、71 類 **2.771s**、組裝 **0.649ms**；另有約 5ms 中途量測開銷。
單 token 的 `decode=0` 是首 token 後没有後續生成，不代表零運算。[方法與設定](docs/METHOD.md)

## 完全平行的理想估算（非實測）

`T = T_prefix + max(T₁ … T₇₁) + T_assembly`

假設全部分支同時執行，且每類仍維持四路測試平均 240.6ms、沒有新增資源競爭：
`0.7042 + 0.2406 + 0.000649 ≈ 0.9455s`，加量測開銷約 **0.95s**。
這以平均值代替未知的最慢分支，是樂觀情境，**不是 5090 的理論下限或速度保證**。
[公式、資料依據與限制](docs/THEORY.md)

## 重現

Linux／WSL2、Python 3.12、相容 NVIDIA 驅動。使用獨立環境，不與其他模型同時搶用 GPU：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-server.txt
bash scripts/serve.sh 6k71
```

另一個終端啟用相同環境：

```bash
python scripts/benchmark.py --profile 6k71 --protocol segmented --out runs/6k71
python scripts/verify_results.py   # 只驗證收錄的歷史結果，不跑 GPU
```

四路分段用 `serve.sh 32k4` 與 `--profile 32k4 --protocol segmented`；
A/B 對照用 `serve.sh 32k4` 與 `--profile 32k4 --protocol paired`。每次使用新的輸出目錄。
腳本會檢查模型版本、完整 token 長度與前綴，不截斷輸入或自動重試推論。

啟動器固定從下載 metadata 恢復的模型 revision；`historical/` 保存實際量測程式原檔，
公開 wrapper 只做分詞預檢並在新目錄執行它們。**本次整理未重新跑 GPU**。
完整歷史依賴 lock 未保留，不保證位元級重現。[變更與更正](docs/CHANGES.md)

## 資料與參考

資料：**[endman100/skill-router-nexus](https://github.com/endman100/skill-router-nexus)** 的 71 類工作目錄快照，
加一則固定中文需求；不是 Jev 或 HF repo 的 presets。
實驗時 `Writing-Craft` 描述有未提交修改，故以本 repo 的 [taxonomy](data/taxonomy.json) 與 [patch](data/router_worktree.patch) 為準。

方法：[Jev／TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、
[MLX Parallel Constrained Decoding](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)、
[Transformers 版本](https://huggingface.co/shreyansh26/Qwen-2.5-1B-RLCD)。
模型：[Unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)；引擎：[vLLM](https://github.com/vllm-project/vllm)。
[來源與版本](docs/SOURCES.md) · [完整驗證](results/verification.json)
