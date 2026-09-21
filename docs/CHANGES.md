# 變更與更正

## 實際測過的改動

| 項目 | 原二元B | 高並行B |
|---|---:|---:|
| client workers | 4 | 71 |
| HTTP連線池 | 8 | 80 |
| context | 32768 | 6144 |
| server max-num-seqs | 4 | 71 |
| batch token budget | 預設，重建為2048 | 同左 |
| weights／prompt／thinking／threshold | 固定／false／0.5 | 不變 |

分段版另加入 exact-prefix priming、每輪獨立cache salt、三個時間邊界metrics、逐類HTTP及client等待時間。
V2 runner關閉、TRITON_ATTN、FlashInfer sampler關閉及eager模式，是早期WSL相容性處置，不代表其他環境必須照用。

## 2026-09-21 恢復連線後的文件更正

- 原資料來源不是未知：已核對公開 `endman100/skill-router-nexus`；實驗工作目錄有未提交修改，隨附實際taxonomy與patch。
- 模型revision不是不可追溯：從下載metadata恢復並固定，但未重算完整權重hash。
- 不再只有對話匯出摘要：現在收錄原精度結果、原始API回覆、階段metrics與實際benchmark程式；2160個正式回覆由新驗證器重新檢查。
- 此次不沿用未核對的重構benchmark作為歷史原碼；`historical/` 是原檔，wrapper只準備新目錄和tokenization預檢。
- 既有速度經重算無須更改；保留6類A/B差異與3類跨配置翻轉，不稱同品質加速。
- 約0.95s僅是無競爭完全重疊的樂觀假設，不是實測或5090硬體下限。
- 本次沒有重跑GPU，也不將歷史health=200當成目前服務狀態；整理時API不可連線。

## 公開啟動器差異

將host改成本機127.0.0.1，明寫prefix caching、2048 token budget與下載revision；移除私人模型路徑，允許用MODEL_PATH指定本地副本。
模型訓練及量化不變。未測的新改動包括4096／8192 token budget、批次Completion API、移除重複grammar、精確尾端共享、CUDA Graphs／torch.compile。
