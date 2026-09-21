# 方法與設定

## 固定条件

RTX 5090 32GB；Windows 11 + WSL2；Python 3.12、vLLM 0.29.0、PyTorch 2.13.0、CUDA 13.0、driver 610.88。
模型 `unsloth/Qwen3.8-27B-NVFP4`，無微調、無 MTP。歷史環境與本次補查環境分開記錄。

| 參數 | 設定 |
|---|---|
| 取樣 | temperature=0、top_p=1、top_k=-1、seed=20260920 |
| Thinking | `chat_template_kwargs.enable_thinking=false` |
| 二元輸出 | max_tokens=1；allowed_token_ids=[1802,3721]；choice=[true,false] |
| 候選機率 | 指定兩 token logprobs；P(true)=exp(lt)/(exp(lt)+exp(lf))，用穩定正規化 |
| 判定與回傳 | P(true)≥0.5；全部71個key，值必須為bool |
| Cache | FP8 E4M3；prefix caching；chunked prefill；Mamba align |
| 執行相容性 | TRITON_ATTN、enforce-eager；V2 runner與FlashInfer sampler關閉 |
| 記憶體 | gpu-memory-utilization=0.90 |
| token batch budget | 原CLI未指定；依當時安裝版本重建為2048，非live scheduler introspection |
| 32k4 | context=32768；client=4；server sequences=4；HTTP pool=8 |
| 6k71 | context=6144；client=71；server sequences=71；HTTP pool=80 |

`data/` 保存原 taxonomy、task、system、實際請求模板與 token 前綴；不是依對話重新猜寫。
最長二元輸入4991 tokens＋1輸出，6K不截斷。快取tokens仍計入context。

## 測試程序

**01 paired：** A完整JSON與B四路各暖機一次，再各做一次獨立冷前綴試跑，均不列正式統計。
接著各10次、交錯A/B順序；每方法固定自己的cache salt。A要求71個必填boolean、禁止額外keys，max_tokens=2048，實測649輸出tokens；B為71次單token請求。

**02／03 segmented：** 整體暖機一輪排除，每正式輪用新cache salt。
1. 71個chat prompts逐token最長共同前綴4953 tokens；Completion API先處理，產生一個丟棄token（max_tokens=0實測HTTP400）。不把該token加入分支輸入。
2. 用同salt提交全部71個完整分類請求，02四路／03最多71路。
3. 從logprobs計算機率，驗證71個bool並JSON序列化。

這是同一個任務重複測量，不是跨71個任務的準確率評測。每類的判斷不條件於其他類的輸出；A的後續輸出則會看到前面已生成tokens，因此不保證語意等價。

## 計時邊界

`總牆鐘 = 請求準備 + prefix HTTP + 中途 metrics + 全部分類HTTP/排隊 + 組裝`

模型載入、暖機、事前tokenization、首尾metrics讀取與檔案儲存不列入。原paired程式的進度寫入包含在其時間內。
伺服器時間來自階段histogram count/sum差值，核對1個prefix及71個分類；並行請求區間重疊，不能將sum當成總牆鐘或純CUDA kernel時間。
單token時首末token相同，所以decode區間為零。

每類輸入4983–4991 tokens、快取命中4704；全部未命中20032。其中共享尾端249×71=17679 tokens重算。不得把88.25%的token減量當作88.25%延遲改善保證。

## 重現邊界

公開wrapper先檢查tokenization，再逐位元複製 `historical/` 對應程式到新輸出目錄執行；原始程式的SHA-256已保存。
啟動器改為本機127.0.0.1、明寫2048 token budget與模型revision，其他效能設定不變。原server綁0.0.0.0。
本次只重驗歷史資料與程式，未重跑推論；新平台仍須實際測試。
沒有人工ground truth，不將任何舊A答案當成真值；機率未校準。跨batch結果翻轉尚未定位。
