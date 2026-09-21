# 原始量測程式

`paired.py`、`segmented4.py`、`segmented71.py` 直接複製實驗當時檔案，SHA-256 見 `results/provenance.json`，不是可攜重構。
它們讀取同目錄下的輸入資料；請用 `scripts/benchmark.py` 在新 `runs/` 目錄准备資料並執行，勿直接在此執行或覆蓋歷史結果。

原paired程式具有checkpoint續跑邏輯，歷史曾因進度檔鎖定續跑；新wrapper要求不存在的输出目錄，不會自動選取較快樣本。
部分metadata為原程式寫死的歷史配置；wrapper會預檢context、版本及tokenization，server sequence上限仍須從啟動參數核對。
