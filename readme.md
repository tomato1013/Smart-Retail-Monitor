## 📂 專案架構

```text
Project_Folder/
├── config.py           # [核心設定] 參數、ROI 區域、Prompt 提示詞
├── main.py             # [主程式] 啟動入口
├── utils.py            # [工具庫] 幾何計算、Server 連線
├── gemini_api.py       # [AI 服務] Gemini API 串接邏輯
├── .env                # [機密] API Key (需自行建立)
├── requirements.txt    # [套件] 專案環境
└── test video/         # [素材] 測試影片存放區
````

## 🚀 快速開始

### 1\. 環境準備

 Python 版本為 3.8 或以上

```bash
# 安裝必要套件
pip install -r requirements.txt
```

### 2\. 設定 API Key

現在的 API KEY 是我的，但如果一直用我的我會破產
所以請修改 `.env` 的檔案，並填入你的 Google API Key：

```ini
GOOGLE_API_KEY=您的_Gemini_API_Key_貼在這裡
```

### 3\. 準備測試影片

請確保影片檔案位於 `test video` 資料夾內，或是前往 `config.py` 修改 `VIDEO_DIR` 與 `VIDEO_FILENAME` 設定。

### 4\. 啟動系統

請執行主程式來開始監控：

```bash
python main.py
```

## ⚙️ 參數調整 (config.py)

若需要調整系統敏感度或區域，請直接修改 `config.py`：

  * **`CONFIDENCE_THRESHOLD`**: 物件偵測信心度門檻。
  * **`SHELF_AREA` / `SCANNER_AREA`**: 電子圍籬 (ROI) 的座標點。
  * **`PROMPT_PICKING`**: 給 Gemini 的取貨判斷提示詞。

## ⚠️ 常見問題 (FAQ)

**Q: 程式顯示 "ImportError: No module named..."**
A: 請確認是否有執行 `pip install -r requirements.txt`。

**Q: 程式顯示 "No API Key" 或 403 錯誤**
A: 請檢查 `.env` 檔案是否存在，且內容格式正確（不需要引號）。

**Q: 畫面一片黑或讀不到影片**
A: 請確認影片檔名是否與 `config.py` 中的 `VIDEO_FILENAME` 完全一致（包含副檔名）。