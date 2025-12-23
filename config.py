import os
import sys
from dotenv import load_dotenv

# 載入 API KEY
load_dotenv()

# ==================== 1. 系統與 API 設定 ====================
BASE_URL = "http://127.0.0.1:8000"

# Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("錯誤: 未偵測到 GOOGLE_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash-lite"

# API 費用 (Gemini Flash 2.5 Lite)
PRICE_PER_1M_INPUT = 0.1
PRICE_PER_1M_OUTPUT = 0.2

# ==================== 2. 相機與模型設定 ====================
CAM_NAME = "驛結世紀PoC Videos (53).mp4"
TASK_NAME = "poc-efs"
MODEL_NAME = "Human"

# 影片來源設定
VIDEO_DIR = "test video" 

# 2. 定義檔名
VIDEO_FILENAME = CAM_NAME 

if VIDEO_FILENAME == "0":
    VIDEO_SRC = "0"
else:
    full_path = os.path.join(VIDEO_DIR, VIDEO_FILENAME)
    VIDEO_SRC = os.path.abspath(full_path)

# 為了防止路徑錯誤，建議加這行檢查 (選用)
if VIDEO_FILENAME != "0" and not os.path.exists(VIDEO_SRC):
    print(f"警告：找不到影片！請檢查路徑：{VIDEO_SRC}")

SAVE_DIR = "training_data"
# ==================== 3. 邏輯判定參數 ====================
CONFIDENCE_THRESHOLD = 0.45 # 信心度門檻

MAX_MISSED_FRAMES = 20     # 容許中斷偵數

# 接觸秒數判定
TRIGGER_SECONDS_PICK = 2.0  # 取貨觸發時間
TRIGGER_SECONDS_SCAN = 1.0  # 掃描觸發時間

# 冷卻時間設定 (秒)
COOLDOWN_NORMAL = 5.0
COOLDOWN_ERROR = 60.0  # 當遇到 API 429 Too Many Requests 時

# 狀態常數
STAGE_PICKING = 1
STAGE_SCANNING = 2

# ==================== 4. 偵測區域 (ROI) ====================
# 電子圍籬 - 貨架 (門口左邊那個貨架誤判率太高，所以先註解掉了)
SHELF_AREA = [
    [{"x": 0.4982, "y": 0.4357}, {"x": 0.5366, "y": 0.0202}, {"x": 0.9991, "y": 0.2892}, {"x": 0.9964, "y": 0.9952}, {"x": 0.6267, "y": 0.9940}], 
    [{"x": 0.0016, "y": 0.1222}, {"x": 0.2691, "y": 0.0322}, {"x": 0.2558, "y": 0.3566}, {"x": 0.2733, "y": 0.5922}, {"x": 0.2433, "y": 0.7077}, {"x": 0.2075, "y": 0.7133}, {"x": 0.1616, "y": 0.9977}, {"x": 0.0016, "y": 0.9966}], 
    #[{"x": 0.278125,"y": 0.056666666666666664},{"x": 0.275,"y": 0.3175},{"x": 0.319375,"y": 0.32083333333333336},{"x": 0.3225,"y": 0.4525},{"x": 0.35875,"y": 0.35},{"x": 0.35,"y": 0.059166666666666666}]
]

# 電子圍籬 - 掃描儀
SCANNER_AREA = [
    [{"x": 0.2857, "y": 0.3210}, {"x": 0.2779, "y": 0.3632}, {"x": 0.2586, "y": 0.3641}, {"x": 0.2586, "y": 0.3485}, {"x": 0.2570, "y": 0.3218}]
]

# ==================== 5. LLM Prompts ====================

# 階段 1: 取貨
PROMPT_PICKING = """
Analyze the image to determine if the person is retrieving a package from the shelf.

Answer YES only if ALL of the following are true:
1. **Physical Contact:** The person's hand is touching or holding a box, package, or item on the shelf. The package may be brown, black or white
2. **Retrieval Action:** The person is grasping the item and pulling it towards themselves or lifting it off the shelf.
3. **Intention:** The person is looking toward the shelf and has intention to retrieve their item. 


Answer NO if:
- The person is just standing near the shelf.
- The person is not standing in front of the shelf. 
- The person's hands are empty and not touching any cargo.
- The person is just resting their hand on the shelf structure (without holding an item).
- The person is using a phone and no intention of going to get package.
- The person is carrying packages and want to go out from door.

Output Format: Start with YES or NO, followed by a very short reason.
"""

# 階段 2: 掃描
PROMPT_SCANNING = """
Analyze the image to determine if the person is scanning an item on the checkout counter platform.

The scanner is a flat platform/window on the desk surface (or a specific area on the counter).

Answer YES if:
1. The person is holding a package/item.
2. The item is being placed onto, sliding over, or hovering directly above the counter/scanner surface.
3. The person is pausing (dwelling) with the item in that specific area to register a scan.

Answer NO if:
- The person is just holding an item far away from the counter surface.
- The person is walking past the counter without stopping.
- The counter is empty.

Output Format: Start with YES or NO, followed by a very short reason.
"""