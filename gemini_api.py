import requests
import json
import base64
import cv2
import config  # 匯入設定檔
import utils   # 匯入工具 (為了計算費用)

def _send_gemini_request(image, prompt_text, task_name="General"):
    """
    (內部函式) 負責將圖片編碼並發送給 Gemini API。
    """
    # 1. 建構 API URL
    if not config.GOOGLE_API_KEY:
        print("❌ Error: API Key is missing.")
        return False, "No API Key"

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={config.GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}

    try:
        # 2. 影像前處理 (OpenCV -> JPEG -> Base64)
        if image is None:
            return False, "Empty Image"
            
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 3. 組合 Payload
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
                ]
            }]
        }
        
        # 4. 發送請求
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10)
        
        if response.status_code == 200:
            result_json = response.json()
            
            # 計算費用 (呼叫 utils 中的工具)
            utils.calculate_and_print_cost(result_json)

            # 解析回傳文字
            if 'candidates' in result_json and result_json['candidates']:
                content = result_json['candidates'][0]['content']['parts'][0]['text']
                cleaned_content = content.strip()
                print(f"   -> Gemini ({task_name}): {cleaned_content}")
                
                # 回傳 (是否為 Yes, 完整回應文字)
                return ("yes" in cleaned_content.lower()), cleaned_content
        else:
            print(f"❌ API Error ({response.status_code}): {response.text}")
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False, str(e)

    return False, "Unknown Error"

# ==================== 公開介面 ====================

def analyze_picking(image):
    """
    Stage 1: 分析是否正在取貨
    """
    print("\n[☁️ Gemini] 分析取貨動作 (Stage 1)...")
    # 使用 config 中設定好的 Prompt
    return _send_gemini_request(image, config.PROMPT_PICKING, task_name="Picking")

def analyze_scanning(image):
    """
    Stage 2: 分析是否正在掃描
    """
    print("\n[☁️ Gemini] 分析掃描動作 (Stage 2)...")
    # 使用 config 中設定好的 Prompt
    return _send_gemini_request(image, config.PROMPT_SCANNING, task_name="Scanning")