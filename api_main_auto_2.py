import requests
import json
import time
import cv2
import numpy as np
import base64
import os
import atexit

# ==================== 1. 設定區 ====================
BASE_URL = "http://127.0.0.1:8000"

# [參數設定]
CAM_NAME = "test_2"
VIDEO_FILENAME = "驛結世紀PoC Videos (3).mp4" 

if VIDEO_FILENAME == "0": VIDEO_SRC = "0"
else: VIDEO_SRC = os.path.abspath(VIDEO_FILENAME)

TASK_NAME = "poc-efs" 
MODEL_NAME = "Human" 

#這是我的 API KEY，請手下留情
GOOGLE_API_KEY = "AIzaSyD6cUg_lU8QG5yq84Na1RGeVdSfymmtV6o" 

# 以 Gemini Flash2.5 Lite 做計算
PRICE_PER_1M_INPUT = 0.1
PRICE_PER_1M_OUTPUT = 0.2

# 電子圍籬 - 貨架 (門口左邊那個貨架誤判率太高，所以先註解掉了)
SHELF_AREA = [
    [{"x": 0.4982142857142857,"y": 0.4357142857142857},{"x": 0.5366071428571427,"y": 0.020238095238095236},{"x": 0.9991071428571429,"y": 0.2892857142857143},{"x": 0.9964285714285712,"y": 0.9952380952380951},{"x": 0.6267857142857143,"y": 0.994047619047619}], 
    [{"x": 0.0016666666666666666,"y": 0.12222222222222222},{"x": 0.26916666666666667,"y": 0.03222222222222222},{"x": 0.2558333333333333,"y": 0.35666666666666674},{"x": 0.27333333333333326,"y": 0.5922222222222222},{"x": 0.24333333333333332,"y": 0.7077777777777778},{"x": 0.2075,"y": 0.7133333333333335},{"x": 0.16166666666666668,"y": 0.9977777777777777},{"x": 0.0016666666666666666,"y": 0.9966666666666667}], 
    #[{"x": 0.278125,"y": 0.056666666666666664},{"x": 0.275,"y": 0.3175},{"x": 0.319375,"y": 0.32083333333333336},{"x": 0.3225,"y": 0.4525},{"x": 0.35875,"y": 0.35},{"x": 0.35,"y": 0.059166666666666666}]
]

# 電子圍籬 - 掃描儀
SCANNER_AREA = [
    [{"x": 0.2857774828577748, "y": 0.32107866371498833}, {"x": 0.27794103706512463, "y": 0.36326275699397204}, {"x": 0.2586501090150725, "y": 0.36418283305359295}, {"x": 0.2586501090150725, "y": 0.3485415400400371}, {"x": 0.25708597971371694, "y": 0.3218593343110302}]
    ]

# 接觸秒數判定
TRIGGER_SECONDS_PICK = 2.0 # 取貨
TRIGGER_SECONDS_SCAN = 1.0 # 掃描

MAX_MISSED_FRAMES = 15  # 允許中斷偵數

# 信心度設定
threshold = 0.4  

SAVE_DIR = "training_data"

# 狀態常數設定
STAGE_PICKING = 1
STAGE_SCANNING = 2

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def calculate_and_print_cost(response_json):
    """解析 Token 使用量並計算費用"""
    usage = response_json.get('usageMetadata', {})
    
    prompt_tokens = usage.get('promptTokenCount', 0)
    candidate_tokens = usage.get('candidatesTokenCount', 0)
    total_tokens = usage.get('totalTokenCount', 0)
    
    # 計算費用
    cost_input = (prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT
    cost_output = (candidate_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT
    total_cost = cost_input + cost_output
    
    print(f"   [💰 Token] In: {prompt_tokens} | Out: {candidate_tokens} | Total: {total_tokens}")
    print(f"   [💸 Cost ] USD: ${total_cost:.7f} (約 TWD {total_cost*32:.4f})")
    
    return total_tokens, total_cost

# ==================== 2. Server 初始化 ====================

def setup_server():
    print("\n[System] 正在初始化 Leda Server (V10.1)...")
    
    # 1. 開啟相機
    print(f"  -> 開啟相機: {CAM_NAME}...")
    cam_form_data = {
        'name': CAM_NAME,
        'cam_type': 'cv', 
        'src': VIDEO_SRC,
        'is_capture': 'false' 
    }
    try:
        requests.post(f"{BASE_URL}/api/camera/open", data=cam_form_data, timeout=5)
    except Exception as e:
        print(f"  ❌ 相機連線失敗: {e}"); return False

    time.sleep(1.0)

    # 2. 初始化 OD 模型
    print(f"  -> 初始化 OD 模型 ({MODEL_NAME})...")
    init_payload = {
        'name': TASK_NAME,
        'model_name': MODEL_NAME,
        'draw_alert_area': False,
        'crop_m': 1, 'crop_n': 1,
        'alert_area': [], 'cls_w_h': []
    } 
    try:
        res = requests.post(f"{BASE_URL}/api/init_od", json=init_payload, timeout=5)
        if res.status_code != 200: 
            print(f"  ❌ OD 初始化失敗: {res.text}"); return False
    except: return False

    print("  ✅ 系統初始化完成")
    return True

def stop_server():
    print("\n[System] 清理資源...")
    try:
        requests.post(f"{BASE_URL}/api/camera/close", json={'name': CAM_NAME}, timeout=2)
        print(f"  ✅ 已關閉相機 ({CAM_NAME})")
    except: pass

atexit.register(stop_server)

# ==================== 3. Call Gemini 函式 ====================

def ask_llm_is_picking(image):
    print("\n[☁️ Gemini] 分析取貨動作 (Stage 1)...")
    model_name = "gemini-2.5-flash-lite" 
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        prompt_text = """
    Analyze the image to determine if the person is retrieving a package from the shelf.
    
    Answer YES only if ALL of the following are true:
    1. **Physical Contact:** The person's hand is touching a box, package, or item on the shelf.
    2. **Retrieval Action:** The person is grasping the item and pulling it towards themselves or lifting it off the shelf.
    
    Answer NO if:
    - The person is just standing near the shelf.
    - The person is not standing in front of the shelf. 
    - The person's hands are empty and not touching any cargo.
    - The person is just resting their hand on the shelf structure (without holding an item).
    - The person is using a phone and no intention of going to get package.
    - The person is carrying packages and want to go out from door.

    Output Format: Start with YES or NO, followed by a very short reason.
    """
        payload = {"contents": [{"parts": [{"text": prompt_text}, {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}]}]}
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            result_json = response.json()
            
            calculate_and_print_cost(result_json)

            if 'candidates' in result_json and result_json['candidates']:
                content = result_json['candidates'][0]['content']['parts'][0]['text']
                print(f"   -> Gemini: {content.strip()}")
                return ("yes" in content.lower()), content
            
    except: pass
    return False, "Error"

def ask_llm_is_scanning(image):
    print("\n[☁️ Gemini] 分析掃描動作 (Stage 2)...")
    model_name = "gemini-2.5-flash-lite" 
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        prompt_text ="""
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
        payload = {"contents": [{"parts": [{"text": prompt_text}, {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}]}]}
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            result_json = response.json()

            # [新增] 計算並印出費用
            calculate_and_print_cost(result_json)

            if 'candidates' in result_json and result_json['candidates']:
                content = result_json['candidates'][0]['content']['parts'][0]['text']
                print(f"   -> Gemini: {content.strip()}")
                return ("yes" in content.lower()), content
            
    except: pass
    return False, "Error"

# ==================== 4. 接觸判斷函式 ====================
def check_body_and_hand_contact(x1, y1, x2, y2, polygons_px):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    body_contact = False; hand_contact = False
    for poly_pts in polygons_px:
        if cv2.pointPolygonTest(poly_pts, (cx, cy), False) >= 0:
            body_contact = True; break
    h = y2 - y1
    probes = [(x1, y1), (x2, y1), (x1, int(y1 + h*0.3)), (x2, int(y1 + h*0.3))]
    for pt in probes:
        for poly_pts in polygons_px:
            if cv2.pointPolygonTest(poly_pts, pt, False) >= 0:
                hand_contact = True; break
        if hand_contact: break
    return body_contact, hand_contact

def check_scanner_overlap(x1, y1, x2, y2, scanner_polygons_px):
    for poly in scanner_polygons_px:
        px, py, pw, ph = cv2.boundingRect(poly)
        ix1, iy1 = max(x1, px), max(y1, py)
        ix2, iy2 = min(x2, px+pw), min(y2, py+ph)
        if ix1 < ix2 and iy1 < iy2: return True
    return False

# ==================== 5. 主迴圈  ====================
def main():
    if not setup_server(): 
        print("❌ 程式終止：Server 初始化失敗")
        return

    print(f"\n[Step 3] 啟動監控 (V10.1 - Direct Name)...")
    print("------------------------------------------------")
    
    detect_url = f"{BASE_URL}/api/detect_frame_od"
    result_url = f"{BASE_URL}/api/detect_frame_result_od"
    sess = requests.Session()
    
    current_stage = STAGE_PICKING
    touch_start_time = 0
    is_touching = False
    missed_frames = 0
    llm_cooldown = 0
    system_msg = "STAGE 1: Waiting for Pick..."
    last_llm_result = None

    while True:
        try:
            # 1. 發送偵測請求
            detect_data = {
                'name': TASK_NAME,
                'cam_name': CAM_NAME,
                'threshold': threshold
            }
            
            img_resp = sess.post(detect_url, data=detect_data, timeout=5)
            
            # 偵錯：如果拿不到圖，印出狀態碼
            if img_resp.status_code != 200 or not img_resp.content:
                # print(f"[Info] 尚未取得影像 (Code: {img_resp.status_code})")
                time.sleep(0.05); continue
            
            # 解碼圖片
            arr = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None: continue
            h, w, _ = frame.shape
            
            # 2. 取得 JSON 結果
            res_resp = sess.get(result_url, params={'name': TASK_NAME}, timeout=5)
            data = res_resp.json()
            
            detections = []
            if 'result' in data and 'last_location' in data['result']:
                detections = data['result']['last_location']
            
            # --- 邏輯處理 ---
            
            active_polygons = []
            target_data = SHELF_AREA if current_stage == STAGE_PICKING else SCANNER_AREA
            for poly in target_data:
                pts = [[int(pt['x'] * w), int(pt['y'] * h)] for pt in poly]
                active_polygons.append(np.array(pts, np.int32))

            detected_valid_action = False
            if detections:
                for obj in detections:
                    lbl = str(obj.get('label') or obj.get('cls') or 'unknown').lower()
                    if 'person' in lbl or 'human' in lbl:
                        if 'x1' in obj:
                            raw_x1, raw_y1, raw_x2, raw_y2 = float(obj['x1']), float(obj['y1']), float(obj['x2']), float(obj['y2'])
                            if raw_x2 <= 1.0: x1, y1, x2, y2 = int(raw_x1 * w), int(raw_y1 * h), int(raw_x2 * w), int(raw_y2 * h)
                            else: x1, y1, x2, y2 = int(raw_x1), int(raw_y1), int(raw_x2), int(raw_y2)

                            hit = False
                            color = (255, 0, 0)
                            if current_stage == STAGE_PICKING:
                                _, hand_hit = check_body_and_hand_contact(x1, y1, x2, y2, active_polygons)
                                if hand_hit: hit = True; color = (0, 0, 255)
                            elif current_stage == STAGE_SCANNING:
                                if check_scanner_overlap(x1, y1, x2, y2, active_polygons): hit = True; color = (0, 255, 0)
                            
                            if hit: detected_valid_action = True
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            if current_stage == STAGE_PICKING:
                                ph_h = y2 - y1
                                probes = [(x1, y1), (x2, y1), (x1, int(y1+ph_h*0.3)), (x2, int(y1+ph_h*0.3))]
                                for pt in probes: cv2.circle(frame, pt, 3, (0, 255, 0), -1)

            poly_color = (0, 255, 255) if current_stage == STAGE_PICKING else (0, 255, 0)
            for poly_np in active_polygons: cv2.polylines(frame, [poly_np], True, poly_color, 2)

            current_time = time.time()
            if detected_valid_action:
                missed_frames = 0
                if not is_touching: touch_start_time = current_time; is_touching = True
                duration = current_time - touch_start_time
                cv2.putText(frame, f"Timer: {duration:.1f}s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                if current_stage == STAGE_PICKING:
                    if duration > TRIGGER_SECONDS_PICK and current_time > llm_cooldown:
                        print(">>> (Stage 1) 呼叫 Gemini...")
                        is_picking, reason = ask_llm_is_picking(frame)
                        last_llm_result = f"Picking: {'YES' if is_picking else 'NO'} ({reason[:15]}...)"
                        if is_picking:
                            print(">>> 取貨確認，進入掃描階段。")
                            current_stage = STAGE_SCANNING
                            system_msg = "STAGE 2: Please Scan Item..."
                            is_touching = False; touch_start_time = 0
                        else: llm_cooldown = current_time + (60.0 if "429" in reason else 5.0)

                elif current_stage == STAGE_SCANNING:
                    if duration > TRIGGER_SECONDS_SCAN and current_time > llm_cooldown:
                        print(">>> (Stage 2) 呼叫 Gemini...")
                        is_scanning, reason = ask_llm_is_scanning(frame)
                        last_llm_result = f"Scanning: {'YES' if is_scanning else 'NO'} ({reason[:15]}...)"
                        if is_scanning:
                            print(">>> 掃描確認！開門！")
                            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
                            cv2.putText(frame, "DOOR OPENED!", (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                            cv2.imshow("Workflow Manager", frame); cv2.waitKey(3000)
                            current_stage = STAGE_PICKING
                            system_msg = "STAGE 1: Waiting for Pick..."
                            is_touching = False; touch_start_time = 0; last_llm_result = None
                        else:
                            llm_cooldown = current_time + (60.0 if "429" in reason else 5.0)
                            is_touching = False; touch_start_time = 0
            else:
                if is_touching:
                    missed_frames += 1
                    if missed_frames > MAX_MISSED_FRAMES: is_touching = False; touch_start_time = 0

            cv2.putText(frame, system_msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if last_llm_result: cv2.putText(frame, last_llm_result, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            if current_time < llm_cooldown: cv2.putText(frame, f"Cooldown: {int(llm_cooldown - current_time)}s", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Workflow Manager", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

        except Exception as e:
            if "timed out" not in str(e):
                print(f"[Loop Error]: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()