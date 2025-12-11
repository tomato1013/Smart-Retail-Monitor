import cv2
import numpy as np
import requests
import time
import config  # 匯入我們剛寫好的設定檔

# ==================== 1. 成本計算工具 ====================

def calculate_and_print_cost(response_json):
    """
    解析 Gemini 回傳的 JSON，計算 Token 使用量並印出費用。
    """
    usage = response_json.get('usageMetadata', {})
    
    prompt_tokens = usage.get('promptTokenCount', 0)
    candidate_tokens = usage.get('candidatesTokenCount', 0)
    total_tokens = usage.get('totalTokenCount', 0)
    
    # 使用 config 中的費率進行計算
    cost_input = (prompt_tokens / 1_000_000) * config.PRICE_PER_1M_INPUT
    cost_output = (candidate_tokens / 1_000_000) * config.PRICE_PER_1M_OUTPUT
    total_cost = cost_input + cost_output
    
    print(f"   [💰 Token] In: {prompt_tokens} | Out: {candidate_tokens} | Total: {total_tokens}")
    print(f"   [💸 Cost ] USD: ${total_cost:.7f} (約 TWD {total_cost*32:.4f})")
    
    return total_tokens, total_cost

# ==================== 2. Leda Server 控制工具 ====================

def setup_server():
    """
    初始化 Server：開啟相機、載入 OD 模型。
    回傳: True (成功) / False (失敗)
    """
    print(f"\n[System] 正在初始化 Leda Server (Base: {config.BASE_URL})...")
    
    # 1. 開啟相機
    print(f"   -> 開啟相機: {config.CAM_NAME}...")
    cam_form_data = {
        'name': config.CAM_NAME,
        'cam_type': 'cv', 
        'src': config.VIDEO_SRC,
        'is_capture': 'false' 
    }
    
    try:
        requests.post(f"{config.BASE_URL}/api/camera/open", data=cam_form_data, timeout=5)
    except Exception as e:
        print(f"   相機連線失敗: {e}")
        return False

    time.sleep(1.0) # 等待相機資源就緒

    # 2. 初始化 OD 模型
    print(f"   -> 初始化 OD 模型 ({config.MODEL_NAME})...")
    init_payload = {
        'name': config.TASK_NAME,
        'model_name': config.MODEL_NAME,
        'draw_alert_area': False,
        'crop_m': 1, 
        'crop_n': 1,
        'alert_area': [], 
        'cls_w_h': []
    } 
    
    try:
        res = requests.post(f"{config.BASE_URL}/api/init_od", json=init_payload, timeout=5)
        if res.status_code != 200: 
            print(f"   OD 初始化失敗: {res.text}")
            return False
    except Exception as e:
        print(f"   OD 初始化連線錯誤: {e}")
        return False

    print("   ✅ 系統初始化完成")
    return True

def stop_server():
    """
    關閉相機資源 (通常在程式結束時呼叫)
    """
    print("\n[System] 清理資源...")
    try:
        requests.post(f"{config.BASE_URL}/api/camera/close", json={'name': config.CAM_NAME}, timeout=2)
        print(f"   ✅ 已關閉相機 ({config.CAM_NAME})")
    except Exception as e:
        print(f"   關閉相機時發生錯誤 (可能已關閉): {e}")

# ==================== 3. 幾何判定邏輯 ====================

def check_body_and_hand_contact(x1, y1, x2, y2, polygons_px):
    """
    判斷人體是否接觸貨架區域 (Stage 1)。
    邏輯：
    1. 身體中心點 (Body Center) 是否在區域內
    2. 上半身探測點 (Probes) 是否在區域內 (模擬手部)
    """
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    body_contact = False
    hand_contact = False
    
    # 檢查身體中心
    for poly_pts in polygons_px:
        if cv2.pointPolygonTest(poly_pts, (cx, cy), False) >= 0:
            body_contact = True
            break
            
    # 檢查手部/上半身 (設定幾個探測點)
    h = y2 - y1
    # 探測點：左上、右上、左肩(30%處)、右肩(30%處)
    probes = [
        (x1, y1), 
        (x2, y1), 
        (x1, int(y1 + h * 0.3)), 
        (x2, int(y1 + h * 0.3))
    ]
    
    for pt in probes:
        for poly_pts in polygons_px:
            if cv2.pointPolygonTest(poly_pts, pt, False) >= 0:
                hand_contact = True
                break
        if hand_contact:
            break
            
    return body_contact, hand_contact

def check_scanner_overlap(x1, y1, x2, y2, scanner_polygons_px):
    """
    判斷人體是否與掃描器區域重疊 (Stage 2)。
    邏輯：計算 Bounding Box 是否有交集 (Overlap)。
    """
    for poly in scanner_polygons_px:
        px, py, pw, ph = cv2.boundingRect(poly)
        
        # 計算重疊區域的座標
        ix1 = max(x1, px)
        iy1 = max(y1, py)
        ix2 = min(x2, px + pw)
        iy2 = min(y2, py + ph)
        
        # 如果交叉區域的寬高都大於 0，表示有重疊
        if ix1 < ix2 and iy1 < iy2:
            return True
            
    return False