import cv2
import time
import numpy as np
import requests
import atexit

import config
import utils
import gemini_api

def main():
    # ==================== 1. 初始化 ====================
    # 啟動 Server 與相機
    if not utils.setup_server():
        print("❌ 程式終止：Server 初始化失敗")
        return

    # 註冊程式結束時的清理動作 (關閉相機)
    atexit.register(utils.stop_server)

    print(f"\n[Step 3] 啟動監控 (Task: {config.TASK_NAME})...")
    print("------------------------------------------------")
    
    # 建立 Session 以重複使用連線
    sess = requests.Session()
    detect_url = f"{config.BASE_URL}/api/detect_frame_od"
    result_url = f"{config.BASE_URL}/api/detect_frame_result_od"
    
    # 狀態變數初始化
    current_stage = config.STAGE_PICKING
    touch_start_time = 0
    is_touching = False
    missed_frames = 0
    llm_cooldown = 0
    
    # UI 顯示文字
    system_msg = "STAGE 1: Waiting for Pick..."
    last_llm_result = None

    # [新增] 設定視窗屬性為 NORMAL，這樣你就可以用滑鼠拉視窗大小了
    cv2.namedWindow("Workflow Manager", cv2.WINDOW_NORMAL) 
    
    # [選用] 如果你不想手動拉，可以強制設定一個初始大小 (例如 1280x720)
    cv2.resizeWindow("Workflow Manager", 1280, 720)

    # ==================== 2. 主迴圈 ====================
    while True:
        try:
            # --- A. 取得影像與偵測結果 ---
            detect_data = {
                'name': config.TASK_NAME,
                'cam_name': config.CAM_NAME,
                'threshold': config.CONFIDENCE_THRESHOLD
            }
            
            # 請求影像
            img_resp = sess.post(detect_url, data=detect_data, timeout=5)
            if img_resp.status_code != 200 or not img_resp.content:
                time.sleep(0.05); continue
            
            # 解碼影像
            arr = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None: continue
            h, w, _ = frame.shape
            
            # 請求 JSON 結果
            res_resp = sess.get(result_url, params={'name': config.TASK_NAME}, timeout=5)
            data = res_resp.json()
            
            detections = []
            if 'result' in data and 'last_location' in data['result']:
                detections = data['result']['last_location']
            
            # --- B. 準備偵測區域 (ROI) ---
            # 根據目前階段，決定要監控「貨架」還是「掃描器」
            active_polygons = []
            target_data = config.SHELF_AREA if current_stage == config.STAGE_PICKING else config.SCANNER_AREA
            
            # 將比例座標轉換為像素座標
            for poly in target_data:
                pts = [[int(pt['x'] * w), int(pt['y'] * h)] for pt in poly]
                active_polygons.append(np.array(pts, np.int32))

            # --- C. 判斷邏輯 ---
            detected_valid_action = False
            
            if detections:
                for obj in detections:
                    lbl = str(obj.get('label') or obj.get('cls') or 'unknown').lower()
                    
                    # 只關心「人」
                    if 'person' in lbl or 'human' in lbl:
                        # 取得座標 (處理歸一化座標)
                        raw_x1, raw_y1, raw_x2, raw_y2 = float(obj['x1']), float(obj['y1']), float(obj['x2']), float(obj['y2'])
                        if raw_x2 <= 1.0: 
                            x1, y1, x2, y2 = int(raw_x1 * w), int(raw_y1 * h), int(raw_x2 * w), int(raw_y2 * h)
                        else: 
                            x1, y1, x2, y2 = int(raw_x1), int(raw_y1), int(raw_x2), int(raw_y2)

                        hit = False
                        color = (255, 0, 0) # 預設藍色 (沒接觸)

                        # 呼叫 utils 進行幾何判定
                        if current_stage == config.STAGE_PICKING:
                            _, hand_hit = utils.check_body_and_hand_contact(x1, y1, x2, y2, active_polygons)
                            if hand_hit: hit = True; color = (0, 0, 255) # 紅色 (取貨中)
                        
                        elif current_stage == config.STAGE_SCANNING:
                            if utils.check_scanner_overlap(x1, y1, x2, y2, active_polygons): 
                                hit = True; color = (0, 255, 0) # 綠色 (掃描中)
                        
                        if hit: detected_valid_action = True
                        
                        # 繪製 Bounding Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # 如果是取貨階段，畫出探測點方便除錯
                        if current_stage == config.STAGE_PICKING:
                            ph_h = y2 - y1
                            probes = [(x1, y1), (x2, y1), (x1, int(y1+ph_h*0.3)), (x2, int(y1+ph_h*0.3))]
                            for pt in probes: cv2.circle(frame, pt, 3, (0, 255, 0), -1)

            # 繪製監控區域 (ROI)
            poly_color = (0, 255, 255) if current_stage == config.STAGE_PICKING else (0, 255, 0)
            for poly_np in active_polygons: 
                cv2.polylines(frame, [poly_np], True, poly_color, 2)

            # --- D. 計時與狀態機觸發 ---
            current_time = time.time()
            
            if detected_valid_action:
                missed_frames = 0 # 重置丟失計數
                if not is_touching: 
                    touch_start_time = current_time
                    is_touching = True
                
                duration = current_time - touch_start_time
                cv2.putText(frame, f"Timer: {duration:.1f}s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                # 判斷是否滿足觸發 AI 的時間門檻
                trigger_threshold = config.TRIGGER_SECONDS_PICK if current_stage == config.STAGE_PICKING else config.TRIGGER_SECONDS_SCAN
                
                if duration > trigger_threshold and current_time > llm_cooldown:
                    
                    # === 狀態 1: 取貨 ===
                    if current_stage == config.STAGE_PICKING:
                        print(">>> (Stage 1) 觸發 AI 判讀...")
                        is_picking, reason = gemini_api.analyze_picking(frame) # 呼叫 gemini_api
                        last_llm_result = f"Picking: {'YES' if is_picking else 'NO'} ({reason[:15]}...)"
                        
                        if is_picking:
                            print(">>> ✅ 取貨確認，切換至掃描階段。")
                            current_stage = config.STAGE_SCANNING
                            system_msg = "STAGE 2: Please Scan Item..."
                            is_touching = False; touch_start_time = 0
                        else:
                            # 判斷是否需要長冷卻 (API Error 429)
                            cooldown_time = config.COOLDOWN_ERROR if "429" in reason else config.COOLDOWN_NORMAL
                            llm_cooldown = current_time + cooldown_time

                    # === 狀態 2: 掃描 ===
                    elif current_stage == config.STAGE_SCANNING:
                        print(">>> (Stage 2) 觸發 AI 判讀...")
                        is_scanning, reason = gemini_api.analyze_scanning(frame) # 呼叫 gemini_api
                        last_llm_result = f"Scanning: {'YES' if is_scanning else 'NO'} ({reason[:15]}...)"
                        
                        if is_scanning:
                            print(">>> ✅ 掃描確認！開門！")
                            # 開門特效
                            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
                            cv2.putText(frame, "DOOR OPENED!", (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                            cv2.imshow("Workflow Manager", frame)
                            cv2.waitKey(3000) # 暫停 3 秒展示開門
                            
                            # 重置回第一階段
                            current_stage = config.STAGE_PICKING
                            system_msg = "STAGE 1: Waiting for Pick..."
                            is_touching = False; touch_start_time = 0; last_llm_result = None
                        else:
                            cooldown_time = config.COOLDOWN_ERROR if "429" in reason else config.COOLDOWN_NORMAL
                            llm_cooldown = current_time + cooldown_time
            else:
                # 容錯機制：如果人短暫消失(閃爍)，不要馬上重置 Timer
                if is_touching:
                    missed_frames += 1
                    if missed_frames > config.MAX_MISSED_FRAMES:
                        is_touching = False
                        touch_start_time = 0

            # --- E. 畫面資訊顯示 ---
            cv2.putText(frame, system_msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if last_llm_result: 
                cv2.putText(frame, last_llm_result, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            if current_time < llm_cooldown: 
                cv2.putText(frame, f"Cooldown: {int(llm_cooldown - current_time)}s", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Workflow Manager", frame)
            
            # 按 Q 離開
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

        except Exception as e:
            if "timed out" not in str(e):
                print(f"[Loop Error]: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()