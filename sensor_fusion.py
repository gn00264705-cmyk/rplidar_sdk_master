import cv2
import numpy as np
import threading
import time
import queue
import math
import serial
import struct
import os

class SensorFusionRanging:
    """
    「攝影機與光達（LiDAR）資料融合測距」類別
    負責管理雙感測器的硬體連線、時間同步、座標系轉換與影像繪圖。
    """
    def __init__(self, lidar_port='/dev/ttyUSB0', baudrate=115200):
        self.lidar_port = lidar_port
        self.baudrate = baudrate
        
        self.is_running = False
        
        # 1. 【時間同步機制】: 建立 Queue
        self.cam_queue = queue.Queue(maxsize=1)
        self.lidar_queue = queue.Queue(maxsize=1)
        
        # 滑鼠點擊查詢參數
        self.click_point = None
        
        # ==========================================
        # 2. 攝影機內參矩陣 (Camera Intrinsic Matrix) K
        # ==========================================
        self.K = np.array([
            [600.0,   0.0, 320.0],  # 嘗試調高焦距 (f)，通常 IMX219 在 640x480 下約為 500-700
            [  0.0, 600.0, 240.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)
        
        # ==========================================
        # 3. 外參矩陣 (Extrinsic Matrix): 光達到相機的空間關係
        # ==========================================
        self.R = np.eye(3, dtype=np.float32)
        self.T = np.array([
            [  0.0],  # X方向平移
            [ 50.0],  # Y方向平移 (mm)
            [  0.0]   # Z方向平移
        ], dtype=np.float32)
        
        # ==========================================
        # 4. AI 物件辨識模型設定
        # ==========================================
        self.PROTOTXT = "MobileNetSSD_deploy.prototxt"
        self.MODEL = "MobileNetSSD_deploy.caffemodel"
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        
        self.net = None
        if os.path.exists(self.PROTOTXT) and os.path.exists(self.MODEL):
            try:
                self.net = cv2.dnn.readNetFromCaffe(self.PROTOTXT, self.MODEL)
                print("[資訊] AI 模型載入成功")
            except Exception as e:
                print(f"[錯誤] AI 模型載入失敗: {e}")
        else:
            print("[警告] 找不到 AI 模型檔案，請確認 prototxt 與 caffemodel 是否存在。")

        # ==========================================
        # 5. 立體視覺 (Stereo Vision) 設定
        # ==========================================
        self.baseline = 60.0  # 鏡頭間距 (mm)，請依硬體實際測量修改
        self.focal_length = self.K[0, 0]  # 從內參矩陣取得焦距 (px)
        
        # 建立立體匹配器 (StereoBM)
        self.stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15) # 增加搜索範圍
        self.stereo.setTextureThreshold(10) # 過濾低紋理區域
        self.stereo.setUniquenessRatio(15)  # 確保匹配的唯一性
        self.stereo.setSpeckleWindowSize(100) # 過濾散斑噪點
        self.stereo.setSpeckleRange(32)

    def _gstreamer_pipeline(self, cam_id, width=640, height=480, fps=30):
        cam_names = [
            "/base/axi/pcie@1000120000/rp1/i2c@88000/imx219@10",
            "/base/axi/pcie@1000120000/rp1/i2c@80000/imx219@10"
        ]
        return (
            f"libcamerasrc camera-name={cam_names[cam_id]} ! "
            f"video/x-raw, format=I420, width={width}, height={height}, framerate={fps}/1 ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=True sync=False"
        )

    def _camera_worker(self):
        """ 攝影機資料讀取執行緒 """
        print("[資訊] 啟動攝影機執行緒...")
        cap_l = cv2.VideoCapture(self._gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
        cap_r = cv2.VideoCapture(self._gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
        
        # 4. 【異常處理】硬體斷線重連機制
        while (not cap_l.isOpened() or not cap_r.isOpened()) and self.is_running:
            print("[警告] 攝影機未連接，等待重試...")
            if not cap_l.isOpened():
                cap_l.open(self._gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
            if not cap_r.isOpened():
                cap_r.open(self._gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
            time.sleep(2)
            
        print("[資訊] 攝影機連線成功")
        
        while self.is_running:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                print("[警告] 影像幀遺失 (Drop frame)，嘗試重新讀取...")
                cap_l.release()
                cap_r.release()
                time.sleep(0.5)
                cap_l.open(self._gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
                cap_r.open(self._gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
                continue
                
            timestamp = time.time()
            
            if self.cam_queue.full():
                try: self.cam_queue.get_nowait()
                except queue.Empty: pass
                
            self.cam_queue.put((timestamp, frame_l, frame_r))
            time.sleep(0.01)
            
        cap_l.release()
        cap_r.release()

    def _lidar_worker(self):
        """ 光達資料讀取執行緒 """
        print("[資訊] 啟動光達執行緒...")
        
        SYNC = 0xA5
        CMD_SCAN = 0x20
        CMD_STOP = 0x25
        
        def parse_packet(raw):
            if len(raw) < 5: return None
            syncbit = raw[0] & 0x01
            syncbit_inv = (raw[0] >> 1) & 0x01
            check_bit = raw[1] & 0x01
            if check_bit != 1 or (syncbit ^ syncbit_inv) != 1:
                return None
            angle = ((raw[1] >> 1) + (raw[2] << 7)) / 64.0
            dist_mm = (raw[3] + (raw[4] << 8)) / 4.0
            return angle, dist_mm, bool(syncbit)
            
        while self.is_running:
            try:
                with serial.Serial(self.lidar_port, self.baudrate, timeout=2) as ser:
                    # 停止並重啟
                    ser.write(bytes([SYNC, CMD_STOP]))
                    time.sleep(0.1)
                    ser.reset_input_buffer()
                    ser.dtr = False # 馬達啟動
                    time.sleep(2)   # 暖機
                    
                    # 發送掃描指令
                    ser.write(bytes([SYNC, CMD_SCAN]))
                    
                    # 等待 header (A5 5A)
                    buf = bytearray()
                    while self.is_running:
                        b = ser.read(1)
                        if not b: continue
                        buf.extend(b)
                        if len(buf) >= 2 and buf[-2:] == b'\xA5\x5A':
                            ser.read(5)
                            break
                            
                    scan_points = []
                    while self.is_running:
                        buf = bytearray(ser.read(5))
                        if len(buf) < 5: continue
                        
                        # 加入滑動視窗機制，避免失步導致永久無法解析
                        res = None
                        for _ in range(20):
                            res = parse_packet(buf)
                            if res: break
                            new_b = ser.read(1)
                            if not new_b: break
                            buf = buf[1:] + bytearray(new_b)
                            
                        if not res: continue
                        
                        angle, dist_mm, is_new_scan = res
                        if is_new_scan:
                            if len(scan_points) > 10:
                                timestamp = time.time()
                                if self.lidar_queue.full():
                                    try: self.lidar_queue.get_nowait()
                                    except queue.Empty: pass
                                self.lidar_queue.put((timestamp, scan_points))
                            scan_points = []
                            
                        if dist_mm > 0:
                            scan_points.append((dist_mm, angle))
                            
            except Exception as e:
                print(f"[警告] 光達斷線或發生異常: {e}，5秒後嘗試重連...")
                time.sleep(5)

    def project_lidar_to_camera(self, distance, angle):
        """
        座標轉換：將 LiDAR 極座標點映射到影像 2D 像素座標
        """
        if distance == 0:
            return None
            
        theta = math.radians(angle)
        
        # 假設 LiDAR 前方為 0 度，Y 軸朝前，X 軸朝右
        x_l = distance * math.sin(theta)
        y_l = distance * math.cos(theta)
        z_l = 0.0
        
        P_lidar = np.array([[x_l], [y_l], [z_l]], dtype=np.float32)
        
        align_matrix = np.array([
            [ 1,  0,  0],
            [ 0,  0, -1],
            [ 0,  1,  0]
        ], dtype=np.float32)
        
        P_lidar_aligned = np.dot(align_matrix, P_lidar)
        P_camera = np.dot(self.R, P_lidar_aligned) + self.T
        
        Z_c = float(P_camera[2, 0])
        if Z_c <= 0:
            return None
            
        p_img = np.dot(self.K, P_camera)
        
        u = int(p_img[0, 0] / Z_c)
        v = int(p_img[1, 0] / Z_c)
        
        if 0 <= u < 640 and 0 <= v < 480:
            return (u, v, distance)
        return None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 限制滑鼠點擊只有在上方「主攝影機畫面」內才生效
            if y < 480:
                self.click_point = (x, y)

    def start(self):
        self.is_running = True
        
        threading.Thread(target=self._camera_worker, daemon=True).start()
        threading.Thread(target=self._lidar_worker, daemon=True).start()
        
        cv2.namedWindow("Sensor Fusion")
        cv2.setMouseCallback("Sensor Fusion", self._mouse_callback)
        
        print("[資訊] 進入資料融合與視覺化迴圈...")
        
        latest_scan = []
        
        while self.is_running:
            # 讀取相機影像 (如果沒影像就等待一小段時間，但不卡死迴圈)
            try:
                cam_ts, frame_l, frame_r = self.cam_queue.get(timeout=0.05)
            except queue.Empty:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
                
            # 讀取光達點雲 (光達更新頻率較慢，約 5-10Hz，相機 30Hz)
            # 使用非阻塞讀取，若有新資料就更新 latest_scan
            try:
                lidar_ts, scan = self.lidar_queue.get_nowait()
                latest_scan = scan
            except queue.Empty:
                pass
                
            # 主要的 AI 與 LiDAR 融合運算都在左攝影機畫面上
            frame = frame_l
            display_img = frame.copy()
            valid_points = []

            # --- 0. 計算立體視差圖 (Stereo Disparity) ---
            # 將影像轉為灰階，這是立體匹配的必要步驟
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            # 計算視差
            raw_disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
            
            # 建立遮罩標記有效區域 (視差必須大於 0)
            valid_mask = raw_disparity > 0
            
            # 計算全畫面深度圖 Z = (f * B) / d
            depth_map = np.zeros_like(raw_disparity)
            depth_map[valid_mask] = (self.focal_length * self.baseline) / raw_disparity[valid_mask]
            depth_map[~valid_mask] = 0 # 無效區域設為 0
            
            # --- 0. 建立 2D 雷達圖 (320x240) ---
            radar_img = np.zeros((240, 320, 3), dtype=np.uint8)
            cx, cy = 160, 120 # 中心點
            max_dist = 4000.0  # 雷達圖顯示半徑為 4000 mm (4公尺)
            radar_radius_px = 110 # 雷達圖在畫布上的半徑(像素)
            
            # 畫雷達距離網格
            for r in [1000, 2000, 3000, 4000]:
                r_pix = int((r / max_dist) * radar_radius_px)
                cv2.circle(radar_img, (cx, cy), r_pix, (50, 50, 50), 1)
                if r < 4000:
                    cv2.putText(radar_img, f"{r//1000}m", (cx + 2, cy - r_pix - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                                
            # 畫相機 FOV 參考線 (約前向 ±30 度)
            cv2.line(radar_img, (cx, cy), (int(cx + radar_radius_px * math.sin(math.radians(30))), int(cy - radar_radius_px * math.cos(math.radians(30)))), (50, 50, 150), 1)
            cv2.line(radar_img, (cx, cy), (int(cx + radar_radius_px * math.sin(math.radians(330))), int(cy - radar_radius_px * math.cos(math.radians(330)))), (50, 50, 150), 1)
            cv2.putText(radar_img, "Cam FOV", (cx - 30, int(cy - radar_radius_px * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 150), 1)
            cv2.circle(radar_img, (cx, cy), 4, (0, 0, 255), -1) # 標示光達位置中心

            # --- 1. LiDAR 點雲投影與繪製 ---
            for dist, angle in latest_scan:
                # [雷達圖] 將 360 度的光達點轉換至雷達畫布
                if 0 < dist <= max_dist:
                    px = int(cx + (dist / max_dist) * radar_radius_px * math.sin(math.radians(angle)))
                    py = int(cy - (dist / max_dist) * radar_radius_px * math.cos(math.radians(angle)))
                    cv2.circle(radar_img, (px, py), 2, (0, 255, 0), -1)

                # [攝影機圖] 僅處理落在相機前方範圍的點
                if angle < 45 or angle > 315:
                    pt = self.project_lidar_to_camera(dist, angle)
                    if pt:
                        u, v, d = pt
                        valid_points.append((u, v, d))
                        
                        ratio = min(max(d / 4000.0, 0.0), 1.0)
                        b = int(255 * ratio)
                        g = int(255 * (1 - abs(ratio - 0.5) * 2))
                        r = int(255 * (1 - ratio))
                        cv2.circle(display_img, (u, v), 3, (b, g, r), -1)
                        
            # --- 2. AI 物件辨識與距離結合 ---
            if self.net is not None:
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # 尋找落在這個 Bounding Box 內的 LiDAR 有效測距點
                        # 同時我們可以從 Stereo Depth Map 取得中心區域的平均深度
                        # 取框內中心 50% 的區域，避免背景干擾
                        bh, bw = endY - startY, endX - startX
                        roi_depth = depth_map[startY+bh//4:endY-bh//4, startX+bw//4:endX-bw//4]
                        
                        valid_roi_depth = roi_depth[roi_depth > 0]
                        if len(valid_roi_depth) > 0:
                            stereo_dist = np.median(valid_roi_depth)
                        else:
                            stereo_dist = -1
                        
                        box_lidar_pts = []
                        for (u, v, d) in valid_points:
                            if startX <= u <= endX and startY <= v <= endY:
                                box_lidar_pts.append(d)

                        # 若框內有光達點，取最短距離當作物件距離（最靠近鏡頭的通常是物體表面）
                        s_label = f"S:{stereo_dist:.0f}" if stereo_dist > 0 else "S:N/A"
                        if box_lidar_pts:
                            obj_dist = min(box_lidar_pts)
                            label = f"{self.CLASSES[idx]} | L:{obj_dist:.0f} {s_label} mm"
                        else:
                            label = f"{self.CLASSES[idx]} | {s_label} mm"

                        cv2.rectangle(display_img, (startX, startY), (endX, endY), self.COLORS[idx], 2)
                        y_pos = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(display_img, label, (startX, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

            # --- 3. 輔助：滑鼠點擊測距 ---
            if self.click_point:
                cx, cy = self.click_point
                min_err = float('inf')
                closest_d = -1
                for u, v, d in valid_points:
                    err = (u - cx)**2 + (v - cy)**2
                    if err < 600:
                        if err < min_err:
                            min_err = err
                            closest_d = d
                            
                if closest_d != -1:
                    text = f"Dist: {closest_d:.1f} mm"
                    cv2.putText(display_img, text, (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(display_img, (cx, cy), 5, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, "No LiDAR data", (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
            # --- 4. 畫面合併與顯示 ---
            # 上方: 左攝影機畫面 (640x480)
            # 下方: 右攝影機畫面 (320x240) + 雷達圖 (320x240)
            frame_r_resized = cv2.resize(frame_r, (320, 240))

            # 額外製作一個視差圖的預覽 (幫助除錯)
            disp_vis = cv2.normalize(raw_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            disp_vis_resized = cv2.resize(disp_vis, (320, 240))
            
            # 將底部面板改為三部分或替換其中之一
            bottom_panel = np.hstack((frame_r_resized, radar_img)) 
            combined_img = np.vstack((display_img, bottom_panel))

            cv2.imshow("Disparity Map (Debugging)", disp_vis_resized)
            cv2.imshow("Sensor Fusion", combined_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        self.stop()

    def stop(self):
        print("[資訊] 系統停止中...")
        self.is_running = False
        time.sleep(0.5)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fusion_app = SensorFusionRanging(lidar_port='/dev/ttyUSB0', baudrate=115200)
    fusion_app.start()
