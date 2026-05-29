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
        # 1.5 IMU (Yahboom 擴展板) 設定
        # ==========================================
        # 改為自動偵測：設為 None，讓程式自動去尋找可用的序列埠
        self.imu_port = None
        self.imu_baudrate = 115200
        # 共享狀態：儲存最新的姿態角
        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.imu_lock = threading.Lock()

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
        
        # 5.1 預留校準參數 (未來可載入 .npz 檔案)
        self.map_l1, self.map_l2 = None, None
        self.map_r1, self.map_r2 = None, None
        self.is_calibrated = False

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
                
            # 如果有校準資料，進行 Rectification
            if self.is_calibrated:
                frame_l = cv2.remap(frame_l, self.map_l1, self.map_l2, cv2.INTER_LINEAR)
                frame_r = cv2.remap(frame_r, self.map_r1, self.map_r2, cv2.INTER_LINEAR)

            timestamp = time.time()
            
            if self.cam_queue.full():
                try: self.cam_queue.get_nowait()
                except queue.Empty: pass
                
            self.cam_queue.put((timestamp, frame_l, frame_r))
            time.sleep(0.01)
            
        cap_l.release()
        cap_r.release()

    def _imu_worker(self):
        """ IMU (Yahboom 擴展板) 資料讀取執行緒 """
        import serial.tools.list_ports
        print("[資訊] 啟動 IMU 執行緒，正在自動尋找可用的設備...")
        
        while self.is_running:
            # --- 自動尋找並過濾光達佔用的 Port ---
            if self.imu_port is None:
                ports = [port.device for port in serial.tools.list_ports.comports()]
                possible_ports = [p for p in ports if 'ttyUSB' in p or 'ttyACM' in p]
                
                if self.lidar_port in possible_ports:
                    possible_ports.remove(self.lidar_port)
                    
                if not possible_ports:
                    print("[警告] 找不到其他的 USB/ACM 設備供 IMU 使用，請檢查接線！3秒後重試...")
                    time.sleep(3)
                    continue
                    
                self.imu_port = possible_ports[0]
                print(f"[資訊] 自動分配 IMU 至: {self.imu_port}")

            try:
                with serial.Serial(self.imu_port, self.imu_baudrate, timeout=1) as ser:
                    print(f"[資訊] IMU 連線成功 ({self.imu_port})")
                    ser.reset_input_buffer()
                    buffer = bytearray()
                    
                    # 加入低通濾波 (Low-Pass Filter) 暫存變數
                    smooth_roll = 0.0
                    smooth_pitch = 0.0
                    smooth_yaw = 0.0
                    alpha = 0.05  # 濾波強度，範圍 0~1，越小越平滑但反應稍慢
                    
                    last_time = time.time()
                    
                    while self.is_running:
                        # 讀取一行資料 (假設為 ASCII 輸出，依 Yahboom 韌體實際情況調整)
                        # 若 Yahboom 底板採用二進制封包(如 0xAA 0x55 開頭)，請替換為封包解析邏輯
                        if ser.in_waiting > 0:
                            try:
                                buffer.extend(ser.read(ser.in_waiting))
                                
                                # 尋找封包標頭 FF FB
                                while len(buffer) >= 4:
                                    if buffer[0] == 0xFF and buffer[1] == 0xFB:
                                        length = buffer[2]
                                        total_packet_len = 2 + length
                                        
                                        if len(buffer) >= total_packet_len:
                                            packet = buffer[:total_packet_len]
                                            buffer = buffer[total_packet_len:] # 移除已處理封包
                                            
                                            cmd = packet[3]
                                            if cmd == 0x0E and length == 0x15: # 0x0E 為 IMU 姿態資料
                                                # 解析 9 個 int16: Gyro(3), Accel(3), Mag(3)
                                                gx, gy, gz, ax, ay, az, mx, my, mz = struct.unpack('<hhhhhhhhh', packet[4:22])
                                                
                                                # --- 計算時間差 (dt) 供陀螺儀積分使用 ---
                                                current_time = time.time()
                                                dt = current_time - last_time
                                                last_time = current_time
                                                if dt > 0.5: dt = 0.01  # 防止中斷重連時的突波
                                                
                                                # 1. 修正平放時的 180 度翻轉問題
                                                # 你的板子平放時 Z 軸加速度 az 是負值，使用 -az 讓它基準變成 0 度
                                                # 修正：根據實體板的物理方向，將 ax 與 ay 互換以對應真實的 Roll/Pitch
                                                raw_roll = math.atan2(ax, -az) * 180.0 / math.pi
                                                # 修正：Y 軸動作相反，將 ay 加上負號反轉
                                                raw_pitch = math.atan2(-ay, math.sqrt(ax*ax + az*az)) * 180.0 / math.pi
                                                
                                                # 2. 進行低通濾波 (消除震動帶來的亂跳)
                                                smooth_roll = (1.0 - alpha) * smooth_roll + alpha * raw_roll
                                                smooth_pitch = (1.0 - alpha) * smooth_pitch + alpha * raw_pitch
                                                
                                                # 3. 處理 Yaw (互補濾波器：結合陀螺儀積分與磁力計)
                                                # 將角速度轉換為度/秒 (假設 MPU9250 在 ±2000dps 時靈敏度約為 16.4 LSB/dps)
                                                gz_dps = gz / 131.0 
                                                
                                                # 磁力計的絕對基準
                                                raw_yaw_mag = math.atan2(my, mx) * 180.0 / math.pi
                                                
                                                # 處理磁力計 180 度邊界跳變
                                                diff_yaw = raw_yaw_mag - smooth_yaw
                                                if diff_yaw > 180.0: diff_yaw -= 360.0
                                                elif diff_yaw < -180.0: diff_yaw += 360.0
                                                
                                                # 互補濾波：98% 相信陀螺儀積分 (順滑)，2% 相信磁力計 (防長期飄移)
                                                smooth_yaw = 0.98 * (smooth_yaw + gz_dps * dt) + 0.02 * (smooth_yaw + diff_yaw)
                                                
                                                if smooth_yaw > 180.0: smooth_yaw -= 360.0
                                                elif smooth_yaw < -180.0: smooth_yaw += 360.0
                                                
                                                with self.imu_lock:
                                                    self.imu_data['roll'] = smooth_roll
                                                    self.imu_data['pitch'] = smooth_pitch
                                                    # 將 Yaw 鎖定為 0.0，避免畫面亂轉
                                                    self.imu_data['yaw'] = 0.0
                                        else:
                                            break # 封包未完整，等待下一次讀取
                                    else:
                                        buffer.pop(0) # 不是標頭，滑動一個 byte 繼續找
                                
                            except Exception as e:
                                pass # 忽略偶發的資料轉換錯誤
                        else:
                            time.sleep(0.01)
                            
            except Exception as e:
                print(f"[警告] IMU 斷線或無法開啟 {self.imu_port}: {e}")
                print("       (3秒後重新自動搜尋可用的 Port...)")
                self.imu_port = None  # 清除紀錄，觸發重新搜尋
                time.sleep(3)

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
                    
                    # 等待 header (A5 5A)，加入超時機制防止卡死
                    buf = bytearray()
                    start_time = time.time()
                    header_found = False
                    
                    while self.is_running and (time.time() - start_time < 3.0):
                        b = ser.read(1)
                        if not b: continue
                        buf.extend(b)
                        if len(buf) >= 2 and buf[-2:] == b'\xA5\x5A':
                            ser.read(5)
                            header_found = True
                            break
                            
                    if not header_found:
                        print(f"[警告] 光達執行緒在 {self.lidar_port} 找不到回應標頭！")
                        print(f"       (可能是 Port 填錯連到 IMU 了？) 3秒後重試...")
                        time.sleep(3)
                        continue # 回到 while self.is_running 重新嘗試連線

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

    def project_lidar_to_camera(self, distance, angle, roll=0.0, pitch=0.0):
        """
        座標轉換：將 LiDAR 極座標點映射到影像 2D 像素座標
        並透過 IMU 姿態計算絕對高度以過濾地面點
        """
        if distance == 0:
            return None
            
        theta = math.radians(angle)
        
        # 假設 LiDAR 前方為 0 度，Y 軸朝前，X 軸朝右
        x_l = distance * math.sin(theta)
        y_l = distance * math.cos(theta)
        z_l = 0.0
        
        P_lidar = np.array([[x_l], [y_l], [z_l]], dtype=np.float32)
        
        # ==========================================
        # IMU 姿態補償 (計算絕對水平面座標 P_horizontal)
        # ==========================================
        r = math.radians(roll)
        p = math.radians(pitch)
        
        # X 軸旋轉矩陣 (Roll)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(r), -math.sin(r)],
            [0, math.sin(r), math.cos(r)]
        ], dtype=np.float32)
        
        # Y 軸旋轉矩陣 (Pitch)
        Ry = np.array([
            [math.cos(p), 0, math.sin(p)],
            [0, 1, 0],
            [-math.sin(p), 0, math.cos(p)]
        ], dtype=np.float32)
        
        R_imu = np.dot(Ry, Rx)
        P_horizontal = np.dot(R_imu, P_lidar)
        
        # 【過濾地面點】
        # 假設 LiDAR 離地安裝高度約 120mm，當點的絕對高度低於 -100mm 時，極有可能是打到地面
        if P_horizontal[2, 0] < -100.0:
            return None

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

    def draw_imu_3d_board(self, img, roll, pitch, yaw, cx=150, cy=150, size=80):
        """
        繪製固定坐標軸與動態旋轉的 3D 實體板，用來直觀顯示 IMU 姿態
        """
        # --- 1. 定義等角投影函數 (Isometric Projection) ---
        def project(pt):
            x, y, z = pt
            cos30 = math.cos(math.radians(30))
            sin30 = math.sin(math.radians(30))
            # 修改視角：X 向右下 (車身右側)，Y 向右上 (車頭)，Z 向上
            screen_x = int(cx + x * cos30 + y * cos30)
            screen_y = int(cy + x * sin30 - y * sin30 - z)
            return (screen_x, screen_y)

        # --- 2. 繪製固定不動的 3D 世界坐標軸 ---
        axes_3d = np.array([
            [0, 0, 0],       # 原點
            [size, 0, 0],    # X 軸 (紅)
            [0, size, 0],    # Y 軸 (綠)
            [0, 0, size]     # Z 軸 (藍)
        ], dtype=np.float32)
        
        p_org, p_x, p_y, p_z = project(axes_3d[0]), project(axes_3d[1]), project(axes_3d[2]), project(axes_3d[3])
        
        # 畫暗色底軸 (作為背景參考)
        cv2.line(img, p_org, p_x, (0, 0, 150), 2)  
        cv2.line(img, p_org, p_y, (0, 150, 0), 2)  
        cv2.line(img, p_org, p_z, (150, 0, 0), 2)  
        cv2.putText(img, "X", p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
        cv2.putText(img, "Y", p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
        cv2.putText(img, "Z", p_z, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 1)

        # --- 3. 定義實體板子的 8 個頂點 (長方體) ---
        # 假設板子寬度(X) 40, 長度(Y) 60, 厚度(Z) 5
        w, l, t = 40, 60, 5
        board_pts = np.array([
            [ w,  l,  t], [-w,  l,  t], [-w, -l,  t], [ w, -l,  t], # 0-3: 頂面
            [ w,  l, -t], [-w,  l, -t], [-w, -l, -t], [ w, -l, -t]  # 4-7: 底面
        ], dtype=np.float32)

        # --- 4. 根據 IMU 數據計算旋轉矩陣 ---
        r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))

        # 旋轉板子頂點並投影到 2D 螢幕
        rotated_pts = np.dot(board_pts, R.T)
        proj_pts = [project(pt) for pt in rotated_pts]

        # --- 5. 繪製旋轉後的實體板 (線框圖) ---
        # 畫底面 (藍色)
        for i in range(4):
            cv2.line(img, proj_pts[4+i], proj_pts[4+(i+1)%4], (255, 50, 50), 2)
        # 畫側面連接線 (綠色)
        for i in range(4):
            cv2.line(img, proj_pts[i], proj_pts[4+i], (50, 255, 50), 2)
        # 畫頂面 (紅色)
        for i in range(4):
            cv2.line(img, proj_pts[i], proj_pts[(i+1)%4], (50, 50, 255), 2)
            
        # 畫一個黃色實心圓代表「車頭/板子前方」(Y軸正方向的邊緣)
        front_center_rot = np.dot(np.array([0, l, 0]), R.T)
        cv2.circle(img, project(front_center_rot), 6, (0, 255, 255), -1)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 限制滑鼠點擊只有在上方「主攝影機畫面」內才生效
            if y < 480:
                self.click_point = (x, y)

    def start(self):
        self.is_running = True
        
        threading.Thread(target=self._camera_worker, daemon=True).start()
        threading.Thread(target=self._imu_worker, daemon=True).start()
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
            # [暫時關閉] 因為尚未進行相機校準，計算視差圖只會產生雜訊且極度消耗 CPU。
            # 未來有印表機可以列印棋盤格時，再將此段程式碼打開。
            # gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            # gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            # raw_disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
            # 
            # valid_mask = raw_disparity > 0
            # depth_map = np.zeros_like(raw_disparity)
            # depth_map[valid_mask] = (self.focal_length * self.baseline) / raw_disparity[valid_mask]
            # depth_map[~valid_mask] = 0
            
            # 給一個空的 depth_map 避免後續程式報錯
            depth_map = np.zeros((frame_l.shape[0], frame_l.shape[1]), dtype=np.float32)
            
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
            # 在每一幀開始前，先鎖定並提取當下最新的 IMU 姿態
            with self.imu_lock:
                current_roll = self.imu_data['roll']
                current_pitch = self.imu_data['pitch']

            for dist, angle in latest_scan:
                # [雷達圖] 將 360 度的光達點轉換至雷達畫布
                if 0 < dist <= max_dist:
                    px = int(cx + (dist / max_dist) * radar_radius_px * math.sin(math.radians(angle)))
                    py = int(cy - (dist / max_dist) * radar_radius_px * math.cos(math.radians(angle)))
                    cv2.circle(radar_img, (px, py), 2, (0, 255, 0), -1)

                # [攝影機圖] 僅處理落在相機前方範圍的點
                if angle < 45 or angle > 315:
                    pt = self.project_lidar_to_camera(dist, angle, current_roll, current_pitch)
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
                        box_lidar_pts = []
                        for (u, v, d) in valid_points:
                            if startX <= u <= endX and startY <= v <= endY:
                                box_lidar_pts.append(d)

                        # 若框內有光達點，取最短距離當作物件距離（最靠近鏡頭的通常是物體表面）
                        if box_lidar_pts:
                            obj_dist = min(box_lidar_pts)
                            label = f"{self.CLASSES[idx]} | L:{obj_dist:.0f} mm"
                        else:
                            label = f"{self.CLASSES[idx]}"

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
                                
            # --- 3.5 顯示 IMU 姿態資訊 ---
            with self.imu_lock:
                r, p, y = self.imu_data['roll'], self.imu_data['pitch'], self.imu_data['yaw']
            
            imu_text = f"IMU | Roll: {r:5.1f}  Pitch: {p:5.1f}  Yaw: {y:5.1f}"
            # 將文字顯示在左上角
            cv2.putText(display_img, imu_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
            # --- 3.6 繪製專屬的 3D 實體板視圖 ---
            imu_vis_img = np.zeros((300, 300, 3), dtype=np.uint8)
            self.draw_imu_3d_board(imu_vis_img, r, p, y, cx=150, cy=150, size=80)

            # --- 4. 畫面合併與顯示 ---
            # 上方: 左攝影機畫面 (640x480)
            # 下方: 右攝影機畫面 (320x240) + 雷達圖 (320x240)
            frame_r_resized = cv2.resize(frame_r, (320, 240))

            # 額外製作一個視差圖的預覽 (幫助除錯)
            # [暫時關閉] 隱藏未校準的視差圖視窗
            # disp_vis = cv2.normalize(raw_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            # if not self.is_calibrated:
            #     cv2.putText(disp_vis, "UNCALIBRATED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # disp_vis_resized = cv2.resize(disp_vis, (320, 240))
            
            # 將底部面板改為三部分或替換其中之一
            bottom_panel = np.hstack((frame_r_resized, radar_img)) 
            combined_img = np.vstack((display_img, bottom_panel))

            cv2.imshow("Sensor Fusion", combined_img)
            cv2.imshow("IMU 3D Visualization", imu_vis_img)
            
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
    # 這裡的光達 port 請與 self.imu_port 區分開來
    # 如果 IMU 是 /dev/ttyUSB1，那光達通常就是 /dev/ttyUSB0
    fusion_app = SensorFusionRanging(lidar_port='/dev/ttyUSB0', baudrate=115200)
    fusion_app.start()
