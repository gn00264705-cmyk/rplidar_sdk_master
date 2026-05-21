"""
RPLidar A1M8 - 互動式單圈 360° 掃描工具
COM4, 115200 baud (或 /dev/ttyUSB0)

流程：掃描 → 收集滿一圈 → 馬達停止 → 等待指令
"""
import serial
import struct
import time
import multiprocessing

PORT     = '/dev/ttyUSB0'
BAUDRATE = 115200

CMD_STOP       = 0x25
CMD_SCAN       = 0x20
CMD_RESET      = 0x40
CMD_GET_INFO   = 0x50
CMD_GET_HEALTH = 0x52

SYNC = 0xA5
MOTOR_SPINUP_SEC = 2.0   # A1M8 馬達暖機時間

# ── 攝影機控制 (獨立 Process) ──────────────────────────────────

def camera_process_func(is_scanning_event):
    """
    在獨立的 Process 中執行 OpenCV 與 GStreamer，
    避免與主程式的迴圈衝突，也確保發生卡死時 OS 能強制回收相機資源。
    """
    import cv2
    import numpy as np

    def gstreamer_pipeline(cam_id=0, width=640, height=480, fps=30):
        # 使用日誌中的完整硬體路徑
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

    cap_l = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("\n[錯誤] 無法開啟雙目相機，請檢查硬體連接。")
        return

    while is_scanning_event.is_set():
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if ret_l and ret_r:
            # 將左右影像水平合併
            combined = np.hstack((frame_l, frame_r))
            cv2.imshow("Stereo View (Left | Right)", combined)

        # 等待 1ms 以更新視窗，同時捕捉事件 (若在視窗按下 'q' 鍵也可關閉視窗)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清除資源
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


# ── Serial 工具 ──────────────────────────────────────────

def send_cmd(ser, cmd, payload=None):
    if payload is None:
        pkt = bytes([SYNC, cmd])
    else:
        size = len(payload)
        checksum = SYNC ^ (cmd | 0x80) ^ size
        for b in payload:
            checksum ^= b
        pkt = bytes([SYNC, cmd | 0x80, size] + list(payload) + [checksum])
    ser.write(pkt)


def read_response_header(ser, timeout=3.0):
    """等待回應 header (A5 5A ...) 並回傳 (ans_type, data_size)"""
    ser.timeout = timeout
    buf = bytearray()
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        buf.extend(b)
        if len(buf) >= 2 and buf[-2:] == bytes([0xA5, 0x5A]):
            rest = ser.read(5)
            if len(rest) < 5:
                return None
            data_size = struct.unpack('<I', rest[:4])[0] & 0x3FFFFFFF
            ans_type  = rest[4]
            return ans_type, data_size
    return None


# ── 封包解析 ─────────────────────────────────────────────

def parse_scan_packet(raw):
    """
    解析 5-byte 標準掃描封包
    回傳 (angle_deg, dist_mm, quality, is_new_scan) 或 None（封包無效）
    """
    if len(raw) < 5:
        return None

    syncbit     =  raw[0] & 0x01
    syncbit_inv = (raw[0] >> 1) & 0x01
    quality     =  raw[0] >> 2
    check_bit   =  raw[1] & 0x01

    if check_bit != 1:
        return None
    if (syncbit ^ syncbit_inv) != 1:
        return None

    angle   = ((raw[1] >> 1) + (raw[2] << 7)) / 64.0
    dist_mm = (raw[3] + (raw[4] << 8)) / 4.0

    return angle, dist_mm, quality, bool(syncbit)


def read_scan_point(ser):
    """讀一筆有效測距點，失步時自動重新對齊（最多滑動 20 次）"""
    buf = bytearray(ser.read(5))
    if len(buf) < 5:
        return None
    for _ in range(20):
        result = parse_scan_packet(buf)
        if result:
            return result
        new_byte = ser.read(1)
        if not new_byte:
            return None
        buf = buf[1:] + bytearray(new_byte)
    return None


# ── 馬達控制 (A1M8 以 DTR 控制) ──────────────────────────

def motor_start(ser):
    ser.dtr = False
    print(f'  馬達啟動，等待 {MOTOR_SPINUP_SEC:.0f} 秒暖機...')
    time.sleep(MOTOR_SPINUP_SEC)


def motor_stop(ser):
    send_cmd(ser, CMD_STOP)
    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.dtr = True
    print('  馬達已停止')


# ── 指令實作 ─────────────────────────────────────────────

def cmd_get_info(ser):
    print('\n[GET_DEVICE_INFO]')
    ser.reset_input_buffer()
    send_cmd(ser, CMD_GET_INFO)
    result = read_response_header(ser)
    if not result:
        print('  無回應（請確認連線）')
        return
    _, data_size = result
    data = ser.read(data_size)
    if len(data) < 20:
        print(f'  資料不足: {data.hex()}')
        return

    model      = data[0]
    fw_minor   = data[1]
    fw_major   = data[2]
    hw_ver     = data[3]
    serial_num = data[4:20].hex().upper()

    print(f'  型號 ID   : {model}')
    print(f'  韌體版本  : {fw_major}.{fw_minor:02d}')
    print(f'  硬體版本  : {hw_ver}')
    print(f'  序號      : {serial_num}')


def cmd_get_health(ser):
    print('\n[GET_HEALTH]')
    ser.reset_input_buffer()
    send_cmd(ser, CMD_GET_HEALTH)
    result = read_response_header(ser)
    if not result:
        print('  無回應（請確認連線）')
        return
    _, data_size = result
    data = ser.read(data_size)
    if len(data) < 3:
        print(f'  資料不足: {data.hex()}')
        return

    status     = data[0]
    error_code = struct.unpack('<H', data[1:3])[0]
    status_str = {0: 'OK', 1: 'Warning', 2: 'Error'}.get(status, '未知')
    print(f'  狀態      : {status_str}')
    print(f'  錯誤碼    : {error_code}')


def cleanup_camera(is_scanning_event, cam_proc):
    """安全關閉攝影機子程序"""
    if cam_proc:
        is_scanning_event.clear()
        cam_proc.join(timeout=2.0)  # 給 OpenCV 2 秒的時間清理資源
        if cam_proc.is_alive():
            # 若 GStreamer 卡死，直接由 OS 強制中止，確保下一次能乾淨啟動
            print("  [警告] 攝影機程序無回應，正在強制中止...")
            cam_proc.terminate()
            cam_proc.join(timeout=1.0)
            if cam_proc.is_alive():
                print("  [警告] terminate 無效，發送 SIGKILL...")
                try:
                    cam_proc.kill()
                except Exception:
                    pass


def cmd_scan_360(ser):
    """
    執行一圈 360° 掃描：
      1. 啟動馬達並暖機 (同時啟動雙目相機子程序)
      2. 送出 SCAN 指令
      3. 等到第一個 syncbit=1（新圈起點）開始收點
      4. 遇到第二個 syncbit=1 代表一圈結束
      5. 停止掃描並關閉馬達
      6. 印出結果統計並關閉相機
    """
    print('\n[360° 掃描]')
    
    # 使用獨立的 Process 啟動相機，以避免 GStreamer / libcamera 卡死主程式
    is_scanning_event = multiprocessing.Event()
    is_scanning_event.set()
    cam_proc = multiprocessing.Process(target=camera_process_func, args=(is_scanning_event,))
    cam_proc.daemon = True
    cam_proc.start()

    motor_start(ser)

    send_cmd(ser, CMD_SCAN)
    result = read_response_header(ser)
    if not result:
        print('  無回應，停止')
        motor_stop(ser)
        cleanup_camera(is_scanning_event, cam_proc)
        return

    print('  等待掃描起始點...')
    ser.timeout = 5.0

    # 等到第一個 syncbit=1（新圈起點）
    while True:
        pt = read_scan_point(ser)
        if pt is None:
            print('  等待逾時')
            motor_stop(ser)
            cleanup_camera(is_scanning_event, cam_proc)
            return
        if pt[3]:   # is_new_scan
            break

    # 從第一個新圈起點開始收集，直到下一個新圈起點
    scan_points = []
    first_point = pt
    if pt[1] > 0:           # dist_mm > 0 才加入
        scan_points.append(pt)

    print('  收集一圈中...')
    ser.timeout = 3.0

    while True:
        pt = read_scan_point(ser)
        if pt is None:
            print('  讀取逾時，提前結束')
            break
        angle, dist_mm, quality, is_new_scan = pt
        if is_new_scan:
            break           # 第二個新圈起點 → 一圈完成
        if dist_mm > 0:
            scan_points.append(pt)

    # 停止並關馬達
    motor_stop(ser)

    # 關閉攝影機 (給予2秒緩衝，超時強制切斷)
    cleanup_camera(is_scanning_event, cam_proc)

    # ── 顯示結果 ──────────────────────────────────────────
    if not scan_points:
        print('  無有效測距點')
        return

    total   = len(scan_points)
    min_d   = min(p[1] for p in scan_points)
    max_d   = max(p[1] for p in scan_points)
    avg_d   = sum(p[1] for p in scan_points) / total
    min_pt  = min(scan_points, key=lambda p: p[1])
    max_pt  = max(scan_points, key=lambda p: p[1])

    print(f'\n  ── 統計 ──────────────────────────')
    print(f'  有效點數  : {total}')
    print(f'  最近距離  : {min_d:.1f} mm  (角度 {min_pt[0]:.2f}°)')
    print(f'  最遠距離  : {max_d:.1f} mm  (角度 {max_pt[0]:.2f}°)')
    print(f'  平均距離  : {avg_d:.1f} mm')

    show = input('\n  顯示所有測距點? (y/n) ').strip().lower()
    if show == 'y':
        print(f'\n  {"角度(°)":>10} {"距離(mm)":>12} {"品質":>6}')
        print('  ' + '-' * 32)
        for angle, dist_mm, quality, _ in sorted(scan_points, key=lambda p: p[0]):
            print(f'  {angle:>10.2f} {dist_mm:>12.2f} {quality:>6}')


def cmd_camera_only():
    """單純開啟雙目攝影機的模式，不牽涉 LiDAR 雷射馬達"""
    print('\n[單純開啟雙目攝影機]')
    is_running_event = multiprocessing.Event()
    is_running_event.set()
    
    # 同樣使用已建立的獨立 Process 來確保穩定性
    cam_proc = multiprocessing.Process(target=camera_process_func, args=(is_running_event,))
    cam_proc.daemon = True
    cam_proc.start()

    try:
        # 阻塞主執行緒，直到使用者按下 Enter
        input('  攝影機運作中... 請在此按下 [Enter] 鍵停止並返回選單：\n')
    except KeyboardInterrupt:
        print('\n  收到中斷訊號')
    finally:
        print('  正在關閉攝影機...')
        cleanup_camera(is_running_event, cam_proc)
        print('  攝影機已關閉。')


def cmd_reset(ser):
    print('\n[RESET] 重置裝置...')
    send_cmd(ser, CMD_RESET)
    time.sleep(2)
    ser.reset_input_buffer()
    print('  重置完成')


# ── 主程式 ───────────────────────────────────────────────

def main():
    print(f'開啟 {PORT} @ {BAUDRATE} baud...')
    with serial.Serial(PORT, BAUDRATE, timeout=2, dsrdtr=False) as ser:
        ser.dtr = True   # 開啟時先停馬達，掃描才啟動
        time.sleep(0.1)
        ser.reset_input_buffer()
        print('連線成功，馬達待機中\n')

        try:
            while True:
                print('=' * 42)
                print('  1. GET_DEVICE_INFO')
                print('  2. GET_HEALTH')
                print('  3. 執行一圈 360° 掃描')
                print('  4. RESET')
                print('  5. 單純開啟雙目攝影機')
                print('  0. 離開')
                choice = input('> ').strip()

                if choice == '1':
                    cmd_get_info(ser)
                elif choice == '2':
                    cmd_get_health(ser)
                elif choice == '3':
                    cmd_scan_360(ser)
                elif choice == '4':
                    cmd_reset(ser)
                elif choice == '5':
                    cmd_camera_only()
                elif choice == '0':
                    print('離開')
                    break
                else:
                    print('無效選項')
        except KeyboardInterrupt:
            print('\n[系統] 收到使用者中斷，準備退出...')
        finally:
            ser.dtr = True  # 確保異常或正常離開時都能關閉馬達


if __name__ == '__main__':
    # 確保 Windows/Linux 下的 Multiprocessing 安全行為
    multiprocessing.freeze_support()
    main()
