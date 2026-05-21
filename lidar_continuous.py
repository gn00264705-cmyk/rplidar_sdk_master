"""
RPLidar A1M8 - 持續量測輸出
COM4, 115200 baud
按 Ctrl+C 停止
"""
import serial
import struct
import time

PORT     = 'COM4'
BAUDRATE = 115200

SYNC         = 0xA5
CMD_SCAN     = 0x20
CMD_STOP     = 0x25
POINT_SIZE   = 5  # 標準掃描每個測距點固定 5 bytes


def send_cmd(ser, cmd):
    ser.write(bytes([SYNC, cmd]))


def wait_response_header(ser):
    """等待並跳過回應 header (A5 5A ...)"""
    buf = bytearray()
    deadline = time.time() + 3.0
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        buf.extend(b)
        if len(buf) >= 2 and buf[-2:] == bytes([0xA5, 0x5A]):
            ser.read(5)  # 吃掉剩餘 header
            return True
    return False


def parse_point(raw):
    """解析一筆 5-byte 測距點，回傳 (angle_deg, dist_mm, quality, is_new_scan)"""
    sync_q      = raw[0]
    angle_raw   = struct.unpack('<H', raw[1:3])[0]
    dist_raw    = struct.unpack('<H', raw[3:5])[0]

    quality     = (sync_q >> 2) & 0x3F
    angle       = (angle_raw >> 1) / 64.0
    dist_mm     = dist_raw / 4.0
    is_new_scan = bool(sync_q & 0x01)

    return angle, dist_mm, quality, is_new_scan


def main():
    print(f'開啟 {PORT} @ {BAUDRATE}...')
    with serial.Serial(PORT, BAUDRATE, timeout=1, dsrdtr=False) as ser:
        ser.dtr = False  # DTR Low → 馬達啟動
        time.sleep(0.1)
        ser.reset_input_buffer()

        print('送出 SCAN 指令...')
        send_cmd(ser, CMD_SCAN)

        if not wait_response_header(ser):
            print('錯誤：未收到回應 header')
            return

        print('開始持續輸出（Ctrl+C 停止）\n')
        print(f'{"角度(°)":>10} {"距離(mm)":>12} {"品質":>6} {"新一圈":>6}')
        print('-' * 40)

        scan_count = 0
        point_count = 0

        try:
            while True:
                raw = ser.read(POINT_SIZE)
                if len(raw) < POINT_SIZE:
                    continue

                angle, dist_mm, quality, is_new_scan = parse_point(raw)

                if is_new_scan:
                    scan_count += 1
                    print(f'--- 第 {scan_count} 圈開始 (已累計 {point_count} 點) ---')

                # 距離為 0 表示無效點，可選擇過濾
                if dist_mm == 0:
                    continue

                point_count += 1
                marker = '★' if is_new_scan else ' '
                print(f'{marker}{angle:>10.2f} {dist_mm:>12.2f} {quality:>6}')

        except KeyboardInterrupt:
            print(f'\n\n共掃描 {scan_count} 圈，{point_count} 個有效點')

        finally:
            print('送出 STOP 指令...')
            send_cmd(ser, CMD_STOP)
            time.sleep(0.1)


if __name__ == '__main__':
    main()
