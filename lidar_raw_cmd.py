"""
RPLidar A1M8 - 直接 Serial Port 指令工具
COM4, 115200 baud
"""
import serial
import struct
import time

PORT = 'COM4'
BAUDRATE = 115200

CMD_STOP           = 0x25
CMD_SCAN           = 0x20
CMD_FORCE_SCAN     = 0x21
CMD_RESET          = 0x40
CMD_GET_INFO       = 0x50
CMD_GET_HEALTH     = 0x52
CMD_GET_SAMPLERATE = 0x59

SYNC = 0xA5


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
    print(f'  → 送出: {pkt.hex(" ")}')


def read_response_header(ser, timeout=2.0):
    """等待並解析回應 header: A5 5A [4 bytes] [1 byte type]"""
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
            size_subtype = struct.unpack('<I', rest[:4])[0]
            data_size = size_subtype & 0x3FFFFFFF
            ans_type  = rest[4]
            print(f'  ← Header: type=0x{ans_type:02X}  data_size={data_size}')
            return ans_type, data_size
    return None


def parse_scan_packet(raw):
    """
    解析 5-byte 標準掃描封包。
    回傳 (angle_deg, dist_mm, quality, is_new_scan) 或 None（封包無效）

    Byte 0: [quality(6)] [syncbit_inv(1)] [syncbit(1)]
    Byte 1: [angle_q6[6:0](7)] [check_bit(1)]  ← check_bit 必須為 1
    Byte 2: [angle_q6[13:7](8)]
    Byte 3: [distance_q2[7:0](8)]
    Byte 4: [distance_q2[15:8](8)]
    """
    if len(raw) < 5:
        return None

    syncbit     = raw[0] & 0x01
    syncbit_inv = (raw[0] >> 1) & 0x01
    quality     = raw[0] >> 2
    check_bit   = raw[1] & 0x01

    # 封包合法性驗證
    if check_bit != 1:
        return None
    if (syncbit ^ syncbit_inv) != 1:   # syncbit 與 syncbit_inv 必須互補
        return None

    angle   = ((raw[1] >> 1) + (raw[2] << 7)) / 64.0
    dist_mm = (raw[3] + (raw[4] << 8)) / 4.0
    is_new_scan = bool(syncbit)

    return angle, dist_mm, quality, is_new_scan


def read_scan_point(ser):
    """讀取一筆有效測距點，封包失步時自動重新同步（最多嘗試 20 次）"""
    buf = bytearray(ser.read(5))
    if len(buf) < 5:
        return None

    for _ in range(20):
        result = parse_scan_packet(buf)
        if result:
            return result
        # 滑動一個 byte 重新對齊
        new_byte = ser.read(1)
        if not new_byte:
            return None
        buf = buf[1:] + bytearray(new_byte)

    return None


# ── 指令函式 ──────────────────────────────────────────────

def cmd_get_info(ser):
    print('\n[GET_DEVICE_INFO]')
    send_cmd(ser, CMD_GET_INFO)
    result = read_response_header(ser)
    if not result:
        print('  無回應')
        return
    ans_type, data_size = result
    data = ser.read(data_size)
    if len(data) < 20:
        print(f'  資料不足: {data.hex()}')
        return

    # struct: model(1) + firmware_version(u16 LE) + hardware_version(1) + serialnum(16)
    model    = data[0]
    fw_minor = data[1]          # firmware_version low byte
    fw_major = data[2]          # firmware_version high byte
    hw_ver   = data[3]
    serial_num = data[4:20].hex().upper()

    print(f'  型號 ID   : {model}')
    print(f'  韌體版本  : {fw_major}.{fw_minor:02d}')
    print(f'  硬體版本  : {hw_ver}')
    print(f'  序號      : {serial_num}')


def cmd_get_health(ser):
    print('\n[GET_HEALTH]')
    send_cmd(ser, CMD_GET_HEALTH)
    result = read_response_header(ser)
    if not result:
        print('  無回應')
        return
    ans_type, data_size = result
    data = ser.read(data_size)
    if len(data) < 3:
        print(f'  資料不足: {data.hex()}')
        return

    status     = data[0]
    error_code = struct.unpack('<H', data[1:3])[0]
    status_str = {0: 'OK', 1: 'Warning', 2: 'Error'}.get(status, '未知')
    print(f'  狀態      : {status_str} ({status})')
    print(f'  錯誤碼    : {error_code}')


def cmd_scan_once(ser, num_points=20):
    print('\n[SCAN] 開始掃描，讀取有效點...')

    # 送 STOP 清除舊資料
    send_cmd(ser, CMD_STOP)
    time.sleep(0.1)
    ser.reset_input_buffer()

    send_cmd(ser, CMD_SCAN)
    result = read_response_header(ser)
    if not result:
        print('  無回應')
        return
    ans_type, data_size = result
    print(f'  掃描回應 type=0x{ans_type:02X}, 每包 {data_size} bytes')

    print(f'\n  {"新圈":4} {"角度(°)":>10} {"距離(mm)":>12} {"品質":>6}')
    print('  ' + '-' * 36)

    count = 0
    while count < num_points:
        point = read_scan_point(ser)
        if point is None:
            print('  讀取失敗（逾時或失步）')
            break
        angle, dist_mm, quality, is_new_scan = point

        if dist_mm == 0:    # 無效測距，跳過
            continue

        marker = '[S]' if is_new_scan else '   '
        print(f'  {marker} {angle:>10.2f} {dist_mm:>12.2f} {quality:>6}')
        count += 1

    print('\n  停止掃描...')
    send_cmd(ser, CMD_STOP)
    time.sleep(0.15)
    ser.reset_input_buffer()


def cmd_reset(ser):
    print('\n[RESET] 重置裝置...')
    send_cmd(ser, CMD_RESET)
    time.sleep(2)
    ser.reset_input_buffer()
    print('  重置完成')


def main():
    print(f'開啟 {PORT} @ {BAUDRATE} baud...')
    with serial.Serial(PORT, BAUDRATE, timeout=2, dsrdtr=False) as ser:
        ser.dtr = False  # DTR Low → 馬達啟動
        time.sleep(0.1)
        ser.reset_input_buffer()

        while True:
            print('\n' + '=' * 40)
            print('選擇指令:')
            print('  1. GET_DEVICE_INFO  (0x50)')
            print('  2. GET_HEALTH       (0x52)')
            print('  3. SCAN             (0x20) - 讀 20 個有效點後停止')
            print('  4. RESET            (0x40)')
            print('  0. 離開')
            choice = input('> ').strip()

            if choice == '1':
                cmd_get_info(ser)
            elif choice == '2':
                cmd_get_health(ser)
            elif choice == '3':
                cmd_scan_once(ser)
            elif choice == '4':
                cmd_reset(ser)
            elif choice == '0':
                send_cmd(ser, CMD_STOP)
                break
            else:
                print('無效選項')


if __name__ == '__main__':
    main()
