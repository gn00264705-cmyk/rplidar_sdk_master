from rplidar import RPLidar

PORT = 'COM4'
BAUDRATE = 115200

lidar = RPLidar(PORT, baudrate=BAUDRATE)

try:
    info = lidar.get_info()
    print('裝置資訊:', info)

    health = lidar.get_health()
    print('健康狀態:', health)

    print('\n開始掃描（按 Ctrl+C 停止）...')
    for i, scan in enumerate(lidar.iter_scans()):
        print(f'掃描 #{i+1}: {len(scan)} 個點')
        for (quality, angle, distance) in scan[:5]:
            print(f'  角度: {angle:.2f}°  距離: {distance:.0f}mm  品質: {quality}')
        if i >= 4:
            break

except Exception as e:
    print('錯誤:', e)

finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    print('已中斷連線')
