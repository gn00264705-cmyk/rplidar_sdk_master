import cv2
import numpy as np

# 建立 GStreamer 管道字串
# IMX219-83 支援的最大解析度很高，但開發階段建議先用 640x480 以維持流暢
def gstreamer_pipeline(cam_id=0, width=640, height=480, fps=30):
    # 使用您日誌中的完整硬體路徑
    cam_names = [
        "/base/axi/pcie@1000120000/rp1/i2c@88000/imx219@10",
        "/base/axi/pcie@1000120000/rp1/i2c@80000/imx219@10"
    ]
    
    # 關鍵修正：加入 format=I420 避免 PiSP 崩潰
    return (
        f"libcamerasrc camera-name={cam_names[cam_id]} ! "
        f"video/x-raw, format=I420, width={width}, height={height}, framerate={fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=True sync=False"
    )

def main():
    # 建立兩個相機的物件
    cap_l = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("錯誤：無法開啟雙目相機，請檢查連接或 config.txt 設定。")
        return

    print("雙目視圖啟動中... 按 'q' 鍵退出。")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if ret_l and ret_r:
            # 將左右影像水平合併
            combined = np.hstack((frame_l, frame_r))
            cv2.imshow("Stereo View (Left | Right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()