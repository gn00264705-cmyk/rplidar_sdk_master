import cv2
import numpy as np

# MobileNet SSD 模型路徑 (請確認當前目錄下有這兩個檔案)
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# MobileNet SSD 能辨識的 20 種分類
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def gstreamer_pipeline(cam_id=0, width=640, height=480, fps=30):
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

def main():
    print("[INFO] 載入 AI 模型中...")
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    except Exception as e:
        print(f"[錯誤] 模型載入失敗，請確認 prototxt 與 caffemodel 檔案存在。({e})")
        return
    
    print("[INFO] 啟動左側攝影機...")
    # 這裡我們先開單邊相機(左邊)來做 AI 測試即可，以節省效能
    cap = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[錯誤] 無法開啟攝影機")
        return

    print("[INFO] 開始影像串流，按 'q' 鍵離開視窗")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        
        # 將影像轉換為神經網路需要的 Blob 格式 (MobileNet 通常接受 300x300 解析度)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        
        # 進行物件偵測
        net.setInput(blob)
        detections = net.forward()

        # 畫出偵測框
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # 過濾掉信心度太低的結果 (設定閾值為 0.5)
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # 繪製方框與文字標籤
                label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        cv2.imshow("AI Object Detection", frame)

        # 按 'q' 離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()