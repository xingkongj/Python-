import cv2

# 加载人脸识别器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

try:
    while True:
        # 读取摄像头帧
        ret, img = cap.read()

        # 将图像转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 进行人脸识别
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 绘制矩形框标记人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示图像
        cv2.imshow('img', img)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

# 释放摄像头资源
cap.release()

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()