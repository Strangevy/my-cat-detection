import os
import cv2
from flask import Flask, render_template, Response
from flask_uploads import UploadSet, IMAGES, configure_uploads
from ultralytics import YOLO

# Flask 应用程序设置
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOADED_IMAGES_DEST'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

# 图片上传配置
images = UploadSet('images', IMAGES)
configure_uploads(app, images)

# 加载 YOLO 模型
# model = YOLO('best.pt')  # 修改为您的模型路径
model = YOLO('yolov8n.pt')  # 修改为您的模型路径

# DroidCam视频流URL
stream_url = 'http://192.168.233.160:4747/video'

@app.route('/')
def index():
    """视频流首页."""
    return render_template('index.html')

def gen():
    """视频流生成器函数."""
    cap = cv2.VideoCapture(stream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 进行对象检测
        results = model(frame)  # 使用模型对象的预测方法

        for result in results:  # 迭代每个检测结果
            if result.boxes is not None and len(result.boxes) > 0:  # 检查是否有检测到的对象
                for box in result.boxes:  # 每个检测结果是一个字典，包含检测到的对象信息
                    if len(box.xyxy) == 4:  # 检查边界框是否具有四个值
                        x_min, y_min, x_max, y_max = box.xyxy  # 获取坐标信息
                        conf = float(box.conf)  # 获取置信度
                        cls = int(box.cls)  # 获取类别标签

                        label = f'{result.names[cls]} {conf:.2f}'

                        # 在帧上绘制边界框
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """视频流路由，将此路由放入img标签的src属性中."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)