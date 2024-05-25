import os
import cv2
from flask import Flask, render_template, Response
# from flask_uploads import UploadSet, IMAGES, configure_uploads  # 不再需要这个导入
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename  # 直接从 werkzeug 导入
from werkzeug.datastructures import FileStorage  # 直接从 werkzeug 导入
import numpy as np

# Flask application settings
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOADED_IMAGES_DEST'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Image upload configuration
# images = UploadSet('images', IMAGES)  # 不再需要这个配置
# configure_uploads(app, images)  # 不再需要这个配置

# Load YOLO model
model = YOLO('best.pt')  # Change to your model path

# DroidCam video stream URL
stream_url = os.environ.get('STREAM_URL', 'http://192.168.233.160:4747/video')

# Set Chinese font
font_path = "./fonts/NotoSansSC-Regular.ttf"  # Replace this path with the path to your downloaded Chinese font file
font_size = 20
font = ImageFont.truetype(font_path, font_size)

@app.route('/')
def index():
    """Video stream homepage."""
    return render_template('index.html')

def draw_chinese_text(image, text, position, font, color=(255, 0, 0)):
    """Draw Chinese text on an image."""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def gen():
    """Video stream generator function."""
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from stream.")
            break

        # Perform object detection
        results = model(frame)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    label = f'{model.names[cls]} {conf:.2f}'

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    
                    # Draw Chinese label on the frame
                    frame = draw_chinese_text(frame, label, (x_min, y_min - 10), font)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video stream route, put this route in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    debug_mode = os.environ.get('DEBUG', 'False') == 'True'
    print(debug_mode)
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
