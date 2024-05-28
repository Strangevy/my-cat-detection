import os
import cv2
import time
import threading
import logging
from collections import Counter
from ultralytics import YOLO
import requests
from contextlib import contextmanager

# Load environment variables
model_path = os.getenv("MODEL_PATH")
stream_url = os.getenv("STREAM_URL")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ultralytics_logger = logging.getLogger('ultralytics')
ultralytics_logger.setLevel(logging.WARNING)

# Load YOLO model
model = YOLO(model_path)

@contextmanager
def open_video_capture(url):
    """Context manager to open and release video capture."""
    cap = cv2.VideoCapture(url)
    try:
        yield cap
    finally:
        cap.release()

def telegram_message(message):
    """Send a message to the Telegram bot."""
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {'chat_id': telegram_chat_id, 'text': message}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info("Message sent successfully.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending message: {e}")

class ObjectDetector(threading.Thread):
    """Thread class for video processing and reporting."""

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.detection_counter = Counter()
        self.lock = threading.Lock()

    def run(self):
        """Main loop for video processing."""
        while not self.stop_event.is_set():
            with open_video_capture(stream_url) as cap:
                logging.info("Connected to video stream.")
                while not self.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logging.error("Failed to read frame from stream. Reconnecting in 5 seconds...")
                        time.sleep(5)
                        break  # Exit inner loop to reconnect

                    logging.info("Processing frame...")
                    results = model(frame)

                    with self.lock:
                        for result in results:
                            if result.boxes is not None and len(result.boxes) > 0:
                                for box in result.boxes:
                                    cls = int(box.cls[0].item())
                                    label = model.names[cls]
                                    self.detection_counter[label] += 1

                    time.sleep(5)  # Control processing rate

        logging.info("Video processing thread stopped.")

    def report_detections(self):
        """Report detections at regular intervals."""
        while not self.stop_event.is_set():
            time.sleep(60)
            with self.lock:
                if self.detection_counter:
                    filtered_detections = {label: count for label, count in self.detection_counter.items() if count >= 3}
                    if filtered_detections:
                        sorted_detections = sorted(filtered_detections.items(), key=lambda item: item[1], reverse=True)
                        message = "Object detection report:\n\n" + "\n".join([f"{label}: {count}" for label, count in sorted_detections])
                        logging.info(f"Sending detection report: {message}")
                        telegram_message(message)
                    self.detection_counter.clear()

    def stop(self):
        """Stop the video processing and reporting."""
        self.stop_event.set()

def main():
    """Main function to start object detection and reporting."""
    detector = ObjectDetector()
    report_thread = threading.Thread(target=detector.report_detections)

    try:
        logging.info("Starting object detector thread.")
        detector.start()
        logging.info("Starting report thread.")
        report_thread.start()
        detector.join()  # Wait for detector thread to finish
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
    finally:
        logging.info("Stopping threads...")
        detector.stop()
        report_thread.join()
        logging.info("Stopped all threads.")

if __name__ == "__main__":
    main()
