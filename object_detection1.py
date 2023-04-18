from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import time
import numpy as np
from darknet import darknet

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'


class VideoCamera(object):
    def __init__(self):
        self.VIDEO = cv2.VideoCapture(0)
        self.detect = False
        self.flipH = False
        self._preview = True

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    def __del__(self):
        self.VIDEO.release()

    def get_frame(self):
        while self.VIDEO.isOpened():
            ret, snap = self.VIDEO.read()
            if self.flipH:
                snap = cv2.flip(snap, 1)
            labels = []
            confidences = []
            if ret == True:
                if self._preview:
                    if self.detect:
                        frame = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                        # Detect objects
                        results = model(frame)
                        labels = []
                        confidences = []
                        for i, det in enumerate(results.xyxy[0]):
                            # get the predicted class label and confidence for the detection
                            label = results.names[int(det[-1])]
                            confidence = float(det[-2])
                            labels.append(label)
                            confidences.append(confidence)

                        # Draw bounding boxes and labels on image
                        img = results.render()
                        # Convert back to BGR for displaying in OpenCV window
                        snap = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    snap = np.zeros((
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ), np.uint8)
                    label = "camera disabled"
                    H, W = snap.shape
                    font = cv2.FONT_HERSHEY_PLAIN
                    color = (255, 255, 255)
                    cv2.putText(snap, label, (W // 2 - 100, H // 2),
                                font, 2, color, 2)

                frame = cv2.imencode(".jpg", snap)[1].tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                time.sleep(0.01)

            else:
                break

        self.VIDEO.release()
        print("off")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@socketio.on('connect')
def test_connect():
    print('Connected')


if __name__ == '__main__':
    # Load YOLOv4-tiny model
    model_cfg = 'yolov4-tiny.cfg'
    model_weights = 'yolov4-tiny.weights'
    model_data = 'coco.data'
    model = darknet.load_model(model_cfg, model_weights)
    darknet.load_metadata(model_data)

    # Start the video camera
    VIDEO = VideoCamera()

    # Start the Flask app and socketio
    socketio.run(app, debug=True)

