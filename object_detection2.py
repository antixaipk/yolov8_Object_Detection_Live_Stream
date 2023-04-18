
import torch
model =  torch.hub.load('ultralytics/yolov5', 'yolov5x')


import time
import os
import sys
from pathlib import Path
import numpy as np
import cv2
# from model_object_det import model_object_detection
from model_logo_det import model_logo_detection, device, run




class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        self.VIDEO = cv2.VideoCapture(0)
        self.VIDEO.set(10, 200)
        self._preview = True
        self._flipH = False
        self._detect = False

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)


    def show(self):
        while(self.VIDEO.isOpened()):
            ret, snap = self.VIDEO.read()
            if self.flipH:
                snap = cv2.flip(snap, 1)
            labels = []
            confidences = []
            if ret == True:
                if self._preview:
                    # snap = cv2.resize(snap, (0, 0), fx=0.5, fy=0.5)
                    if self.detect:
                        frame = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        ))
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
                        print(labels)
                        print(confidences)
                        # Draw bounding boxes and labels on image
                        img = results.render()
                        print(type(img))

                        img, labels = run(model = model_logo_detection, device= device, testImg=frame, imageName= "imgtest")
                        print(labels)
                        # cv2.imshow("", img)
                        # cv2.waitKey(0)
                        # Convert back to BGR for displaying in OpenCV window
                        snap = img
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
                    cv2.putText(snap, label, (W//2 - 100, H//2),
                                font, 2, color, 2)

                frame = cv2.imencode(".jpg", snap)[1].tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                                                                     # b'Accuracy: ' + confidences + b'\r\n')

                time.sleep(0.01)

            else:
                break

        self.VIDEO.release()
        print("off")
