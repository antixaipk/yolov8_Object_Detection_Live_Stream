# from ultralytics import YOLO
# import cv2
# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#
# # Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# results = model("laptop.jpg")  # predict on an image
# # print(results[0].names)
#
# image, labels = results[0].plot()
# print(labels)
# cv2.imwrite("out.jpg", image)
#
# # success = model.export(format="onnx")  # export the model to ONNX format
import cv2
from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch

model = YOLO("best.pt")  # load a pretrained model (recommended for training)
model.overrides[""]
results = model("Capture.JPG")  # predict on an image

image, labels = results[0].plot()
print(labels)
cv2.imwrite("out.jpg", image)

# Use the model
# model.train(data="custom_data.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format