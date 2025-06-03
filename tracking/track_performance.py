import cv2 as cv
import numpy as np
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

factory = PiGPIOFactory()
servo_x = AngularServo(12, min_angle=-90, max_angle=90,
                       min_pulse_width=0.0006, max_pulse_width=0.0024,
                       pin_factory=factory)

net = cv.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

cap = cv.VideoCapture(0)
FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    best_box = None
    best_confidence = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * FRAME_WIDTH)
                width = int(detection[2] * FRAME_WIDTH)
                x1 = center_x - width // 2
                x2 = center_x + width // 2

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, x2, center_x, class_id)

    if best_box:
        x1, x2, box_cx, class_id = best_box
        x_angle = ((box_cx / FRAME_WIDTH) * 180) - 90
        x_angle = max(-90, min(90, x_angle))
        servo_x.angle = x_angle

        label = f'{classes[class_id]} {best_confidence:.2f}'
        cv.rectangle(frame, (x1, 100), (x2, 300), (255, 0, 0), 2)
        cv.putText(frame, label, (x1, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.imshow("YOLOv4-tiny Tracking", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
