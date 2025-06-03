from ultralytics import YOLO
import cv2 as cv
from gpiozero import AngularServo
from time import sleep

model = YOLO('yolov8s.pt')
cap = cv.VideoCapture(0)
servo_x = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)
servo_y = AngularServo(17, min_pulse_width=0.0006, max_pulse_width=0.0023)

FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, stream=True)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2

            x_angle = int((box_cx / FRAME_WIDTH) * 180) - 90
            y_angle = int((box_cy / FRAME_HEIGHT) * 180) - 90

            servo_x.angle = x_angle
            servo_y.angle = y_angle

            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f'{model.names[int(box.cls[0])]} {float(box.conf[0]):.2f}'
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv.imshow('YOLOv8 Tracking', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
