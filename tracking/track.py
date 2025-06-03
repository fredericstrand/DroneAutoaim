from ultralytics import YOLO
import cv2 as cv
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

factory = PiGPIOFactory()

servo_x = AngularServo(12, min_angle=-90, max_angle=90,
                       min_pulse_width=0.0006, max_pulse_width=0.0024,
                       pin_factory=factory)

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the webcam
cap = cv.VideoCapture(0)
FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, stream=True)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_cx = (x1 + x2) // 2

            # Convert X center to angle between -90 and 90
            x_angle = ((box_cx / FRAME_WIDTH) * 180) - 90
            x_angle = max(-90, min(90, x_angle))  # Clamp

            servo_x.angle = x_angle

            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f'{model.names[int(box.cls[0])]} {float(box.conf[0]):.2f}'
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv.imshow('YOLOv8 Tracking', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
