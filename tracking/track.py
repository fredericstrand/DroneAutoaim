from ultralytics import YOLO
import cv2 as cv
import serial
import time

# model path: ../pretraning/models/weights/best.pt

model = YOLO('yolov8s.pt')

# Open webcam
cap = cv.VideoCapture(0)

arduino = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)

# Frame size
FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'

            # Compute the center
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2

            x_angle = int((box_cx / FRAME_WIDTH) * 180)
            y_angle = int((box_cy / FRAME_HEIGHT) * 180)

            arduino.write(f"{x_angle},{y_angle}\n".encode())

            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            print(f'{label}: ({x1}, {y1}), ({x2}, {y2}) , Confidence: {conf:.2f}')

    cv.imshow('YOLOv8 Tracking', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

