from ultralytics import YOLO
import cv2 as cv
import serial
import time

arduino = serial.Serial('/dev/tty.usbmodem2113401', 9600)
time.sleep(2)

model = YOLO('yolov8m.pt')
cap = cv.VideoCapture(0)
FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, stream=True)
    person_found = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_cx = (x1 + x2) // 2
                x_angle = ((box_cx / FRAME_WIDTH) * 180) - 90
                x_angle = max(-90, min(90, x_angle))
                arduino.write(f"{int(x_angle)}\n".encode())
                person_found = True
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv.putText(frame, f'{label} {float(box.conf[0]):.2f}',
                           (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                break

    if not person_found:
        arduino.write(b"0\n")

    cv.imshow('YOLOv8 Tracking', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
