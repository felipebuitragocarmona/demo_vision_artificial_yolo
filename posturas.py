from ultralytics import YOLO
import cv2

# Modelo Pose
model = YOLO("models/yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

while True:

    ok, frame = cap.read()

    if not ok:
        break

    # Inferencia pose
    results = model(frame, verbose=False)

    # Dibujar resultados
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()