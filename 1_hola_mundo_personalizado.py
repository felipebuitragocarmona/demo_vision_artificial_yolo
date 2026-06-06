"""
YOLO con filtro de clases
"""

import cv2
from ultralytics import YOLO

# Objetos que quiero detectar
objetos_interes = [

    "car",
    "dog"
]

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.4)

    for result in results:

        for box in result.boxes:

            clase_id = int(box.cls[0])
            nombre_clase = model.names[clase_id]

            # Filtrar solo las clases deseadas
            if nombre_clase not in objetos_interes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{nombre_clase} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Filtrado", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()