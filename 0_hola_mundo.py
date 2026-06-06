"""
Demo simple de YOLO con webcam
Instalar:
pip install ultralytics opencv-python
"""

import cv2
from ultralytics import YOLO

# Cargar modelo YOLO
model = YOLO("yolov8n.pt")

# Abrir cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detectar objetos
    results = model(frame, conf=0.4)

    # Dibujar detecciones automáticamente
    frame_detectado = results[0].plot()

    # Mostrar imagen
    cv2.imshow("Demo YOLO - Presiona Q para salir", frame_detectado)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()