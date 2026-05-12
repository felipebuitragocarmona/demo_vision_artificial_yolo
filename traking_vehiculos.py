"""
YOLO - Tracking + conteo por línea configurable
Detecta clases configurables desde una lista.

Dependencias:
pip install ultralytics opencv-python
"""

import cv2
import json
import os
from datetime import datetime
from ultralytics import YOLO

# ─── Configuración ────────────────────────────────────────────────────────────
MODEL_NAME   = "models/yolov8n.pt"
CAMERA_INDEX = 0
CONF_THRESH  = 0.4
JSON_PATH    = "points/detecciones.json"
JSON_EVERY   = 5
TRACKER      = "bytetrack.yaml"

LINE_MODE = "vertical"  # "vertical" o "horizontal"

# Clases a detectar y contar
TARGET_CLASSES = [
    "car",
    # "person",
    # "bus",
    # "truck",
    # "motorcycle",
    # "bicycle",
]
# ──────────────────────────────────────────────────────────────────────────────


def save_json(json_path, frame_n, detections, contadores, line_mode, line_pos):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "modelo": MODEL_NAME,
            "tracker": TRACKER,
            "line_mode": line_mode,
            "target_classes": TARGET_CLASSES,
            "frames": []
        }

    entrada = {
        "frame": frame_n,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "line_mode": line_mode,
        "line_pos": line_pos,
        "target_classes": TARGET_CLASSES,
        "entradas": contadores["entradas"],
        "salidas": contadores["salidas"],
        "total_objetos": len(detections),
        "objetos": detections
    }

    data["frames"].append(entrada)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def procesar_frame(frame, results, posiciones_previas, contadores, line_mode):
    detections = []

    h, w = frame.shape[:2]

    if line_mode == "vertical":
        line_pos = w // 2
        cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 0, 0), 2)

        cv2.putText(
            frame,
            "LINEA VERTICAL",
            (line_pos + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    elif line_mode == "horizontal":
        line_pos = h // 2
        cv2.line(frame, (0, line_pos), (w, line_pos), (255, 0, 0), 2)

        cv2.putText(
            frame,
            "LINEA HORIZONTAL",
            (10, line_pos - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    else:
        raise ValueError("LINE_MODE debe ser 'vertical' o 'horizontal'")

    for result in results:
        for box in result.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]

            # Detectar solo las clases configuradas
            if label not in TARGET_CLASSES:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color = (
                (track_id * 37) % 256,
                (track_id * 97) % 256,
                (track_id * 173) % 256
            )

            # Caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Punto rojo en el centro
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)

            # Texto
            texto = f"ID {track_id} | {label} {conf:.0%}"

            cv2.putText(
                frame,
                texto,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Usar X si la línea es vertical, Y si es horizontal
            if line_mode == "vertical":
                pos_actual = cx
            else:
                pos_actual = cy

            # Conteo por cruce de línea
            if track_id in posiciones_previas:
                pos_anterior = posiciones_previas[track_id]

                if pos_anterior < line_pos <= pos_actual:
                    contadores["entradas"] += 1
                    print(f"Entrada detectada | ID {track_id} | {label}")

                elif pos_anterior > line_pos >= pos_actual:
                    contadores["salidas"] += 1
                    print(f"Salida detectada | ID {track_id} | {label}")

            posiciones_previas[track_id] = pos_actual

            detections.append({
                "id": track_id,
                "label": label,
                "confianza": round(conf, 3),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "centro": {
                    "cx": cx,
                    "cy": cy
                }
            })

    cv2.putText(
        frame,
        f"Entradas: {contadores['entradas']}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Salidas: {contadores['salidas']}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Detectando: {', '.join(TARGET_CLASSES)}",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return frame, detections, line_pos


def main():
    print(f"Cargando modelo {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara índice {CAMERA_INDEX}")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo": MODEL_NAME,
                "tracker": TRACKER,
                "line_mode": LINE_MODE,
                "target_classes": TARGET_CLASSES,
                "frames": []
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    posiciones_previas = {}

    contadores = {
        "entradas": 0,
        "salidas": 0
    }

    frame_n = 0

    print("Cámara abierta.")
    print(f"Modo de línea: {LINE_MODE}")
    print(f"Detectando: {TARGET_CLASSES}")
    print("Presiona [Q] para salir.\n")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Error leyendo frame.")
            break

        frame_n += 1

        results = model.track(
            frame,
            conf=CONF_THRESH,
            persist=True,
            tracker=TRACKER,
            verbose=False
        )

        frame, detections, line_pos = procesar_frame(
            frame,
            results,
            posiciones_previas,
            contadores,
            LINE_MODE
        )

        if JSON_EVERY == 0 or frame_n % JSON_EVERY == 0:
            save_json(
                JSON_PATH,
                frame_n,
                detections,
                contadores,
                LINE_MODE,
                line_pos
            )

        cv2.imshow("YOLO Tracking + Conteo [Q] salir", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nDemo finalizado.")
    print(f"Entradas: {contadores['entradas']}")
    print(f"Salidas: {contadores['salidas']}")
    print(f"JSON guardado en: {os.path.abspath(JSON_PATH)}")


if __name__ == "__main__":
    main()