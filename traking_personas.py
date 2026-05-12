"""
Demo YOLO - Tracking en tiempo real con webcam
Dependencias: pip install ultralytics opencv-python
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
PRINT_EVERY  = 10
JSON_PATH    = "points/detecciones.json"
JSON_EVERY   = 5
TRACKER      = "bytetrack.yaml"
# ──────────────────────────────────────────────────────────────────────────────


def draw_detections(frame, results):
    """Dibuja bounding boxes con ID y retorna lista de detecciones."""
    detections = []

    for result in results:
        for box in result.boxes:

            # Si no hay ID, saltar esta detección
            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = result.names[cls]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            color = (
                (track_id * 37) % 256,
                (track_id * 97) % 256,
                (track_id * 173) % 256
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"ID {track_id} | {label} {conf:.0%}"

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1
            )

            cv2.rectangle(
                frame,
                (x1, y1 - th - 8),
                (x1 + tw + 6, y1),
                color,
                -1
            )

            cv2.putText(
                frame,
                text,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            detections.append(
                (track_id, label, conf, x1, y1, x2, y2, cx, cy)
            )

    return frame, detections


def print_detections(frame_n, detections):
    """Imprime tabla de detecciones del frame actual."""
    print(f"\n{'─'*80}")
    print(f"  Frame #{frame_n:<6}  |  {len(detections)} objeto(s) detectado(s)")
    print(f"{'─'*80}")

    if detections:
        print(
            f"  {'ID':>4} {'Objeto':<14} {'Conf':>5}  "
            f"{'x1':>5} {'y1':>5} {'x2':>5} {'y2':>5}  "
            f"{'cx':>5} {'cy':>5}"
        )

        print(f"  {'─'*4} {'─'*14} {'─'*5}  {'─'*5} {'─'*5} {'─'*5} {'─'*5}  {'─'*5} {'─'*5}")

        for track_id, label, conf, x1, y1, x2, y2, cx, cy in detections:
            print(
                f"  {track_id:>4} {label:<14} {conf:>4.0%}  "
                f"{x1:>5} {y1:>5} {x2:>5} {y2:>5}  "
                f"{cx:>5} {cy:>5}"
            )
    else:
        print("  (sin detecciones)")

    print(f"{'─'*80}")


def save_json(json_path, frame_n, detections):
    """Carga el JSON existente, agrega el frame actual y guarda."""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"modelo": MODEL_NAME, "tracker": TRACKER, "frames": []}

    entrada = {
        "frame": frame_n,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "total_objetos": len(detections),
        "objetos": [
            {
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
            }
            for track_id, label, conf, x1, y1, x2, y2, cx, cy in detections
        ]
    }

    data["frames"].append(entrada)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
                "frames": []
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Cámara abierta. JSON guardando en: {os.path.abspath(JSON_PATH)}")
    print("Presiona [Q] para salir.\n")

    frame_n = 0

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

        frame, detections = draw_detections(frame, results)

        if PRINT_EVERY == 0 or frame_n % PRINT_EVERY == 0:
            print_detections(frame_n, detections)

        if JSON_EVERY == 0 or frame_n % JSON_EVERY == 0:
            save_json(JSON_PATH, frame_n, detections)

        fps_text = f"FPS: {cap.get(cv2.CAP_PROP_FPS):.0f}"

        cv2.putText(
            frame,
            fps_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLO Tracking - Detección con ID [Q] salir", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDemo finalizado. Detecciones guardadas en: {os.path.abspath(JSON_PATH)}")


if __name__ == "__main__":
    main()