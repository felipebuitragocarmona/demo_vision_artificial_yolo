"""
Demo YOLO - Tracking + conteo por línea configurable
Con redimensionamiento a VGA (640×480)
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

LINE_MODE    = "vertical"   # "vertical" o "horizontal"
VGA_W, VGA_H = 640, 480     # Resolución objetivo
# ──────────────────────────────────────────────────────────────────────────────


def resize_vga(frame, width=VGA_W, height=VGA_H):
    """Redimensiona el frame a VGA (640×480) de forma simple."""
    return cv2.resize(frame, (width, height))


def save_json(json_path, frame_n, detections, contadores, line_mode, line_pos):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "modelo": MODEL_NAME,
            "tracker": TRACKER,
            "line_mode": line_mode,
            "frames": []
        }

    entrada = {
        "frame": frame_n,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "line_mode": line_mode,
        "line_pos": line_pos,
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

            if label != "person":
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color = (
                (track_id * 37) % 256,
                (track_id * 97) % 256,
                (track_id * 173) % 256
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Punto rojo en el centro de la persona
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)

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

            if line_mode == "vertical":
                pos_actual = cx
            else:
                pos_actual = cy

            if track_id in posiciones_previas:
                pos_anterior = posiciones_previas[track_id]

                if pos_anterior < line_pos <= pos_actual:
                    contadores["entradas"] += 1
                    print(f"Entrada detectada | ID {track_id}")

                elif pos_anterior > line_pos >= pos_actual:
                    contadores["salidas"] += 1
                    print(f"Salida detectada | ID {track_id}")

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

    return frame, detections, line_pos


def main():
    print(f"Cargando modelo {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture("videos/video1.MOV")

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video")

    # Mostrar resolución original
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución original: {orig_w}×{orig_h}")
    print(f"Redimensionando a VGA: {VGA_W}×{VGA_H}")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo": MODEL_NAME,
                "tracker": TRACKER,
                "line_mode": LINE_MODE,
                "resolucion": f"{VGA_W}x{VGA_H}",
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

    print("Video abierto.")
    print(f"Modo de línea: {LINE_MODE}")
    print("Presiona [Q] para salir.\n")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Fin del video o error leyendo frame.")
            break

        frame_n += 1

        # ── Redimensionar a VGA antes de inferencia ──
        frame = resize_vga(frame)

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

        # Resolución en pantalla
        cv2.putText(
            frame,
            f"{VGA_W}x{VGA_H}",
            (VGA_W - 100, VGA_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
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