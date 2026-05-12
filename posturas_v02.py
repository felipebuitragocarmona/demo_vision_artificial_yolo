from ultralytics import YOLO
import cv2
import json
import os
from datetime import datetime

# ─── Configuración ────────────────────────────────────────────────────────────
MODEL_NAME = "models/yolov8n-pose.pt"
JSON_PATH = "points/pose_keypoints.json"
CAMERA_INDEX = 0
CONF_THRESH = 0.5
JSON_EVERY = 5
# ──────────────────────────────────────────────────────────────────────────────

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


def save_json(json_path, frame_n, personas):
    entrada = {
        "frame": frame_n,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "total_personas": len(personas),
        "personas": personas
    }

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(entrada)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print(f"Cargando modelo {MODEL_NAME}...")

    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    # Crear JSON limpio como lista
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    frame_n = 0

    print("Presiona [Q] para salir.\n")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_n += 1

        results = model(
            frame,
            conf=CONF_THRESH,
            verbose=False
        )

        annotated_frame = results[0].plot()
        personas_json = []

        if results[0].keypoints is not None:
            keypoints_xy = results[0].keypoints.xy
            keypoints_conf = results[0].keypoints.conf

            for person_idx, person_points in enumerate(keypoints_xy):
                persona = {
                    "persona_id": person_idx,
                    "keypoints": []
                }

                for kp_idx, point in enumerate(person_points):
                    x = float(point[0])
                    y = float(point[1])

                    conf = float(keypoints_conf[person_idx][kp_idx])
                    nombre = KEYPOINT_NAMES[kp_idx]

                    persona["keypoints"].append({
                        "nombre": nombre,
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "confianza": round(conf, 3)
                    })

                    if x > 0 and y > 0:
                        cv2.putText(
                            annotated_frame,
                            nombre,
                            (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 255),
                            1
                        )

                personas_json.append(persona)

        if JSON_EVERY == 0 or frame_n % JSON_EVERY == 0:
            save_json(
                JSON_PATH,
                frame_n,
                personas_json
            )

        cv2.imshow("YOLO Pose", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nJSON guardado en: {os.path.abspath(JSON_PATH)}")


if __name__ == "__main__":
    main()