import time
from collections import deque, Counter

import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "best.pt"
CAMERA_INDEX = 0          # 0 suele ser la webcam principal
IMGSZ = 512
CONF = 0.7              # súbelo si da falsos positivos (0.5-0.7). bájalo si no detecta (0.25-0.4)
IOU = 0.5
DEVICE = "cpu"              # "0" = GPU NVIDIA, "cpu" = sin GPU

# Estabilidad (para que no “parpadee” el nombre)
HISTORY = 12              # frames a considerar
MIN_HITS = 7              # mínimo de frames (de HISTORY) con la misma carta para “confirmar”
COOLDOWN_S = 1.0          # segundos para volver a anunciar otra carta

# Voz (opcional)
USE_TTS = True            # si no instalaste pyttsx3, ponlo en False
DEBUG_FRAMES = True       # imprime info de cada frame para diagnosticar


def _init_tts():
    if not USE_TTS:
        return None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        return engine
    except Exception:
        print("[AVISO] No pude iniciar TTS (pyttsx3). Sigo solo con texto.")
        return None


def main():
    model = YOLO(MODEL_PATH)
    names = model.names  # {id: "clase"}

    # Mapeo solicitado: traducir ciertas etiquetas numéricas a nombres de lotería
    rename_map = {
        "2": "melon",
        "0": "catrin",
        "4": "paraguas",
        "7": "escaleras",
        "5": "soldado",
        "8": "muerte",
        "9": "rosa"
    }

    tts = _init_tts()
    last_spoken = None
    last_spoken_time = 0.0

    hist = deque(maxlen=HISTORY)

    cap = cv2.VideoCapture(CAMERA_INDEX)  # Sin CAP_DSHOW en Mac
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir la cámara index={CAMERA_INDEX}. Prueba 0, 1, 2...")

    # Opcional: fija resolución (si tu cámara lo soporta)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No pude leer frame de cámara.")
            break

        # Inferencia
        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False
        )
        r = results[0]

        detected_label = None
        detected_conf = None

        if r.boxes is not None and len(r.boxes) > 0:
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy()
            best_i = int(confs.argmax())
            detected_conf = float(confs[best_i])
            raw_label = names[int(clss[best_i])]
            detected_label = rename_map.get(raw_label, raw_label)

        hist.append(detected_label)

        # Decide etiqueta “estable”
        stable = None
        counts = Counter([x for x in hist if x is not None])
        if counts:
            label, hits = counts.most_common(1)[0]
            if hits >= MIN_HITS:
                stable = label

        if DEBUG_FRAMES:
            print(f"[DEBUG] detected={detected_label} stable={stable} counts={dict(counts)} hist_len={len(hist)}")

        # Render con cajas
        annotated = r.plot()  # dibuja bounding boxes y labels del modelo

        # Texto grande arriba
        display_text = stable if stable is not None else "..."
        cv2.putText(
            annotated,
            f"Carta: {display_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        # Anunciar cuando cambie de carta (y está estable)
        now = time.time()
        # Hablar cuando haya una carta estable nueva o cuando haya pasado el cooldown aunque sea distinta.
        if stable is not None and (stable != last_spoken) and (now - last_spoken_time) >= COOLDOWN_S:
            msg = f"Detectada: {stable}"
            if detected_conf is not None:
                msg += f" (conf {detected_conf:.2f})"
            print(msg)

            if tts is not None:
                try:
                    tts.say(stable)
                    tts.runAndWait()
                except Exception:
                    pass

            last_spoken = stable
            last_spoken_time = now

        cv2.imshow("Loteria YOLO - Cam", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q o ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
