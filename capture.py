# capture.py
import cv2
import os
import argparse
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(description="Captura rostos usando Haar Cascade e salva no dataset.")
    ap.add_argument("--name", required=True, help="Nome da pessoa (cria pasta dataset/<name>)")
    ap.add_argument("--num", type=int, default=100, help="Quantidade de amostras a capturar")
    ap.add_argument("--camera", type=int, default=None, help="Índice da webcam (ex: 0 ou 1)")
    ap.add_argument("--video", type=str, default=None, help="Arquivo de vídeo em vez da webcam (ex: teste.mp4)")
    args = ap.parse_args()

    # Verifica fonte
    if args.camera is None and args.video is None:
        raise RuntimeError("Use --camera ou --video para capturar imagens.")

    person = args.name.strip().replace(" ", "_")
    out_dir = os.path.join("dataset", person)
    os.makedirs(out_dir, exist_ok=True)

    # Carrega Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Erro ao carregar Haar Cascade.")

    # Fonte: Webcam ou vídeo
    if args.video:
        print(f"[INFO] Usando vídeo como entrada: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"[INFO] Usando webcam {args.camera} como entrada...")
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a fonte de vídeo.")

    print(f"[INFO] Capturando {args.num} imagens de {person}...")
    saved = 0

    while saved < args.num:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Frame inválido. Fim do vídeo ou falha.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            filename = f"{person}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            cv2.imwrite(os.path.join(out_dir, filename), face)
            saved += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{person} ({saved}/{args.num})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Captura (pressione Q para sair)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[INFO] Finalizado! Total salvo: {saved}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
