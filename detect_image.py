# detect_image.py
import cv2
import argparse
import os

def main():
    ap = argparse.ArgumentParser(description="Detecção de rostos (Haar Cascade) em uma imagem.")
    ap.add_argument("--image", required=True, help="Caminho da imagem de entrada.")
    ap.add_argument("--out", default=None, help="Caminho de saída (imagem anotada). Opcional.")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Não foi possível abrir: {args.image}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Falha ao carregar Haar Cascade.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("Detecção - q para sair", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.out:
        cv2.imwrite(args.out, img)
        print(f"[OK] Salvo em {args.out}")

if __name__ == "__main__":
    main()