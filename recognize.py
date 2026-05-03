# recognize_image.py
import cv2
import pickle
import argparse
import os

def load_model(model_path, labels_path):
    if not os.path.exists(model_path):
        raise RuntimeError(f"Modelo não encontrado: {model_path}")
    if not os.path.exists(labels_path):
        raise RuntimeError(f"Labels não encontrados: {labels_path}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(labels_path, "rb") as f:
        id_map = pickle.load(f)

    return recognizer, id_map

def main():
    ap = argparse.ArgumentParser(description="Reconhecimento facial em IMAGEM usando Haar + LBPH")
    ap.add_argument("--image", required=True, help="Caminho da imagem (jpg, jpeg, png)")
    ap.add_argument("--model", default="model_lbph.yml")
    ap.add_argument("--labels", default="labels.pkl")
    ap.add_argument("--max_conf", type=float, default=80)
    args = ap.parse_args()

    # carrega modelo
    recognizer, id_map = load_model(args.model, args.labels)

    # carrega classificador Haar para detectar rosto
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # abre imagem
    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError("Erro ao abrir a imagem. Verifique o caminho.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label_id, confidence = recognizer.predict(face)

        if confidence <= args.max_conf:
            name = id_map.get(label_id, "Desconhecido")
            color = (0, 255, 0)
        else:
            name = "Desconhecido"
            color = (0, 0, 255)

        text = f"{name} ({confidence:.1f})"
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    scale_percent = 40  # diminuir para 40% do tamanho original
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height))

    cv2.imshow("Reconhecimento - Aperte Q para sair", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
