# train.py
import os
import cv2
import numpy as np
import pickle

DATASET_DIR = "dataset"
MODEL_PATH = "model_lbph.yml"
LABELS_PATH = "labels.pkl"

def load_dataset(dataset_dir):
    images = []
    labels = []
    label_map = {}   # nome -> id
    id_map = {}      # id -> nome
    current_id = 0

    for person in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue
        label_map[person] = current_id
        id_map[current_id] = person

        for fname in os.listdir(person_dir):
            fpath = os.path.join(person_dir, fname)
            if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png")):
                continue
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Redimensiona para um padrão (compatível com capture.py)
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(current_id)

        current_id += 1

    return images, np.array(labels, dtype=np.int32), id_map

def main():
    if not os.path.isdir(DATASET_DIR):
        raise RuntimeError(f"Pasta '{DATASET_DIR}' não encontrada. Rode capture.py antes.")

    images, labels, id_map = load_dataset(DATASET_DIR)
    if len(images) == 0:
        raise RuntimeError("Dataset vazio. Capture imagens antes de treinar.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    print(f"[INFO] Treinando LBPH com {len(images)} imagens de {len(id_map)} pessoas...")
    recognizer.train(images, labels)

    recognizer.write(MODEL_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(id_map, f)
    print(f"[OK] Modelo salvo em {MODEL_PATH} e labels em {LABELS_PATH}")

if __name__ == "__main__":
    main()