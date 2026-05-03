import os
import cv2
import numpy as np
import pickle

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ================= CONFIG =================
MODEL_PATH = "model_lbph.yml"
LABELS_PATH = "labels.pkl"
MAX_CONF = 70  # ajuste fino aqui

# ================= LOAD =================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)

# inverter dict se necessário
if isinstance(list(labels.keys())[0], str):
    labels = {v: k for k, v in labels.items()}


# ================= RECONHECIMENTO =================
def reconhecer_imagem(path):
    frame = cv2.imread(path)

    if frame is None:
        return None, "Erro ao abrir imagem"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return frame, "Nenhum rosto detectado"

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (200, 200))
            id_pred, conf = recognizer.predict(face)
        except:
            continue

        if conf <= MAX_CONF:
            nome = labels.get(id_pred, "Desconhecido")
            label = f"{nome} ({int(conf)})"
            color = (0, 255, 0)
        else:
            label = f"Desconhecido ({int(conf)})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return frame, "Reconhecimento concluído"


# ================= REDIMENSIONAR =================
def redimensionar_para_tela(frame, max_width=800, max_height=550):
    h, w = frame.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    return frame


# ================= INTERFACE =================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Teste de Reconhecimento Facial - HAAR")
        self.root.geometry("950x750")

        self.imagens = []
        self.indice_atual = 0
        self.tk_image = None

        titulo = tk.Label(
            root,
            text="Reconhecimento Facial com Haar Cascade",
            font=("Arial", 18, "bold")
        )
        titulo.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_selecionar = tk.Button(
            btn_frame,
            text="Selecionar imagem(ns)",
            command=self.selecionar_imagens,
            font=("Arial", 12),
            width=22
        )
        self.btn_selecionar.grid(row=0, column=0, padx=5)

        self.btn_anterior = tk.Button(
            btn_frame,
            text="Anterior",
            command=self.anterior,
            font=("Arial", 12),
            width=12
        )
        self.btn_anterior.grid(row=0, column=1, padx=5)

        self.btn_proxima = tk.Button(
            btn_frame,
            text="Próxima",
            command=self.proxima,
            font=("Arial", 12),
            width=12
        )
        self.btn_proxima.grid(row=0, column=2, padx=5)

        self.lbl_status = tk.Label(
            root,
            text="Nenhuma imagem selecionada",
            font=("Arial", 11)
        )
        self.lbl_status.pack(pady=5)

        self.lbl_imagem = tk.Label(root)
        self.lbl_imagem.pack(pady=10)

    def selecionar_imagens(self):
        paths = filedialog.askopenfilenames(
            title="Selecione imagens",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not paths:
            return

        self.imagens = list(paths)
        self.indice_atual = 0
        self.exibir_imagem_atual()

    def exibir_imagem_atual(self):
        if not self.imagens:
            return

        path = self.imagens[self.indice_atual]
        frame, status = reconhecer_imagem(path)

        if frame is None:
            messagebox.showerror("Erro", status)
            return

        frame = redimensionar_para_tela(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(frame_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)

        self.lbl_imagem.config(image=self.tk_image)

        nome_arquivo = os.path.basename(path)
        self.lbl_status.config(
            text=f"{self.indice_atual + 1}/{len(self.imagens)} - {nome_arquivo} - {status}"
        )

    def proxima(self):
        if self.imagens:
            self.indice_atual = (self.indice_atual + 1) % len(self.imagens)
            self.exibir_imagem_atual()

    def anterior(self):
        if self.imagens:
            self.indice_atual = (self.indice_atual - 1) % len(self.imagens)
            self.exibir_imagem_atual()


# ================= MAIN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()