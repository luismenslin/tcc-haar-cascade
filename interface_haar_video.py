import cv2
import pickle
import tkinter as tk
from tkinter import filedialog

# ================= CONFIG =================
MODEL_PATH = "model_lbph.yml"
LABELS_PATH = "labels.pkl"
MAX_CONF = 100

# ================= LOAD =================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)

# inverter labels se necessário
if isinstance(list(labels.keys())[0], str):
    labels = {v: k for k, v in labels.items()}


# ================= RECONHECIMENTO =================
def reconhecer_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

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

    return frame


# ================= PLAYER =================
class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial - Vídeo (HAAR)")
        self.root.geometry("400x150")

        self.cap = None
        self.rodando = False

        tk.Label(
            root,
            text="Reconhecimento Facial em Vídeo (HAAR)",
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        tk.Button(
            root,
            text="Selecionar Vídeo",
            command=self.selecionar_video,
            width=25
        ).pack(pady=5)

        tk.Button(
            root,
            text="Parar",
            command=self.parar,
            width=25
        ).pack(pady=5)

    def selecionar_video(self):
        path = filedialog.askopenfilename(
            title="Selecione um vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv")]
        )

        if not path:
            return

        self.cap = cv2.VideoCapture(path)
        self.rodando = True
        self.processar()

    def processar(self):
        if not self.rodando or self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.parar()
            return

        frame = reconhecer_frame(frame)

        cv2.imshow("Reconhecimento - HAAR", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            self.parar()
            return

        self.root.after(10, self.processar)

    def parar(self):
        self.rodando = False

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()


# ================= MAIN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()