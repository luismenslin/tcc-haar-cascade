import os
import csv
import time
import pickle
import argparse
import statistics
from pathlib import Path

import cv2


MODEL_PATH = "model_lbph.yml"
LABELS_PATH = "labels.pkl"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
THRESHOLD = 90.0
EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def carregar_imagens(entrada):
    """Recebe arquivo ou pasta e retorna a lista de imagens encontradas."""
    caminho = Path(entrada)

    if caminho.is_file():
        if caminho.suffix.lower() in EXTENSOES_VALIDAS:
            return [str(caminho)]
        return []

    if caminho.is_dir():
        imagens = []
        for arquivo in caminho.rglob("*"):
            if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS:
                imagens.append(str(arquivo))
        return sorted(imagens)

    return []


def carregar_labels(labels_path):
    """Carrega labels.pkl aceitando formatos comuns: {id:nome} ou {nome:id}."""
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    if isinstance(labels, dict):
        # Caso esteja como {"Luis": 0}, inverte para {0: "Luis"}
        if labels and all(isinstance(v, int) for v in labels.values()):
            return {v: k for k, v in labels.items()}
        return labels

    return labels


def criar_reconhecedor_lbph():
    """Cria o reconhecedor LBPH. Exige opencv-contrib-python."""
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face não encontrado. Instale o OpenCV contrib: pip install opencv-contrib-python"
        )
    return cv2.face.LBPHFaceRecognizer_create()


def reconhecer_imagem(path, face_cascade, recognizer, labels, threshold):
    """
    Processa uma imagem e retorna somente dados para relatório.
    O tempo medido considera: leitura da imagem, conversão, detecção facial e reconhecimento.
    Não exibe imagem e não exibe nomes reconhecidos.
    """
    inicio = time.perf_counter()

    frame = cv2.imread(path)

    if frame is None:
        tempo_ms = (time.perf_counter() - inicio) * 1000
        return {
            "imagem": os.path.basename(path),
            "caminho": path,
            "status": "Erro ao abrir imagem",
            "qtd_faces": 0,
            "tempo_ms": tempo_ms,
        }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        tempo_ms = (time.perf_counter() - inicio) * 1000
        return {
            "imagem": os.path.basename(path),
            "caminho": path,
            "status": "Nenhum rosto detectado",
            "qtd_faces": 0,
            "tempo_ms": tempo_ms,
        }

    qtd_faces_processadas = 0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        if roi_gray.size == 0:
            continue

        # O reconhecimento é executado, mas o nome/label não é exibido no relatório.
        label_id, confidence = recognizer.predict(roi_gray)

        # Mantém a lógica de decisão apenas para executar o fluxo completo.
        _nome = labels.get(label_id, "Desconhecido") if isinstance(labels, dict) else str(label_id)
        _resultado = _nome if confidence <= threshold else "Desconhecido"

        qtd_faces_processadas += 1

    tempo_ms = (time.perf_counter() - inicio) * 1000

    return {
        "imagem": os.path.basename(path),
        "caminho": path,
        "status": "Reconhecimento concluído",
        "qtd_faces": qtd_faces_processadas,
        "tempo_ms": tempo_ms,
    }


def salvar_csv(resultados, caminho_csv):
    with open(caminho_csv, "w", newline="", encoding="utf-8-sig") as f:
        campos = ["imagem", "caminho", "status", "qtd_faces", "tempo_ms"]
        writer = csv.DictWriter(f, fieldnames=campos, delimiter=";")
        writer.writeheader()

        for r in resultados:
            linha = r.copy()
            linha["tempo_ms"] = f"{linha['tempo_ms']:.3f}".replace(".", ",")
            writer.writerow(linha)


def gerar_resumo(resultados):
    tempos = [r["tempo_ms"] for r in resultados]

    if not tempos:
        return {
            "total": 0,
            "media": 0.0,
            "mediana": 0.0,
            "desvio": 0.0,
            "minimo": 0.0,
            "maximo": 0.0,
        }

    return {
        "total": len(tempos),
        "media": statistics.mean(tempos),
        "mediana": statistics.median(tempos),
        "desvio": statistics.stdev(tempos) if len(tempos) > 1 else 0.0,
        "minimo": min(tempos),
        "maximo": max(tempos),
    }


def imprimir_relatorio(resultados, resumo):
    print("\n===== RELATÓRIO DE PROCESSAMENTO HAAR CASCADE / LBPH =====")
    print("\nTempo individual por imagem:")

    for i, r in enumerate(resultados, start=1):
        print(
            f"{i:03d} | {r['imagem']} | "
            f"Status: {r['status']} | "
            f"Faces: {r['qtd_faces']} | "
            f"Tempo: {r['tempo_ms']:.3f} ms"
        )

    print("\n===== MÉTRICAS GERAIS =====")
    print(f"Total de imagens processadas: {resumo['total']}")
    print(f"Tempo médio: {resumo['media']:.3f} ms")
    print(f"Mediana: {resumo['mediana']:.3f} ms")
    print(f"Desvio padrão: {resumo['desvio']:.3f} ms")
    print(f"Menor tempo: {resumo['minimo']:.3f} ms")
    print(f"Maior tempo: {resumo['maximo']:.3f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Executa reconhecimento facial Haar Cascade/LBPH em lote via terminal e gera relatório de tempo em ms."
    )
    parser.add_argument(
        "entrada",
        help="Caminho de uma imagem ou de uma pasta com imagens."
    )
    parser.add_argument(
        "--saida",
        default="resultados_processamento_haar.csv",
        help="Caminho do arquivo CSV de saída. Padrão: resultados_processamento_haar.csv"
    )
    parser.add_argument(
        "--modelo",
        default=MODEL_PATH,
        help=f"Caminho do modelo .yml. Padrão: {MODEL_PATH}"
    )
    parser.add_argument(
        "--labels",
        default=LABELS_PATH,
        help=f"Caminho do arquivo labels.pkl. Padrão: {LABELS_PATH}"
    )
    parser.add_argument(
        "--cascade",
        default=CASCADE_PATH,
        help=f"Caminho do haarcascade_frontalface_default.xml. Padrão: {CASCADE_PATH}"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Limite de confiança do LBPH. Padrão: {THRESHOLD}"
    )

    args = parser.parse_args()

    imagens = carregar_imagens(args.entrada)

    if not imagens:
        print("Nenhuma imagem válida encontrada.")
        print("Extensões aceitas: .jpg, .jpeg, .png, .bmp, .webp")
        return

    print("Carregando Haar Cascade, modelo LBPH e labels...")

    face_cascade = cv2.CascadeClassifier(args.cascade)
    if face_cascade.empty():
        print(f"Erro ao carregar o arquivo Haar Cascade: {args.cascade}")
        return

    recognizer = criar_reconhecedor_lbph()
    recognizer.read(args.modelo)
    labels = carregar_labels(args.labels)

    print(f"Total de imagens encontradas: {len(imagens)}")
    print("Iniciando processamento...\n")

    resultados = []

    for i, path in enumerate(imagens, start=1):
        resultado = reconhecer_imagem(
            path=path,
            face_cascade=face_cascade,
            recognizer=recognizer,
            labels=labels,
            threshold=args.threshold,
        )
        resultados.append(resultado)
        print(f"{i}/{len(imagens)} - {resultado['imagem']} - {resultado['tempo_ms']:.3f} ms")

    resumo = gerar_resumo(resultados)
    salvar_csv(resultados, args.saida)
    imprimir_relatorio(resultados, resumo)

    print(f"\nCSV salvo em: {args.saida}")


if __name__ == "__main__":
    main()
