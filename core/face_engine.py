"""Detecção (YuNet) + reconhecimento (SFace) usando o módulo DNN do OpenCV.

Ambos são modelos ONNX leves do OpenCV Zoo, rodam em CPU e dispensam
dlib/onnxruntime/GPU — ideais para o Raspberry Pi 3B.

Fluxo:
  detect(frame)          -> lista de "faces" (linhas Nx15 do YuNet)
  embed(frame, face)     -> vetor (128,) float32 L2-normalizado
  match(vec, gallery)    -> (person_id, score_cosseno) do mais parecido
"""

import cv2
import numpy as np

from .config import project_path

# O YuNet "2023mar" usa formas de entrada dinâmicas e SÓ carrega no OpenCV >= 4.8.
# Se você estiver preso a um OpenCV mais antigo (ex.: o python3-opencv 4.6 do
# Raspberry Pi OS Bookworm), use o modelo "2022mar" — daí este mapa de fallback.
_DETECTOR_FALLBACKS = {
    "face_detection_yunet_2023mar.onnx": "face_detection_yunet_2022mar.onnx",
    "face_detection_yunet_2023mar_int8.onnx": "face_detection_yunet_2022mar.onnx",
}


def _cv_version() -> tuple[int, int]:
    parts = cv2.__version__.split(".")
    return int(parts[0]), int(parts[1])


def _resolve_model(rel_path: str, kind: str) -> str:
    """Devolve um caminho de modelo existente e compatível com este OpenCV."""
    path = project_path(rel_path)
    major, minor = _cv_version()
    too_old = (major, minor) < (4, 8)

    candidates = [path]
    if kind == "detector":
        fb = _DETECTOR_FALLBACKS.get(path.name)
        if fb:
            fb_path = path.with_name(fb)
            if too_old:
                candidates.insert(0, fb_path)   # com OpenCV antigo, o legado vem 1º
            else:
                candidates.append(fb_path)

    for cand in candidates:
        if cand.exists():
            if kind == "detector" and too_old and "2023mar" in cand.name:
                raise RuntimeError(
                    f"OpenCV {cv2.__version__} é antigo demais para {cand.name} "
                    "(o YuNet 2023mar exige OpenCV >= 4.8). Corrija com:\n"
                    "  pip install 'opencv-contrib-python-headless>=4.9,<5'\n"
                    "(é o que o install_pi.sh faz). Alternativa: colocar o modelo "
                    "face_detection_yunet_2022mar.onnx na pasta models/, que este "
                    "código usa automaticamente quando o OpenCV é antigo."
                )
            return str(cand)

    tried = ", ".join(c.name for c in candidates)
    raise FileNotFoundError(
        f"Modelo de {kind} não encontrado (tentei: {tried}) em {path.parent}. "
        "Rode: python models/download_models.py"
    )


class FaceEngine:
    def __init__(self, cfg):
        m = cfg.models
        det_path = _resolve_model(m.detector, "detector")
        rec_path = _resolve_model(m.recognizer, "recognizer")

        self.detector = cv2.FaceDetectorYN.create(
            det_path,
            "",
            (m.detector_input_width, m.detector_input_height),
            float(m.score_threshold),
            float(m.nms_threshold),
            int(m.top_k),
        )
        self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
        self.detector_path = det_path
        self.recognizer_path = rec_path

        self.detect_width = int(m.get("detect_width", 0))  # 0 = não redimensiona
        self.cosine_threshold = float(cfg.recognition.cosine_threshold)
        self.min_face_size = int(cfg.recognition.min_face_size)

    def detect(self, image):
        """Detecta rostos. Para acelerar, pode reduzir o frame antes de detectar
        e reescalar as coordenadas de volta para a resolução original."""
        h, w = image.shape[:2]
        scale = 1.0
        img = image
        if self.detect_width and w > self.detect_width:
            scale = self.detect_width / float(w)
            img = cv2.resize(image, (int(w * scale), int(h * scale)))

        ih, iw = img.shape[:2]
        self.detector.setInputSize((iw, ih))
        _, faces = self.detector.detect(img)
        if faces is None:
            return []

        out = []
        for face in faces:
            face = face.copy().astype(np.float32)
            if scale != 1.0:
                # x, y, w, h e os 5 landmarks (10 valores) voltam à escala original
                face[:14] = face[:14] / scale
            if min(face[2], face[3]) < self.min_face_size:
                continue
            out.append(face)
        return out

    def embed(self, image, face):
        """Alinha o rosto e extrai o embedding L2-normalizado (para cosseno = produto interno)."""
        aligned = self.recognizer.alignCrop(image, face)
        feat = self.recognizer.feature(aligned)
        vec = np.asarray(feat, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def match(self, vec, gallery_matrix, gallery_ids):
        """Retorna (person_id, score) do rosto mais parecido da galeria."""
        if gallery_matrix is None or len(gallery_ids) == 0:
            return None, 0.0
        sims = gallery_matrix @ vec  # cosseno (tudo normalizado)
        idx = int(np.argmax(sims))
        return gallery_ids[idx], float(sims[idx])

    @staticmethod
    def best_face(faces):
        """Maior rosto (maior área) — usado no cadastro."""
        if not faces:
            return None
        return max(faces, key=lambda f: float(f[2]) * float(f[3]))
