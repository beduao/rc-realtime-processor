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


class FaceEngine:
    def __init__(self, cfg):
        m = cfg.models
        det_path = str(project_path(m.detector))
        rec_path = str(project_path(m.recognizer))

        self.detector = cv2.FaceDetectorYN.create(
            det_path,
            "",
            (m.detector_input_width, m.detector_input_height),
            float(m.score_threshold),
            float(m.nms_threshold),
            int(m.top_k),
        )
        self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")

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
