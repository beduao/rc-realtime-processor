"""Rastreamento de rostos entre frames, para separar CAPTURA de RECONHECIMENTO.

Por que existe
--------------
A detecção (YuNet) custa ~57 ms por frame no Pi 3B, **independente de quantos
rostos** haja — é uma passada única da rede. O reconhecimento (SFace) custa
~285 ms **por rosto**. Reconhecer todo rosto de todo frame é o que trava o Pi
quando várias pessoas passam juntas.

Este módulo agrupa as detecções de uma mesma pessoa ao longo dos frames em uma
"trilha" (track) e guarda apenas os melhores recortes dela. O reconhecimento
roda depois, uma vez por trilha, sobre esses recortes. Ganhos:

- a captura acompanha o vídeo mesmo com muita gente no frame;
- cada pessoa tem VÁRIAS chances de aparecer bem, em vez de um único frame;
- com vários recortes é possível decidir por votação, o que reduz muito o falso
  positivo em comparação com a decisão baseada num frame só.

O rastreador é deliberadamente simples (associação por IoU): pessoas andando num
corredor não exigem filtro de Kalman, e no Pi cada ciclo de CPU conta.
"""

import time

import cv2
import numpy as np

# Layout de uma linha do YuNet (15 valores):
#   0..3   -> x, y, w, h
#   4..13  -> 5 pontos (olho dir, olho esq, nariz, canto boca dir, canto boca esq)
#   14     -> confiança do detector
FACE_BBOX = slice(0, 4)
FACE_LANDMARKS = slice(4, 14)


def iou(a, b) -> float:
    """Interseção sobre união de duas caixas (x, y, w, h)."""
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    return float(inter / (aw * ah + bw * bh - inter))


def sharpness(gray) -> float:
    """Variância do Laplaciano: mede nitidez.

    É o critério que mais importa aqui. Criança correndo gera borrão de
    movimento, e embedding de rosto borrado é a principal fonte de erro de
    identificação — pior que rosto pequeno.
    """
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_with_landmarks(frame, face, margin: float = 0.4):
    """Recorta o rosto com margem e devolve `(recorte, face_em_coordenadas_do_recorte)`.

    A margem existe porque o `alignCrop` do SFace usa os 5 pontos para alinhar e
    precisa de contexto ao redor. Os pontos são transladados para o sistema de
    coordenadas do recorte — sem isso o alinhamento na fase 2 sairia errado, o
    que degradaria o embedding de forma silenciosa.
    """
    h, w = frame.shape[:2]
    x, y, fw, fh = (float(v) for v in face[FACE_BBOX])
    mx, my = fw * margin, fh * margin
    x0, y0 = int(max(0, x - mx)), int(max(0, y - my))
    x1, y1 = int(min(w, x + fw + mx)), int(min(h, y + fh + my))
    if x1 <= x0 or y1 <= y0:
        return None, None

    recorte = frame[y0:y1, x0:x1].copy()
    local = np.asarray(face, dtype=np.float32).copy()
    local[0] -= x0
    local[1] -= y0
    for i in range(4, 14, 2):       # pares (x, y) dos landmarks
        local[i] -= x0
        local[i + 1] -= y0
    return recorte, local


class Track:
    """Uma pessoa acompanhada ao longo dos frames."""

    __slots__ = ("id", "bbox", "last_frame", "frames", "started_at", "ended_at",
                 "crops", "_max_crops")

    def __init__(self, track_id: int, face, frame_idx: int, now: float,
                 max_crops: int):
        self.id = track_id
        self.bbox = [float(v) for v in face[FACE_BBOX]]
        self.last_frame = frame_idx
        self.frames = 1
        self.started_at = now
        self.ended_at = now
        self.crops = []            # lista de (qualidade, recorte, face_local)
        self._max_crops = max_crops

    def atualizar(self, face, frame_idx: int, now: float):
        self.bbox = [float(v) for v in face[FACE_BBOX]]
        self.last_frame = frame_idx
        self.ended_at = now
        self.frames += 1

    def considerar_recorte(self, frame, face):
        """Guarda o recorte se ele for melhor que os já guardados."""
        recorte, local = crop_with_landmarks(frame, face)
        if recorte is None:
            return
        lado = min(float(face[2]), float(face[3]))
        cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        # nitidez pesa mais; tamanho entra como fator limitado para não deixar
        # um rosto enorme e borrado ganhar de um menor e nítido
        qualidade = sharpness(cinza) * min(1.0, lado / 100.0)

        if len(self.crops) < self._max_crops:
            self.crops.append((qualidade, recorte, local))
            self.crops.sort(key=lambda c: c[0], reverse=True)
        elif qualidade > self.crops[-1][0]:
            self.crops[-1] = (qualidade, recorte, local)
            self.crops.sort(key=lambda c: c[0], reverse=True)

    @property
    def duracao(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


class FaceTracker:
    """Associa detecções a trilhas por IoU e encerra as que saíram de cena."""

    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 8,
                 crops_per_track: int = 3, min_track_frames: int = 2):
        self.iou_threshold = float(iou_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self.crops_per_track = int(crops_per_track)
        self.min_track_frames = int(min_track_frames)
        self.ativos: dict[int, Track] = {}
        self._proximo_id = 1
        self.descartadas = 0        # trilhas curtas demais (provável falso positivo)

    def update(self, faces, frame, frame_idx: int, now: float = None) -> list[Track]:
        """Processa um frame. Devolve as trilhas ENCERRADAS neste frame."""
        now = time.time() if now is None else now

        # --- associação gulosa por IoU ---------------------------------------
        # Para cada par (trilha, detecção) com IoU acima do limiar, casa o de
        # maior IoU primeiro. Suficiente para pessoas caminhando; evita a
        # complexidade de um algoritmo de atribuição ótima.
        pares = []
        for tid, tr in self.ativos.items():
            for di, face in enumerate(faces):
                v = iou(tr.bbox, face)
                if v >= self.iou_threshold:
                    pares.append((v, tid, di))
        pares.sort(reverse=True)

        trilhas_usadas, deteccoes_usadas = set(), set()
        for _, tid, di in pares:
            if tid in trilhas_usadas or di in deteccoes_usadas:
                continue
            trilhas_usadas.add(tid)
            deteccoes_usadas.add(di)
            tr = self.ativos[tid]
            tr.atualizar(faces[di], frame_idx, now)
            tr.considerar_recorte(frame, faces[di])

        # --- detecções sem par viram trilhas novas ---------------------------
        for di, face in enumerate(faces):
            if di in deteccoes_usadas:
                continue
            tr = Track(self._proximo_id, face, frame_idx, now, self.crops_per_track)
            tr.considerar_recorte(frame, face)
            self.ativos[self._proximo_id] = tr
            self._proximo_id += 1

        # --- encerra quem não aparece há tempo demais ------------------------
        encerradas = []
        for tid in list(self.ativos):
            tr = self.ativos[tid]
            if frame_idx - tr.last_frame > self.max_missing_frames:
                del self.ativos[tid]
                if tr.frames >= self.min_track_frames and tr.crops:
                    encerradas.append(tr)
                else:
                    self.descartadas += 1
        return encerradas

    def flush(self) -> list[Track]:
        """Encerra todas as trilhas (usar ao parar o worker)."""
        restantes = [t for t in self.ativos.values()
                     if t.frames >= self.min_track_frames and t.crops]
        self.ativos.clear()
        return restantes
