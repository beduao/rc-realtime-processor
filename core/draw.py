"""Anotações simples sobre o frame (retângulo + nome)."""

import cv2

KNOWN_COLOR = (0, 200, 0)      # verde
UNKNOWN_COLOR = (0, 0, 255)    # vermelho


def draw_face(frame, face, label: str, score=None, known: bool = True):
    x, y, w, h = (int(v) for v in face[:4])
    color = KNOWN_COLOR if known else UNKNOWN_COLOR
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text = label if score is None else f"{label} {score:.2f}"
    tw = max(80, 9 * len(text))
    cv2.rectangle(frame, (x, max(0, y - 22)), (x + tw, y), color, -1)
    cv2.putText(
        frame, text, (x + 3, max(12, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def crop_face(frame, face, margin: float = 0.25):
    """Recorta o rosto (com margem) para usar como miniatura da amostra."""
    h, w = frame.shape[:2]
    x, y, fw, fh = (int(v) for v in face[:4])
    mx, my = int(fw * margin), int(fh * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
    return frame[y0:y1, x0:x1].copy()
