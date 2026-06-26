"""Leitor RTSP resiliente para câmeras IP (Intelbras/Dahua).

A leitura roda em uma thread dedicada que mantém apenas o *último* frame.
Isso evita o acúmulo do buffer interno do RTSP (que causa atraso/lag) e
reconecta automaticamente se o stream cair — importante no Raspberry Pi.
"""

import os
import threading
import time

# Passo 2 — opções de baixa latência para o backend FFmpeg do OpenCV.
# Precisa ser definido ANTES da criação do VideoCapture. `setdefault` deixa o
# usuário sobrescrever via variável de ambiente (ex.: trocar tcp por udp).
#   rtsp_transport=tcp  -> mais estável que o UDP padrão
#   fflags=nobuffer     -> não acumula buffer de entrada
#   flags=low_delay     -> prioriza latência baixa no decode
# (O CAP_PROP_BUFFERSIZE costuma ser ignorado em RTSP; estas opções resolvem.)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import cv2  # noqa: E402  (importado após configurar a env do FFmpeg)


class Camera:
    def __init__(self, rtsp_url: str, reconnect_delay: float = 3.0):
        self.url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # Buffer mínimo: queremos sempre o frame mais recente, não o histórico.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def start(self) -> "Camera":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    time.sleep(self.reconnect_delay)
                    continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        """Retorna uma cópia do último frame, ou None se ainda não houver."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def read_wait(self, timeout: float = 10.0):
        """Bloqueia até haver um frame disponível ou estourar o timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
