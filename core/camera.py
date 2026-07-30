"""Leitor RTSP resiliente para câmeras IP (Intelbras/Dahua).

A leitura roda em uma thread dedicada que mantém apenas o *último* frame.
Isso evita o acúmulo do buffer interno do RTSP (que causa atraso/lag) e
reconecta automaticamente se o stream cair — importante no Raspberry Pi.

No Raspberry Pi 3B prefira `read_new()` em vez de `read()`: ele só devolve um
frame quando existe um NOVO (contador de sequência) e por padrão não copia a
imagem. O `read()` copiava o mesmo frame centenas de vezes por segundo, o que
sozinho consumia uma fatia relevante da CPU do Pi.
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


def _parse_source(source):
    """Interpreta a fonte de vídeo e escolhe o backend certo do OpenCV.

    Aceita:
      "rtsp://..." / caminho de arquivo  -> FFmpeg (câmera IP, vídeo de teste)
      0, "0", "1"                        -> índice de webcam USB     -> V4L2
      "/dev/video0"                      -> dispositivo V4L2 explícito

    Isso permite usar uma webcam USB ou a câmera CSI do Pi quando a câmera IP
    não expõe RTSP (caso das câmeras de nuvem, linha Mibo).
    """
    if isinstance(source, int):
        return source, cv2.CAP_V4L2
    text = str(source).strip()
    if text.isdigit():
        return int(text), cv2.CAP_V4L2
    if text.startswith("/dev/video"):
        return text, cv2.CAP_V4L2
    return text, cv2.CAP_FFMPEG


class Camera:
    def __init__(self, rtsp_url, reconnect_delay: float = 3.0,
                 width: int = 0, height: int = 0, fps: int = 0, mjpeg: bool = True):
        self.url = rtsp_url
        self.source, self.backend = _parse_source(rtsp_url)
        self.is_local_device = self.backend == cv2.CAP_V4L2
        # Só usados quando a fonte é uma webcam USB / câmera CSI:
        self.width, self.height, self.fps, self.mjpeg = width, height, fps, mjpeg
        self.reconnect_delay = reconnect_delay
        self._cap = None
        self._frame = None
        self._seq = 0                 # incrementa a cada frame novo
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        # estatísticas simples (usadas pelo scripts/check_pi.py)
        self.frames_received = 0
        self.reconnects = 0
        self.last_frame_at = 0.0

    # ---- ciclo de vida --------------------------------------------------- #
    def _open(self):
        cap = cv2.VideoCapture(self.source, self.backend)
        # Buffer mínimo: queremos sempre o frame mais recente, não o histórico.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if self.is_local_device:
            # MJPEG ANTES da resolução: em USB 2.0 o formato cru (YUYV) satura o
            # barramento e limita a 640x480@10fps, além de custar CPU no Pi.
            if self.mjpeg:
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
            if self.width and self.height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
            if self.fps:
                cap.set(cv2.CAP_PROP_FPS, int(self.fps))
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
                    self.reconnects += 1
                    time.sleep(self.reconnect_delay)
                    continue
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                self.reconnects += 1
                time.sleep(self.reconnect_delay)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
                self.frames_received += 1
                self.last_frame_at = time.time()

    # ---- leitura --------------------------------------------------------- #
    def read(self):
        """Retorna uma CÓPIA do último frame, ou None se ainda não houver.

        Mantido por compatibilidade (API de cadastro). Em loops contínuos
        (worker) use `read_new()`.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def read_new(self, last_seq: int = 0, copy: bool = False):
        """Retorna `(frame, seq)` só quando há frame mais novo que `last_seq`.

        Sem novidade devolve `(None, last_seq)`.

        `copy=False` (padrão) entrega a referência do array sem memcpy — é
        seguro porque a thread de captura sempre cria um array novo a cada
        leitura, então o frame já entregue continua íntegro. Peça `copy=True`
        apenas se for desenhar diretamente sobre o frame recebido.
        """
        with self._lock:
            if self._frame is None or self._seq <= last_seq:
                return None, last_seq
            frame = self._frame
            seq = self._seq
        return (frame.copy() if copy else frame), seq

    def read_wait(self, timeout: float = 10.0):
        """Bloqueia até haver um frame disponível ou estourar o timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def stats(self) -> dict:
        return {
            "frames_received": self.frames_received,
            "reconnects": self.reconnects,
            "seconds_since_last_frame": (
                round(time.time() - self.last_frame_at, 2) if self.last_frame_at else None
            ),
        }

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


def camera_from_config(cfg) -> "Camera":
    """Cria a Camera a partir do config.yaml (usado por worker, API e scripts).

    Além de `rtsp_url`, aceita opcionalmente (só valem para webcam USB/CSI):
        camera:
          width: 640
          height: 480
          fps: 15
          mjpeg: true
    """
    cam_cfg = cfg.camera
    return Camera(
        cam_cfg.rtsp_url,
        cam_cfg.get("reconnect_delay_seconds", 3.0),
        width=int(cam_cfg.get("width", 0) or 0),
        height=int(cam_cfg.get("height", 0) or 0),
        fps=int(cam_cfg.get("fps", 0) or 0),
        mjpeg=bool(cam_cfg.get("mjpeg", True)),
    )
