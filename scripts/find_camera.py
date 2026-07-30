"""Descobre a câmera na rede e prova se ela expõe RTSP.

Responde de forma definitiva a pergunta "essa câmera serve para o projeto?",
sem depender do que a ficha técnica diz.

Roda no Windows, no macOS ou no Pi — o que importa é estar na MESMA REDE da
câmera. A varredura usa só a biblioteca padrão do Python; o OpenCV é necessário
apenas para o teste dos caminhos RTSP (`--user`).

Uso:
    python scripts/find_camera.py                          # varre a sub-rede
    python scripts/find_camera.py --ip 192.168.1.77         # sonda um IP só
    python scripts/find_camera.py --ip 192.168.1.77 \
        --user admin --password "MinhaSenha"                # testa caminhos RTSP

O que ele faz:
  1. Descobre a sub-rede do Pi e varre as portas típicas de câmera em cada host.
  2. Classifica cada host encontrado (RTSP? ONVIF? só HTTP? protocolo de nuvem?).
  3. Com usuário e senha, tenta os caminhos RTSP mais comuns (Intelbras/Dahua,
     Hikvision, genéricos) e diz qual funcionou — já no formato do config.yaml.

Sem dependências além do OpenCV que já está instalado.
"""

import argparse
import base64
import hashlib
import ipaddress
import os
import re
import socket
import subprocess
import sys
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Portas que denunciam o tipo de equipamento.
PORTS = {
    554: "RTSP (é isto que o projeto precisa)",
    8554: "RTSP alternativo",
    80: "interface web HTTP",
    443: "interface web HTTPS",
    8000: "ONVIF/HTTP alternativo (comum em Hikvision)",
    8080: "HTTP alternativo",
    8899: "ONVIF (comum em câmeras chinesas genéricas)",
    37777: "porta proprietária Dahua/Intelbras (SDK)",
    34567: "porta proprietária XMeye/Sofia",
    9000: "protocolo de nuvem (Tuya e similares)",
}

# Portas tratadas como "tem RTSP". `--rtsp-port` adiciona a informada aqui,
# senão o script varreria sem olhar justamente a porta que você pediu.
RTSP_PORTS = {554, 8554}

# Caminhos RTSP mais comuns por fabricante.
RTSP_PATHS = [
    ("/cam/realmonitor?channel=1&subtype=1", "Intelbras/Dahua — substream"),
    ("/cam/realmonitor?channel=1&subtype=0", "Intelbras/Dahua — principal"),
    ("/Streaming/Channels/102", "Hikvision — substream"),
    ("/Streaming/Channels/101", "Hikvision — principal"),
    ("/onvif1", "ONVIF genérico 1"),
    ("/onvif2", "ONVIF genérico 2"),
    ("/live/ch00_1", "genérico — substream"),
    ("/live/ch00_0", "genérico — principal"),
    ("/h264_stream", "genérico"),
    ("/11", "genérico (Sofia/XMeye)"),
    ("/stream1", "genérico"),
    ("/video1", "genérico"),
    ("", "raiz (sem caminho)"),
]


def local_subnet() -> str | None:
    """Descobre a sub-rede /24 da interface que tem rota para fora."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))     # não envia pacote; só resolve a rota
        ip = s.getsockname()[0]
        s.close()
        return str(ipaddress.ip_network(f"{ip}/24", strict=False))
    except OSError:
        return None


def port_open(ip: str, port: int, timeout: float = 0.6) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((ip, port)) == 0


MAC_RE = re.compile(r"^([0-9a-fA-F]{1,2}[:-]){5}[0-9a-fA-F]{1,2}$")


def arp_vendor(ip: str) -> str:
    """MAC do host pela tabela ARP (ajuda a identificar o fabricante).

    Cada sistema tem um comando diferente. Duas armadilhas tratadas aqui:

    1. No Windows, `arp -n` é opção inválida e o comando responde com o TEXTO DE
       AJUDA — que contém um MAC de exemplo (`00-aa-00-62-c6-09`). Sem cuidado,
       todo host aparece com esse MAC falso. Por isso escolhemos o comando pela
       plataforma.
    2. Só aceitamos um MAC que esteja na MESMA LINHA que o IP consultado.

    É informativo: devolver vazio não prejudica o diagnóstico.
    """
    if sys.platform.startswith("win"):
        commands = (["arp", "-a", ip],)
    else:
        commands = (["ip", "neigh", "show", ip], ["arp", "-n", ip])

    for cmd in commands:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=4).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            if ip not in line:
                continue
            for token in line.replace(",", " ").split():
                if MAC_RE.match(token):
                    return token.lower()
    return ""


def scan_host(ip: str) -> dict | None:
    open_ports = [p for p in PORTS if port_open(ip, p)]
    if not open_ports:
        return None
    return {"ip": ip, "ports": open_ports, "mac": arp_vendor(ip)}


def scan_subnet(cidr: str, workers: int = 128) -> list[dict]:
    net = ipaddress.ip_network(cidr, strict=False)
    hosts = [str(h) for h in net.hosts()]
    print(f"Varrendo {cidr} ({len(hosts)} endereços)... pode levar ~1 minuto.\n")
    found = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(scan_host, hosts):
            if res:
                found.append(res)
                names = ", ".join(f"{p} ({PORTS[p]})" for p in res["ports"])
                print(f"  {res['ip']:<16} {res['mac']:<18} {names}")
    return found


def classify(host: dict) -> str:
    ports = set(host["ports"])
    if not ports:
        # Diferente de "só HTTP": aqui o host não respondeu nada.
        return ("não respondeu em nenhuma porta conhecida — IP errado, "
                "equipamento desligado, ou firewall/isolamento na rede")
    if ports & RTSP_PORTS:
        return "TEM RTSP — provavelmente serve para o projeto"
    if ports & {8899, 8000, 37777, 34567}:
        return ("sem RTSP, mas com porta de ONVIF/SDK — talvez dê para habilitar "
                "RTSP nas configurações do equipamento")
    if ports <= {80, 443, 8080, 9000}:
        # Não afirmamos que é câmera: roteador, impressora, NAS e TV também
        # respondem só em HTTP. O que importa é que não serve como fonte.
        return ("só HTTP, RTSP fechado — pode ser roteador, impressora ou "
                "câmera de nuvem. Não serve como fonte de vídeo")
    return "tipo indefinido"


# --------------------------------------------------------------------------- #
# Sondagem RTSP nativa (só biblioteca padrão — NÃO precisa de OpenCV).
#
# Falar RTSP direto no socket dá uma informação que o OpenCV esconde: o CÓDIGO
# de resposta do servidor. É ele que separa as duas causas que se confundem:
#     401 Unauthorized -> usuário ou senha errados (o caminho pode estar certo)
#     404 Not Found    -> credencial ACEITA, mas o caminho está errado
#     200 OK           -> caminho e credencial corretos
# O cabeçalho WWW-Authenticate ainda revela o "realm", que em muitas câmeras
# traz o modelo do equipamento.
# --------------------------------------------------------------------------- #

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _parse_auth_header(value: str) -> tuple[str, dict]:
    """Separa o esquema ('Digest'/'Basic') dos parâmetros do WWW-Authenticate."""
    value = value.strip()
    scheme, _, rest = value.partition(" ")
    params = {}
    for match in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([^,\s]+))', rest):
        params[match.group(1).lower()] = match.group(2) or match.group(3) or ""
    return scheme.lower(), params


def _auth_value(scheme: str, params: dict, user: str, password: str,
                uri: str, method: str = "DESCRIBE") -> str:
    """Monta o cabeçalho Authorization para Basic ou Digest."""
    if scheme == "basic":
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        return f"Basic {token}"

    realm = params.get("realm", "")
    nonce = params.get("nonce", "")
    qop = params.get("qop", "")
    ha1 = _md5(f"{user}:{realm}:{password}")
    ha2 = _md5(f"{method}:{uri}")

    fields = [f'username="{user}"', f'realm="{realm}"', f'nonce="{nonce}"',
              f'uri="{uri}"']
    if "auth" in qop.split(","):
        cnonce = uuid.uuid4().hex[:16]
        nc = "00000001"
        response = _md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
        fields += [f'qop=auth', f'nc={nc}', f'cnonce="{cnonce}"']
    else:
        response = _md5(f"{ha1}:{nonce}:{ha2}")
    fields.append(f'response="{response}"')
    if params.get("opaque"):
        fields.append(f'opaque="{params["opaque"]}"')
    return "Digest " + ", ".join(fields)


def _rtsp_request(sock, method: str, uri: str, cseq: int,
                  auth: str = "") -> tuple[int, dict, str]:
    """Envia uma requisição RTSP e devolve (status, cabeçalhos, corpo)."""
    lines = [f"{method} {uri} RTSP/1.0", f"CSeq: {cseq}",
             "User-Agent: find_camera.py"]
    if method == "DESCRIBE":
        lines.append("Accept: application/sdp")
    if auth:
        lines.append(f"Authorization: {auth}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())

    raw = b""
    while b"\r\n\r\n" not in raw:
        chunk = sock.recv(4096)
        if not chunk:
            break
        raw += chunk
    if not raw:
        raise ConnectionError("servidor fechou a conexão sem responder")

    head, _, body = raw.partition(b"\r\n\r\n")
    text = head.decode("utf-8", "replace")
    first, *rest = text.splitlines()
    try:
        status = int(first.split()[1])
    except (IndexError, ValueError):
        raise ConnectionError(f"resposta não parece RTSP: {first[:60]!r}")
    headers = {}
    for line in rest:
        k, _, v = line.partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers, body.decode("utf-8", "replace")


def rtsp_describe(ip: str, path: str, user: str = "", password: str = "",
                  port: int = 554, timeout: float = 5.0) -> dict:
    """DESCRIBE em um caminho. Devolve status, realm e SDP quando houver."""
    uri = f"rtsp://{ip}:{port}{path}"
    result = {"status": None, "realm": "", "scheme": "", "sdp": "", "error": ""}
    try:
        with socket.create_connection((ip, port), timeout) as sock:
            sock.settimeout(timeout)
            status, headers, body = _rtsp_request(sock, "DESCRIBE", uri, 1)

            if status == 401 and (user or password):
                scheme, params = _parse_auth_header(headers.get("www-authenticate", ""))
                result["realm"] = params.get("realm", "")
                result["scheme"] = scheme
                if scheme in ("digest", "basic"):
                    auth = _auth_value(scheme, params, user, password, uri)
                    try:
                        status, headers, body = _rtsp_request(sock, "DESCRIBE", uri, 2, auth)
                    except (ConnectionError, OSError):
                        # vários firmwares encerram a conexão após o 401
                        with socket.create_connection((ip, port), timeout) as s2:
                            s2.settimeout(timeout)
                            status, headers, body = _rtsp_request(s2, "DESCRIBE", uri, 2, auth)
            elif status == 401:
                _, params = _parse_auth_header(headers.get("www-authenticate", ""))
                result["realm"] = params.get("realm", "")

            result["status"] = status
            result["sdp"] = body
    except (OSError, ConnectionError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _sdp_summary(sdp: str) -> str:
    """Extrai codec e resolução do SDP, quando informados."""
    bits = []
    for line in sdp.splitlines():
        low = line.lower()
        if low.startswith("a=rtpmap:"):
            codec = line.split()[-1].split("/")[0]
            if codec.upper() in ("H264", "H265", "HEVC", "MJPEG", "JPEG"):
                bits.append(codec.upper())
        elif "framesize" in low or low.startswith("a=x-dimensions"):
            dims = re.search(r"(\d{3,4})[-x,](\d{3,4})", line)
            if dims:
                bits.append(f"{dims.group(1)}x{dims.group(2)}")
    return " ".join(dict.fromkeys(bits))


def probe_paths(ip: str, user: str, password: str, port: int = 554) -> dict:
    """Sonda todos os caminhos conhecidos. Devolve o resumo classificado."""
    print(f"\nSondando {len(RTSP_PATHS)} caminhos RTSP em {ip}:{port} "
          f"(usuário '{user or '(nenhum)'}')...\n")
    ok, unauthorized, notfound, other = [], [], [], []
    realm = ""
    for path, label in RTSP_PATHS:
        r = rtsp_describe(ip, path, user, password, port)
        realm = realm or r["realm"]
        status, err = r["status"], r["error"]
        if err:
            print(f"  [erro]  {label}: {err}")
            other.append((path, label, err))
        elif status == 200:
            extra = _sdp_summary(r["sdp"])
            print(f"  [OK 200] {label}{(' — ' + extra) if extra else ''}")
            ok.append((path, label, extra))
        elif status == 401:
            print(f"  [401]   {label}: credencial recusada")
            unauthorized.append((path, label))
        elif status in (404, 451, 455):
            print(f"  [{status}]   {label}: caminho inexistente (credencial OK!)")
            notfound.append((path, label))
        else:
            print(f"  [{status}]   {label}")
            other.append((path, label, str(status)))
    return {"ok": ok, "401": unauthorized, "404": notfound, "other": other,
            "realm": realm}


def _build_url(ip: str, port: int, path: str, user: str, password: str) -> str:
    """URL final para o config.yaml, com usuário e senha codificados."""
    creds = ""
    if user:
        u = urllib.parse.quote(user, safe="")
        p = urllib.parse.quote(password, safe="")
        creds = f"{u}:{p}@"
    return f"rtsp://{creds}{ip}:{port}{path}"


def grab_frame(url: str) -> str:
    """Confirma o stream capturando 1 frame. Só roda se o OpenCV existir."""
    try:
        import cv2
    except ImportError:
        return ""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return ""
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            return f"{w}x{h}"
    finally:
        cap.release()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Descobre a câmera e testa RTSP.")
    ap.add_argument("--ip", help="sonda apenas este IP (pula a varredura)")
    ap.add_argument("--subnet", help="sub-rede a varrer, ex.: 192.168.1.0/24")
    ap.add_argument("--user", default="", help="usuário da câmera (ex.: admin)")
    ap.add_argument("--password", default="", help="senha da câmera (sem codificar)")
    ap.add_argument("--rtsp-port", type=int, default=554)
    ap.add_argument("--try-users", action="store_true",
                    help="testa também os usuários de fábrica comuns (admin, root, "
                         "user...). CUIDADO: câmeras podem bloquear após várias "
                         "tentativas falhas")
    args = ap.parse_args()

    # Garante que a porta RTSP informada seja realmente varrida e reconhecida.
    if args.rtsp_port not in PORTS:
        PORTS[args.rtsp_port] = "RTSP (porta informada em --rtsp-port)"
    RTSP_PORTS.add(args.rtsp_port)

    if args.ip:
        hosts = [scan_host(args.ip) or {"ip": args.ip, "ports": [], "mac": arp_vendor(args.ip)}]
        h = hosts[0]
        names = ", ".join(f"{p} ({PORTS[p]})" for p in h["ports"]) or "nenhuma porta conhecida aberta"
        print(f"\n{h['ip']}  {h['mac']}\n  portas: {names}")
    else:
        cidr = args.subnet or local_subnet()
        if not cidr:
            print("Não descobri a sub-rede. Use --subnet 192.168.1.0/24")
            return 1
        hosts = scan_subnet(cidr)
        if not hosts:
            print("Nenhum host com portas de câmera encontrado. Possíveis motivos:")
            print("  - a câmera está em outra sub-rede (rede de visitantes? 5 GHz")
            print("    separada do 2,4 GHz? o Pi no cabo e a câmera no Wi-Fi?);")
            print("  - o roteador tem isolamento de clientes (AP/client isolation),")
            print("    que impede um dispositivo de ver o outro;")
            print("  - a câmera realmente não abre nenhuma porta local (nuvem).")
            print(f"\n  Sua máquina está em {cidr}. Confira no roteador em que")
            print("  sub-rede a câmera aparece e use --subnet para varrer aquela.")
            return 1

    print("\n--- Diagnóstico ---")
    candidates = []
    for h in hosts:
        verdict = classify(h)
        print(f"  {h['ip']:<16} {verdict}")
        if set(h["ports"]) & RTSP_PORTS:
            candidates.append(h)

    if not candidates:
        print("\nNenhum equipamento com a porta RTSP (554) aberta.")
        print("Se a sua câmera é da linha Mibo, isso é o esperado: ela conversa")
        print("apenas com o aplicativo, pela nuvem. Veja as alternativas na")
        print("seção 3.6 do GUIA_RASPBERRY_PI.md.")
        return 1

    if not args.user:
        alvo = candidates[0]
        print("\nHá RTSP aberto. Rode de novo com as credenciais DA CÂMERA "
              "(não as do seu computador):")
        print(f'  python scripts/find_camera.py --ip {alvo["ip"]} '
              f'--user admin --password "SENHA_DA_CAMERA"')
        if 80 in alvo["ports"] or 443 in alvo["ports"]:
            esquema = "https" if 443 in alvo["ports"] and 80 not in alvo["ports"] else "http"
            print(f"\nA porta web também está aberta — abra {esquema}://{alvo['ip']} "
                  "no navegador.")
            print("  É por ali que você confirma o usuário/senha e configura o")
            print("  substream (H.264, 640x480, 10 fps) descrito na seção 3.1 do guia.")
        if 37777 in alvo["ports"]:
            print("\nA porta 37777 (SDK Dahua) aberta indica firmware base Dahua:")
            print("  o caminho /cam/realmonitor?channel=1&subtype=1 deve funcionar.")
        return 0

    users = [args.user]
    if args.try_users:
        # Ordem: o que a pessoa informou primeiro, depois os padrões de fábrica.
        for extra in ("admin", "root", "user", "Admin", "administrator"):
            if extra not in users:
                users.append(extra)
        print("\n⚠  --try-users vai tentar vários usuários. Algumas câmeras")
        print("   BLOQUEIAM o acesso temporariamente após várias falhas de login.")
        print("   Se isso acontecer, espere alguns minutos ou reinicie a câmera.")

    for h in candidates:
        for user in users:
            res = probe_paths(h["ip"], user, args.password, args.rtsp_port)

            if res["ok"]:
                path, label, extra = res["ok"][0]
                url = _build_url(h["ip"], args.rtsp_port, path, user, args.password)
                frame = grab_frame(url)
                print("\n--- Cole no config.yaml ---")
                print("camera:")
                print(f'  rtsp_url: "{url}"')
                print(f"\n(funcionou: {label}"
                      f"{' — ' + extra if extra else ''}"
                      f"{'; frame capturado ' + frame if frame else ''})")
                if len(res["ok"]) > 1:
                    print("\nOutros caminhos também responderam — prefira o substream:")
                    for p, lb, ex in res["ok"][1:]:
                        print(f"  {lb}: {p}")
                if not frame:
                    print("\nO servidor aceitou o DESCRIBE. Para confirmar a imagem,")
                    print("rode este mesmo comando no Pi (que tem OpenCV), ou instale:")
                    print("  pip install opencv-contrib-python-headless")
                return 0

            if res["404"]:
                print(f"\nA credencial do usuário '{user}' FOI ACEITA "
                      "(os caminhos responderam 404, não 401),")
                print("mas nenhum caminho conhecido existe nesta câmera.")
                if res["realm"]:
                    print(f"Identificação do dispositivo (realm): {res['realm']}")
                print("Procure a URL RTSP no manual do modelo e use-a direto no "
                      "config.yaml.")
                return 1

        # Nem 200, nem 404, nem 401: as sondagens não chegaram a falar RTSP.
        if not res["401"]:
            print(f"\nA porta {args.rtsp_port} não respondeu a RTSP em nenhuma "
                  "tentativa.")
            print("Nenhuma credencial foi avaliada — o problema é anterior a isso.")
            erro = next((d for _, _, d in res["other"]), "")
            if erro:
                print(f"Erro observado: {erro}")
            print("\nVerifique:")
            print(f"  - se a porta RTSP é mesmo a {args.rtsp_port} "
                  "(use --rtsp-port para outra);")
            print("  - se o serviço RTSP está habilitado nas configurações da câmera;")
            print("  - se o firewall do seu computador está bloqueando a saída.")
            return 1

        print(f"\nA porta {args.rtsp_port} responde, mas TODAS as tentativas "
              "voltaram 401 Unauthorized.")
        print("Isso significa: o servidor RTSP existe e está saudável — o que ele")
        print("recusou foi o usuário/senha. O caminho ainda não foi avaliado.")
        if res["realm"]:
            print(f"\nIdentificação do dispositivo (realm): {res['realm']}")
        print("\nO que verificar, em ordem:")
        print("  1. A senha do RTSP pode não ser a do aplicativo. Procure uma")
        print("     etiqueta na câmera com 'código de verificação' / 'password'.")
        print("  2. O usuário pode não ser 'admin'. Tente:")
        print(f'       python scripts/find_camera.py --ip {candidates[0]["ip"]} '
              f'--try-users --password "SUA_SENHA"')
        print("  3. Alguns firmwares exigem habilitar o RTSP/ONVIF e definir uma")
        print("     senha específica para ele nas configurações do dispositivo.")
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
