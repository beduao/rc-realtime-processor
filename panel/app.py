"""Painel web (Streamlit) — cliente HTTP fino da API.

Na Fase 1 roda no seu computador apontando para http://localhost:8000.
Na Fase 2 roda no PC; basta trocar api.base_url no config.yaml para o IP do Pi.
Não usa OpenCV nem toca a câmera: tudo passa pela API.

Rodar com:  streamlit run panel/app.py
"""

import os
import sys
import time

# garante que o pacote `core` (na raiz do projeto) seja importável
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
import streamlit as st  # noqa: E402

from core.config import load_config  # noqa: E402

cfg = load_config()
API = cfg.api.base_url.rstrip("/")

# Sessão única com o token, quando configurado. Se a API do Pi exigir token e
# o config DESTE computador não tiver, as chamadas voltam 401 — os dois
# arquivos precisam do mesmo valor em api.token.
S = requests.Session()
_token = str((cfg.get("api") or {}).get("token") or "").strip()
if not _token:
    # Aceita a forma por consumidor, usando o token chamado "painel" (ou o
    # primeiro, se houver outro nome).
    _tokens = (cfg.get("api") or {}).get("tokens") or {}
    _token = str(_tokens.get("painel")
                 or next(iter(_tokens.values()), "") or "").strip()
if _token:
    S.headers["X-API-Token"] = _token

st.set_page_config(page_title="Reconhecimento Facial", page_icon="📷", layout="wide")


def api_up() -> bool:
    """Saúde da API, com cache curto.

    Streamlit reexecuta o script inteiro a cada interação, então sem cache
    esta chamada virava uma ida à rede ANTES de qualquer coisa renderizar —
    em todo clique, em toda tecla Enter. 10 segundos é curto o bastante para
    perceber a API caindo e longo o bastante para não pesar na navegação.
    """
    agora = time.time()
    st.session_state.setdefault("_saude", (0.0, False))
    quando, valor = st.session_state["_saude"]
    if agora - quando < 10:
        return valor
    try:
        valor = S.get(f"{API}/health", timeout=3).ok
    except requests.RequestException:
        valor = False
    st.session_state["_saude"] = (agora, valor)
    return valor


# Cache de imagens do próprio processo. Fotos de snapshot são IMUTÁVEIS: um
# caminho sempre devolve os mesmos bytes, então rebaixá-las é desperdício puro.
#
# Isso virou necessário quando as imagens passaram a ser buscadas pelo painel
# (para enviar o token) em vez de pelo navegador. O navegador cacheava; o
# `requests` não. Em Reconhecimentos, com 60 fotos, cada clique custava 60
# downloads — e era a maior parte da lentidão percebida.
#
# O preview ao vivo NÃO entra aqui: ele muda a cada instante e a URL carrega um
# cache-buster, então cachear só encheria a memória.
_CACHE_MAX = 400


def _cache_imagem():
    return st.session_state.setdefault("_img_cache", {})


def imagem(caminho: str, **kwargs):
    """Baixa a imagem COM o token e entrega bytes para o `st.image`.

    Passar URL para o `st.image` faz o Streamlit devolvê-la sem modificação
    (verificado na fonte: `if isinstance(image, str)` e a URL é retornada
    direto), então quem busca o arquivo é o NAVEGADOR — sem o cabeçalho
    X-API-Token que está nesta Session. Resultado: bastava ligar o token para
    todas as imagens do painel quebrarem.

    Baixando aqui, três coisas melhoram de uma vez:
      1. o token é enviado, então o painel funciona com a API protegida;
      2. com HTTPS, só este processo precisa confiar no certificado — nada a
         instalar no navegador nem no sistema operacional;
      3. o navegador deixa de precisar alcançar o Pi, então o painel passa a
         funcionar em qualquer topologia de rede (VPN, sub-rede diferente).
    """
    url = caminho if caminho.startswith("http") else f"{API}{caminho}"

    # Preview ao vivo muda a cada instante e traz cache-buster na URL: cachear
    # só encheria a memória. Snapshot é imutável e vale guardar.
    cacheavel = "/enroll/preview" not in url
    cache = _cache_imagem()
    if cacheavel and url in cache:
        st.image(cache[url], **kwargs)
        return

    try:
        r = S.get(url, timeout=10)
    except requests.RequestException as exc:
        st.caption(f"⚠ imagem indisponível ({type(exc).__name__})")
        return
    if not r.ok:
        # Mostra o `detail` que a API mandou, em vez de deduzir a causa pelo
        # código HTTP. A primeira versão fazia essa dedução e errava feio: o
        # mesmo 503 é usado pelo middleware ("sem token configurado") e pelo
        # /enroll/preview ("ainda sem imagem, o worker está rodando?"). O
        # painel anunciava problema de token quando o problema era câmera,
        # mandando procurar no lugar errado.
        detalhe = ""
        try:
            detalhe = str(r.json().get("detail") or "")
        except ValueError:
            detalhe = (r.text or "").strip()[:300]
        if not detalhe:
            detalhe = {401: "token inválido ou ausente neste computador",
                       404: "foto não está mais no disco (retenção)"}.get(
                           r.status_code, f"HTTP {r.status_code}")
        st.caption(f"⚠ {detalhe}")
        return

    # Só sucesso entra no cache: guardar um 404 ou 503 faria o painel repetir
    # o erro depois de resolvido.
    if cacheavel:
        if len(cache) >= _CACHE_MAX:
            cache.clear()          # simples de propósito; são bytes, não estado
        cache[url] = r.content
    st.image(r.content, **kwargs)


st.sidebar.title("📷 Reconhecimento Facial")
st.sidebar.caption(f"API: {API}")
if api_up():
    st.sidebar.success("API conectada")
else:
    st.sidebar.error("API offline — suba `uvicorn api:app` e o `worker.py`.")

page = st.sidebar.radio(
    "Menu", ["Chamada", "Cadastrar", "Reconhecimentos", "Pessoas", "Ao vivo"])


# --------------------------------------------------------------------------- #
if page == "Chamada":
    import datetime

    st.header("Chamada")
    ss = st.session_state
    ss.setdefault("autor", "")

    # vertical_alignment="bottom": sem isso o botão sobe até o topo da coluna,
    # porque ele não tem rótulo e os campos ao lado têm.
    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    dia = c1.date_input("Dia", value=datetime.date.today(), format="DD/MM/YYYY")
    dia_iso = dia.strftime("%Y-%m-%d")
    if c2.button("Atualizar", width="stretch"):
        st.rerun()
    ss.autor = c3.text_input(
        "Quem está conferindo", value=ss.autor,
        placeholder="seu nome",
        help="Fica registrado na correção e no fechamento. Não é autenticação — "
             "serve para saber quem conferiu.")

    try:
        r = S.get(f"{API}/attendance", params={"dia": dia_iso}, timeout=20)
    except requests.RequestException as exc:
        st.error(f"Falha ao falar com a API: {exc}")
        st.stop()
    if r.status_code == 404:
        st.error("A API do Pi não tem o endpoint de chamada — versão antiga.\n\n"
                 "No Pi: `git pull && sudo systemctl restart facial-api`")
        st.stop()
    if not r.ok:
        st.error(f"A API respondeu {r.status_code}: {r.text}")
        st.stop()

    d = r.json()
    presentes, ausentes = d["presentes"], d["ausentes"]
    corr = d["correcoes"]

    # --- estado da chamada -------------------------------------------------- #
    m1, m2, m3, m4 = st.columns(4)
    # "de N" como valor, não como delta: delta desenha seta e cor de variação,
    # o que sugeria alta/baixa onde não há comparação nenhuma.
    m1.metric("Presentes", f"{d['total_presentes']} de {d['total_cadastrados']}")
    m2.metric("Ausentes", len(ausentes))
    m3.metric("Correções", corr["total"])
    m4.metric("Situação", "Conferida" if d["conferida"] else "Em aberto")

    if not d["completo"]:
        st.error(f"**{d['trilhas_pendentes']} trilha(s) aguardando reconhecimento.** "
                 "A chamada está incompleta — rode o lote antes de conferir:\n\n"
                 "`python scripts/recognize_batch.py`")
    if d.get("sem_inep"):
        st.warning(f"**{d['sem_inep']} aluno(s) sem ID INEP.** O sistema de gestão "
                   "da escola não consegue casar esses registros. Preencha em "
                   "**Pessoas**.")
    if d["conferida"]:
        f = d["fechamento"]
        # ISO -> "26/08 às 11:29"
        quando = f"{f['em'][8:10]}/{f['em'][5:7]} às {f['em'][11:16]}"
        st.success(f"Conferida em {quando}" +
                   (f" por {f['autor']}" if f["autor"] else "") +
                   ". Para alterar, reabra abaixo.")
    else:
        st.warning("Chamada **em aberto**. Quem não foi identificado pode ter "
                   "passado sem ser detectado — confira antes de tratar como falta.")

    st.divider()

    # --- conferência -------------------------------------------------------- #
    ICONES = {"automatico": "✅ detectado",
              "manual_presente": "✏️ marcado presente",
              "manual_ausente": "✏️ marcado ausente",
              "nao_identificado": "❔ não identificado"}

    def _linha(pessoa, marcado_default):
        """Uma pessoa na conferência. Devolve o valor do checkbox.

        A FOTO da melhor detecção fica ao lado, porque sem ela a conferência é
        feita no escuro: marcar "presente" vira confirmação de um nome, não de
        um rosto. E é dessa confirmação que saem os rótulos da calibração e o
        denominador da medição de recall — conferir sem olhar contamina as duas.

        O nome é o RÓTULO da caixa, não um texto ao lado. Com colunas, a
        largura sobrando na coluna da caixa virava um vão vazio — e a área de
        clique ficava restrita ao quadradinho. Assim o nome inteiro é clicável.
        """
        col_foto, col_dados = st.columns([1, 9], vertical_alignment="center")

        with col_foto:
            if pessoa.get("foto_url"):
                imagem(pessoa["foto_url"], width=64)
            else:
                # Ausência de foto é informação: significa que o sistema não
                # viu essa pessoa hoje. Marcar presente aqui é justamente o
                # falso negativo que a medição de recall procura.
                st.markdown(
                    "<div style='width:64px;height:64px;border:1px dashed #555;"
                    "border-radius:4px;display:flex;align-items:center;"
                    "justify-content:center;color:#777;font-size:0.7em;"
                    "text-align:center'>sem<br>registro</div>",
                    unsafe_allow_html=True)

        with col_dados:
            valor = st.checkbox(f"**{pessoa['nome']}**", value=marcado_default,
                                key=f"pres-{dia_iso}-{pessoa['person_id']}",
                                disabled=d["conferida"])
            detalhe = [ICONES[pessoa["origem"]]]
            if not pessoa.get("inep_id"):
                detalhe.append("⚠ sem ID INEP")
            if pessoa.get("primeira_vez"):
                detalhe.append(pessoa["primeira_vez"][11:16])
            if pessoa.get("melhor_score") is not None:
                detalhe.append(f"score {pessoa['melhor_score']:.2f}")
            if pessoa.get("correcao") and pessoa["correcao"]["motivo"]:
                detalhe.append(f"motivo: {pessoa['correcao']['motivo']}")
            # recuo alinha o detalhe com o texto do rótulo, não com a caixa
            st.markdown(
                f"<div style='color:gray;font-size:0.85em;"
                f"margin:-0.7rem 0 0.6rem 2rem'>"
                f"{' · '.join(detalhe)}</div>", unsafe_allow_html=True)
        return valor

    with st.form(f"conferencia-{dia_iso}"):
        st.caption("Marque quem esteve presente. Só as diferenças em relação ao "
                   "que o sistema detectou são gravadas como correção.")

        if presentes:
            st.subheader(f"Presentes ({len(presentes)})")
            marcados = {p["person_id"]: _linha(p, True) for p in presentes}
        else:
            marcados = {}
            st.info("Ninguém registrado como presente neste dia.")

        if ausentes:
            st.subheader(f"Ausentes ({len(ausentes)})")
            marcados.update({p["person_id"]: _linha(p, False) for p in ausentes})

        motivo = st.text_input(
            "Motivo das correções (opcional)",
            placeholder="ex.: chegou antes da câmera ligar",
            disabled=d["conferida"])
        enviar = st.form_submit_button("Salvar correções", type="primary",
                                       disabled=d["conferida"])

    if enviar:
        detectado = {p["person_id"]: p["detectado_pelo_sistema"]
                     for p in presentes + ausentes}
        criadas = removidas = 0
        erros = []
        for pid, marcado in marcados.items():
            # Regra: correção existe só quando a pessoa discorda da máquina.
            if marcado != detectado[pid]:
                resp = S.post(f"{API}/attendance/override", timeout=15, json={
                    "dia": dia_iso, "person_id": pid, "presente": marcado,
                    "motivo": motivo, "autor": ss.autor})
                criadas += 1 if resp.ok else 0
                if not resp.ok:
                    erros.append(f"{pid}: {resp.text}")
            else:
                resp = S.delete(f"{API}/attendance/override", timeout=15,
                                params={"dia": dia_iso, "person_id": pid})
                if resp.ok:
                    removidas += resp.json().get("removidas", 0)
        if erros:
            st.error("Algumas correções falharam:\n\n" + "\n".join(erros))
        else:
            st.success(f"{criadas} correção(ões) gravada(s), "
                       f"{removidas} desfeita(s).")
            st.rerun()

    st.divider()

    # --- fechamento --------------------------------------------------------- #
    if d["conferida"]:
        st.caption("Reabrir permite corrigir de novo. O fechamento anterior é "
                   "substituído.")
        if st.button("Reabrir chamada"):
            resp = S.delete(f"{API}/attendance/close", params={"dia": dia_iso},
                            timeout=15)
            if resp.ok:
                st.rerun()
            else:
                st.error(resp.text)
    else:
        st.caption("Fechar registra que uma pessoa conferiu esta chamada. Só "
                   "depois disso ela deveria alimentar falta em outro sistema.")
        if st.button("Fechar chamada", type="primary",
                     disabled=not d["completo"]):
            resp = S.post(f"{API}/attendance/close", timeout=20,
                          json={"dia": dia_iso, "autor": ss.autor})
            if resp.ok:
                st.rerun()
            else:
                st.error(resp.json().get("detail", resp.text))

    # --- o que as correções dizem sobre o sistema --------------------------- #
    if corr["total"]:
        st.divider()
        st.subheader("O que isso diz sobre o reconhecimento")
        total_real = len(presentes)
        perdidos = corr["marcados_presentes"]
        errados = corr["marcados_ausentes"]
        if total_real:
            st.write(f"Neste dia o sistema **deixou passar {perdidos}** de "
                     f"{total_real} presentes "
                     f"({100 * perdidos / total_real:.0f}%) e "
                     f"**identificou {errados} por engano**.")
        st.caption("Cada correção sua é um erro medido do reconhecimento. "
                   "Acumulando alguns dias, esses números dizem se vale ajustar "
                   "o limiar, o enquadramento ou o cadastro.")


# --------------------------------------------------------------------------- #
elif page == "Cadastrar":
    st.header("Cadastrar pessoa (ao vivo)")
    ss = st.session_state
    ss.setdefault("enroll_session", None)
    ss.setdefault("enroll_name", "")
    ss.setdefault("enroll_target", 5)
    ss.setdefault("samples", [])

    # --- sem sessão: pedir nome e iniciar ---
    if ss.enroll_session is None:
        st.write("Informe o nome e inicie o cadastro para abrir o preview da câmera.")
        cn, ci = st.columns([2, 1])
        name = cn.text_input("Nome")
        inep = ci.text_input(
            "ID INEP", placeholder="12 dígitos",
            help="Identificação única do aluno no Censo Escolar. É por ela que o "
                 "sistema de gestão da escola casa os registros — casar por nome "
                 "é frágil. Pode ficar em branco e ser preenchida depois.")
        if st.button("Iniciar cadastro", type="primary", disabled=not name.strip()):
            try:
                r = S.post(f"{API}/enroll/start", timeout=30,
                           json={"name": name.strip(), "inep_id": inep.strip()})
            except requests.RequestException as exc:
                st.error(f"Falha ao falar com a API: {exc}")
            else:
                if r.ok:
                    data = r.json()
                    ss.enroll_session = data["session_id"]
                    ss.enroll_name = data["name"]
                    ss.enroll_target = data.get("target", 5)
                    ss.samples = []
                    st.rerun()
                else:
                    st.error(r.json().get("detail", r.text))

    # --- sessão ativa: preview + captura individual ---
    else:
        target = ss.enroll_target
        n = len(ss.samples)
        st.info(f"Cadastrando **{ss.enroll_name}** — {n}/{target} amostras capturadas.")

        col_preview, col_samples = st.columns([2, 1])

        with col_preview:
            @st.fragment(run_every="0.8s")
            def _preview():
                imagem(f"/enroll/preview?t={time.time()}", width="stretch",
                       caption="Preview ao vivo — caixa verde = rosto detectado")
            _preview()

            if st.button(f"📸 Capturar amostra {n + 1}", type="primary",
                         disabled=n >= target, width="stretch"):
                try:
                    r = S.post(f"{API}/enroll/capture",
                                      json={"session_id": ss.enroll_session}, timeout=30)
                    data = r.json()
                except requests.RequestException as exc:
                    st.error(f"Falha ao capturar: {exc}")
                else:
                    if r.ok and data.get("ok"):
                        ss.samples = data["samples"]
                        st.rerun()
                    else:
                        # `message` é a recusa explicada da API (sem rosto, sem
                        # imagem). `detail` é o corpo de uma HTTPException, e
                        # HTTP 500 sem corpo útil significa exceção no servidor.
                        # A versão anterior caía num "Não foi possível capturar"
                        # genérico nos dois últimos casos, escondendo a única
                        # informação que permitiria diagnosticar.
                        motivo = (data.get("message") or data.get("detail")
                                  or "").strip()
                        if motivo:
                            st.warning(motivo)
                        else:
                            st.error(
                                f"A API respondeu {r.status_code} sem explicar. "
                                "Provável exceção no servidor — o traceback "
                                "está no terminal do uvicorn.")
                            st.caption(f"corpo: {r.text[:400]}")

        with col_samples:
            st.caption("Amostras")
            if not ss.samples:
                st.write("_nenhuma ainda_")
            for s in ss.samples:
                imagem(s["snapshot_url"], width=110)
                if st.button("🗑 remover", key=f"rm-{s['index']}"):
                    r = S.post(f"{API}/enroll/sample/delete",
                                      json={"session_id": ss.enroll_session, "index": s["index"]},
                                      timeout=30)
                    if r.ok:
                        ss.samples = r.json()["samples"]
                    st.rerun()

        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("✅ Concluir cadastro", type="primary", disabled=n < 1, width="stretch"):
            r = S.post(f"{API}/enroll/finish",
                              json={"session_id": ss.enroll_session}, timeout=30)
            if r.ok:
                st.success(f"{ss.enroll_name} cadastrado(a) com {r.json()['captured']} amostras.")
                ss.enroll_session = None
                ss.samples = []
            else:
                st.error(r.json().get("detail", r.text))
        if c2.button("Cancelar", width="stretch"):
            S.post(f"{API}/enroll/cancel", json={"session_id": ss.enroll_session}, timeout=30)
            ss.enroll_session = None
            ss.samples = []
            st.rerun()


# --------------------------------------------------------------------------- #
elif page == "Reconhecimentos":
    import datetime

    st.header("Histórico de reconhecimentos")

    try:
        pessoas_r = S.get(f"{API}/people", timeout=10)
        pessoas_lista = pessoas_r.json() if pessoas_r.ok else []
    except requests.RequestException:
        pessoas_lista = []

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")

    opcoes = {"Todas as pessoas": 0}
    opcoes.update({p["name"]: p["id"] for p in pessoas_lista})
    escolha = c1.selectbox("Pessoa", list(opcoes), index=0)
    pid_filtro = opcoes[escolha]

    usar_dia = c2.checkbox("Filtrar dia", value=False)
    dia_r = c3.date_input("Dia", value=datetime.date.today(),
                          format="DD/MM/YYYY", disabled=not usar_dia,
                          label_visibility="collapsed" if not usar_dia else "visible")
    # 24 de padrão: a primeira renderização baixa uma imagem por detecção, e
    # 60 custavam segundos antes de qualquer coisa aparecer. Depois do cache
    # ficam instantâneas, mas a primeira impressão é a que conta.
    limit = c4.number_input("Quantidade", 10, 500, 24, step=10)

    params = {"limit": int(limit), "person_id": pid_filtro}
    if usar_dia:
        params["dia"] = dia_r.strftime("%Y-%m-%d")

    try:
        r = S.get(f"{API}/detections", params=params, timeout=15)
    except requests.RequestException as exc:
        st.error(f"Falha ao buscar detecções: {exc}")
        st.stop()
    if r.status_code == 404:
        st.error("A API não tem o endpoint /detections — versão antiga.\n\n"
                 "No Pi: `git pull && sudo systemctl restart facial-api`")
        st.stop()
    if not r.ok:
        st.error(f"A API respondeu {r.status_code}: {r.text[:300]}")
        st.stop()

    deteccoes = r.json()["deteccoes"]

    if not deteccoes:
        modo = pendentes = None
        try:
            h = S.get(f"{API}/health", timeout=5).json()
            modo, pendentes = h.get("worker_mode"), h.get("tracks_pending")
        except (requests.RequestException, ValueError):
            pass
        if pendentes:
            st.info(f"Nada aqui, mas há **{pendentes} trilha(s)** aguardando "
                    "reconhecimento. Rode o lote:\n\n"
                    "```\npython scripts/recognize_batch.py\n```")
        elif pid_filtro:
            st.info(f"Nenhuma detecção de **{escolha}** no período.")
        elif modo is None:
            st.warning("Nada encontrado, e não consegui falar com a API para "
                       "saber o modo do worker. Ele está rodando?")
        else:
            st.info("Nenhuma detecção ainda. Deixe o worker rodando e passe "
                    "na frente da câmera.")
    else:
        errados = sum(1 for d in deteccoes if d["rotulo"] == "errado")
        resumo = f"{len(deteccoes)} detecção(ões)"
        if errados:
            resumo += f" · **{errados}** marcada(s) como erro"
        st.caption(resumo + " · marcar aqui **não** altera a chamada")

        cols = st.columns(4)
        for i, d in enumerate(deteccoes):
            with cols[i % 4]:
                if d["foto_url"]:
                    imagem(d["foto_url"], width="stretch")
                else:
                    st.caption("_sem foto_")

                quando = d["quando"][11:19]
                tag = "✅" if d["is_known"] else "❓"
                origem = "lote" if d["fonte"] == "trilha" else "ao vivo"
                st.caption(f"{tag} **{d['nome']}** · {d['score']:.2f}\n\n"
                           f"{quando} · {origem}")

                if d["rotulo"] == "errado":
                    st.error("marcada como erro", icon="🚫")
                    if st.button("desfazer", key=f"undo-{d['chave']}",
                                 width="stretch"):
                        S.delete(f"{API}/detections/label",
                                 params={"fonte": d["fonte"],
                                         "detection_id": d["id"]}, timeout=10)
                        st.rerun()
                elif d["is_known"]:
                    if st.button("🚫 não é essa pessoa", key=f"bad-{d['chave']}",
                                 width="stretch"):
                        resp = S.post(f"{API}/detections/label",
                                      json={"fonte": d["fonte"],
                                            "detection_id": d["id"],
                                            "rotulo": "errado",
                                            "autor": st.session_state.get("autor", "")},
                                      timeout=10)
                        if not resp.ok:
                            st.error(resp.text[:200])
                        st.rerun()

        st.divider()
        st.caption(
            "Marcar erros aqui alimenta o `calibrate_threshold.py`, que usa "
            "esses rótulos para calcular o limiar. Quanto mais erros reais "
            "marcados, melhor a sugestão — e um erro marcado vale mais que "
            "dez acertos, porque define o teto do limiar.")


# --------------------------------------------------------------------------- #
elif page == "Pessoas":
    st.header("Pessoas cadastradas")

    # Acima de qual similaridade duas amostras contam como redundantes. 0.95 é
    # alto de propósito: abaixo disso a variação entre elas ainda ajuda.
    LIMITE_REDUNDANCIA = 0.95

    try:
        rp = S.get(f"{API}/people", timeout=10)
        rp.raise_for_status()
        people = rp.json()
    except requests.RequestException as exc:
        st.error(f"Falha ao buscar pessoas: {exc}")
        st.stop()

    if not people:
        st.info("Ninguém cadastrado ainda. Vá em **Cadastrar**.")
        st.stop()

    # O worker publica o frame que a API usa no preview. Sem ele rodando, não
    # há como capturar amostra nova — melhor avisar do que mostrar imagem quebrada.
    try:
        saude = S.get(f"{API}/health", timeout=5).json()
    except requests.RequestException:
        saude = {}
    camera_viva = bool(saude.get("worker_mode"))

    st.caption(f"{len(people)} pessoa(s) cadastrada(s). "
               "Selecione uma para ver e gerenciar as amostras.")
    def _rotulo(p):
        inep = p.get("inep_id") or "sem ID INEP"
        return f"{p['name']}  —  {inep}  ({p['embeddings']} amostra(s))"
    rotulos = {_rotulo(p): p for p in people}
    escolhido = st.selectbox("Pessoa", list(rotulos))
    p = rotulos[escolhido]

    try:
        r = S.get(f"{API}/people/{p['id']}/samples", timeout=15)
    except requests.RequestException as exc:
        st.error(f"Falha ao buscar amostras: {exc}")
        st.stop()

    # Verificar o status é essencial aqui: sem isso um 404 (API numa versão
    # antiga, sem este endpoint) viraria "nenhuma amostra" e esconderia a causa.
    if r.status_code == 404 and "Not Found" in r.text:
        st.error(
            "A API do Pi não tem o endpoint de amostras — ela está rodando uma "
            "versão anterior do código.\n\n"
            "No Raspberry Pi:\n"
            "```\ngit pull\nsudo systemctl restart facial-api facial-worker\n```")
        st.caption(f"Conferir direto: {API}/people/{p['id']}/samples")
        st.stop()
    if not r.ok:
        st.error(f"A API respondeu {r.status_code}: "
                 f"{r.json().get('detail', r.text) if r.text else '(vazio)'}")
        st.stop()

    dados = r.json()
    amostras = dados.get("samples", [])
    if dados.get("sem_foto"):
        st.warning(
            f"**{dados['sem_foto']} amostra(s) sem foto guardada.** Elas continuam "
            "valendo para o reconhecimento, mas não é possível revisá-las aqui.\n\n"
            "Isso acontece quando o cadastro foi feito **antes** de a API do Pi ser "
            "atualizada. Se este cadastro é recente e a API já está atualizada, "
            "verifique no Pi se a gravação da foto falhou:\n\n"
            "```\nsudo journalctl -u facial-api | grep 'não movi o recorte'\n```\n"
            "Para ter as fotos: use **Reforçar cadastro** abaixo para criar "
            "amostras novas e depois exclua as antigas, ou remova a pessoa e "
            "cadastre de novo.")

    # --- grade de amostras -------------------------------------------------- #
    st.subheader("Amostras")
    if not amostras:
        # Não deveria acontecer: /people contou embeddings para esta pessoa.
        # Se cair aqui, banco e API estão vendo arquivos diferentes.
        st.error(
            f"A API não retornou amostras, mas a lista de pessoas diz que "
            f"**{p['embeddings']}** existem. Isso indica que a API e o painel "
            "estão olhando bancos diferentes, ou que a API é de uma versão antiga.")
        st.caption(f"Conferir direto no navegador: {API}/people/{p['id']}/samples")
        st.stop()

    colunas = st.columns(4)
    for i, a in enumerate(amostras):
        col = colunas[i % 4]
        with col:
            if a["snapshot_url"]:
                imagem(a["snapshot_url"], width="stretch")
            else:
                st.info("sem foto")

            legenda = [f"#{a['id']}"]
            if a.get("quality") is not None:
                legenda.append(f"nitidez {a['quality']:.0f}")
            if a.get("created_at"):
                legenda.append(time.strftime("%d/%m %H:%M",
                                             time.localtime(a["created_at"])))
            st.caption(" · ".join(legenda))

            sim = a.get("similaridade_maxima")
            if sim is not None and sim >= LIMITE_REDUNDANCIA:
                st.warning(f"redundante — {sim:.2f} com a #{a['parecida_com']}")
            elif sim is not None:
                st.caption(f"similaridade máx. {sim:.2f}")

            if st.button("Excluir", key=f"delemb-{a['id']}"):
                r = S.delete(f"{API}/embeddings/{a['id']}", timeout=15)
                if r.ok:
                    st.rerun()
                else:
                    st.error(r.json().get("detail", r.text))

    redundantes = [a for a in amostras
                   if (a.get("similaridade_maxima") or 0) >= LIMITE_REDUNDANCIA]
    if redundantes:
        st.info(f"{len(redundantes)} amostra(s) quase idênticas a outras. Excluir "
                "uma delas não piora o reconhecimento: o que ajuda é variação de "
                "ângulo e iluminação, não quantidade.")
    elif len(amostras) < 3:
        st.info("Poucas amostras. Mais amostras variadas elevam o score dos "
                "acertos, o que permite subir o limiar sem deixar de reconhecer "
                "a pessoa.")

    st.divider()

    # --- reforçar cadastro -------------------------------------------------- #
    st.subheader("Reforçar cadastro")
    if not camera_viva:
        st.warning("O worker não está publicando imagem, então não dá para "
                   "capturar agora. Verifique: sudo systemctl status facial-worker")
    else:
        ca, cb = st.columns([2, 3])
        with ca:
            st.caption("A pessoa deve estar em frente à câmera. Varie o ângulo "
                       "em relação às amostras que já existem.")
            if st.button("Capturar nova amostra", type="primary",
                         key=f"add-{p['id']}"):
                try:
                    r = S.post(f"{API}/people/{p['id']}/samples", timeout=30)
                    data = r.json()
                except requests.RequestException as exc:
                    st.error(f"Falha: {exc}")
                else:
                    if r.ok and data.get("ok"):
                        st.success(f"Amostra adicionada (total: {data['total']}).")
                        st.rerun()
                    else:
                        st.warning(data.get("message") or data.get("detail") or r.text)
            if st.button("Atualizar prévia", key=f"prev-{p['id']}"):
                st.rerun()
        with cb:
            imagem(f"/enroll/preview?t={time.time()}",
                   caption="prévia da câmera", width="stretch")

    st.divider()

    # --- renomear e remover ------------------------------------------------- #
    st.subheader("Editar")
    cr, cd = st.columns([3, 1])
    with cr:
        novo = st.text_input("Nome", value=p["name"], key=f"nome-{p['id']}")
        novo_inep = st.text_input(
            "ID INEP", value=p.get("inep_id") or "", key=f"inep-{p['id']}",
            help="Deixe em branco para limpar. Dois alunos não podem ter o mesmo.")
        mudou = (novo.strip() != p["name"] or
                 novo_inep.strip() != (p.get("inep_id") or ""))
        if st.button("Salvar", key=f"ren-{p['id']}",
                     disabled=not novo.strip() or not mudou):
            r = S.patch(f"{API}/people/{p['id']}", timeout=10,
                        json={"name": novo.strip(), "inep_id": novo_inep.strip()})
            if r.ok:
                d = r.json()
                if d.get("aviso"):
                    st.warning(d["aviso"])
                    time.sleep(2)
                st.rerun()
            else:
                st.error(r.json().get("detail", r.text))
    with cd:
        st.caption("**Excluir pessoa**")
        modo = st.radio(
            "O que fazer com o histórico de passagens?",
            ["Anonimizar", "Apagar tudo"],
            key=f"modo-{p['id']}",
            captions=["Mantém a contagem das passagens sem identificar quem "
                      "passou. As fotos são apagadas.",
                      "Remove também os registros de passagem. Não há como "
                      "desfazer."])
        confirmo = st.checkbox("Confirmo a exclusão", key=f"conf-{p['id']}")
        if st.button("Excluir", key=f"del-{p['id']}", disabled=not confirmo,
                     type="secondary"):
            anon = modo == "Anonimizar"
            r = S.delete(f"{API}/people/{p['id']}",
                         params={"anonimizar": str(anon).lower()}, timeout=20)
            if r.ok:
                d = r.json()
                st.success(f"Excluída ({d['modo']}): {d['fotos_removidas']} foto(s) "
                           f"e {d['recortes_removidos']} recorte(s) removidos.")
                st.rerun()
            else:
                st.error(r.text)


# --------------------------------------------------------------------------- #
elif page == "Ao vivo":
    st.header("Câmera ao vivo")
    auto = st.checkbox("Atualizar automaticamente (~1s)", value=False)
    placeholder = st.empty()
    placeholder.image(f"{API}/live.jpg?t={time.time()}", width="stretch")
    if auto:
        time.sleep(1)
        st.rerun()
