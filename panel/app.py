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
if _token:
    S.headers["X-API-Token"] = _token

st.set_page_config(page_title="Reconhecimento Facial", page_icon="📷", layout="wide")


def api_up() -> bool:
    try:
        return S.get(f"{API}/health", timeout=3).ok
    except requests.RequestException:
        return False


st.sidebar.title("📷 Reconhecimento Facial")
st.sidebar.caption(f"API: {API}")
if api_up():
    st.sidebar.success("API conectada")
else:
    st.sidebar.error("API offline — suba `uvicorn api:app` e o `worker.py`.")

page = st.sidebar.radio("Menu", ["Cadastrar", "Reconhecimentos", "Pessoas", "Ao vivo"])


# --------------------------------------------------------------------------- #
if page == "Cadastrar":
    st.header("Cadastrar pessoa (ao vivo)")
    ss = st.session_state
    ss.setdefault("enroll_session", None)
    ss.setdefault("enroll_name", "")
    ss.setdefault("enroll_target", 5)
    ss.setdefault("samples", [])

    # --- sem sessão: pedir nome e iniciar ---
    if ss.enroll_session is None:
        st.write("Informe o nome e inicie o cadastro para abrir o preview da câmera.")
        name = st.text_input("Nome")
        if st.button("Iniciar cadastro", type="primary", disabled=not name.strip()):
            try:
                r = S.post(f"{API}/enroll/start", json={"name": name.strip()}, timeout=30)
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
                st.image(f"{API}/enroll/preview?t={time.time()}", width="stretch",
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
                        st.warning(data.get("message", "Não foi possível capturar."))

        with col_samples:
            st.caption("Amostras")
            if not ss.samples:
                st.write("_nenhuma ainda_")
            for s in ss.samples:
                st.image(f"{API}{s['snapshot_url']}", width=110)
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
    st.header("Histórico de reconhecimentos")
    col_a, col_b = st.columns([1, 3])
    limit = col_a.number_input("Quantidade", 10, 500, 60, step=10)
    if col_b.button("Atualizar"):
        st.rerun()

    try:
        events = S.get(f"{API}/events", params={"limit": int(limit)}, timeout=10).json()
    except requests.RequestException as exc:
        st.error(f"Falha ao buscar eventos: {exc}")
        events = []

    if not events:
        st.info("Nenhum evento ainda. Deixe o worker rodando e passe na frente da câmera.")
    else:
        cols = st.columns(4)
        for i, ev in enumerate(events):
            with cols[i % 4]:
                if ev.get("snapshot_path"):
                    st.image(f"{API}/snapshots/{ev['snapshot_path']}", width="stretch")
                quando = time.strftime("%d/%m %H:%M:%S", time.localtime(ev["ts"]))
                tag = "✅" if ev["is_known"] else "❓"
                st.caption(f"{tag} **{ev['name']}** · {ev['score']:.2f}\n\n{quando}")


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
    rotulos = {f"{p['name']}  ({p['embeddings']} amostra(s))": p for p in people}
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
                st.image(f"{API}{a['snapshot_url']}", width="stretch")
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
            st.image(f"{API}/enroll/preview?t={time.time()}",
                     caption="prévia da câmera", width="stretch")

    st.divider()

    # --- renomear e remover ------------------------------------------------- #
    st.subheader("Editar")
    cr, cd = st.columns([3, 1])
    with cr:
        novo = st.text_input("Nome", value=p["name"], key=f"nome-{p['id']}")
        if st.button("Renomear", key=f"ren-{p['id']}",
                     disabled=not novo.strip() or novo.strip() == p["name"]):
            r = S.patch(f"{API}/people/{p['id']}",
                               json={"name": novo.strip()}, timeout=10)
            if r.ok:
                st.rerun()
            else:
                st.error(r.json().get("detail", r.text))
    with cd:
        st.caption("Apaga a pessoa, as amostras e as fotos dela. O histórico de "
                   "reconhecimentos é mantido.")
        if st.button("Remover pessoa", key=f"del-{p['id']}"):
            r = S.delete(f"{API}/people/{p['id']}", timeout=15)
            if r.ok:
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
