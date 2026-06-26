"""Painel web (Streamlit) — cliente HTTP fino da API.

Na Fase 1 roda no Mac apontando para http://localhost:8000.
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

st.set_page_config(page_title="Reconhecimento Facial", page_icon="📷", layout="wide")


def api_up() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=3).ok
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
                r = requests.post(f"{API}/enroll/start", json={"name": name.strip()}, timeout=30)
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
                    r = requests.post(f"{API}/enroll/capture",
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
                    r = requests.post(f"{API}/enroll/sample/delete",
                                      json={"session_id": ss.enroll_session, "index": s["index"]},
                                      timeout=30)
                    if r.ok:
                        ss.samples = r.json()["samples"]
                    st.rerun()

        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("✅ Concluir cadastro", type="primary", disabled=n < 1, width="stretch"):
            r = requests.post(f"{API}/enroll/finish",
                              json={"session_id": ss.enroll_session}, timeout=30)
            if r.ok:
                st.success(f"{ss.enroll_name} cadastrado(a) com {r.json()['captured']} amostras.")
                ss.enroll_session = None
                ss.samples = []
            else:
                st.error(r.json().get("detail", r.text))
        if c2.button("Cancelar", width="stretch"):
            requests.post(f"{API}/enroll/cancel", json={"session_id": ss.enroll_session}, timeout=30)
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
        events = requests.get(f"{API}/events", params={"limit": int(limit)}, timeout=10).json()
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
    try:
        people = requests.get(f"{API}/people", timeout=10).json()
    except requests.RequestException as exc:
        st.error(f"Falha ao buscar pessoas: {exc}")
        people = []

    if not people:
        st.info("Ninguém cadastrado ainda. Vá em **Cadastrar**.")
    for p in people:
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.write(f"**{p['name']}**")
        c2.write(f"{p['embeddings']} amostras")
        if c3.button("Remover", key=f"del-{p['id']}"):
            requests.delete(f"{API}/people/{p['id']}", timeout=10)
            st.rerun()


# --------------------------------------------------------------------------- #
elif page == "Ao vivo":
    st.header("Câmera ao vivo")
    auto = st.checkbox("Atualizar automaticamente (~1s)", value=False)
    placeholder = st.empty()
    placeholder.image(f"{API}/live.jpg?t={time.time()}", width="stretch")
    if auto:
        time.sleep(1)
        st.rerun()
