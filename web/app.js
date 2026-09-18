/* Painel — Chamada e Reconhecimentos.
 *
 * A API é servida pela mesma origem que esta página, então todas as chamadas
 * são relativas e o cookie de sessão vai sozinho. Nenhum token aparece aqui.
 *
 * Princípio que guia o arquivo: nada de recarregar a página. Cada ação manda
 * uma requisição e atualiza só o pedaço que mudou. É justamente o que o
 * Streamlit não fazia — lá cada clique reexecutava o script inteiro, e com 50
 * alunos isso eram 50 fotos rebaixadas por interação.
 */

"use strict";

const $ = (sel) => document.querySelector(sel);
const criar = (tag, props = {}) => Object.assign(document.createElement(tag), props);

/* --------------------------------------------------------------- rede ---- */

async function api(caminho, opcoes = {}) {
  const resp = await fetch(caminho, {
    credentials: "same-origin",
    headers: opcoes.corpo ? { "Content-Type": "application/json" } : {},
    method: opcoes.metodo || "GET",
    body: opcoes.corpo ? JSON.stringify(opcoes.corpo) : undefined,
  });
  // 401/503 depois de autenticado significa sessão expirada: volta ao login
  // em vez de deixar a tela quebrada sem explicação.
  if (resp.status === 401 || resp.status === 503) {
    const dados = await resp.json().catch(() => ({}));
    mostrarLogin(dados.detail || "Sessão expirada. Entre novamente.");
    throw new Error(dados.detail || "não autenticado");
  }
  if (!resp.ok) {
    let detalhe = `HTTP ${resp.status}`;
    try {
      const j = await resp.json();
      detalhe = j.detail || j.message || detalhe;
    } catch { /* corpo não-JSON: fica o status */ }
    throw new Error(detalhe);
  }
  return resp.status === 204 ? null : resp.json();
}

/* ---------------------------------------------------------------- lupa --- */

/* Abre a foto em tamanho grande, com navegação entre as fotos da mesma lista.
 *
 * Miniatura de 60px não responde a pergunta "é essa pessoa mesmo?", que é
 * justamente a decisão que a conferência da chamada e a marcação de erro
 * exigem. Sem isso, a foto na tela é decoração.
 *
 * `itens` é a lista inteira para as setas funcionarem: revisar 24 detecções
 * abrindo e fechando uma a uma é o tipo de atrito que faz ninguém revisar.
 */
let lupa = { itens: [], i: 0 };

function abrirLupa(itens, indice) {
  lupa = { itens, i: indice };
  $("#lupa").hidden = false;
  document.body.style.overflow = "hidden";   // não rolar a página atrás
  desenharLupa();
}

function fecharLupa() {
  $("#lupa").hidden = true;
  document.body.style.overflow = "";
}

function desenharLupa() {
  const it = lupa.itens[lupa.i];
  if (!it) return fecharLupa();

  $("#lupa-img").src = it.url;
  $("#lupa-img").alt = it.titulo;
  $("#lupa-titulo").textContent = it.titulo;
  $("#lupa-meta").textContent = it.meta || "";
  $("#lupa-pos").textContent =
    lupa.itens.length > 1 ? `${lupa.i + 1} de ${lupa.itens.length}` : "";

  $("#lupa-anterior").disabled = lupa.i === 0;
  $("#lupa-proxima").disabled = lupa.i >= lupa.itens.length - 1;

  const acoes = $("#lupa-acoes");
  acoes.innerHTML = "";
  for (const a of it.acoes || []) {
    const b = criar("button", { textContent: a.rotulo, className: a.classe || "" });
    b.addEventListener("click", async () => { await a.aoClicar(); });
    acoes.append(b);
  }
}

function andarLupa(passo) {
  const novo = lupa.i + passo;
  if (novo >= 0 && novo < lupa.itens.length) { lupa.i = novo; desenharLupa(); }
}

$("#lupa-fechar").addEventListener("click", fecharLupa);
$("#lupa-anterior").addEventListener("click", () => andarLupa(-1));
$("#lupa-proxima").addEventListener("click", () => andarLupa(1));
// Clicar no fundo fecha; clicar na imagem ou nos controles, não.
$("#lupa").addEventListener("click", (ev) => {
  if (ev.target.id === "lupa") fecharLupa();
});
document.addEventListener("keydown", (ev) => {
  if ($("#lupa").hidden) return;
  if (ev.key === "Escape") fecharLupa();
  if (ev.key === "ArrowLeft") andarLupa(-1);
  if (ev.key === "ArrowRight") andarLupa(1);
});

/* -------------------------------------------------------------- sessão --- */

function mostrarLogin(motivo = "") {
  $("#app").hidden = true;
  $("#login").hidden = false;
  $("#login-motivo").textContent = motivo;
}

async function iniciar() {
  let s;
  try {
    s = await fetch("/session", { credentials: "same-origin" }).then(r => r.json());
  } catch {
    $("#estado-api").textContent = "API inacessível";
    $("#estado-api").className = "estado ruim";
    return;
  }

  if (!s.autenticado) {
    mostrarLogin(s.motivo || (s.token_exigido
      ? "Esta API exige token para acesso pela rede."
      : "Acesso não autorizado."));
    // Sem token configurado, digitar não resolve: esconde o formulário.
    $("#form-login").hidden = !s.token_exigido;
    return;
  }

  $("#login").hidden = true;
  $("#app").hidden = false;
  $("#estado-api").textContent =
    s.consumidor === "local" ? "acesso local" : `sessão: ${s.consumidor}`;

  const hoje = new Date().toISOString().slice(0, 10);
  $("#dia").value = hoje;
  $("#dia-rec").value = hoje;

  await Promise.all([carregarChamada(), recarregarListasDePessoas()]);
  atualizarSaude();
  setInterval(atualizarSaude, 15000);
}

$("#form-login").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const erro = $("#login-erro");
  erro.hidden = true;
  try {
    await api("/session", { metodo: "POST", corpo: { token: $("#token").value } });
    $("#token").value = "";
    location.reload();          // recomeça já com o cookie
  } catch (e) {
    erro.textContent = e.message;
    erro.hidden = false;
  }
});

async function atualizarSaude() {
  try {
    const h = await api("/health");
    const partes = [];
    if (h.worker_mode) partes.push(`modo ${h.worker_mode}`);
    if (h.tracks_pending) partes.push(`${h.tracks_pending} pendentes`);
    if (h.worker_live_age_seconds != null && h.worker_live_age_seconds < 30) {
      partes.push("worker ativo");
    } else {
      partes.push("worker parado?");
    }
    $("#estado-api").textContent = partes.join(" · ");
    $("#estado-api").className = "estado";
  } catch { /* o erro já apareceu em outro lugar */ }
}

/* ------------------------------------------------------------- abas ------ */

const ABAS = ["chamada", "reconhecimentos", "cadastrar", "pessoas", "aovivo"];

document.querySelectorAll("nav button").forEach((b) => {
  b.addEventListener("click", () => trocarAba(b.dataset.aba));
});

function trocarAba(alvo) {
  document.querySelectorAll("nav button").forEach(o =>
    o.classList.toggle("ativa", o.dataset.aba === alvo));
  for (const a of ABAS) $(`#aba-${a}`).hidden = a !== alvo;

  // Preview é polling: parar ao sair da aba não é otimização, é correção.
  // Cada ciclo é uma requisição, e no /enroll/preview a API mantém a câmera
  // aberta enquanto está sendo consultada — deixar rodando em aba invisível
  // gasta rede e segura o dispositivo sem motivo.
  previewVivo.parar();
  previewAdd.parar();
  if (alvo !== "cadastrar") previewCad.parar();

  if (alvo === "reconhecimentos") carregarDeteccoes();
  if (alvo === "pessoas") { recarregarListasDePessoas(); previewAdd.iniciar(); }
  if (alvo === "cadastrar" && sessaoCad) previewCad.iniciar();
  if (alvo === "aovivo" && $("#vivo-ligado").checked) previewVivo.iniciar();
  atualizarEstadoVivo();
}

/* ---------------------------------------------------------- chamada ------ */

let chamada = null;

const ICONES = {
  automatico: "detectado",
  manual_presente: "marcado presente",
  manual_ausente: "marcado ausente",
  nao_identificado: "não identificado",
};

async function carregarChamada() {
  const dia = $("#dia").value;
  try {
    chamada = await api(`/attendance?dia=${dia}`);
  } catch (e) {
    $("#avisos").innerHTML = "";
    $("#avisos").append(faixa("erro", `Falha ao carregar: ${e.message}`));
    return;
  }
  desenharChamada();
}

function faixa(tipo, html) {
  const d = criar("div", { className: `faixa ${tipo}` });
  d.innerHTML = html;
  return d;
}

function metrica(rotulo, valor) {
  const d = criar("div", { className: "metrica" });
  d.append(criar("div", { className: "rotulo", textContent: rotulo }));
  d.append(criar("div", { className: "valor", textContent: valor }));
  return d;
}

function desenharChamada() {
  const d = chamada;

  const m = $("#metricas");
  m.innerHTML = "";
  m.append(metrica("Presentes", `${d.total_presentes} de ${d.total_cadastrados}`));
  m.append(metrica("Ausentes", d.ausentes.length));
  m.append(metrica("Correções", d.correcoes.total));
  m.append(metrica("Situação", d.conferida ? "Conferida" : "Em aberto"));

  const av = $("#avisos");
  av.innerHTML = "";
  if (!d.completo) {
    av.append(faixa("erro",
      `<b>${d.trilhas_pendentes} trilha(s) aguardando reconhecimento.</b> ` +
      "A chamada está incompleta — rode o lote antes de conferir: " +
      "<code>python scripts/recognize_batch.py</code>"));
  }
  if (d.sem_matricula) {
    av.append(faixa("aviso",
      `<b>${d.sem_matricula} aluno(s) sem matrícula.</b> ` +
      "O sistema de gestão da escola não consegue casar esses registros."));
  }
  if (d.conferida) {
    const f = d.fechamento;
    av.append(faixa("ok", `Conferida em ${f.em.slice(8, 10)}/${f.em.slice(5, 7)} ` +
      `às ${f.em.slice(11, 16)}` + (f.autor ? ` por ${f.autor}` : "") +
      ". Para alterar, reabra abaixo."));
  } else {
    av.append(faixa("aviso",
      "Chamada <b>em aberto</b>. Quem não foi identificado pode ter passado " +
      "sem ser detectado — confira antes de tratar como falta."));
  }

  $("#n-presentes").textContent = `(${d.presentes.length})`;
  $("#n-ausentes").textContent = `(${d.ausentes.length})`;
  encher($("#presentes"), d.presentes, true);
  encher($("#ausentes"), d.ausentes, false);

  $("#fechar").hidden = d.conferida;
  $("#reabrir").hidden = !d.conferida;
}

function encher(container, pessoas, presente) {
  container.innerHTML = "";
  if (!pessoas.length) {
    container.append(criar("p", { className: "nota", textContent: "ninguém" }));
    return;
  }
  for (const p of pessoas) container.append(linhaPessoa(p, presente));
}

function linhaPessoa(p, presente) {
  const l = criar("label", { className: "pessoa" });
  if (p.origem.startsWith("manual")) l.classList.add("corrigida");

  // A foto é o que permite conferir. Sem ela, marcar "presente" confirma um
  // nome, não um rosto — e são essas confirmações que alimentam a calibração
  // e a medição de recall.
  if (p.foto_url) {
    const img = criar("img", { src: p.foto_url, loading: "lazy",
                               alt: `melhor captura de ${p.nome}` });
    img.addEventListener("click", (ev) => {
      // CRÍTICO: a linha é um <label> em volta do checkbox, então um clique
      // na foto marcaria presença. Conferir uma imagem não pode alterar a
      // chamada — daí o preventDefault antes de abrir a lupa.
      ev.preventDefault();
      ev.stopPropagation();
      abrirLupaDaChamada(p);
    });
    l.append(img);
  } else {
    const v = criar("div", { className: "sem-foto" });
    v.innerHTML = "sem<br>registro";
    l.append(v);
  }

  const cx = criar("input", { type: "checkbox", checked: presente });
  cx.disabled = chamada.conferida;
  cx.addEventListener("change", () => corrigir(p, cx.checked));
  l.append(cx);

  // `.texto` cresce (flex:1) e ocupa a largura restante — sem isso a linha
  // terminava logo depois do nome e sobrava um vão à direita.
  const txt = criar("div", { className: "texto" });
  txt.append(criar("div", { className: "nome", textContent: p.nome }));

  const det = [ICONES[p.origem]];
  if (!p.matricula) det.push("sem matrícula");
  if (p.primeira_vez) det.push(p.primeira_vez.slice(11, 16));
  if (p.melhor_score != null) det.push(`score ${p.melhor_score.toFixed(2)}`);
  if (p.correcao && p.correcao.motivo) det.push(`motivo: ${p.correcao.motivo}`);
  txt.append(criar("div", { className: "detalhe", textContent: det.join(" · ") }));
  l.append(txt);
  return l;
}

/* Abre a lupa percorrendo só quem TEM foto, na ordem da tela.
 *
 * A ação de presença fica dentro da lupa porque é a decisão que motiva abrir
 * a foto: ampliar, olhar, decidir. Ter que fechar e procurar a linha de novo
 * quebraria o fluxo justamente no momento da decisão.
 */
function abrirLupaDaChamada(pessoa) {
  const comFoto = [...chamada.presentes, ...chamada.ausentes]
    .filter(p => p.foto_url);
  const itens = comFoto.map(p => {
    const presente = chamada.presentes.includes(p);
    const det = [ICONES[p.origem]];
    if (p.primeira_vez) det.push(p.primeira_vez.slice(11, 16));
    if (p.melhor_score != null) det.push(`score ${p.melhor_score.toFixed(2)}`);
    return {
      url: p.foto_url,
      titulo: p.nome,
      meta: det.join(" · ") + (p.foto_score != null
        ? ` · melhor captura do dia (${p.foto_score.toFixed(2)})` : ""),
      acoes: chamada.conferida ? [] : [{
        rotulo: presente ? "Marcar ausente" : "Marcar presente",
        classe: presente ? "perigo" : "primario",
        aoClicar: async () => { fecharLupa(); await corrigir(p, !presente); },
      }],
    };
  });
  abrirLupa(itens, Math.max(0, comFoto.indexOf(pessoa)));
}

/* Grava a correção NA HORA, sem botão de salvar.
 *
 * O Streamlit usava um formulário: marcava tudo e enviava no fim. Aqui cada
 * caixa é uma requisição — o que só é aceitável porque não há rerun, e o
 * resto da tela nem pisca. Em troca, ninguém perde o trabalho por fechar a
 * aba antes de enviar. */
async function corrigir(pessoa, marcado) {
  const dia = $("#dia").value;
  const autor = $("#autor").value.trim();
  const detectado = pessoa.detectado_pelo_sistema;

  try {
    if (marcado === detectado) {
      // Voltou a concordar com o sistema: a correção deixa de existir. Guardar
      // "correção que concorda" encheria a tabela e estragaria a métrica de
      // acerto, que é justamente a diferença entre automático e corrigido.
      await api(`/attendance/override?dia=${dia}&person_id=${pessoa.person_id}`,
                { metodo: "DELETE" });
    } else {
      await api("/attendance/override", { metodo: "POST", corpo: {
        dia, person_id: pessoa.person_id, presente: marcado, autor, motivo: "" } });
    }
    piscarSalvo();
    await carregarChamada();
  } catch (e) {
    $("#avisos").prepend(faixa("erro", `Não gravou: ${e.message}`));
    await carregarChamada();       // volta ao estado real do servidor
  }
}

let timerSalvo;
function piscarSalvo() {
  const s = $("#salvo");
  s.hidden = false;
  clearTimeout(timerSalvo);
  timerSalvo = setTimeout(() => { s.hidden = true; }, 1200);
}

$("#recarregar").addEventListener("click", carregarChamada);
$("#dia").addEventListener("change", carregarChamada);

$("#fechar").addEventListener("click", async () => {
  if (!confirm("Fechar a chamada deste dia? Depois é preciso reabrir para editar."))
    return;
  try {
    await api("/attendance/close", { metodo: "POST", corpo: {
      dia: $("#dia").value, autor: $("#autor").value.trim() } });
    await carregarChamada();
  } catch (e) {
    $("#avisos").prepend(faixa("erro", e.message));
  }
});

$("#reabrir").addEventListener("click", async () => {
  try {
    await api(`/attendance/close?dia=${$("#dia").value}`, { metodo: "DELETE" });
    await carregarChamada();
  } catch (e) {
    $("#avisos").prepend(faixa("erro", e.message));
  }
});

/* -------------------------------------------------- reconhecimentos ------ */

async function carregarDeteccoes() {
  const p = new URLSearchParams({
    person_id: $("#filtro-pessoa").value,
    limit: $("#limite").value,
  });
  if ($("#usar-dia-rec").checked) p.set("dia", $("#dia-rec").value);

  const grade = $("#grade");
  grade.innerHTML = "";
  $("#resumo-rec").textContent = "carregando…";

  let dados;
  try {
    dados = await api(`/detections?${p}`);
  } catch (e) {
    $("#resumo-rec").textContent = "";
    grade.append(faixa("erro", e.message));
    return;
  }

  const ds = dados.deteccoes;
  if (!ds.length) {
    $("#resumo-rec").textContent = "";
    grade.append(faixa("aviso", "Nenhuma detecção com esses filtros."));
    return;
  }

  const errados = ds.filter(d => d.rotulo === "errado").length;
  $("#resumo-rec").textContent =
    `${ds.length} detecção(ões)` +
    (errados ? ` · ${errados} marcada(s) como erro` : "") +
    " · marcar aqui não altera a chamada";

  deteccoesNaTela = ds;
  for (const d of ds) grade.append(cartaoDeteccao(d));
}

/* Guardada para a lupa poder navegar entre todas com as setas. Revisar 24
 * detecções abrindo e fechando uma a uma é o atrito que faz ninguém revisar —
 * e sem revisão marcada a calibração não tem teto para o limiar. */
let deteccoesNaTela = [];

async function marcarErro(d) {
  await api("/detections/label", { metodo: "POST", corpo: {
    fonte: d.fonte, detection_id: d.id, rotulo: "errado",
    autor: $("#autor").value.trim() } });
}

async function desmarcarErro(d) {
  await api(`/detections/label?fonte=${d.fonte}&detection_id=${d.id}`,
            { metodo: "DELETE" });
}

function abrirLupaDeDeteccoes(alvo) {
  const comFoto = deteccoesNaTela.filter(d => d.foto_url);
  const itens = comFoto.map(d => ({
    url: d.foto_url,
    titulo: `${d.nome} · ${d.score.toFixed(2)}`,
    meta: `${d.quando.slice(0, 10)} ${d.quando.slice(11, 19)} · ` +
          `${d.fonte === "trilha" ? "lote" : "ao vivo"}` +
          (d.rotulo === "errado" ? " · marcada como erro" : ""),
    acoes: !d.is_known ? [] : [d.rotulo === "errado" ? {
      rotulo: "Desfazer marcação",
      aoClicar: async () => { await desmarcarErro(d); fecharLupa(); carregarDeteccoes(); },
    } : {
      rotulo: "Não é essa pessoa",
      classe: "perigo",
      // Segue para a próxima em vez de fechar: revisão em lote fica fluida,
      // e é assim que se acumula o rótulo de erro que a calibração precisa.
      aoClicar: async () => {
        await marcarErro(d);
        d.rotulo = "errado";
        if (lupa.i < lupa.itens.length - 1) andarLupa(1); else desenharLupa();
      },
    }],
  }));
  abrirLupa(itens, Math.max(0, comFoto.indexOf(alvo)));
}

function cartaoDeteccao(d) {
  const c = criar("div", { className: "cartao" });
  if (d.rotulo === "errado") c.classList.add("errada");

  if (d.foto_url) {
    const img = criar("img", { src: d.foto_url, loading: "lazy",
                               alt: `detecção de ${d.nome}` });
    img.addEventListener("click", () => abrirLupaDeDeteccoes(d));
    c.append(img);
  }

  const info = criar("div", { className: "info" });
  info.append(criar("div", { className: "nome",
    textContent: `${d.is_known ? "" : "? "}${d.nome} · ${d.score.toFixed(2)}` }));
  info.append(criar("div", { className: "meta",
    textContent: `${d.quando.slice(11, 19)} · ${d.fonte === "trilha" ? "lote" : "ao vivo"}` }));
  c.append(info);

  if (d.rotulo === "errado") {
    const b = criar("button", { textContent: "desfazer" });
    b.addEventListener("click", async () => {
      await desmarcarErro(d);
      carregarDeteccoes();
    });
    c.append(b);
  } else if (d.is_known) {
    const b = criar("button", { textContent: "não é essa pessoa" });
    b.addEventListener("click", async () => {
      await marcarErro(d);
      carregarDeteccoes();
    });
    c.append(b);
  }
  return c;
}

/* ------------------------------------------------------ preview ao vivo -- */

/* Troca a imagem só depois que a nova terminou de carregar.
 *
 * Mudar o `src` direto deixa o <img> em branco enquanto baixa, e a 1 Hz isso
 * pisca sem parar. Carregando num objeto Image em memória e trocando no
 * onload, a transição é invisível. O Streamlit não tinha como fazer isso: lá
 * o fragmento redesenhava o elemento inteiro a cada ciclo. */
function criarAtualizador(img, caminho, intervalo = 700) {
  let timer = null, parado = true, falhas = 0;

  function tick() {
    const nova = new Image();
    nova.onload = () => {
      img.src = nova.src;
      falhas = 0;
      if (!parado) timer = setTimeout(tick, intervalo);
    };
    nova.onerror = () => {
      falhas += 1;
      // Recua quando falha: sem imagem, insistir a 1 Hz só enche o log do
      // servidor. Volta ao ritmo normal sozinho quando a imagem voltar.
      if (!parado) timer = setTimeout(tick, Math.min(intervalo * falhas, 5000));
    };
    nova.src = `${caminho}${caminho.includes("?") ? "&" : "?"}t=${Date.now()}`;
  }

  return {
    iniciar() { if (parado) { parado = false; tick(); } },
    parar() { parado = true; clearTimeout(timer); },
    get falhas() { return falhas; },
  };
}

/* ---------------------------------------------------------- cadastro ----- */

let sessaoCad = null;
const previewCad = criarAtualizador($("#cad-preview"), "/enroll/preview");

function validarInicio() {
  $("#cad-iniciar").disabled = !$("#cad-nome").value.trim();
}
$("#cad-nome").addEventListener("input", validarInicio);

$("#cad-iniciar").addEventListener("click", async () => {
  try {
    sessaoCad = await api("/enroll/start", { metodo: "POST", corpo: {
      name: $("#cad-nome").value.trim(),
      matricula: $("#cad-matricula").value.trim() } });
  } catch (e) {
    alert(`Não foi possível iniciar: ${e.message}`);
    return;
  }
  $("#cad-inicio").hidden = true;
  $("#cad-sessao").hidden = false;
  $("#cad-quem").textContent = sessaoCad.name;
  desenharCadastro([]);
  previewCad.iniciar();
});

function desenharCadastro(amostras) {
  const alvo = sessaoCad.target;
  $("#cad-progresso").textContent =
    `${amostras.length} de ${alvo} amostra(s) capturada(s).`;
  $("#cad-capturar").disabled = amostras.length >= alvo;
  $("#cad-concluir").disabled = amostras.length < 1;

  const box = $("#cad-amostras");
  box.innerHTML = "";
  if (!amostras.length) {
    box.append(criar("p", { className: "nota", textContent: "nenhuma ainda" }));
  }
  for (const a of amostras) {
    const t = criar("div", { className: "tira" });
    t.append(criar("img", { src: a.snapshot_url, alt: `amostra ${a.index + 1}` }));
    const b = criar("button", { textContent: "remover" });
    b.addEventListener("click", async () => {
      const r = await api("/enroll/sample/delete", { metodo: "POST", corpo: {
        session_id: sessaoCad.session_id, index: a.index } });
      desenharCadastro(r.samples);
    });
    t.append(b);
    box.append(t);
  }
}

$("#cad-capturar").addEventListener("click", async () => {
  const erro = $("#cad-erro");
  erro.hidden = true;
  $("#cad-capturar").disabled = true;
  try {
    const r = await api("/enroll/capture", { metodo: "POST", corpo: {
      session_id: sessaoCad.session_id } });
    if (r.ok) {
      desenharCadastro(r.samples);
    } else {
      // A API explica o motivo (sem rosto, sem imagem, conflito de câmera).
      // Repassar em vez de inventar texto genérico foi a lição do painel
      // anterior, que dizia "não foi possível capturar" e escondia a causa.
      erro.textContent = r.message || "não capturou";
      erro.hidden = false;
      $("#cad-capturar").disabled = false;
    }
  } catch (e) {
    erro.textContent = e.message;
    erro.hidden = false;
    $("#cad-capturar").disabled = false;
  }
});

$("#cad-concluir").addEventListener("click", async () => {
  try {
    const r = await api("/enroll/finish", { metodo: "POST", corpo: {
      session_id: sessaoCad.session_id } });
    encerrarCadastro();
    alert(`${r.name} cadastrado(a) com ${r.captured} amostra(s).`);
    await recarregarListasDePessoas();
  } catch (e) {
    $("#cad-erro").textContent = e.message;
    $("#cad-erro").hidden = false;
  }
});

$("#cad-cancelar").addEventListener("click", async () => {
  try {
    await api("/enroll/cancel", { metodo: "POST", corpo: {
      session_id: sessaoCad.session_id } });
  } catch { /* sessão já podia ter expirado */ }
  encerrarCadastro();
});

function encerrarCadastro() {
  previewCad.parar();
  sessaoCad = null;
  $("#cad-sessao").hidden = true;
  $("#cad-inicio").hidden = false;
  $("#cad-nome").value = "";
  $("#cad-matricula").value = "";
  validarInicio();
}

/* ----------------------------------------------------------- pessoas ----- */

let pessoas = [];
let pessoaAtual = null;
const previewAdd = criarAtualizador($("#add-preview"), "/enroll/preview");

async function recarregarListasDePessoas() {
  try {
    pessoas = await api("/people");
  } catch {
    return;
  }
  // Filtro da aba de reconhecimentos
  const f = $("#filtro-pessoa");
  const antes = f.value;
  f.innerHTML = "";
  f.append(criar("option", { value: "0", textContent: "Todas" }));
  for (const p of pessoas) {
    f.append(criar("option", { value: p.id,
      textContent: `${p.name} (${p.embeddings} amostra(s))` }));
  }
  f.value = antes || "0";

  // Seletor da aba Pessoas
  const s = $("#sel-pessoa");
  const antesP = s.value;
  s.innerHTML = "";
  for (const p of pessoas) {
    s.append(criar("option", { value: p.id,
      textContent: `${p.name} — ${p.matricula || "sem matrícula"} ` +
                   `(${p.embeddings} amostra(s))` }));
  }
  if (antesP && pessoas.some(p => String(p.id) === antesP)) s.value = antesP;
  $("#resumo-pessoas").textContent = `${pessoas.length} pessoa(s) cadastrada(s).`;
  if (pessoas.length) await carregarAmostras();
  else limparPessoa();
}

function limparPessoa() {
  pessoaAtual = null;
  $("#amostras-pessoa").innerHTML = "";
  $("#avisos-pessoa").innerHTML = "";
  $("#ed-nome").value = "";
  $("#ed-matricula").value = "";
}

async function carregarAmostras() {
  const id = $("#sel-pessoa").value;
  if (!id) return limparPessoa();

  let dados;
  try {
    dados = await api(`/people/${id}/samples`);
  } catch (e) {
    $("#avisos-pessoa").innerHTML = "";
    $("#avisos-pessoa").append(faixa("erro", e.message));
    return;
  }
  pessoaAtual = dados.person;
  $("#ed-nome").value = pessoaAtual.name;
  $("#ed-matricula").value = pessoaAtual.matricula || "";
  validarEdicao();

  const av = $("#avisos-pessoa");
  av.innerHTML = "";
  if (dados.sem_foto) {
    av.append(faixa("aviso", `${dados.sem_foto} amostra(s) sem foto. São de ` +
      "antes de o sistema guardar a imagem de cada amostra — o vetor " +
      "funciona, só não há o que conferir."));
  }
  const redundantes = dados.samples.filter(
    a => (a.similaridade_maxima || 0) >= 0.95).length;
  if (redundantes) {
    av.append(faixa("aviso", `${redundantes} amostra(s) quase idênticas a ` +
      "outras. Excluir uma delas não piora o reconhecimento: o que ajuda é " +
      "variação de ângulo e luz, não quantidade."));
  } else if (dados.samples.length < 3) {
    av.append(faixa("aviso", "Poucas amostras. Três ou mais, com ângulos " +
      "diferentes, melhoram bastante o reconhecimento."));
  }

  amostrasNaTela = dados.samples;
  const g = $("#amostras-pessoa");
  g.innerHTML = "";
  for (const a of dados.samples) g.append(cartaoAmostra(a, dados.samples.length));
}

let amostrasNaTela = [];

function abrirLupaDeAmostras(alvo) {
  const comFoto = amostrasNaTela.filter(a => a.snapshot_url);
  const itens = comFoto.map(a => {
    const meta = [`#${a.id}`];
    if (a.quality != null) meta.push(`nitidez ${Math.round(a.quality)}`);
    if (a.similaridade_maxima != null) {
      meta.push(`similaridade máx. ${a.similaridade_maxima.toFixed(2)}` +
                (a.similaridade_maxima >= 0.95
                 ? ` — redundante com a #${a.parecida_com}` : ""));
    }
    return {
      url: a.snapshot_url,
      titulo: pessoaAtual ? pessoaAtual.name : "amostra",
      meta: meta.join(" · "),
      acoes: comFoto.length <= 1 ? [] : [{
        rotulo: "Excluir amostra",
        classe: "perigo",
        aoClicar: async () => {
          try {
            await api(`/embeddings/${a.id}`, { metodo: "DELETE" });
            fecharLupa();
            await recarregarListasDePessoas();
          } catch (e) {
            $("#avisos-pessoa").prepend(faixa("erro", e.message));
            fecharLupa();
          }
        },
      }],
    };
  });
  abrirLupa(itens, Math.max(0, comFoto.indexOf(alvo)));
}

function cartaoAmostra(a, total) {
  const c = criar("div", { className: "cartao" });
  if (a.snapshot_url) {
    const img = criar("img", { src: a.snapshot_url, loading: "lazy",
                               alt: `amostra ${a.id}` });
    img.addEventListener("click", () => abrirLupaDeAmostras(a));
    c.append(img);
  }
  const info = criar("div", { className: "info" });
  const partes = [`#${a.id}`];
  if (a.quality != null) partes.push(`nitidez ${Math.round(a.quality)}`);
  if (a.created_at) {
    partes.push(new Date(a.created_at * 1000).toLocaleDateString("pt-BR"));
  }
  info.append(criar("div", { className: "meta", textContent: partes.join(" · ") }));

  if (a.similaridade_maxima != null) {
    const redundante = a.similaridade_maxima >= 0.95;
    info.append(criar("div", {
      className: redundante ? "redundante" : "meta",
      textContent: redundante
        ? `redundante — ${a.similaridade_maxima.toFixed(2)} com a #${a.parecida_com}`
        : `similaridade máx. ${a.similaridade_maxima.toFixed(2)}`,
    }));
  }
  c.append(info);

  const b = criar("button", { textContent: "Excluir" });
  b.disabled = total <= 1;   // a API recusa com 409; desabilitar explica antes
  b.title = total <= 1 ? "É a única amostra. Adicione outra antes de remover." : "";
  b.addEventListener("click", async () => {
    try {
      await api(`/embeddings/${a.id}`, { metodo: "DELETE" });
      await recarregarListasDePessoas();
    } catch (e) {
      $("#avisos-pessoa").prepend(faixa("erro", e.message));
    }
  });
  c.append(b);
  return c;
}

function validarEdicao() {
  if (!pessoaAtual) { $("#ed-salvar").disabled = true; return; }
  const nome = $("#ed-nome").value.trim();
  const mudou = nome !== pessoaAtual.name ||
                $("#ed-matricula").value.trim() !== (pessoaAtual.matricula || "");
  $("#ed-salvar").disabled = !nome || !mudou;
}
$("#ed-nome").addEventListener("input", validarEdicao);
$("#ed-matricula").addEventListener("input", validarEdicao);

$("#ed-salvar").addEventListener("click", async () => {
  try {
    await api(`/people/${pessoaAtual.id}`, { metodo: "PATCH", corpo: {
      name: $("#ed-nome").value.trim(),
      matricula: $("#ed-matricula").value.trim() } });
    piscarSalvo();
    await recarregarListasDePessoas();
  } catch (e) {
    $("#avisos-pessoa").prepend(faixa("erro", e.message));
  }
});

$("#add-amostra").addEventListener("click", async () => {
  const erro = $("#add-erro");
  erro.hidden = true;
  try {
    const r = await api(`/people/${pessoaAtual.id}/samples`, { metodo: "POST" });
    if (r.ok) {
      await recarregarListasDePessoas();
    } else {
      erro.textContent = r.message || "não capturou";
      erro.hidden = false;
    }
  } catch (e) {
    erro.textContent = e.message;
    erro.hidden = false;
  }
});

$("#ex-pessoa").addEventListener("click", async () => {
  const anon = $("#ex-anonimizar").checked;
  const aviso = anon
    ? `Excluir ${pessoaAtual.name}, preservando as passagens sem identificação?`
    : `Excluir ${pessoaAtual.name} e TODO o histórico dela? Não tem desfazer.`;
  if (!confirm(aviso)) return;
  try {
    const r = await api(`/people/${pessoaAtual.id}?anonimizar=${anon}`,
                        { metodo: "DELETE" });
    limparPessoa();
    await recarregarListasDePessoas();
    alert(`Removido. ${JSON.stringify(r)}`);
  } catch (e) {
    $("#avisos-pessoa").prepend(faixa("erro", e.message));
  }
});

$("#sel-pessoa").addEventListener("change", carregarAmostras);
$("#recarregar-pessoas").addEventListener("click", recarregarListasDePessoas);

/* ----------------------------------------------------------- ao vivo ----- */

const previewVivo = criarAtualizador($("#vivo"), "/live.jpg", 600);

$("#vivo-ligado").addEventListener("change", (ev) => {
  if (ev.target.checked) previewVivo.iniciar();
  else previewVivo.parar();
  atualizarEstadoVivo();
});

function atualizarEstadoVivo() {
  $("#vivo-estado").textContent = !$("#vivo-ligado").checked ? "pausado"
    : previewVivo.falhas ? "sem imagem — o worker está rodando?" : "";
}
setInterval(atualizarEstadoVivo, 1500);

$("#recarregar-rec").addEventListener("click", carregarDeteccoes);
$("#filtro-pessoa").addEventListener("change", carregarDeteccoes);
$("#limite").addEventListener("change", carregarDeteccoes);
$("#dia-rec").addEventListener("change", carregarDeteccoes);
$("#usar-dia-rec").addEventListener("change", (ev) => {
  $("#dia-rec").disabled = !ev.target.checked;
  carregarDeteccoes();
});

iniciar();
