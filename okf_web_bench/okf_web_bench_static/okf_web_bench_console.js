const LANE_LABEL = {
  flat: "Flat retrieval",
  structured: "Structured retrieval",
};

// Starter questions come from the development question set, never the held-out set.
const TEMPLATES = [
  "Which stores are supercenters with a pharmacy?",
  "Which stores belong to the Northeast region?",
  "Which promotions run at neighborhood stores?",
  "In which country is the supplier of Televisions based?",
  "How many square feet does Brightmart Riverside occupy?",
];

const ENCODER_ID = "BAAI/bge-base-en-v1.5";
const ENCODER_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a";
const GENERATOR_ID = "Qwen/Qwen3-0.6B";

const $ = (id) => document.getElementById(id);

const sessions = [];
let activeSessionId = null;
let trialCounter = 0;
let healthDevice = "cpu";
let healthCuda = false;
let corpusDoc = null;
let corpusHighlight = { flat: null, struct: null };

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
  })[c]);
}

function activeSession() {
  return sessions.find((s) => s.id === activeSessionId) ?? null;
}

function ensureSession() {
  if (activeSession()) return activeSession();
  return newSession("Comparison 1");
}

function newSession(title) {
  const session = {
    id: Date.now() + Math.random(),
    title,
    trials: [],
  };
  sessions.unshift(session);
  activeSessionId = session.id;
  renderSessions();
  renderTrials();
  return session;
}

function sessionMeta(session) {
  const gpuOn = $("gpu").dataset.on === "true";
  const n = session.trials.length;
  return `${n} trial${n === 1 ? "" : "s"} · GPU ${gpuOn ? "on" : "off"}`;
}

function renderSessions() {
  const list = $("runs");
  list.innerHTML = "";
  sessions.forEach((session) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    if (session.id === activeSessionId) btn.classList.add("on");
    btn.innerHTML = `<span class="t">${esc(session.title)}</span><span class="m">${esc(sessionMeta(session))}</span>`;
    btn.onclick = () => {
      activeSessionId = session.id;
      renderSessions();
      renderTrials();
    };
    li.appendChild(btn);
    list.appendChild(li);
  });
}

function gpuOn() {
  return $("gpu").dataset.on === "true";
}

function updateGpuState() {
  const on = gpuOn();
  const label = on && healthCuda ? `on · ${healthDevice}` : on ? "on · cpu" : "off · cpu";
  $("gpuState").textContent = label;
}

async function loadHealth() {
  try {
    const res = await fetch("/api/okf-web-bench/health");
    if (!res.ok) return;
    const h = await res.json();
    healthDevice = h.device || "cpu";
    healthCuda = Boolean(h.cuda);
    populateLaneSelects(h.arms);
    updateGpuState();
    const hint = $("hint");
    if (hint) {
      hint.textContent =
        `postgres ${h.postgres ? "ok" : "down"} · bge ${h.bge ? "ok" : "down"} · qwen ${h.qwen ? "ok" : "down"} · ${h.device} · enter to run`;
    }
  } catch {
    /* health optional on first paint */
  }
}

function populateLaneSelects(arms) {
  const list = arms?.length ? arms : ["flat", "structured"];
  const prevA = $("modeA").value;
  const prevB = $("modeB").value;
  [$("modeA"), $("modeB")].forEach((sel) => {
    sel.innerHTML = "";
    list.forEach((arm) => {
      const opt = document.createElement("option");
      opt.value = arm;
      opt.textContent = laneLabel(arm);
      sel.appendChild(opt);
    });
  });
  $("modeA").value = list.includes(prevA) ? prevA : list[0];
  $("modeB").value = list.includes(prevB) ? prevB : list[Math.min(1, list.length - 1)];
}

function currentConfig() {
  return {
    corpus: $("corpus").value,
    gpu: gpuOn(),
    device: gpuOn() && healthCuda ? healthDevice : "cpu",
    topK: +$("topk").value,
    laneA: $("modeA").value,
    laneB: $("modeB").value,
    encoder: { id: ENCODER_ID, revision: ENCODER_REVISION },
    generator: { id: GENERATOR_ID },
  };
}

function laneLabel(mode) {
  return LANE_LABEL[mode] || mode;
}

function formatSource(src) {
  const path = src.doc_path || "?";
  const cid = src.chunk_id != null ? `#${src.chunk_id}` : "";
  const cite = src.citation != null ? `[${src.citation}] ` : "";
  return `${cite}${path}${cid}`;
}

function renderMetrics(m) {
  if (!m) return "";
  const clip =
    m.clipped
      ? '<span class="clip"><b>yes</b> clipped</span>'
      : "<span><b>no</b> clipped</span>";
  return `
    <div class="metrics">
      <span><b>${Math.round(m.retrieve_ms ?? 0)}</b> retrieve ms</span>
      <span><b>${Math.round(m.pack_ms ?? 0)}</b> pack ms</span>
      <span><b>${Math.round(m.generate_ms ?? 0)}</b> generate ms</span>
      <span><b>${m.chunks ?? 0}</b> chunks</span>
      <span><b>${m.tokens_in ?? "—"}</b> ctx tokens</span>
      ${clip}
    </div>`;
}

function laneBody(lane) {
  if (lane.loading) {
    return '<div class="answer skeleton"><span></span><span></span><span></span></div>';
  }
  if (lane.fetchError) {
    return `<div class="answer"><p class="err">${esc(lane.fetchError)}</p></div>`;
  }
  const data = lane.data;
  if (!data) {
    return '<div class="answer skeleton"><span></span><span></span><span></span></div>';
  }
  if (data.error && !data.answer) {
    return `<div class="answer"><p class="err">${esc(data.error)}</p></div>${renderSources(data)}${renderMetrics(data.metrics)}`;
  }
  const answer = data.answer
    ? `<div class="answer"><p>${esc(data.answer)}</p></div>`
    : `<div class="answer"><p class="err">${esc(data.error || "no answer")}</p></div>`;
  return `${answer}${renderSources(data)}${renderMetrics(data.metrics)}`;
}

function renderSources(data) {
  const sources = data?.sources || [];
  if (!sources.length) return "";
  return `<div class="sources">${sources
    .map((s) => `<span class="src">${esc(formatSource(s))}</span>`)
    .join("")}</div>`;
}

function renderTrial(trial) {
  const el = document.createElement("article");
  el.className = "trial";
  el.id = `t${trial.id}`;
  const cfg = trial.config;
  el.innerHTML = `
    <div class="q">
      <span class="n">${String(trial.id).padStart(2, "0")}</span>
      <div>
        <h2>${esc(trial.question)}</h2>
        <div class="cfg">${esc(cfg.corpus)} · k=${cfg.topK} · gpu ${cfg.gpu ? "on" : "off"} · ${new Date(trial.ts).toLocaleTimeString()}</div>
      </div>
    </div>
    <div class="lanes">
      ${renderLane("A", trial.laneA, cfg.laneA)}
      ${renderLane("B", trial.laneB, cfg.laneB)}
    </div>
    <div class="verdict">
      <span class="lbl">Closer to correct</span>
      ${["A", "tie", "B"]
        .map(
          (v) =>
            `<button type="button" aria-pressed="${trial.verdict === v}" data-verdict="${v}">${v === "tie" ? "Tie" : "Lane " + v}</button>`
        )
        .join("")}
    </div>`;
  el.querySelectorAll("[data-verdict]").forEach((btn) => {
    btn.onclick = () => setVerdict(trial.id, btn.dataset.verdict);
  });
  return el;
}

function renderLane(key, lane, mode) {
  return `
    <section class="lane">
      <div class="lane-head lane-${key.toLowerCase()}-top">
        <span class="name">Lane ${key}</span>
        <span class="mode">${esc(laneLabel(mode))}</span>
      </div>
      ${laneBody(lane)}
    </section>`;
}

function setVerdict(id, v) {
  const session = activeSession();
  if (!session) return;
  const trial = session.trials.find((x) => x.id === id);
  if (!trial) return;
  trial.verdict = trial.verdict === v ? null : v;
  const node = $(`t${id}`);
  if (node) node.replaceWith(renderTrial(trial));
}

function renderTrials() {
  const container = $("trials");
  const session = activeSession();
  container.innerHTML = "";
  if (!session || !session.trials.length) {
    container.innerHTML = `
      <div class="empty" id="empty">
        <h1>Ask once, answer twice.</h1>
        <p>Every question runs through both retrieval paths with the same generator, so the only variable is how the context was found. Pick a starter below or write your own.</p>
      </div>`;
    return;
  }
  session.trials.forEach((t) => container.appendChild(renderTrial(t)));
}

async function runTrial() {
  const question = $("q").value.trim();
  if (!question) return;

  const session = ensureSession();
  if (session.trials.length === 0 && session.title.startsWith("Comparison")) {
    session.title = question.length > 42 ? question.slice(0, 42) + "…" : question;
    renderSessions();
  }

  const config = currentConfig();
  const trial = {
    id: ++trialCounter,
    question,
    ts: new Date().toISOString(),
    config,
    verdict: null,
    laneA: { loading: true, data: null, fetchError: null },
    laneB: { loading: true, data: null, fetchError: null },
  };
  session.trials.push(trial);
  $("q").value = "";
  renderTrials();
  $("stream").scrollTop = 1e6;
  renderSessions();

  try {
    const res = await fetch("/api/okf-web-bench/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        corpus: config.corpus,
        lane_a: config.laneA,
        lane_b: config.laneB,
        top_k: config.topK,
        gpu: config.gpu,
      }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      const msg = detail.detail || res.statusText || "compare failed";
      trial.laneA = { loading: false, data: null, fetchError: msg };
      trial.laneB = { loading: false, data: null, fetchError: msg };
    } else {
      const body = await res.json();
      trial.laneA = { loading: false, data: body.lane_a, fetchError: null };
      trial.laneB = { loading: false, data: body.lane_b, fetchError: null };
    }
  } catch (err) {
    const msg = err?.message || "network error";
    trial.laneA = { loading: false, data: null, fetchError: msg };
    trial.laneB = { loading: false, data: null, fetchError: msg };
  }

  const node = $(`t${trial.id}`);
  if (node) node.replaceWith(renderTrial(trial));
  renderSessions();
}

function downloadJSON() {
  const session = activeSession();
  const trials = (session?.trials || []).map((t) => ({
    question: t.question,
    timestamp: t.ts,
    verdict: t.verdict,
    config: { ...t.config },
    laneA: {
      mode: t.config.laneA,
      answer: t.laneA.data?.answer ?? null,
      error: t.laneA.data?.error ?? t.laneA.fetchError,
      sources: t.laneA.data?.sources ?? [],
      metrics: t.laneA.data?.metrics ?? null,
    },
    laneB: {
      mode: t.config.laneB,
      answer: t.laneB.data?.answer ?? null,
      error: t.laneB.data?.error ?? t.laneB.fetchError,
      sources: t.laneB.data?.sources ?? [],
      metrics: t.laneB.data?.metrics ?? null,
    },
  }));

  const payload = {
    exportedAt: new Date().toISOString(),
    config: currentConfig(),
    trials,
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `retrieval-bench-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  const h = $("hint");
  h.textContent = `exported ${trials.length} trial${trials.length === 1 ? "" : "s"}`;
}

function switchTab(tab) {
  const compare = tab === "compare";
  $("tabCompare").classList.toggle("on", compare);
  $("tabCorpus").classList.toggle("on", !compare);
  $("panelCompare").hidden = !compare;
  $("panelCorpus").hidden = compare;
  document.body.classList.toggle("corpus-mode", !compare);
  if (!compare) loadCorpusDocuments();
}

async function loadCorpusDocuments() {
  const corpus = $("corpusMap").value;
  const list = $("docList");
  list.innerHTML = '<div class="empty-detail">Loading…</div>';
  corpusDoc = null;
  corpusHighlight = { flat: null, struct: null };
  renderCorpusDetail();

  try {
    const res = await fetch(`/api/okf-web-bench/corpus-map?corpus=${encodeURIComponent(corpus)}`);
    if (!res.ok) throw new Error("failed to load documents");
    const body = await res.json();
    list.innerHTML = "";
    (body.documents || []).forEach((doc) => {
      const btn = document.createElement("button");
      btn.innerHTML = `<span class="path">${esc(doc.path)}</span><span class="meta">${doc.title ? esc(doc.title) + " · " : ""}${doc.n_flat} flat · ${doc.n_struct} struct${doc.identical ? '<span class="badge">identical</span>' : ""}</span>`;
      btn.onclick = () => selectDocument(corpus, doc.path, btn);
      list.appendChild(btn);
    });
    if (!body.documents?.length) {
      list.innerHTML = '<div class="empty-detail">No documents.</div>';
    }
  } catch (err) {
    list.innerHTML = `<div class="empty-detail">${esc(err.message)}</div>`;
  }
}

async function selectDocument(corpus, path, btn) {
  document.querySelectorAll("#docList button").forEach((b) => b.classList.remove("on"));
  btn.classList.add("on");
  corpusHighlight = { flat: null, struct: null };
  $("corpusDetail").innerHTML = '<div class="empty-detail">Loading document…</div>';

  try {
    const res = await fetch(
      `/api/okf-web-bench/corpus-map/doc?corpus=${encodeURIComponent(corpus)}&path=${encodeURIComponent(path)}`
    );
    if (!res.ok) throw new Error("document not found");
    corpusDoc = await res.json();
    renderCorpusDetail();
  } catch (err) {
    $("corpusDetail").innerHTML = `<div class="empty-detail">${esc(err.message)}</div>`;
  }
}

function highlightedChunkIds() {
  const flatOn = new Set();
  const structOn = new Set();
  if (!corpusDoc) return { flatOn, structOn };

  const overlaps = corpusDoc.overlap_ids || { flat: {}, struct: {} };

  if (corpusHighlight.flat != null) {
    flatOn.add(corpusHighlight.flat);
    const ids = overlaps.flat?.[String(corpusHighlight.flat)] || [];
    ids.forEach((id) => structOn.add(id));
  }
  if (corpusHighlight.struct != null) {
    structOn.add(corpusHighlight.struct);
    const ids = overlaps.struct?.[String(corpusHighlight.struct)] || [];
    ids.forEach((id) => flatOn.add(id));
  }
  return { flatOn, structOn };
}

function renderChunkButton(chunk, policy, highlightSets) {
  const { flatOn, structOn } = highlightSets;
  const highlight =
    policy === "flat" ? flatOn.has(chunk.chunk_id) : structOn.has(chunk.chunk_id);

  const btn = document.createElement("button");
  if (highlight) btn.classList.add("on");
  const heading =
    policy === "struct" && chunk.heading_path?.length
      ? chunk.heading_path.join(" › ") + " · "
      : "";
  const meta =
    policy === "flat"
      ? `ord ${chunk.ord} · ${chunk.n_tokens} BGE tokens`
      : `ord ${chunk.ord}`;
  btn.innerHTML = `<div class="chunk-meta">${esc(heading)}${esc(meta)}</div><div class="chunk-text">${esc(chunk.text)}</div>`;
  btn.onclick = () => {
    if (policy === "flat") {
      corpusHighlight = { flat: chunk.chunk_id, struct: null };
    } else {
      corpusHighlight = { flat: null, struct: chunk.chunk_id };
    }
    renderCorpusDetail();
  };
  return btn;
}

function renderCorpusDetail() {
  const root = $("corpusDetail");
  if (!corpusDoc) {
    root.innerHTML = '<div class="empty-detail">Select a document to compare flat vs struct chunks.</div>';
    return;
  }

  root.innerHTML = `
    <div class="split-cols">
      <div class="split-col">
        <h3 class="flat">Flat chunks</h3>
        <ul class="chunk-list" id="flatChunks"></ul>
      </div>
      <div class="split-col">
        <h3 class="struct">Struct chunks</h3>
        <ul class="chunk-list" id="structChunks"></ul>
      </div>
    </div>
    <div class="edges-panel">
      <h3>Authored edges</h3>
      <ul class="edges-list" id="edgesList"></ul>
    </div>`;

  const highlightSets = highlightedChunkIds();

  const flatList = $("flatChunks");
  (corpusDoc.flat || []).forEach((ch) => {
    const li = document.createElement("li");
    li.appendChild(renderChunkButton(ch, "flat", highlightSets));
    flatList.appendChild(li);
  });

  const structList = $("structChunks");
  (corpusDoc.struct || []).forEach((ch) => {
    const li = document.createElement("li");
    li.appendChild(renderChunkButton(ch, "struct", highlightSets));
    structList.appendChild(li);
  });

  const edgesList = $("edgesList");
  const edges = corpusDoc.edges || [];
  if (!edges.length) {
    edgesList.innerHTML = "<li>No authored edges for this document.</li>";
  } else {
    edges.forEach((e) => {
      const li = document.createElement("li");
      li.textContent = `${e.src_path || e.path || "?"} → ${e.dst_path || "?"} (${e.edge_kind || "edge"})`;
      edgesList.appendChild(li);
    });
  }
}

function init() {
  const tpl = $("templates");
  TEMPLATES.forEach((t) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = t;
    b.onclick = () => {
      $("q").value = t;
      $("q").focus();
    };
    tpl.appendChild(b);
  });

  $("gpuSwitch").onclick = () => {
    const on = $("gpu").dataset.on !== "true";
    $("gpu").dataset.on = on;
    $("gpuSwitch").setAttribute("aria-checked", on);
    updateGpuState();
  };

  $("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runTrial();
    }
  });

  $("newRun").onclick = () => {
    newSession(`Comparison ${sessions.length + 1}`);
  };

  $("runBtn").onclick = () => runTrial();
  $("downloadBtn").onclick = () => downloadJSON();

  $("tabCompare").onclick = () => switchTab("compare");
  $("tabCorpus").onclick = () => switchTab("corpus");

  $("corpusMap").onchange = () => loadCorpusDocuments();

  newSession("Comparison 1");
  loadHealth();
  switchTab("compare");
}

document.addEventListener("DOMContentLoaded", init);
