// Interactive Client Logic for @AmazonHelp AI Support Agent

const presets = [
  "@AmazonHelp My package 112-9847291 was supposed to be delivered yesterday. The tracker says 'Out for delivery' since 8 AM yesterday. Any update?",
  "@AmazonHelp ALERT: Someone hacked into my Amazon account, changed the 2FA phone number and charged $2400 in digital gift cards! HELP ME SHUT IT DOWN NOW!",
  "@AmazonHelp I ordered a glass dining table and it arrived completely shattered into a million pieces inside the box. Box looked like it fell off a cliff.",
  "@AmazonHelp Does Kohl's still accept returns if I don't have the original brown shipping box?",
  "@AmazonHelp Why did you charge my credit card $149 for an annual Prime membership without asking me? I haven't used Amazon in 6 months! Refund it now.",
  "huge thanks to @AmazonHelp for sorting out my package so quickly yesterday! Best customer service ever! 🎉"
];

let globalGoldenData = [];

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initConsoleEvents();
  loadBenchmarkData();
  loadGoldenSet();
  loadHumanAgreementData();
});

// 1. Navigation Tabs
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const tabTitle = document.getElementById("tab-title");
  const tabDesc = document.getElementById("tab-desc");

  const tabMeta = {
    "live-console": {
      title: "Live Agent Console",
      desc: "Test real-time classification, triage decision reasoning, historical RAG retrieval, and grounded reply drafting."
    },
    "benchmark-view": {
      title: "Benchmark & Baselines Comparison",
      desc: "Quantitative evaluation across 200 hand-curated golden samples comparing Trivial, Simple, and Proposed systems."
    },
    "golden-set": {
      title: "Golden Evaluation Dataset (200 Samples)",
      desc: "Hand-curated, stratified benchmark dataset annotated with ground truth intent, escalation rules, and reference resolutions."
    },
    "human-judge": {
      title: "Human-Judge Agreement & Calibration",
      desc: "Inter-rater reliability analysis (Cohen's Kappa, Pearson r, Adjacent Agreement) across 50 human double-scored samples."
    },
    "failure-analysis": {
      title: "Top 5 Failure Modes & Hypotheses",
      desc: "Concrete failure traces, input examples, and architectural mitigation strategies."
    },
    "decision-log": {
      title: "Decision Log & Non-Obvious Choices",
      desc: "Key architectural decisions, trade-offs, and product justifications."
    }
  };

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetTab = item.getAttribute("data-tab");
      navItems.forEach(n => n.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      item.classList.add("active");
      document.getElementById(targetTab).classList.add("active");

      if (tabMeta[targetTab]) {
        tabTitle.textContent = tabMeta[targetTab].title;
        tabDesc.textContent = tabMeta[targetTab].desc;
      }
    });
  });
}

// 2. Preset loader
function loadPreset(idx) {
  if (presets[idx]) {
    const input = document.getElementById("tweet-input");
    input.value = presets[idx];
    runAgentPrediction();
  }
}

// 3. Live Console Events
function initConsoleEvents() {
  const btnSubmit = document.getElementById("btn-submit-tweet");
  const btnRerun = document.getElementById("btn-re-run-benchmark");

  btnSubmit.addEventListener("click", runAgentPrediction);
  btnRerun.addEventListener("click", triggerRerunBenchmark);
}

async function runAgentPrediction() {
  const input = document.getElementById("tweet-input");
  const text = input.value.trim();
  if (!text) return;

  const spinner = document.getElementById("loading-spinner");
  spinner.classList.remove("hidden");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_text: text })
    });
    const data = await res.json();
    renderPredictionResult(data);
  } catch (err) {
    console.error("Prediction error:", err);
  } finally {
    spinner.classList.add("hidden");
  }
}

function renderPredictionResult(data) {
  document.getElementById("latency-tag").textContent = `${data.processing_time_ms} ms`;
  document.getElementById("res-intent").textContent = data.intent;
  
  const confPct = Math.round(data.intent_confidence * 100);
  document.getElementById("res-confidence-bar").style.width = `${confPct}%`;
  document.getElementById("res-confidence-val").textContent = `${data.intent_confidence.toFixed(2)}`;

  // Triage
  const triageCard = document.getElementById("res-triage-card");
  const triageBadge = document.getElementById("res-triage-badge");
  const reasonCode = document.getElementById("res-reason-code");
  const reasonDetails = document.getElementById("res-reason-details");

  if (data.auto_handled) {
    triageCard.className = "triage-card auto-handled";
    triageBadge.className = "badge badge-success";
    triageBadge.textContent = "AUTO-HANDLED (BOT)";
  } else {
    triageCard.className = "triage-card";
    triageBadge.className = "badge badge-danger";
    triageBadge.textContent = "ESCALATE TO HUMAN";
  }

  reasonCode.textContent = data.escalation_reason_category;
  reasonDetails.textContent = data.escalation_reason_details;

  // Reply
  const draftReply = document.getElementById("res-draft-reply");
  draftReply.textContent = data.draft_reply;
  document.getElementById("char-count").textContent = `${data.draft_reply.length} / 280 chars`;

  // Retrieved context list
  const listContainer = document.getElementById("retrieved-list");
  listContainer.innerHTML = "";
  if (data.retrieved_resolutions && data.retrieved_resolutions.length > 0) {
    data.retrieved_resolutions.forEach((item, idx) => {
      const card = document.createElement("div");
      card.className = "retrieved-item";
      card.innerHTML = `
        <div class="retrieved-meta">
          <span>#${idx + 1} Match: ${item.intent}</span>
          <span>Similarity: ${(item.similarity_score * 100).toFixed(1)}%</span>
        </div>
        <p><strong>Query:</strong> "${item.historical_query}"</p>
        <p><strong>Reply:</strong> "${item.resolution_reply}"</p>
      `;
      listContainer.appendChild(card);
    });
  } else {
    listContainer.innerHTML = `<p class="hint-text">No historical resolution retrieved for this query.</p>`;
  }
}

// 4. Benchmark Data Loading
async function loadBenchmarkData() {
  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();
    renderBenchmarkTable(data.model_results);
  } catch (err) {
    console.error("Benchmark load error:", err);
  }
}

async function triggerRerunBenchmark() {
  const btn = document.getElementById("btn-re-run-benchmark");
  btn.innerHTML = "<span>⏳ Benchmarking...</span>";
  btn.disabled = true;

  try {
    const res = await fetch("/api/benchmark/run", { method: "POST" });
    const data = await res.json();
    renderBenchmarkTable(data.model_results);
    alert("Benchmark successfully completed across all 200 Golden Evaluation samples!");
  } catch (err) {
    console.error("Error running benchmark:", err);
  } finally {
    btn.innerHTML = "<span>🔄 Re-run Benchmark</span>";
    btn.disabled = false;
  }
}

function renderBenchmarkTable(results) {
  const tbody = document.getElementById("benchmark-table-body");
  tbody.innerHTML = "";

  const keys = Object.keys(results);
  const m1 = results[keys[0]];
  const m2 = results[keys[1]];
  const m3 = results[keys[2]];

  const rows = [
    { label: "Intent Classification Accuracy", v1: `${(m1.intent_metrics.accuracy * 100).toFixed(1)}%`, v2: `${(m2.intent_metrics.accuracy * 100).toFixed(1)}%`, v3: `${(m3.intent_metrics.accuracy * 100).toFixed(1)}%` },
    { label: "Intent Macro F1-Score", v1: m1.intent_metrics.macro_f1.toFixed(3), v2: m2.intent_metrics.macro_f1.toFixed(3), v3: m3.intent_metrics.macro_f1.toFixed(3) },
    { label: "Escalation Triage Precision", v1: `${(m1.escalation_metrics.precision * 100).toFixed(1)}%`, v2: `${(m2.escalation_metrics.precision * 100).toFixed(1)}%`, v3: `${(m3.escalation_metrics.precision * 100).toFixed(1)}%` },
    { label: "Escalation Triage Recall", v1: `${(m1.escalation_metrics.recall * 100).toFixed(1)}%`, v2: `${(m2.escalation_metrics.recall * 100).toFixed(1)}%`, v3: `${(m3.escalation_metrics.recall * 100).toFixed(1)}%` },
    { label: "Escalation Triage F1-Score", v1: m1.escalation_metrics.f1_score.toFixed(3), v2: m2.escalation_metrics.f1_score.toFixed(3), v3: m3.escalation_metrics.f1_score.toFixed(3) },
    { label: "ROUGE-L F1 (Lexical Match)", v1: m1.reply_quality.rougeL_f1.toFixed(3), v2: m2.reply_quality.rougeL_f1.toFixed(3), v3: m3.reply_quality.rougeL_f1.toFixed(3) },
    { label: "Official Link Adherence Rate", v1: `${(m1.reply_quality.official_link_rate * 100).toFixed(1)}%`, v2: `${(m2.reply_quality.official_link_rate * 100).toFixed(1)}%`, v3: `${(m3.reply_quality.official_link_rate * 100).toFixed(1)}%` },
    { label: "LLM Judge Composite Score", v1: `${m1.judge_quality.composite_score_out_of_5.toFixed(2)} / 5.0`, v2: `${m2.judge_quality.composite_score_out_of_5.toFixed(2)} / 5.0`, v3: `${m3.judge_quality.composite_score_out_of_5.toFixed(2)} / 5.0` },
    { label: "Mean Latency (ms)", v1: `${m1.latency.mean_ms.toFixed(1)} ms`, v2: `${m2.latency.mean_ms.toFixed(1)} ms`, v3: `${m3.latency.mean_ms.toFixed(1)} ms` }
  ];

  rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${r.label}</strong></td>
      <td>${r.v1}</td>
      <td>${r.v2}</td>
      <td class="highlight-col">${r.v3}</td>
    `;
    tbody.appendChild(tr);
  });

  // Render Rubric Bars
  const rubricContainer = document.getElementById("rubric-bars");
  rubricContainer.innerHTML = "";
  const dimNames = {
    "groundedness": "1. Groundedness",
    "actionability": "2. Actionability",
    "brand_tone": "3. Brand Tone",
    "safety_and_escalation": "4. Safety Triage",
    "conciseness": "5. Conciseness"
  };

  for (const [k, title] of Object.entries(dimNames)) {
    const score = m3.judge_quality.dimensions[k] || 4.5;
    const box = document.createElement("div");
    box.className = "rubric-dim-box";
    box.innerHTML = `
      <div class="rubric-dim-name">${title}</div>
      <div class="rubric-dim-score">${score.toFixed(2)}</div>
      <span class="hint-text">out of 5.0</span>
    `;
    rubricContainer.appendChild(box);
  }
}

// 5. Golden Set Loading & Filtering
async function loadGoldenSet() {
  try {
    const res = await fetch("/api/golden_set");
    globalGoldenData = await res.json();
    populateIntentFilter(globalGoldenData);
    renderGoldenTable(globalGoldenData);
    initFilterListeners();
  } catch (err) {
    console.error("Error loading golden set:", err);
  }
}

function populateIntentFilter(data) {
  const select = document.getElementById("filter-intent");
  const uniqueIntents = [...new Set(data.map(d => d.ground_truth_intent))];
  uniqueIntents.forEach(intent => {
    const opt = document.createElement("option");
    opt.value = intent;
    opt.textContent = intent;
    select.appendChild(opt);
  });
}

function initFilterListeners() {
  const fIntent = document.getElementById("filter-intent");
  const fEsc = document.getElementById("filter-escalate");
  const fTier = document.getElementById("filter-tier");

  const applyFilters = () => {
    let filtered = [...globalGoldenData];
    if (fIntent.value !== "ALL") {
      filtered = filtered.filter(d => d.ground_truth_intent === fIntent.value);
    }
    if (fEsc.value !== "ALL") {
      const isEsc = (fEsc.value === "TRUE");
      filtered = filtered.filter(d => d.ground_truth_escalate === isEsc);
    }
    if (fTier.value !== "ALL") {
      filtered = filtered.filter(d => d.difficulty_tier === fTier.value);
    }
    renderGoldenTable(filtered);
  };

  fIntent.addEventListener("change", applyFilters);
  fEsc.addEventListener("change", applyFilters);
  fTier.addEventListener("change", applyFilters);
}

function renderGoldenTable(data) {
  const tbody = document.getElementById("golden-set-tbody");
  tbody.innerHTML = "";

  data.slice(0, 50).forEach(row => {
    const tr = document.createElement("tr");
    const escBadge = row.ground_truth_escalate
      ? `<span class="badge badge-danger">Escalate</span>`
      : `<span class="badge badge-success">Auto-Handle</span>`;

    tr.innerHTML = `
      <td><code>${row.id}</code></td>
      <td>${row.customer_text}</td>
      <td><span class="pill-badge">${row.ground_truth_intent}</span></td>
      <td>${escBadge}</td>
      <td><small><code>${row.escalation_reason_category}</code></small></td>
      <td><small>${row.difficulty_tier}</small></td>
    `;
    tbody.appendChild(tr);
  });
}

// 6. Human Agreement Data
async function loadHumanAgreementData() {
  try {
    const res = await fetch("/api/human_agreement");
    const data = await res.json();
    if (data && data.dimension_breakdown) {
      document.getElementById("kpi-exact").textContent = `${data.overall_exact_agreement_pct}%`;
      document.getElementById("kpi-adj").textContent = `${data.overall_adjacent_agreement_pct}%`;
      document.getElementById("kpi-r").textContent = `${data.pearson_correlation_r}`;

      const tbody = document.getElementById("human-agreement-tbody");
      tbody.innerHTML = "";
      for (const [dim, info] of Object.entries(data.dimension_breakdown)) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${dim.replace(/_/g, " ").toUpperCase()}</strong></td>
          <td>${info.cohen_kappa.toFixed(3)}</td>
          <td>${info.quadratic_weighted_kappa.toFixed(3)}</td>
          <td>${info.exact_agreement_pct.toFixed(1)}%</td>
          <td class="highlight-col">${info.adjacent_agreement_pct.toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
      }
    }
  } catch (err) {
    console.error("Error loading human agreement:", err);
  }
}
