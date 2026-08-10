/* Seeds Before Objectives — interactive companion, vanilla JS */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const el = (t, c, h) => { const n = document.createElement(t); if (c) n.className = c; if (h != null) n.innerHTML = h; return n; };
  const ACCENT = "#b5442e", SLATE = "#3f5b6b", GREEN = "#4d7a5a", GOLD = "#c69749", INK = "#22201c", INKSOFT = "#6a6459", LINE = "#e8e0d2";

  document.addEventListener("DOMContentLoaded", () => {
    $("#thesis-text").textContent = DATA.thesis;
    renderFindings();
    renderModels();
    renderSignificance();
    renderPower();
    renderHighlights();
    renderErrors();
    renderSetup();
    renderLimitations();
    drawSeedChart();
    buildObjToggle();
    drawProbeChart();
    setupReveal();
  });

  function renderFindings() {
    const g = $("#findings-grid");
    DATA.findings.forEach(f => g.appendChild(el("div", "finding",
      `<div class="fn">FINDING 0${f.n}</div><h3>${f.title}</h3><p>${f.detail}</p>`)));
  }

  function renderModels() {
    const tb = $("#model-table tbody");
    DATA.models.forEach(m => {
      const note = m.note ? `<span class="tag-note">${m.note}</span>` : "";
      tb.appendChild(el("tr", m.ours ? "ours" : null,
        `<td>${m.name}${m.ours ? '<span class="tag-ours">OURS</span>' : ''}${note}</td>
         <td>${m.params}</td>
         <td class="wer">${m.wer.toFixed(2)} <span style="color:${INKSOFT};font-size:.85em">±${m.std.toFixed(2)}</span></td>`));
    });
  }

  function renderSignificance() {
    const tb = $("#sig-body");
    DATA.significance.forEach(s => tb.appendChild(el("tr", null,
      `<td>${s.pair}</td><td style="font-family:var(--mono)">${s.meanDiff}</td><td style="font-family:var(--mono)">${s.p}</td><td><span class="ns">not sig.</span></td>`)));
  }

  function renderPower() {
    $("#pw-have").textContent = "5";
    $("#pw-need").textContent = DATA.power.seedsNeeded;
    $("#pw-note").textContent = DATA.power.note;
  }

  function renderHighlights() {
    const o = DATA.objectives;
    const std = o.find(x => x.name.startsWith("Standard")), mat = o.find(x => x.name.startsWith("Matra"));
    $("#matra-big").innerHTML = `Standard <b style="color:var(--ink)">${std.matraErr}%</b> &nbsp;vs&nbsp; Matra-weighted <b style="color:var(--ink)">${mat.matraErr}%</b> &nbsp;matra error`;
    const a = DATA.augmentation;
    $("#aug-big").innerHTML = `${a.trainBefore} → <b>${a.trainAfter}</b> training clips`;
    $("#aug-note").textContent = a.effect + ` (${a.factors} speed perturbation.)`;
  }

  function renderErrors() {
    const box = $("#err-bars");
    const max = Math.max(...DATA.errorProfile.map(e => parseFloat(e.pct)));
    DATA.errorProfile.forEach(e => {
      const w = (parseFloat(e.pct) / max * 100).toFixed(1);
      box.appendChild(el("div", "errrow",
        `<span>${e.cat}</span><div class="bar" style="width:${w}%"></div><span class="pct">${e.pct}%</span>`));
    });
    $("#err-note").textContent = DATA.errorNote;
  }

  function renderSetup() {
    const g = $("#setup-grid");
    const s = DATA.setup;
    const rows = [
      ["Encoder", s.encoder], ["CTC head", s.head], ["Features", s.features],
      ["Decoding", s.decode], ["Seeds", s.seeds], ["Compute", s.compute]
    ];
    rows.forEach(([k, v]) => g.appendChild(el("div", "srow", `<div class="k">${k}</div><div class="v">${v}</div>`)));
  }

  function renderLimitations() {
    const u = $("#limitations");
    DATA.limitations.forEach(x => u.appendChild(el("li", null, x)));
  }

  /* ---- Seed scatter: the signature visual ---- */
  function drawSeedChart() {
    const p = DATA.perSeed;
    // x = seed index, y = WER; three series
    const mk = (arr, color) => arr.map((y, i) => ({ x: i, y }));
    const cfg = {
      type: "scatter",
      data: {
        datasets: [
          { label: "Standard CTC", data: mk(p.standard), backgroundColor: ACCENT, pointRadius: 7, pointHoverRadius: 9 },
          { label: "Focal CTC", data: mk(p.focal), backgroundColor: SLATE, pointRadius: 7, pointHoverRadius: 9 },
          { label: "Matra-weighted", data: mk(p.matra), backgroundColor: GOLD, pointRadius: 7, pointHoverRadius: 9 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: INK, font: { family: "'JetBrains Mono', monospace", size: 11 }, usePointStyle: true } },
          tooltip: {
            backgroundColor: INK, padding: 10,
            callbacks: { title: (it) => "Seed " + p.seeds[it[0].parsed.x], label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(2)}% WER` }
          }
        },
        scales: {
          x: {
            type: "linear",
            min: -0.5, max: 4.5,
            afterBuildTicks: (axis) => { axis.ticks = [0, 1, 2, 3, 4].map(v => ({ value: v })); },
            ticks: {
              color: INKSOFT,
              autoSkip: false,
              callback: (v) => (p.seeds[v] != null) ? "seed " + p.seeds[v] : "",
              font: { family: "'JetBrains Mono', monospace" }
            },
            grid: { color: "rgba(34,32,28,.05)" }
          },
          y: {
            title: { display: true, text: "Corpus WER (%)  ·  lower is better", color: INKSOFT, font: { size: 11 } },
            ticks: { color: INKSOFT, font: { family: "'JetBrains Mono', monospace" } },
            grid: { color: "rgba(34,32,28,.06)" }
          }
        }
      }
    };
    new Chart($("#seedChart"), cfg);
    $("#seed-caption").textContent = "The three objectives' per-seed WERs interleave — no clean separation. Holm-corrected paired tests confirm no pair differs significantly. This is exactly why a single run is not enough.";
  }

  /* ---- Objective toggle chart ---- */
  let objChart, objMode = "aug";
  function buildObjToggle() {
    const t = $("#obj-toggle");
    [["aug", "With speed augmentation"], ["noaug", "Without augmentation"]].forEach(([k, label], i) => {
      const b = el("button", "tbtn" + (i === 0 ? " active" : ""), label);
      b.onclick = () => { objMode = k; [...t.children].forEach(x => x.classList.remove("active")); b.classList.add("active"); drawObjChart(); };
      t.appendChild(b);
    });
    drawObjChart();
  }
  function drawObjChart() {
    const o = DATA.objectives;
    const labels = o.map(x => x.name);
    const vals = o.map(x => objMode === "aug" ? x.aug : x.noaug);
    const errs = o.map(x => objMode === "aug" ? x.augStd : x.noaugStd);
    const colors = [ACCENT, SLATE, GOLD];
    const cfg = {
      type: "bar",
      data: { labels, datasets: [{ data: vals, backgroundColor: colors, borderRadius: 6, barPercentage: 0.6 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { backgroundColor: INK, padding: 10, callbacks: { label: (c) => `${c.parsed.y.toFixed(2)}% WER ±${errs[c.dataIndex].toFixed(2)}` } }
        },
        scales: {
          x: { ticks: { color: INK, font: { family: "'JetBrains Mono', monospace", size: 11 } }, grid: { display: false } },
          y: {
            min: 45, max: objMode === "aug" ? 50 : 51,
            title: { display: true, text: "WER (%)  ·  lower is better", color: INKSOFT, font: { size: 11 } },
            ticks: { color: INKSOFT, font: { family: "'JetBrains Mono', monospace" } }, grid: { color: "rgba(34,32,28,.06)" }
          }
        }
      }
    };
    if (objChart) objChart.destroy();
    objChart = new Chart($("#objChart"), cfg);
  }

  /* ---- Probing line ---- */
  function drawProbeChart() {
    const p = DATA.probing;
    const cfg = {
      type: "line",
      data: {
        labels: p.layers,
        datasets: [
          { label: "Fine-tuned", data: p.finetuned, borderColor: ACCENT, backgroundColor: ACCENT, tension: .3, borderWidth: 3, pointRadius: 5 },
          { label: "Base (pretrained)", data: p.base, borderColor: SLATE, backgroundColor: SLATE, tension: .3, borderWidth: 3, pointRadius: 5, borderDash: [5, 4] }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: INK, font: { family: "'JetBrains Mono', monospace", size: 11 }, usePointStyle: true } },
          tooltip: { backgroundColor: INK, padding: 10, callbacks: { title: (it) => "Layer " + it[0].label, label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}% WER` } }
        },
        scales: {
          x: { title: { display: true, text: "Encoder layer", color: INKSOFT, font: { size: 11 } }, ticks: { color: INKSOFT, font: { family: "'JetBrains Mono', monospace" } }, grid: { color: "rgba(34,32,28,.05)" } },
          y: { title: { display: true, text: "Probe WER (%)", color: INKSOFT, font: { size: 11 } }, ticks: { color: INKSOFT, font: { family: "'JetBrains Mono', monospace" } }, grid: { color: "rgba(34,32,28,.06)" } }
        }
      }
    };
    new Chart($("#probeChart"), cfg);
    $("#probe-note").textContent = p.note;
  }

  function setupReveal() {
    const io = new IntersectionObserver((es) => {
      es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: .1 });
    document.querySelectorAll(".reveal").forEach(n => io.observe(n));
  }
})();
