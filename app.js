/**
 * GPT-2 Writer (HF Edition) — Frontend Application
 * Connects to Cloudflare Worker proxy → Hugging Face Inference API
 *
 * API_BASE  →  your deployed Cloudflare Worker URL
 *              e.g. "https://gpt2-proxy.YOUR-SUBDOMAIN.workers.dev"
 *
 * Change PROXY_URL before deploying / testing locally.
 */

(function () {
  "use strict";

  // ─────────────────────────────────────────────────────────────
  // ⚙️  Config — update PROXY_URL after deploying your worker
  // ─────────────────────────────────────────────────────────────
  const PROXY_URL      = "https://gpt2-proxy.singhggaurav14.workers.dev"; // ← CHANGE THIS
  const DEBOUNCE_DELAY = 500;   // ms after typing → call /predict
  const GHOST_WORDS    = 5;     // ghost lookahead words
  const TOP_N          = 10;    // sidebar predictions count

  // ─────────────────────────────────────────────────────────────
  // DOM refs
  // ─────────────────────────────────────────────────────────────
  const editor       = document.getElementById("editor");
  const ghostOverlay = document.getElementById("ghost-overlay");
  const generateBtn  = document.getElementById("generate-btn");
  const clearBtn     = document.getElementById("clear-btn");
  const tempRange    = document.getElementById("temperature-range");
  const tempValue    = document.getElementById("temp-value");
  const maxTokens    = document.getElementById("max-tokens-input");
  const predList     = document.getElementById("predictions-list");
  const statusBadge  = document.getElementById("status-badge");
  const statusLabel  = document.getElementById("status-label");
  const modelLabel   = document.getElementById("model-label");
  const tempBadge    = document.getElementById("predict-temp-badge");
  const toast        = document.getElementById("toast");

  // ─────────────────────────────────────────────────────────────
  // State
  // ─────────────────────────────────────────────────────────────
  let ghostSuggestion = "";
  let debounceTimer   = null;
  let isGenerating    = false;
  let apiReady        = false;

  // ─────────────────────────────────────────────────────────────
  // Utilities
  // ─────────────────────────────────────────────────────────────
  function getPrompt() { return editor.innerText || ""; }
  function getTemp()   { return parseFloat(tempRange.value); }
  function getMaxTok() { return parseInt(maxTokens.value, 10) || 50; }

  let toastTimer = null;
  function showToast(msg, type = "") {
    toast.textContent = msg;
    toast.className   = "toast show" + (type ? " " + type : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.className = "toast"; }, 3500);
  }

  function setStatus(type, label) {
    statusBadge.className   = "status-badge status-" + type;
    statusLabel.textContent = label;
  }

  // ─────────────────────────────────────────────────────────────
  // Health check — ping /health on startup
  // ─────────────────────────────────────────────────────────────
  async function checkHealth() {
    setStatus("loading", "Connecting…");
    try {
      const res  = await fetch(PROXY_URL + "/health");
      const data = await res.json();
      if (data.status === "ok") {
        setStatus("ok", "HF Ready");
        modelLabel.textContent = data.model || "meta-llama/Llama-3.2-1B-Instruct";
        modelLabel.hidden      = false;
        apiReady = true;
      } else {
        setStatus("error", "Proxy error");
      }
    } catch {
      setStatus("error", "Proxy offline");
      showToast("⚠ Proxy not reachable. Check PROXY_URL in app.js.", "error");
    }
  }

  checkHealth();

  // ─────────────────────────────────────────────────────────────
  // Temperature sync
  // ─────────────────────────────────────────────────────────────
  tempRange.addEventListener("input", () => {
    const v = parseFloat(tempRange.value).toFixed(2);
    tempValue.textContent = v;
    tempBadge.textContent = "t=" + v;
  });

  // ─────────────────────────────────────────────────────────────
  // Ghost text helpers
  // ─────────────────────────────────────────────────────────────
  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\n/g, "<br>");
  }

  function setGhost(prompt, continuation) {
    ghostSuggestion = continuation;
    ghostOverlay.innerHTML =
      "<span style='color:transparent'>" + escapeHtml(prompt) + "</span>" +
      "<span class='ghost-text' style='color:var(--ghost)'>" + escapeHtml(continuation) + "</span>";
  }

  function clearGhost() {
    ghostSuggestion = "";
    ghostOverlay.innerHTML = "";
  }

  function acceptGhost() {
    if (!ghostSuggestion) return;
    editor.innerText = getPrompt() + ghostSuggestion + " ";
    moveCursorToEnd(editor);
    clearGhost();
    schedulePredict();
  }

  function moveCursorToEnd(el) {
    const range = document.createRange();
    const sel   = window.getSelection();
    range.selectNodeContents(el);
    range.collapse(false);
    sel.removeAllRanges();
    sel.addRange(range);
  }

  // ─────────────────────────────────────────────────────────────
  // Predict — top-10 next tokens + ghost text
  // ─────────────────────────────────────────────────────────────
  async function predict() {
    const rawPrompt = getPrompt();
    if (!rawPrompt.trim() || !apiReady) {
      renderPredictionsPlaceholder();
      clearGhost();
      return;
    }

    const prompt = rawPrompt;

    try {
      const res = await fetch(PROXY_URL + "/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          temperature: getTemp(),
          n_predict:   TOP_N,
        }),
      });

      if (!res.ok) throw new Error("Predict failed");
      const data = await res.json();

      if (data.error) {
        console.warn("Predict warning:", data.error, data.detail);
        return;
      }

      const predictions = data.top_predictions || [];
      renderPredictions(predictions);

      // Build ghost text by chaining top-word predictions
      if (predictions.length > 0) {
        buildGhostChain(prompt, predictions[0].word, GHOST_WORDS);
      }

    } catch (e) {
      console.warn("Predict error:", e);
    }
  }

  /**
   * Builds a ghost suggestion by calling /predict N times,
   * each time appending the most probable next token.
   */
  async function buildGhostChain(prompt, firstWord, n) {
    let accumulated = firstWord;
    let currentPrompt = prompt;

    if (currentPrompt.length > 0 && !/\\s$/.test(currentPrompt) && !/^[.,;?!'"]/.test(firstWord)) {
      accumulated = " " + firstWord;
    }
    currentPrompt += accumulated;

    for (let i = 1; i < n; i++) {
      try {
        const res = await fetch(PROXY_URL + "/predict", {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt:      currentPrompt,
            temperature: getTemp(),
            n_predict:   1,
          }),
        });
        if (!res.ok) break;
        const d = await res.json();
        const top = d.top_predictions?.[0];
        if (!top) break;

        let nextWord = top.word;
        if (!/^[.,;?!'"]/.test(nextWord)) {
          nextWord = " " + nextWord;
        }

        accumulated    += nextWord;
        currentPrompt  += nextWord;
      } catch { break; }
    }

    setGhost(prompt, accumulated);
  }

  function schedulePredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(predict, DEBOUNCE_DELAY);
  }

  // ─────────────────────────────────────────────────────────────
  // Render predictions sidebar
  // ─────────────────────────────────────────────────────────────
  function renderPredictionsPlaceholder() {
    predList.innerHTML = `
      <div class="predictions-placeholder">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.2" opacity=".3"/>
          <path d="M12 7v5l3 3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" opacity=".5"/>
        </svg>
        <p>Start typing to see predictions…</p>
      </div>`;
  }

  function escapeAttr(s) {
    return s.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function renderPredictions(predictions) {
    if (!predictions || predictions.length === 0) {
      renderPredictionsPlaceholder();
      return;
    }

    const maxProb = predictions[0].probability;

    predList.innerHTML = predictions.map((p, i) => {
      const pct       = maxProb > 0 ? ((p.probability / maxProb) * 100).toFixed(1) : 0;
      const probPct   = (p.probability * 100).toFixed(2);
      const rankClass = i < 3 ? `rank-${i + 1}` : "";
      const wordDisp  = p.word.replace(/\n/g, "↵").replace(/\t/g, "→") || "⏎";

      return `<div
          class="pred-item ${rankClass}"
          style="--prob-w:${pct}%; animation-delay:${i * 30}ms"
          data-word="${escapeAttr(p.word)}"
          title="Click to insert '${escapeAttr(p.word)}'"
          role="button"
          tabindex="0"
        >
          <span class="pred-rank">${i + 1}</span>
          <span class="pred-word">${escapeHtml(wordDisp)}</span>
          <span class="pred-prob">${probPct}%</span>
        </div>`;
    }).join("");

    predList.querySelectorAll(".pred-item").forEach(el => {
      el.addEventListener("click", () => insertWord(el.dataset.word));
      el.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") insertWord(el.dataset.word);
      });
    });
  }

  function insertWord(word) {
    clearGhost();
    let p = getPrompt();
    if (p.length > 0 && !/\\s$/.test(p) && !/^[.,;?!'"]/.test(word)) {
      p += " ";
    }
    editor.innerText = p + word + " ";
    moveCursorToEnd(editor);
    schedulePredict();
    editor.focus();
  }

  // ─────────────────────────────────────────────────────────────
  // Generate — full continuation
  // ─────────────────────────────────────────────────────────────
  async function generate() {
    if (isGenerating) return;
    if (!apiReady) {
      showToast("Proxy not connected yet", "error");
      return;
    }

    const prompt = getPrompt().trim();
    if (!prompt) {
      showToast("Type something to generate from", "error");
      return;
    }

    isGenerating = true;
    generateBtn.classList.add("loading");
    generateBtn.disabled = true;
    clearGhost();
    setStatus("loading", "Generating…");

    try {
      const res = await fetch(PROXY_URL + "/generate", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: getMaxTok(),
          temperature:    getTemp(),
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Generate failed");
      }

      const data = await res.json();
      editor.innerText = data.generated_text;
      moveCursorToEnd(editor);
      showToast(`Generated ${data.new_tokens} new tokens ✓`, "success");
      schedulePredict();
      setStatus("ok", "HF Ready");

    } catch (e) {
      showToast(e.message || "Generation failed", "error");
      setStatus("ok", "HF Ready");
    } finally {
      isGenerating = false;
      generateBtn.classList.remove("loading");
      generateBtn.disabled = false;
    }
  }

  // ─────────────────────────────────────────────────────────────
  // Editor events
  // ─────────────────────────────────────────────────────────────
  editor.addEventListener("input", () => {
    clearGhost();
    schedulePredict();
  });

  editor.addEventListener("keydown", e => {
    if ((e.key === "Tab" || e.key === "ArrowRight") && ghostSuggestion) {
      e.preventDefault();
      acceptGhost();
      return;
    }
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      generate();
    }
  });

  editor.addEventListener("paste", e => {
    e.preventDefault();
    const text = (e.clipboardData || window.clipboardData).getData("text/plain");
    document.execCommand("insertText", false, text);
  });

  generateBtn.addEventListener("click", generate);

  clearBtn.addEventListener("click", () => {
    editor.innerText = "";
    clearGhost();
    renderPredictionsPlaceholder();
    editor.focus();
  });

  document.addEventListener("keydown", e => {
    if (e.ctrlKey && e.shiftKey && e.key === "X") clearBtn.click();
  });

  // ─────────────────────────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────────────────────────
  editor.focus();
})();
