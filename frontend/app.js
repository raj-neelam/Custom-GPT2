/**
 * GPT-2 Writer — Frontend Application
 * Handles: editor, ghost text, prediction sidebar, generate, health polling
 */

(function () {
  "use strict";

  // ─────────────────────────────────────────────────────────────
  // Config
  // ─────────────────────────────────────────────────────────────
  const API_BASE        = "/api";          // nginx proxies /api → backend:8000
  const DEBOUNCE_DELAY  = 450;            // ms after typing to call /predict
  const HEALTH_INTERVAL = 8000;           // ms between health polls
  const GHOST_WORDS     = 5;             // number of ghost words to show

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
  const topKInput    = document.getElementById("topk-input");
  const predList     = document.getElementById("predictions-list");
  const statusBadge  = document.getElementById("status-badge");
  const statusLabel  = document.getElementById("status-label");
  const deviceLabel  = document.getElementById("device-label");
  const tempBadge    = document.getElementById("predict-temp-badge");
  const toast        = document.getElementById("toast");

  // ─────────────────────────────────────────────────────────────
  // State
  // ─────────────────────────────────────────────────────────────
  let ghostSuggestion  = "";      // current ghost suggestion string
  let debounceTimer    = null;
  let isGenerating     = false;
  let modelReady       = false;

  // ─────────────────────────────────────────────────────────────
  // Utilities
  // ─────────────────────────────────────────────────────────────
  function getPrompt() {
    return editor.innerText || "";
  }

  function getTemp()   { return parseFloat(tempRange.value); }
  function getMaxTok() { return parseInt(maxTokens.value, 10) || 50; }
  function getTopK()   { return parseInt(topKInput.value, 10) || 50; }

  let toastTimer = null;
  function showToast(msg, type = "") {
    toast.textContent = msg;
    toast.className   = "toast show" + (type ? " " + type : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.className = "toast"; }, 3200);
  }

  // ─────────────────────────────────────────────────────────────
  // Health polling
  // ─────────────────────────────────────────────────────────────
  async function checkHealth() {
    try {
      const res  = await fetch(API_BASE + "/health");
      const data = await res.json();
      if (data.model_loaded) {
        setStatus("ok", "Model ready");
        deviceLabel.textContent = data.device.toUpperCase();
        deviceLabel.hidden      = false;
        modelReady = true;
      } else {
        setStatus("error", data.error || "Model not loaded");
        modelReady = false;
      }
    } catch {
      setStatus("error", "Backend offline");
      modelReady = false;
    }
  }

  function setStatus(type, label) {
    statusBadge.className = "status-badge status-" + type;
    statusLabel.textContent = label;
  }

  checkHealth();
  setInterval(checkHealth, HEALTH_INTERVAL);

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
  function setGhost(prompt, continuation) {
    ghostSuggestion = continuation;
    // overlay: prompt visible text + ghost text
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
    // Append ghost text to editor content
    const current = getPrompt();
    editor.innerText = current + ghostSuggestion;
    // Move caret to end
    moveCursorToEnd(editor);
    clearGhost();
    // Trigger new prediction
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

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\n/g, "<br>");
  }

  // ─────────────────────────────────────────────────────────────
  // Predict — top-10 next tokens + ghost text
  // ─────────────────────────────────────────────────────────────
  async function predict() {
    const prompt = getPrompt().trim();
    if (!prompt || !modelReady) {
      renderPredictionsPlaceholder();
      clearGhost();
      return;
    }

    try {
      const res = await fetch(API_BASE + "/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          temperature: getTemp(),
          n_predict:   10,
        }),
      });

      if (!res.ok) throw new Error("Predict failed");
      const data = await res.json();
      renderPredictions(data.top_predictions);

      // Build ghost text from top-N words
      if (data.top_predictions.length > 0) {
        // greedily pick top word GHOST_WORDS times for ghost suggestion
        const ghost = pickGhostWords(data.top_predictions.slice(0, 3), GHOST_WORDS);
        setGhost(getPrompt(), ghost);
      }
    } catch (e) {
      console.warn("Predict error:", e);
    }
  }

  /**
   * Build a multi-word ghost suggestion by repeatedly using the #1 token.
   * We just string together top-1 tokens as a simple lookahead.
   */
  async function pickGhostWords(top3, n) {
    // For a fast ghost we just use the top word repeated-predicted naively
    // We'll chain n calls but limit via a quick predict on growing prompts
    let accumulated = "";
    let currentPrompt = getPrompt();

    for (let i = 0; i < n; i++) {
      accumulated += top3[0]?.word ?? "";
      currentPrompt += top3[0]?.word ?? "";
      if (i < n - 1) {
        try {
          const res = await fetch(API_BASE + "/predict", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: currentPrompt, temperature: getTemp(), n_predict: 1 }),
          });
          if (!res.ok) break;
          const d = await res.json();
          top3 = d.top_predictions;
          if (!top3.length) break;
        } catch { break; }
      }
    }
    setGhost(getPrompt(), accumulated);
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

  function renderPredictions(predictions) {
    if (!predictions || predictions.length === 0) {
      renderPredictionsPlaceholder();
      return;
    }

    const maxProb = predictions[0].probability;

    predList.innerHTML = predictions.map((p, i) => {
      const pct      = maxProb > 0 ? ((p.probability / maxProb) * 100).toFixed(1) : 0;
      const probPct  = (p.probability * 100).toFixed(2);
      const rankClass= i < 3 ? `rank-${i + 1}` : "";
      const wordDisplay = p.word.replace(/\n/g, "↵").replace(/\t/g, "→") || "⏎";

      return `<div
          class="pred-item ${rankClass}"
          style="--prob-w:${pct}%; animation-delay:${i * 30}ms"
          data-word="${escapeAttr(p.word)}"
          title="Click to insert '${escapeAttr(p.word)}'"
          role="button"
          tabindex="0"
        >
          <span class="pred-rank">${i + 1}</span>
          <span class="pred-word">${escapeHtml(wordDisplay)}</span>
          <span class="pred-prob">${probPct}%</span>
        </div>`;
    }).join("");

    // Click to insert word
    predList.querySelectorAll(".pred-item").forEach(el => {
      el.addEventListener("click", () => {
        insertWord(el.dataset.word);
      });
      el.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") insertWord(el.dataset.word);
      });
    });
  }

  function insertWord(word) {
    clearGhost();
    const current = getPrompt();
    editor.innerText = current + word;
    moveCursorToEnd(editor);
    schedulePredict();
    editor.focus();
  }

  function escapeAttr(s) {
    return s.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // ─────────────────────────────────────────────────────────────
  // Generate (full continuation)
  // ─────────────────────────────────────────────────────────────
  async function generate() {
    if (isGenerating) return;
    if (!modelReady) {
      showToast("Model not ready yet", "error");
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

    try {
      const res = await fetch(API_BASE + "/generate", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: getMaxTok(),
          temperature:    getTemp(),
          top_k:          getTopK(),
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Generate failed");
      }

      const data = await res.json();
      // Replace editor content with full generated text
      editor.innerText = data.generated_text;
      moveCursorToEnd(editor);
      showToast(`Generated ${data.new_tokens} new tokens`, "success");
      schedulePredict();
    } catch (e) {
      showToast(e.message || "Generation failed", "error");
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
    // Accept ghost: Tab or ArrowRight at end of content
    if ((e.key === "Tab" || e.key === "ArrowRight") && ghostSuggestion) {
      e.preventDefault();
      acceptGhost();
      return;
    }
    // Generate: Ctrl+Enter
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      generate();
      return;
    }
  });

  // Paste: strip formatting
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

  // Keyboard shortcut: Ctrl+Shift+X → clear
  document.addEventListener("keydown", e => {
    if (e.ctrlKey && e.shiftKey && e.key === "X") {
      clearBtn.click();
    }
  });

  // ─────────────────────────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────────────────────────
  editor.focus();
})();
