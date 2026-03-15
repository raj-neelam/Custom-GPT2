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
  const DEBOUNCE_DELAY = 1000;   // ms after typing → fetch ghost text

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
    let p = getPrompt();
    let cont = ghostSuggestion;
    // ensure space if needed
    if (p.length > 0 && !/\\s$/.test(p) && !/^\\s/.test(cont) && !/^[.,;?!'"]/.test(cont)) {
      cont = " " + cont;
    }
    editor.innerText = p + cont;
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
  // Predict — ghost text
  // ─────────────────────────────────────────────────────────────
  async function predict() {
    const rawPrompt = getPrompt();
    if (!rawPrompt.trim() || !apiReady) {
      clearGhost();
      return;
    }

    const prompt = rawPrompt;

    try {
      const res = await fetch(PROXY_URL + "/generate", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: 25,
          temperature: getTemp(),
        }),
      });

      if (!res.ok) throw new Error("Predict failed");
      const data = await res.json();

      if (data.error) {
        console.warn("Predict warning:", data.error, data.detail);
        return;
      }

      let newText = data.generated_text.slice(prompt.length);
      if (newText) {
        setGhost(prompt, newText);
      } else {
        clearGhost();
      }

    } catch (e) {
      console.warn("Predict error:", e);
    }
  }

  function schedulePredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(predict, DEBOUNCE_DELAY);
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
