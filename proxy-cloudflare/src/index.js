/**
 * Cloudflare Worker — HuggingFace Inference Proxy
 *
 * Routes:
 *   POST /generate  → Chat Completions adapted for raw text continuation
 *   POST /predict   → Chat Completions adapted for top-1 next word
 *   GET  /health    → { status: "ok", model: "..." }
 *
 * Secret:  HF_TOKEN  (set via `wrangler secret put HF_TOKEN`)
 *
 * Model: meta-llama/Llama-3.2-1B-Instruct
 *   - Used via the new v1/chat/completions endpoint
 *   - Replaces older models completely unsupported by the free serverless router
 */

const HF_URL = "https://router.huggingface.co/v1/chat/completions";
const MODEL  = "meta-llama/Llama-3.2-1B-Instruct";

// ─── CORS Headers ──────────────────────────────────────────────
function corsHeaders(origin) {
  const allow =
    origin && (origin.includes("github.io") || origin.includes("localhost") || origin.includes("127.0.0.1"))
      ? origin
      : "*";
  return {
    "Access-Control-Allow-Origin":  allow,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age":       "86400",
  };
}

function jsonResponse(data, status = 200, origin = "*") {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders(origin),
    },
  });
}

// ─── Main Handler ───────────────────────────────────────────────
export default {
  async fetch(request, env) {
    const url    = new URL(request.url);
    const origin = request.headers.get("Origin") || "*";

    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    // ── Health check ──────────────────────────────────────────
    if (url.pathname === "/health" && request.method === "GET") {
      return jsonResponse({ status: "ok", model: MODEL }, 200, origin);
    }

    // ── Generate — full text continuation ─────────────────────
    if (url.pathname === "/generate" && request.method === "POST") {
      let body;
      try { body = await request.json(); } catch {
        return jsonResponse({ error: "Invalid JSON body" }, 400, origin);
      }

      const { prompt = "", max_new_tokens = 50, temperature = 0.9 } = body;

      if (!prompt.trim()) {
        return jsonResponse({ error: "prompt is required" }, 422, origin);
      }

      // Translate text continuation request into a Chat Completion payload
      const chatPayload = {
        model: MODEL,
        messages: [
          { role: "system", content: "You are a text continuation engine. Output ONLY the perfect logical continuation of the user's text. Do not add conversational replies, prefixes, formatting, or explanations." },
          { role: "user", content: prompt }
        ],
        max_tokens: Math.min(max_new_tokens, 500),
        temperature: Math.max(0.01, temperature),
        top_p: 0.95,
      };

      try {
        const hfRes = await fetch(HF_URL, {
          method:  "POST",
          headers: {
            Authorization:  `Bearer ${env.HF_TOKEN}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(chatPayload),
        });

        if (!hfRes.ok) {
          const errText = await hfRes.text();
          return jsonResponse({ error: `HF API error: ${hfRes.status}`, detail: errText }, hfRes.status, origin);
        }

        const hfData = await hfRes.json();
        const newText = hfData.choices?.[0]?.message?.content ?? "";
        
        // Frontend expects the full generated text including the original prompt
        const generatedFull = prompt + newText;
        const newTokens = newText.trim().split(/\s+/).filter(Boolean).length;

        return jsonResponse({ generated_text: generatedFull, new_tokens: newTokens }, 200, origin);

      } catch (err) {
        return jsonResponse({ error: "Proxy error", detail: err.message }, 500, origin);
      }
    }

    // ── Predict — generate 1 token → return as top-1 prediction ──
    if (url.pathname === "/predict" && request.method === "POST") {
      let body;
      try { body = await request.json(); } catch {
        return jsonResponse({ error: "Invalid JSON body" }, 400, origin);
      }

      const { prompt = "", temperature = 0.9 } = body;

      if (!prompt.trim()) {
        return jsonResponse({ top_predictions: [] }, 200, origin);
      }

      // Translate top-k/next-token prediction into a 1-token Chat Completion
      const chatPayload = {
        model: MODEL,
        messages: [
          { role: "system", content: "You are a text continuation engine. Output ONLY the very next word to continue the user's text. No explanations." },
          { role: "user", content: prompt }
        ],
        max_tokens: 1,
        temperature: Math.max(0.01, temperature),
        top_p: 0.95,
      };

      try {
        const hfRes = await fetch(HF_URL, {
          method:  "POST",
          headers: {
            Authorization:  `Bearer ${env.HF_TOKEN}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(chatPayload),
        });

        if (!hfRes.ok) {
          const errText = await hfRes.text();
          return jsonResponse({ error: `HF API error: ${hfRes.status}`, detail: errText }, hfRes.status, origin);
        }

        const hfData = await hfRes.json();
        const nextWord = hfData.choices?.[0]?.message?.content ?? "";

        // Frontend relies on top_predictions array
        const predictions = nextWord
          ? [{ word: nextWord, probability: 1.0, logprob: 0 }]
          : [];

        return jsonResponse({ top_predictions: predictions }, 200, origin);

      } catch (err) {
        return jsonResponse({ error: "Proxy error", detail: err.message }, 500, origin);
      }
    }

    // ── 404 ────────────────────────────────────────────────────
    return jsonResponse({ error: "Not found" }, 404, origin);
  },
};
