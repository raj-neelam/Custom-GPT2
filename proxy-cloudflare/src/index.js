/**
 * Cloudflare Worker — HuggingFace Inference Proxy
 *
 * Routes:
 *   POST /generate  → text-generation (full continuation)
 *   POST /predict   → text-generation with details + top_n_tokens=10
 *   GET  /health    → always returns { status: "ok", model: "gpt2" }
 *
 * Secret:  HF_TOKEN  (set via `wrangler secret put HF_TOKEN`)
 * CORS:    allows GitHub Pages origin OR * for testing
 */

const HF_BASE = "https://api-inference.huggingface.co/models";
const MODEL   = "gpt2";  // change to your fine-tuned model if needed, e.g. "username/my-gpt2"

// ─── CORS Headers ──────────────────────────────────────────────
function corsHeaders(origin) {
  const allow =
    origin && (origin.includes("github.io") || origin.includes("localhost"))
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

      const { prompt = "", max_new_tokens = 50, temperature = 0.9, top_k = 50 } = body;

      if (!prompt.trim()) {
        return jsonResponse({ error: "prompt is required" }, 422, origin);
      }

      const hfPayload = {
        inputs: prompt,
        parameters: {
          max_new_tokens:     Math.min(max_new_tokens, 500),
          temperature:        Math.max(0.01, temperature),
          top_k:              top_k,
          do_sample:          temperature > 0.01,
          return_full_text:   true,
        },
      };

      try {
        const hfRes = await fetch(`${HF_BASE}/${MODEL}`, {
          method:  "POST",
          headers: {
            Authorization:  `Bearer ${env.HF_TOKEN}`,
            "Content-Type": "application/json",
            "x-wait-for-model": "true",   // wait if model is loading (cold start)
          },
          body: JSON.stringify(hfPayload),
        });

        if (!hfRes.ok) {
          const errText = await hfRes.text();
          return jsonResponse({ error: `HF API error: ${hfRes.status}`, detail: errText }, hfRes.status, origin);
        }

        const hfData = await hfRes.json();
        // HF returns: [{ generated_text: "..." }]
        const generatedFull = hfData[0]?.generated_text ?? "";
        const generatedNew  = generatedFull.slice(prompt.length);
        const newTokens     = generatedNew.split(/\s+/).filter(Boolean).length;

        return jsonResponse({
          generated_text: generatedFull,
          new_tokens:     newTokens,
        }, 200, origin);

      } catch (err) {
        return jsonResponse({ error: "Proxy error", detail: err.message }, 500, origin);
      }
    }

    // ── Predict — top-10 next tokens with probabilities ────────
    if (url.pathname === "/predict" && request.method === "POST") {
      let body;
      try { body = await request.json(); } catch {
        return jsonResponse({ error: "Invalid JSON body" }, 400, origin);
      }

      const { prompt = "", temperature = 0.9, n_predict = 10 } = body;

      if (!prompt.trim()) {
        return jsonResponse({ top_predictions: [] }, 200, origin);
      }

      /**
       * Strategy to get top-N token probabilities:
       * We request 1 new token with details=true and top_n_tokens=10.
       * The HF TGI inference API returns each token's alternatives
       * with logprob values. We convert logprob → probability.
       *
       * Note: the free HF inference API (non-TGI) may not always
       * return details. We handle both response shapes gracefully.
       */
      const hfPayload = {
        inputs: prompt,
        parameters: {
          max_new_tokens: 1,
          temperature:    Math.max(0.01, temperature),
          do_sample:      temperature > 0.01,
          details:        true,
          top_n_tokens:   Math.min(n_predict, 10),
          return_full_text: false,
        },
      };

      try {
        const hfRes = await fetch(`${HF_BASE}/${MODEL}`, {
          method:  "POST",
          headers: {
            Authorization:  `Bearer ${env.HF_TOKEN}`,
            "Content-Type": "application/json",
            "x-wait-for-model": "true",
          },
          body: JSON.stringify(hfPayload),
        });

        if (!hfRes.ok) {
          const errText = await hfRes.text();
          return jsonResponse({ error: `HF API error: ${hfRes.status}`, detail: errText }, hfRes.status, origin);
        }

        const hfData = await hfRes.json();

        // ── Parse response ──────────────────────────────────
        // Shape A (TGI details): hfData[0].details.prefill + .tokens[0].top_tokens
        // Shape B (no details):  hfData[0].generated_text only
        let predictions = [];

        const details = hfData[0]?.details;
        if (details && details.tokens && details.tokens[0]?.top_tokens) {
          // Convert logprobs to probabilities
          const topTokens = details.tokens[0].top_tokens;
          const logprobs  = topTokens.map(t => t.logprob);
          const maxLogp   = Math.max(...logprobs);
          const exps      = logprobs.map(lp => Math.exp(lp - maxLogp));
          const sumExps   = exps.reduce((a, b) => a + b, 0);

          predictions = topTokens.map((t, i) => ({
            word:        t.text,
            probability: exps[i] / sumExps,
            logprob:     t.logprob,
          }));

          // Sort descending by probability
          predictions.sort((a, b) => b.probability - a.probability);
        } else {
          // Fallback: we only have the top-1 generated token
          const generatedText = hfData[0]?.generated_text ?? "";
          if (generatedText) {
            predictions = [{
              word:        generatedText,
              probability: 1.0,
              logprob:     0,
            }];
          }
        }

        return jsonResponse({ top_predictions: predictions.slice(0, 10) }, 200, origin);

      } catch (err) {
        return jsonResponse({ error: "Proxy error", detail: err.message }, 500, origin);
      }
    }

    // ── 404 ────────────────────────────────────────────────────
    return jsonResponse({ error: "Not found" }, 404, origin);
  },
};
