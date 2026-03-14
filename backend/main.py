"""
FastAPI Backend for GPT-2 Text Generation
Endpoints:
  GET  /health           — model & system status
  POST /generate         — full text generation
  POST /predict          — top-10 next token probabilities
"""

import os
import time
import torch
import tiktoken
import torch.nn.functional as F
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import GPT, GPTConfig, load_model


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "../Models/GPT2-Pretrained.pth")

app_state: dict = {
    "model":        None,
    "enc":          None,
    "device":       "cpu",
    "model_loaded": False,
    "load_error":   None,
    "load_time":    None,
}


# ---------------------------------------------------------------------------
# Lifespan — load model at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, device = load_model(MODEL_PATH, device)
        enc = tiktoken.get_encoding("gpt2")

        app_state["model"]        = model
        app_state["enc"]          = enc
        app_state["device"]       = device
        app_state["model_loaded"] = True
        app_state["load_time"]    = round(time.time() - t0, 2)
        print(f"✅ Model loaded on {device} in {app_state['load_time']}s")
    except Exception as e:
        app_state["load_error"] = str(e)
        print(f"❌ Model load failed: {e}")
    yield
    # cleanup
    app_state["model"] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GPT-2 Text Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt:         str   = Field(..., min_length=1, description="Input text prompt")
    max_new_tokens: int   = Field(50,  ge=1,  le=500,  description="Number of new tokens to generate")
    temperature:    float = Field(0.9, ge=0.1, le=2.0, description="Sampling temperature")
    top_k:          int   = Field(50,  ge=1,  le=200,  description="Top-k filtering")


class PredictRequest(BaseModel):
    prompt:      str   = Field(..., min_length=1, description="Input text context")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    n_predict:   int   = Field(10,  ge=1,  le=50,   description="Number of predictions to return")


class TokenPrediction(BaseModel):
    word:        str
    probability: float


class GenerateResponse(BaseModel):
    generated_text: str
    prompt_tokens:  int
    new_tokens:     int
    device:         str


class PredictResponse(BaseModel):
    top_predictions: list[TokenPrediction]
    device:          str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    device:       str
    model_path:   str
    load_time_s:  float | None
    error:        str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_model():
    if not app_state["model_loaded"] or app_state["model"] is None:
        err = app_state.get("load_error", "Model not loaded")
        raise HTTPException(status_code=503, detail=f"Model unavailable: {err}")


@torch.no_grad()
def _generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> tuple[str, int]:
    model: GPT = app_state["model"]
    enc         = app_state["enc"]
    device      = app_state["device"]

    tokens = enc.encode(prompt)
    if len(tokens) == 0:
        tokens = [enc.encode(" ")[0]]  # fallback single space token

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = x.shape[1]

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature

        # top-k filtering
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)

    generated = enc.decode(x[0].tolist())
    new_tokens = x.shape[1] - prompt_len
    return generated, new_tokens


@torch.no_grad()
def _predict(prompt: str, temperature: float, n: int) -> list[TokenPrediction]:
    model: GPT = app_state["model"]
    enc         = app_state["enc"]
    device      = app_state["device"]

    tokens = enc.encode(prompt)
    if len(tokens) == 0:
        tokens = [enc.encode(" ")[0]]

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    x_cond = x[:, -model.config.block_size:]

    logits, _ = model(x_cond)
    logits = logits[:, -1, :] / temperature
    probs  = F.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, n)
    top_probs   = top_probs[0].tolist()
    top_indices = top_indices[0].tolist()

    results = []
    for prob, idx in zip(top_probs, top_indices):
        word = enc.decode([idx])
        results.append(TokenPrediction(word=word, probability=round(prob, 6)))

    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    """Check if the model is loaded and the backend is healthy."""
    return HealthResponse(
        status       = "ok" if app_state["model_loaded"] else "error",
        model_loaded = app_state["model_loaded"],
        device       = app_state["device"],
        model_path   = MODEL_PATH,
        load_time_s  = app_state.get("load_time"),
        error        = app_state.get("load_error"),
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(req: GenerateRequest):
    """Generate text continuation from a prompt."""
    _require_model()
    try:
        enc = app_state["enc"]
        prompt_tokens = len(enc.encode(req.prompt))
        generated, new_tokens = _generate(
            req.prompt, req.max_new_tokens, req.temperature, req.top_k
        )
        return GenerateResponse(
            generated_text = generated,
            prompt_tokens  = prompt_tokens,
            new_tokens     = new_tokens,
            device         = app_state["device"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest):
    """Return top-N next token predictions with probabilities."""
    _require_model()
    try:
        predictions = _predict(req.prompt, req.temperature, req.n_predict)
        return PredictResponse(
            top_predictions = predictions,
            device          = app_state["device"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
