"""
api.py
FastAPI — serveur de traduction FR → Bambara.
Démarrage : uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

# ── Modèles de requête/réponse ────────────────────────────────────────────────

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    input:   str
    bambara: str

# ── Initialisation unique au démarrage ───────────────────────────────────────

_engine: TranslationEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    db      = Neo4jClient()
    _engine = TranslationEngine(db)
    yield

# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "KumaGen MT",
    description = "Traduction français → bambara",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "engine": _engine is not None}


@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text est vide")

    # Le pipeline est synchrone — on l'exécute dans un thread pour ne pas
    # bloquer la boucle d'événements (important quand plusieurs requêtes arrivent).
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _engine.translate, req.text)

    return TranslationResponse(input=req.text, bambara=result["bambara"])
