"""FastAPI service exposing the product cards ranker.

    POST /rank-product-cards  {"query": "...", "top_k": 5, "min_relevance": 0.1}
        -> [{document_id, product_name, product_summary, relevance}, ...]

    GET    /catalog/cards            list card ids
    PUT    /catalog/cards/{card_id}  add / replace one card   (hot, zero-downtime)
    DELETE /catalog/cards/{card_id}  remove one card          (hot, zero-downtime)
    GET    /health

The catalog (products.json) is the single source of truth. Every request
checks the file's mtime; if it changed (edited by hand, by the catalog
endpoints, or by a nightly rebuild) the in-memory index is rebuilt in place
and the query cache is dropped. Adds/deletes never need a redeploy.

Run:
    uvicorn app:app --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ranker import ProductCardRanker

ROOT = Path(__file__).parent
CATALOG_PATH = Path(os.environ.get("CATALOG_PATH", ROOT / "products.json"))
EVAL_JSON = ROOT / "eval_results.json"
HOLDOUT_JSON = ROOT / "eval_holdout.json"
DASHBOARD_HTML = ROOT / "static" / "index.html"
CACHE_SIZE = int(os.environ.get("RANK_CACHE_SIZE", "4096"))
_LATENCY_WINDOW = 500  # last N request latencies kept for the dashboard

log = logging.getLogger("ranker")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="DSK Smart Search — Product Cards Ranker", version="0.2.0")


# --------------------------------------------------------------------------- #
# Catalog state: ranker + mtime, guarded by a lock for rebuilds.
# Reads are lock-free (the ranker is immutable after _build).
# --------------------------------------------------------------------------- #
class _Catalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = Lock()
        self.ranker: Optional[ProductCardRanker] = None
        self.mtime: float = 0.0

    def get(self) -> tuple[ProductCardRanker, float]:
        mtime = self.path.stat().st_mtime
        if self.ranker is None or mtime != self.mtime:
            with self.lock:
                if self.ranker is None or mtime != self.mtime:
                    t0 = time.perf_counter()
                    self.ranker = ProductCardRanker.from_json(self.path)
                    self.mtime = mtime
                    _cached_rank.cache_clear()
                    log.info("catalog (re)loaded: %d cards in %.0f ms", len(self.ranker), (time.perf_counter() - t0) * 1000)
        return self.ranker, self.mtime

    def read_cards(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write_cards(self, cards: list[dict]) -> None:
        """Atomic replace: write to a temp file, then rename over products.json."""
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


catalog = _Catalog(CATALOG_PATH)
_latencies: list[float] = []  # ring buffer of recent request latencies (ms)


# The frontend fires the same prefixes from many users; the cache key includes
# the catalog mtime, so a reload can never serve stale results.
@lru_cache(maxsize=CACHE_SIZE)
def _cached_rank(query: str, top_k: int, min_relevance: float, _mtime: float) -> tuple:
    ranker, _ = catalog.get()
    return tuple(tuple(d.items()) for d in ranker.rank(query, top_k=top_k, min_relevance=min_relevance))


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class RankRequest(BaseModel):
    query: str = Field(..., description="Raw user query — sent as typed, no preprocessing.")
    top_k: int = Field(5, ge=1, le=50)
    min_relevance: float = Field(0.1, ge=0.0, le=1.0, description="Drop cards below this relevance.")


class RankedItem(BaseModel):
    document_id: int
    product_name: str
    product_summary: str
    relevance: float


class CardIn(BaseModel):
    """Minimal card. Everything beyond these fields is optional — see products.json."""
    product_name_bg: str = ""
    product_name_en: str = ""
    summary_bg: str = ""
    summary_en: str = ""
    document_ids: list[int]
    name_variants: list[str] = []
    aliases: list[str] = []
    category: str = ""
    keywords: list[str] = []
    text_bg: str = ""
    text_en: str = ""
    priority_boost: float = 1.0
    exact_match_pins: list[str] = []


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, Any]:
    ranker, mtime = catalog.get()
    return {"status": "ok", "cards_indexed": len(ranker), "catalog_mtime": mtime, "cache": _cached_rank.cache_info()._asdict()}


@app.post("/rank-product-cards", response_model=list[RankedItem])
def rank_product_cards(req: RankRequest, response: Response, request: Request) -> list[dict]:
    t0 = time.perf_counter()
    _, mtime = catalog.get()  # hot-reload check happens BEFORE the cache lookup
    items = [dict(t) for t in _cached_rank(req.query, req.top_k, req.min_relevance, mtime)]
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Latency-Ms"] = f"{ms:.3f}"
    _latencies.append(ms)
    if len(_latencies) > _LATENCY_WINDOW:
        del _latencies[: len(_latencies) - _LATENCY_WINDOW]
    # Structured request log: this is the data that feeds boost/alias tuning.
    log.info(json.dumps({
        "q": req.query, "k": req.top_k, "ms": round(ms, 3),
        "top": [(i["document_id"], i["relevance"]) for i in items[:5]],
    }, ensure_ascii=False))
    return items


# --------------------------------------------------------------------------- #
# Dashboard (GET /) + the read-only endpoints it uses
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    if not DASHBOARD_HTML.exists():
        raise HTTPException(404, "dashboard not built: static/index.html missing")
    return DASHBOARD_HTML.read_text(encoding="utf-8")


@app.get("/analyze")
def analyze(q: str = Query(..., description="Raw query")) -> dict:
    """How the ranker interpreted each query token (exact / prefix / translit / fuzzy)."""
    ranker, _ = catalog.get()
    from ranker import _TOKEN_RE, _STOP_BG, _STOP_EN  # debug view only

    raws = [t.lower() for t in _TOKEN_RE.findall(q)]
    raws = [t for t in raws if t not in _STOP_BG and t not in _STOP_EN] or raws
    groups = ranker.analyze(q)
    return {"query": q, "tokens": [
        {"token": r, "matches": [{"term": m.term, "quality": m.quality} for m in g]}
        for r, g in zip(raws, groups)
    ]}


@app.get("/eval-results")
def eval_results(set: str = Query("dev", pattern="^(dev|holdout)$")) -> dict:
    """dev = curated 73-query set (tuned against); holdout = 770 generated queries (never tuned against)."""
    path = HOLDOUT_JSON if set == "holdout" else EVAL_JSON
    if not path.exists():
        raise HTTPException(404, f"run: python eval.py {'--holdout ' if set == 'holdout' else ''}--json {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/catalog/stats")
def catalog_stats() -> dict:
    cards = catalog.read_cards()
    ranker, mtime = catalog.get()
    cats: dict[str, int] = {}
    for c in cards:
        head = (c.get("category") or "").split(" ")
        # first two breadcrumb words in BG, e.g. "кредитиране жилищни"
        bg = [w for w in head if any("а" <= ch <= "я" for ch in w)]
        key = " ".join(bg[:2]) or "other"
        cats[key] = cats.get(key, 0) + 1
    lat = sorted(_latencies)
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None  # noqa: E731
    return {
        "cards": len(cards), "documents": sum(len(c["document_ids"]) for c in cards),
        "flagships": [c["product_name_bg"] or c["product_name_en"] for c in cards if c.get("priority_boost", 1) > 1],
        "summary_source": {s: sum(1 for c in cards if c.get("summary_source") == s) for s in ("tagline", "first_sentence", "llm")},
        "by_category": sorted(cats.items(), key=lambda kv: -kv[1]),
        "catalog_mtime": mtime, "core_vocab": len(ranker._core_set), "vocab": len(ranker._idf),
        "latency": {"n": len(lat), "p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99), "recent": _latencies[-120:]},
        "cache": _cached_rank.cache_info()._asdict(),
    }


@app.get("/catalog/cards")
def list_cards() -> list[dict]:
    return [
        {"card_id": c["card_id"], "product_name_bg": c.get("product_name_bg"), "document_ids": c["document_ids"]}
        for c in catalog.read_cards()
    ]


@app.put("/catalog/cards/{card_id}", status_code=200)
def upsert_card(card_id: str, card: CardIn) -> dict:
    if not (card.product_name_bg or card.product_name_en):
        raise HTTPException(400, "product_name_bg or product_name_en is required")
    with catalog.lock:
        cards = catalog.read_cards()
        payload = card.model_dump() if hasattr(card, "model_dump") else card.dict()
        payload["card_id"] = card_id
        payload["canonical_key"] = card_id.replace("_", " ")
        payload.setdefault("name_variants", [])
        if not payload["name_variants"]:
            payload["name_variants"] = [n for n in (card.product_name_bg, card.product_name_en) if n]
        idx = next((i for i, c in enumerate(cards) if c["card_id"] == card_id), None)
        action = "updated" if idx is not None else "created"
        if idx is not None:
            cards[idx] = payload
        else:
            cards.append(payload)
        catalog.write_cards(cards)
    ranker, _ = catalog.get()
    return {"card_id": card_id, "action": action, "cards_indexed": len(ranker)}


@app.delete("/catalog/cards/{card_id}")
def delete_card(card_id: str) -> dict:
    with catalog.lock:
        cards = catalog.read_cards()
        kept = [c for c in cards if c["card_id"] != card_id]
        if len(kept) == len(cards):
            raise HTTPException(404, f"card {card_id!r} not found")
        catalog.write_cards(kept)
    ranker, _ = catalog.get()
    return {"card_id": card_id, "action": "deleted", "cards_indexed": len(ranker)}
