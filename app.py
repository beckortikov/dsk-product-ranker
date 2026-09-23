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

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ranker import ProductCardRanker

CATALOG_PATH = Path(os.environ.get("CATALOG_PATH", Path(__file__).parent / "products.json"))
CACHE_SIZE = int(os.environ.get("RANK_CACHE_SIZE", "4096"))

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
    # Structured request log: this is the data that feeds boost/alias tuning.
    log.info(json.dumps({
        "q": req.query, "k": req.top_k, "ms": round(ms, 3),
        "top": [(i["document_id"], i["relevance"]) for i in items[:5]],
    }, ensure_ascii=False))
    return items


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
