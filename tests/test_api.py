"""API tests: contract, hot-reload, add/delete endpoints. Run: pytest -q"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Each test gets its own copy of the catalog so the real one is untouched.
    catalog = tmp_path / "products.json"
    shutil.copy(ROOT / "products.json", catalog)
    monkeypatch.setenv("CATALOG_PATH", str(catalog))
    import app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app), catalog


def test_rank_contract(client):
    c, _ = client
    r = c.post("/rank-product-cards", json={"query": "DSK Mobile", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body) <= 3
    assert set(body[0]) == {"document_id", "product_name", "product_summary", "relevance"}
    assert body[0]["product_name"] == "DSK Mobile"
    assert body[0]["relevance"] == 1.0
    assert "X-Latency-Ms" in r.headers


def test_empty_and_junk_queries(client):
    c, _ = client
    assert c.post("/rank-product-cards", json={"query": "   "}).json() == []
    assert c.post("/rank-product-cards", json={"query": "зззз"}).json() == []
    assert c.post("/rank-product-cards", json={"query": "x", "top_k": 0}).status_code == 422


def test_health(client):
    c, _ = client
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["cards_indexed"] > 50


def test_hot_reload_after_manual_edit_even_for_cached_query(client):
    c, catalog = client
    q = {"query": "DSK Mobile", "top_k": 1}
    assert c.post("/rank-product-cards", json=q).json()[0]["product_name"] == "DSK Mobile"  # now cached
    cards = json.loads(catalog.read_text(encoding="utf-8"))
    for card in cards:
        if card["canonical_key"] == "dsk mobile":
            card["product_name_bg"] = "DSK Mobile (renamed)"
    catalog.write_text(json.dumps(cards, ensure_ascii=False), encoding="utf-8")
    os.utime(catalog, None)
    assert c.post("/rank-product-cards", json=q).json()[0]["product_name"] == "DSK Mobile (renamed)"


def test_add_then_delete_card_via_api(client):
    c, _ = client
    n0 = c.get("/health").json()["cards_indexed"]
    new = {
        "product_name_bg": "Зелена ипотека", "product_name_en": "Green mortgage",
        "summary_bg": "По-ниска лихва за енергийно ефективен дом", "document_ids": [999999],
        "aliases": ["еко кредит"],
    }
    r = c.put("/catalog/cards/zelena_ipoteka", json=new)
    assert r.status_code == 200 and r.json()["action"] == "created"
    assert c.get("/health").json()["cards_indexed"] == n0 + 1
    top = c.post("/rank-product-cards", json={"query": "green mortgage", "top_k": 1}).json()[0]
    assert top["document_id"] == 999999 and top["product_summary"].startswith("По-ниска")

    r = c.delete("/catalog/cards/zelena_ipoteka")
    assert r.status_code == 200 and r.json()["action"] == "deleted"
    assert c.get("/health").json()["cards_indexed"] == n0
    assert c.delete("/catalog/cards/zelena_ipoteka").status_code == 404
    top = c.post("/rank-product-cards", json={"query": "green mortgage", "top_k": 1}).json()
    assert not top or top[0]["document_id"] != 999999
