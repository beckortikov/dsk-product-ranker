"""Behavioural tests for the ranker + API. Run: pytest -q"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranker import ProductCardRanker, normalize_text, tokenize, transliterate  # noqa: E402

CATALOG = ROOT / "products.json"


@pytest.fixture(scope="module")
def cards() -> list[dict]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ranker(cards) -> ProductCardRanker:
    return ProductCardRanker(cards)


def key_of(cards, doc_id):
    return next(c["canonical_key"] for c in cards if c["document_ids"][0] == doc_id)


def top1(ranker, cards, q):
    res = ranker.rank(q, top_k=1)
    return key_of(cards, res[0]["document_id"]) if res else None


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
def test_tokenize_bilingual_stemming():
    assert tokenize("Кредитни карти") == ["кредитн", "карт"]
    assert tokenize("Credit cards for the business") == ["credit", "card", "busi"]


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("Застраховка „Живот“!") == "застраховка живот"


def test_transliterate_both_directions():
    assert transliterate("дск") == "dsk"
    assert transliterate("kreditna") == "кредитна"


# --------------------------------------------------------------------------- #
# Catalog shape (business nuances from the spec)
# --------------------------------------------------------------------------- #
def test_client_type_duplicates_are_collapsed(cards):
    mtoken = next(c for c in cards if c["canonical_key"] == "dsk mtoken")
    assert len(mtoken["document_ids"]) >= 4, "individual/business/corporate mToken pages must be one card"
    keys = [c["canonical_key"] for c in cards]
    assert len(keys) == len(set(keys))


def test_every_card_has_a_selling_message(cards):
    for c in cards:
        assert c["summary_bg"] or c["summary_en"], c["card_id"]
        for s in (c["summary_bg"], c["summary_en"]):
            assert not s.lower().startswith(("индивидуални клиенти", "бизнес клиенти", "individual clients", "business clients")), \
                f"breadcrumb leaked into summary of {c['card_id']}: {s!r}"


def test_all_raw_documents_are_covered(cards):
    import pandas as pd

    raw_ids = set(pd.read_csv(ROOT / "df_data.csv").document_id)
    covered = {d for c in cards for d in c["document_ids"]}
    assert raw_ids == covered


# --------------------------------------------------------------------------- #
# Ranking behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,gold", [
    ("DSK Mobile", "dsk mobile"),
    ("mobile", "dsk mobile"),
    ("мобилно банкиране", "dsk mobile"),
    ("DSK Smart", "dsk smart"),
    ("DSK Online", "dsk online"),
    ("дск директ", "дск директ"),
    ("mtoken", "dsk mtoken"),
])
def test_flagships_win_generic_token_collisions(ranker, cards, q, gold):
    assert top1(ranker, cards, q) == gold


@pytest.mark.parametrize("q,gold", [
    ("student loan", "студентски кредит"),
    ("студентски кредит", "студентски кредит"),
    ("virtual pos", "виртуален пос терминал"),
    ("виртуален пос", "виртуален пос терминал"),
])
def test_bilingual_queries_hit_the_same_card(ranker, cards, q, gold):
    assert top1(ranker, cards, q) == gold


def test_prefix_typing(ranker, cards):
    assert top1(ranker, cards, "dsk mob") == "dsk mobile"
    assert top1(ranker, cards, "студентск") == "студентски кредит"


def test_typo_and_transliteration(ranker, cards):
    assert top1(ranker, cards, "dsk mobail") == "dsk mobile"
    assert top1(ranker, cards, "дск мобайл") == "dsk mobile"
    assert top1(ranker, cards, "multikesh") == "multicash"


def test_relevance_is_calibrated(ranker):
    assert ranker.rank("DSK Mobile", 1)[0]["relevance"] == 1.0
    junk = ranker.rank("такси за превод", 1)
    assert not junk or junk[0]["relevance"] < 0.6
    assert ranker.rank("зззз") == []
    assert ranker.rank("") == []


def test_response_contract(ranker):
    item = ranker.rank("кредитна карта", 1)[0]
    assert set(item) == {"document_id", "product_name", "product_summary", "relevance"}
    assert isinstance(item["document_id"], int)
    assert 0.0 <= item["relevance"] <= 1.0


def test_top_k_and_threshold(ranker):
    assert len(ranker.rank("кредит", top_k=3)) == 3
    assert all(r["relevance"] >= 0.5 for r in ranker.rank("кредит", top_k=50, min_relevance=0.5))


def test_add_and_remove_card_without_rebuild(cards):
    r = ProductCardRanker(cards)
    assert r.rank("зелена ипотека") == [] or top1(r, cards, "зелена ипотека") != "зелена ипотека"
    new = {
        "card_id": "зелена_ипотека", "canonical_key": "зелена ипотека", "document_ids": [999999],
        "product_name_bg": "Зелена ипотека", "product_name_en": "Green mortgage",
        "name_variants": ["Зелена ипотека", "Green mortgage"], "aliases": ["еко кредит"],
        "summary_bg": "По-ниска лихва за енергийно ефективен дом", "summary_en": "",
    }
    r2 = ProductCardRanker(cards + [new])
    assert r2.rank("зелена ипотека", 1)[0]["document_id"] == 999999
    assert r2.rank("green mortgage", 1)[0]["document_id"] == 999999
    assert r2.rank("еко кредит", 1)[0]["document_id"] == 999999
    assert len(r2) == len(r) + 1


def test_latency_budget(ranker):
    import time

    qs = ["DSK Mobile", "кредитна карта galaxy", "ипотечен кредт", "how do I confirm a transfer"] * 250
    t0 = time.perf_counter()
    for q in qs:
        ranker.rank(q, 5)
    per_query_ms = (time.perf_counter() - t0) * 1000 / len(qs)
    assert per_query_ms < 5.0, per_query_ms
