"""Evaluation harness for the product cards ranker.

No click logs exist, so the eval set is hand-curated from the spec's
concerns. Every query has a gold `canonical_key` (see products.json).

    python eval.py                 # our ranker + lexical ablations
    python eval.py --embeddings    # + multilingual embedding baseline (downloads a model)
    python eval.py --show-fails    # print misses

Metrics: Hit@1, Hit@3, MRR over positive queries; for negative
(out-of-catalog) queries we report the share correctly returned empty or
below the relevance threshold (`abstain`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ranker import ProductCardRanker

ROOT = Path(__file__).resolve().parent

# Any credit card is a correct #1 for a generic "credit card" query.
_CREDIT_CARDS = "кредитна карта galaxy|кредитна карта platinum|кредитна карта dsk shopping|кредитна карта dsk wizz air|кредитна карта visa business gold|visa business credit"

# (query, gold canonical_key(s) separated by "|", tag)
# tags: flagship | generic | product | prefix | xlang | typo | translit | long
EVAL: list[tuple[str, str, str]] = [
    # ---- flagship apps & collisions with the generic "mobile" / "online" tokens
    ("DSK Mobile", "dsk mobile", "flagship"),
    ("dsk mobile app", "dsk mobile", "flagship"),
    ("мобилно банкиране", "dsk mobile", "flagship"),
    ("mobile banking", "dsk mobile", "flagship"),
    ("mobile", "dsk mobile", "generic"),
    ("ново мобилно приложение", "dsk mobile", "generic"),
    ("DSK Smart", "dsk smart", "flagship"),
    ("дск смарт", "dsk smart", "flagship"),
    ("DSK Online", "dsk online", "flagship"),
    ("онлайн банкиране", "dsk online", "flagship"),
    ("online banking", "dsk online", "flagship"),
    ("дск директ", "дск директ", "flagship"),
    ("dsk direct", "дск директ", "flagship"),
    ("DSK Business", "dsk business", "flagship"),
    ("mtoken", "dsk mtoken", "flagship"),
    ("мтокен", "dsk mtoken", "flagship"),
    ("потвърждаване на преводи", "dsk mtoken", "generic"),
    # ---- prefix / mid-typing (frontend fires every 10 chars)
    ("dsk mob", "dsk mobile", "prefix"),
    ("dsk sma", "dsk smart", "prefix"),
    ("ипотечен кре", "ипотечен кредит за покупка на недвижим имот", "prefix"),
    ("студентск", "студентски кредит", "prefix"),
    ("кредитна карта gal", "кредитна карта galaxy", "prefix"),
    ("virtual po", "виртуален пос терминал", "prefix"),
    # ---- ordinary products, BG
    ("кредитна карта galaxy", "кредитна карта galaxy", "product"),
    ("студентски кредит", "студентски кредит", "product"),
    ("кредит за рефинансиране", "кредит за рефинансиране", "product"),
    ("ипотечен кредит", "ипотечен кредит за покупка на недвижим имот", "product"),
    ("детски влог", "детски влог", "product"),
    ("овърдрафт", "кредит овърдрафт", "product"),
    ("виртуален пос", "виртуален пос терминал", "product"),
    ("структурирани продукти", "структурирани продукти", "product"),
    ("застраховка живот", "застраховка живот с protectinvest", "product"),
    ("застраховка за пътуване", "застраховка моето пътуване|застраховка помощ при пътуване", "product"),
    ("стандартна разплащателна сметка", "стандартна разплащателна сметка", "product"),
    ("сметка за деца", "сметки за деца и тийнейджъри", "product"),
    ("потребителски кредит", "стандартен потребителски кредит|потребителски кредит онлайн", "product"),
    ("кредит онлайн", "потребителски кредит онлайн", "product"),
    ("wizz air", "кредитна карта dsk wizz air", "product"),
    ("банкови гаранции", "банкови гаранции", "product"),
    ("валутен суап", "валутен суап", "product"),
    # ---- ordinary products, EN (foreign clients)
    ("credit card galaxy", "кредитна карта galaxy", "xlang"),
    ("student loan", "студентски кредит", "xlang"),
    ("mortgage refinancing", "кредит за рефинансиране", "xlang"),
    ("visa platinum", "дебитна карта visa platinum", "xlang"),
    ("visa infinite", "дебитна карта visa infinite debit", "xlang"),
    ("multicash", "multicash", "xlang"),
    ("home insurance", "застраховка любим дом стандарт", "xlang"),
    ("life insurance", "застраховка живот с protectinvest", "xlang"),
    ("travel insurance", "застраховка моето пътуване|застраховка помощ при пътуване", "xlang"),
    ("structured products", "структурирани продукти", "xlang"),
    ("virtual pos", "виртуален пос терминал", "xlang"),
    ("children savings account", "детски влог", "xlang"),
    ("overdraft", "кредит овърдрафт", "xlang"),
    ("bank guarantees", "банкови гаранции", "xlang"),
    ("interest rate swap", "лихвен суап", "xlang"),
    ("donation account", "дарителска сметка дск пулс", "xlang"),
    # ---- cross-language: query language ≠ the only language the page exists in
    ("mobile app for individuals", "dsk mobile", "xlang"),   # DSK Mobile page is BG-only
    ("dsk online banking", "dsk online", "xlang"),           # DSK Online page is BG-only
    ("депозит", "безсрочен влог", "xlang"),                   # BG page says "влог", EN says "deposit"
    # ---- typos & transliteration
    ("кредитна крата", _CREDIT_CARDS, "typo"),
    ("credit crad galaxy", "кредитна карта galaxy", "typo"),
    ("ипотечен кредт", "ипотечен кредит за покупка на недвижим имот", "typo"),
    ("dsk mobail", "dsk mobile", "typo"),
    ("mtokn", "dsk mtoken", "typo"),
    ("дск мобайл", "dsk mobile", "translit"),
    ("dsk direkt", "дск директ", "translit"),
    ("студентски кредит", "студентски кредит", "translit"),
    ("kreditna karta", _CREDIT_CARDS, "translit"),
    ("multikesh", "multicash", "translit"),
    # ---- natural-language questions (user presses Enter → agent, but we still rank)
    ("как да си открия сметка за дете", "сметки за деца и тийнейджъри", "long"),
    ("искам кредит за покупка на апартамент", "ипотечен кредит за покупка на недвижим имот|кредит за покупка и довършителни работи", "long"),
    ("how do I confirm a transfer on my phone", "dsk mtoken", "long"),
    ("card with no annual fee", "кредитна карта galaxy", "long"),
]

# Queries with no matching product: the ranker should return nothing above threshold.
NEGATIVES: list[str] = [
    "зззз", "asdfgh", "пица", "weather tomorrow", "bitcoin",
    "кола под наем", "самолетни билети", "12345", "hello", "здравей",
]

def _by_doc(cards: list[dict]) -> dict[int, str]:
    return {c["document_ids"][0]: c["canonical_key"] for c in cards}


def evaluate(rank_fn, cards: list[dict], *, min_relevance: float = 0.1, show_fails: bool = False) -> dict:
    by = _by_doc(cards)
    hit1 = hit3 = 0
    rr = 0.0
    per_tag: dict[str, list[int]] = {}
    fails = []
    for q, gold, tag in EVAL:
        res = rank_fn(q, 10)
        keys = [by.get(r["document_id"], "") for r in res]
        golds = set(gold.split("|"))
        rank = next((i + 1 for i, k in enumerate(keys) if k in golds), 0)
        hit1 += rank == 1
        hit3 += 1 <= rank <= 3
        rr += 1.0 / rank if rank else 0.0
        per_tag.setdefault(tag, []).append(int(rank == 1))
        if rank != 1:
            fails.append((q, gold, rank, keys[:3]))
    abstain = 0
    for q in NEGATIVES:
        res = rank_fn(q, 3)
        if not res or res[0]["relevance"] < min_relevance:
            abstain += 1
        elif show_fails:
            print(f"   neg   {q!r:38} returned {[(by.get(r['document_id'], '')[:24], r['relevance']) for r in res[:2]]}")
    n = len(EVAL)
    out = {
        "n": n, "hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": rr / n,
        "abstain": abstain / len(NEGATIVES),
        "by_tag": {t: sum(v) / len(v) for t, v in per_tag.items()},
    }
    if show_fails:
        for q, gold, rank, top in fails:
            print(f"   miss  {q!r:38} gold={gold[:32]!r:36} rank={rank} top3={top}")
    return out


def latency(rank_fn, n: int = 2000) -> dict:
    qs = [q for q, _, _ in EVAL]
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        rank_fn(qs[i % len(qs)], 5)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"p50_ms": statistics.median(ts), "p95_ms": ts[int(0.95 * n)], "p99_ms": ts[int(0.99 * n)]}


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def lexical_ablation(cards: list[dict], *, boosts: bool, fuzzy: bool) -> ProductCardRanker:
    """Same engine with flagship config and/or query-fallbacks switched off."""
    import ranker as R

    stripped = []
    for c in cards:
        c = dict(c)
        if not boosts:
            c["priority_boost"] = 1.0
            c["exact_match_pins"] = []
            c["aliases"] = [v.lower() for v in c.get("name_variants", [])]
        stripped.append(c)
    r = ProductCardRanker(stripped)
    if not fuzzy:
        r._fuzzy_matches = lambda stem: []  # type: ignore[assignment]
        r._prefix_matches = lambda stem: []  # type: ignore[assignment]
        R_transliterate = R.transliterate
        R.transliterate = lambda tok: tok  # type: ignore[assignment]
        r._restore = lambda: setattr(R, "transliterate", R_transliterate)  # type: ignore[attr-defined]
    return r


class EmbeddingRanker:
    """Reference point only (spec: 'ok to use as a point of reference, not in
    the final ranker'). Multilingual sentence embeddings, cosine over
    name + summary + first 1 500 chars of body per card."""

    def __init__(self, cards: list[dict], model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        import numpy as np
        from fastembed import TextEmbedding

        self.np = np
        self.model = TextEmbedding(model_name=model)
        self.cards = cards
        docs = [
            " | ".join(filter(None, [
                " / ".join(c.get("name_variants", [])), c.get("summary_bg", ""), c.get("summary_en", ""),
                (c.get("text_bg", "") + " " + c.get("text_en", ""))[:1500],
            ]))
            for c in cards
        ]
        cache = ROOT / ".cache" / f"emb_{abs(hash((model, tuple(docs))))}.npy"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            self.doc_emb = np.load(cache)
        else:
            self.doc_emb = np.array(list(self.model.embed(docs)))
            np.save(cache, self.doc_emb)
        self.doc_emb /= np.linalg.norm(self.doc_emb, axis=1, keepdims=True)

    def rank(self, query: str, top_k: int = 5) -> list[dict]:
        q = self.np.array(list(self.model.embed([query])))[0]
        q /= self.np.linalg.norm(q)
        sims = self.doc_emb @ q
        order = self.np.argsort(-sims)[:top_k]
        return [
            {"document_id": self.cards[i]["document_ids"][0], "product_name": self.cards[i]["product_name_bg"],
             "product_summary": self.cards[i]["summary_bg"], "relevance": round(float(sims[i]), 4)}
            for i in order
        ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(ROOT / "products.json"))
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--show-fails", action="store_true")
    ap.add_argument("--md", default=None, help="write results table to this markdown file")
    args = ap.parse_args()

    cards = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    rows = []

    systems = [
        ("BM25F only (no boosts, no prefix/typo/translit)", lambda: lexical_ablation(cards, boosts=False, fuzzy=False)),
        ("BM25F + prefix/typo/translit", lambda: lexical_ablation(cards, boosts=False, fuzzy=True)),
        ("Ours: + curated aliases, pins, boosts", lambda: ProductCardRanker(cards)),
    ]
    for name, make in systems:
        r = make()
        print(f"\n== {name}")
        m = evaluate(lambda q, k: r.rank(q, k, min_relevance=0.0), cards, show_fails=args.show_fails)
        lat = latency(lambda q, k: r.rank(q, k))
        if hasattr(r, "_restore"):
            r._restore()
        rows.append((name, m, lat))
        print(f"   Hit@1 {m['hit@1']:.1%}  Hit@3 {m['hit@3']:.1%}  MRR {m['mrr']:.3f}  abstain {m['abstain']:.0%}  "
              f"p50 {lat['p50_ms']:.2f} ms  p95 {lat['p95_ms']:.2f} ms")
        print("   by tag:", {t: f"{v:.0%}" for t, v in m["by_tag"].items()})

    if args.embeddings:
        print("\n== Embedding baseline (paraphrase-multilingual-MiniLM-L12-v2, cosine)")
        e = EmbeddingRanker(cards)
        m = evaluate(e.rank, cards, min_relevance=0.45, show_fails=args.show_fails)
        lat = latency(e.rank, n=200)
        rows.append(("Embeddings (multilingual MiniLM, reference only)", m, lat))
        print(f"   Hit@1 {m['hit@1']:.1%}  Hit@3 {m['hit@3']:.1%}  MRR {m['mrr']:.3f}  abstain {m['abstain']:.0%}  "
              f"p50 {lat['p50_ms']:.2f} ms  p95 {lat['p95_ms']:.2f} ms")
        print("   by tag:", {t: f"{v:.0%}" for t, v in m["by_tag"].items()})

    if args.md:
        tags = sorted({t for _, _, t in EVAL})
        lines = ["| system | Hit@1 | Hit@3 | MRR | abstain (neg) | p50 ms | p95 ms | " + " | ".join(tags) + " |",
                 "|---|---|---|---|---|---|---|" + "---|" * len(tags)]
        for name, m, lat in rows:
            lines.append(
                f"| {name} | {m['hit@1']:.1%} | {m['hit@3']:.1%} | {m['mrr']:.3f} | {m['abstain']:.0%} | "
                f"{lat['p50_ms']:.2f} | {lat['p95_ms']:.2f} | "
                + " | ".join(f"{m['by_tag'].get(t, 0):.0%}" for t in tags) + " |"
            )
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
