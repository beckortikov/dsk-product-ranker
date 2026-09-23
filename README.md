# DSK Bank — Product Cards Ranker

*Русская версия: [README.ru.md](README.ru.md)*

Lightweight bilingual (BG + EN) ranker for the Smart Search engine on dskbank.bg.
The frontend fires the raw user query every 10 typed characters; this service
returns a ranked list of product cards in well under a millisecond.

## TL;DR

| spec requirement | how it is met |
|---|---|
| Ranked list of `{document_id, product_name, product_summary, relevance}` | `POST /rank-product-cards` — see contract below |
| `product_summary` = short selling message (not in the data) | extracted offline from each page's **hero tagline** (the `##` under the title — the bank's own one-liner), 81/85 cards; fallback = first sentence of "What is it?"; optional `--llm` rewrite with Claude |
| **Lightweight** (embeddings only as a reference point) | pure-Python BM25F + coverage scoring; **p50 0.1 ms, p95 0.3 ms** per query on 85 cards; embeddings used only in `eval.py` as the reference baseline |
| **Scalable** — easy to add / delete cards | catalog is one `products.json`; `PUT/DELETE /catalog/cards/{id}` or edit the file → index rebuilds in-process on the next request, no retrain, no redeploy |
| **Bulgarian & English** | one index over both languages of every page; per-script stemming; query-time transliteration (`kreditna karta`, `дск мобайл`) |
| Near-duplicate pages (individual / business / corporate) | collapsed into one card at build time (201 pages → 85 cards); all `document_id`s kept, first returned |
| Flagships drowning in generic tokens (*DSK Mobile* vs "mobile") | per-card `priority_boost`, curated `aliases`, `exact_match_pins`, phrase/head-of-name bonus; **flagship queries: 100 % Hit@1** |
| Mid-typing (every 10 chars) | prefix expansion of the last token; typo tolerance (Damerau-Levenshtein ≤ 1–2) |

**Evaluation** (73 bilingual queries + 10 out-of-catalog negatives, `python eval.py`):

| system | Hit@1 | Hit@3 | MRR | abstain on negatives | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| BM25F only | 82.2 % | 87.7 % | 0.857 | 80 % | 0.08 | 0.12 |
| + prefix / typo / transliteration | 90.4 % | 97.3 % | 0.937 | 80 % | 0.10 | 0.35 |
| **Ours** (+ curated aliases, pins, boosts) | **98.6 %** | **100 %** | **0.993** | 80 % | 0.10 | 0.31 |
| Multilingual embeddings, cosine (reference only) | 56.2 % | 74.0 % | 0.648 | 100 % | 29.6 | 56.3 |

Per-tag Hit@1 for ours: flagship 100 %, generic-token 100 %, product 100 %, prefix 100 %, cross-language 100 %, transliteration 100 %, typo 80 %, natural-language questions 100 %.

The embedding reference (`paraphrase-multilingual-MiniLM-L12-v2`, ONNX, `python eval.py --embeddings`) is where the spec allows semantic models: as a point of comparison. On this catalog it loses to the lexical ranker on every tag except generic single words, and it does **not** solve the flagship problem (29 % Hit@1 — DSK Mobile / Smart / Online / Business embed almost identically), while costing ~100× the latency and a 120 MB model at runtime. That is the argument for keeping semantics *offline* (aliases, summaries) and the runtime lexical. Full table: `eval_results_full.md`.

### Overfitting check (held-out set)

The 73-query set above is a **dev set**: the same person wrote it and tuned the
ranker against its misses, and the flagship pins/aliases literally contain some
of its queries (14/73 hit an `exact_match_pin`). So `python eval.py --holdout`
generates a second set **mechanically from the catalog** (seed 0, no human
picking, never tuned against): full BG/EN names, 55 % prefixes, adjacent-letter
typos, transliterated names, random 2-word subsets, and the hero taglines as
queries. 770 queries.

| system | Hit@1 | Hit@3 | MRR | prefix | typo | translit | 2-word subset | tagline |
|---|---|---|---|---|---|---|---|---|
| BM25F only | 76.5 % | 86.8 % | 0.816 | 74–80 % | 71–73 % | 16 % | 80 % | 88–90 % |
| + prefix / typo / translit | 91.7 % | 98.4 % | 0.950 | 78–86 % | 99 % | 95 % | 80 % | 88–90 % |
| **Ours** | **91.9 %** | **98.4 %** | **0.952** | 78–86 % | 99 % | 98 % | 80 % | 88–90 % |
| Embeddings (reference) | 54.4 % | 74.8 % | 0.665 | 22–43 % | 51–57 % | 44 % | 37 % | 64–71 % |

What this says, honestly:

* The curated-set number (98.6 %) is optimistic by ~7 points; **91.9 % Hit@1 /
  98.4 % Hit@3** is the fair estimate for query shapes nobody hand-picked.
  The gap is concentrated in *prefix* and *2-word subset* queries, which are
  genuinely ambiguous (55 % of "Застраховка „Кредитна защита“ за…" matches
  three products) — Hit@3 at 98.4 % shows the gold is almost always right there.
* The hand-written flagship config (pins + aliases) is worth **+8 points on the
  curated set and +0.2 on the held-out one**: it fixes a specific business
  requirement (DSK Mobile must beat "mobile"), it does not inflate general
  quality. The generic mechanisms (prefix / typo / transliteration) carry the
  general gain: 76.5 → 91.7.
* The embedding reference stays far behind on both sets, so the lexical-vs-
  semantic conclusion is not an artefact of the dev set.
* No learned parameters exist; what *was* tuned by hand (field weights, bonus
  sizes, thresholds) was tuned on the dev set. Before production, the bank's
  real query log becomes the test set — the dashboard/API log is built for that.

## Files

| file | purpose |
|---|---|
| `df_data.csv`, `df_mapping.csv` | provided raw data (201 scraped pages, BG↔EN mapping) |
| `build_catalog.py` | **offline**: pair BG/EN, collapse duplicates, clean markup, extract selling messages, categories, aliases, keywords, flagship config → `products.json` |
| `products.json` | the catalog — **single source of truth** consumed at runtime |
| `ranker.py` | `ProductCardRanker`: tokenisation, query understanding, BM25F + coverage scoring |
| `app.py` | FastAPI service: `/rank-product-cards`, `/catalog/cards` (add/delete), `/health`, hot-reload, cache, request log; serves the dashboard at `/` |
| `static/index.html` | **live dashboard** (`GET /`): search box that ranks on every keystroke with a relevance chart and the token-level query analysis, eval charts (systems × metrics, Hit@1 by query type), live latency, catalog composition |
| `eval.py` | eval set, metrics, lexical ablations, embedding baseline |
| `demo.ipynb` | executed demo: queries, interactive widget, query analysis, eval table, latency, live add/delete |
| `tests/` | 29 pytest tests (ranker behaviour, catalog invariants, API contract, hot-reload, add/delete) |
| `Dockerfile` | production image (`uvicorn`, 2 workers, healthcheck) |

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python build_catalog.py        # df_*.csv → products.json (85 cards)
.venv/bin/python eval.py                  # metrics; add --embeddings for the reference baseline
.venv/bin/python -m pytest -q             # 29 tests
.venv/bin/jupyter notebook demo.ipynb     # interactive demo
.venv/bin/uvicorn app:app --port 8000     # API + dashboard → open http://localhost:8000/
```

```bash
curl -s -X POST localhost:8000/rank-product-cards -H 'Content-Type: application/json' \
  -d '{"query": "мобилно банкиране", "top_k": 3}'
```

```json
[
  {"document_id": 10217, "product_name": "DSK Mobile", "product_summary": "Повече възможности, стабилност и лекота", "relevance": 1.0},
  {"document_id": 10229, "product_name": "DSK Smart",  "product_summary": "Банкиране на Банка ДСК за индивидуални клиенти", "relevance": 0.67},
  {"document_id": 10572, "product_name": "DSK Business", "product_summary": "Едно приложение - много решения за бизнеса Ви", "relevance": 0.65}
]
```

Docker: `docker build -t dsk-ranker . && docker run -p 8000:8000 dsk-ranker`

## How it works

```
 offline (build_catalog.py)                       runtime (ranker.py, every 10 keystrokes)
 ─────────────────────────────                    ─────────────────────────────────────────
 df_data.csv + df_mapping.csv                     POST /rank-product-cards {query, top_k}
   │ pair BG ↔ EN pages                              │
   │ collapse client-type variants                   ▼
   │ strip <img>/breadcrumbs/markup               tokenize (BG/EN by script, stem, stop-words)
   │ hero tagline → product_summary                  │  per token: exact ▸ prefix (last) ▸ translit ▸ fuzzy
   │ URL breadcrumb → category                       ▼
   │ curated aliases + TF-IDF keywords            score every card
   │ flagship boosts + pins                          relevance = idf-weighted field coverage ∈ [0,1]
   ▼                                                            × specificity + phrase/head bonus + boost
 products.json  ───────────────────────────►        tie-break = BM25F
 (hot-reloaded on mtime change)                       pins → 1.0
                                                     ▼
                                                  drop < min_relevance, top_k
```

### Scoring — why two scores

BM25 is a good *ranking* signal but its absolute value means nothing to a
frontend that has to decide "show this card or not" for a half-typed query.
So each card gets:

* **`relevance`** — idf-weighted **field coverage**: for every query term, the
  weight of the best field it was found in (`name` 1.0 · `aliases` 0.6 ·
  `category` 0.5 · `summary` 0.4 · `keywords` 0.3 · `body` 0.2), discounted by
  match quality (exact 1.0 · prefix 0.85 · transliteration 0.75 · fuzzy 0.6),
  averaged over the query's idf mass. Unmatched tokens count at half weight
  (so `credit crad` < `credit card`). Then: a specificity factor (how much of
  the card's *name* the query covers), +0.15 if the query is a substring of the
  name, +0.10 more if the name *starts* with it, `+0.2·(priority_boost−1)`,
  `exact_match_pins` → 1.0.
  Result: 1.0 = "this card is named that", ~0.2 = "the words occur somewhere in
  the page". Comparable across queries → a single threshold works.
* **BM25F** (tf saturation, idf, length normalisation, field weights) breaks
  ties between cards with equal coverage — e.g. among five credit cards.

### Query understanding (all runtime, no ML)

| user types | what happens |
|---|---|
| `dsk mob` | last token prefix-expanded against the core vocabulary (name/aliases/category/summary first, body only if nothing else) |
| `кредитна крата`, `mtokn` | Damerau-Levenshtein ≤ 1 (≤ 2 for 7+ chars) against the core vocabulary, same first letter required |
| `kreditna karta`, `дск мобайл`, `депозит` | transliteration latin↔cyrillic, then exact/prefix/fuzzy again |
| `DSK Mobile`, `мобилно банкиране` | `exact_match_pins` on the flagship card → relevance 1.0, hard #1 |

### Business nuances

* **Near-duplicates.** Pages for individual / business / corporate clients are
  the same product. `build_catalog.py` groups by the stemmed canonical name
  with client-type suffixes stripped, and additionally merges pairs whose EN
  name is identical *and* whose BG names nest ("Бинарна опция" ⊂ "Бинарна
  валутна опция") — but not two different "Loan Protection Insurance"
  products. 201 pages → 85 cards; a card keeps every `document_id` and returns
  the first as representative.
* **Flagships.** DSK Mobile / Smart / Online / Business / Direct / mToken carry
  `priority_boost`, curated bilingual `aliases` ("мобилно банкиране", "mobile
  banking", "мобайл") and `exact_match_pins`. All of it lives in
  `products.json`, i.e. it is content, not code — the bank's team can tune it
  without a deploy. Ablation shows what it buys: flagship Hit@1 79 % → 100 %.
* **Cross-language recall.** Both languages of a page are indexed in one card,
  so an EN query hits a BG-only page (DSK Mobile, DSK Online exist only in BG)
  through its curated aliases and the category/body text; BG queries hit EN
  names through transliteration + fuzzy ("депозит" → "deposit").

## Production notes (system design)

* **Runtime deps:** `snowballstemmer`, `rapidfuzz`, FastAPI. No torch, no
  model files, ~50 MB image. Index for 85 cards builds in ~100 ms, memory < 20 MB.
* **Latency:** p50 0.1 ms / p95 0.3 ms in-process; end-to-end on localhost single-digit ms.
  Identical `(query, top_k, min_relevance, catalog_mtime)` are served from a
  4096-entry LRU — the 10-keystroke polling from many users hits the same prefixes.
* **Catalog lifecycle.** `products.json` is the deployable artefact.
  Add/delete = `PUT/DELETE /catalog/cards/{id}` (atomic write) or a file edit;
  every request compares the file mtime and rebuilds in place under a lock
  (reads stay lock-free; the cache key contains the mtime so nothing stale is
  ever served). For fleets: ship the file via object storage / config map and
  let every replica reload; or push through the endpoint behind an admin auth.
* **Nightly rebuild.** `build_catalog.py` re-runs against the fresh scrape and
  emits a new `products.json`; a diff on `card_id`s is the release note.
  Curated fields (`aliases`, `priority_boost`, `exact_match_pins`, LLM
  summaries) are merged from the previous catalog, not regenerated.
* **Observability.** Every request is logged as JSON `{q, k, ms, top}` plus an
  `X-Latency-Ms` header; that log is the training signal for the next tuning
  round (which queries abstain, which return low relevance, which flagship
  lost to what). Ship it to ClickHouse/BigQuery; alert on p95 and on the
  abstain rate.
* **Threshold.** The frontend should hide cards below `min_relevance` (default
  0.1; 0.2–0.25 hides body-only matches). Weakly related queries return 0.2–0.4
  by design so the UI can show them dimmed or not at all.
* **Scaling the catalog.** Linear scan over cards per query — fine to ~5 000
  cards (≈ 20 ms). Beyond that: inverted index over the core vocabulary (all
  the pieces are already term-keyed), or shard by client segment.
* **Quality loop.** Add an eval case for every reported miss (`eval.py`), fix
  via catalog data first (alias / pin), code second. `pytest` guards the
  contract and the flagship behaviour.
* **Better summaries.** `python build_catalog.py --llm` rewrites the taglines
  with Claude (`claude-opus-5`) offline; runtime is unchanged.

## Trade-offs we explicitly accept

* Bulgarian stemming is a conservative rule-based suffix stripper (Snowball
  ships no BG algorithm). Under-stems on purpose; good enough for ~100 cards.
* Relevance is *not* a probability — it is a calibrated coverage score. Good
  enough for "show / dim / hide"; if the UI needs probabilities, fit an
  isotonic map on click logs later.
* The eval set is hand-curated (no logs exist yet); gold labels for generic
  queries accept any card that a reasonable user would accept (e.g. any credit
  card for "credit card").
* Typo tolerance is deliberately narrow (≥ 5 chars, same first letter) to keep
  out-of-catalog queries from matching junk; the residual typo misses are
  ambiguous ties, not failures to recall.
