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
| **Lightweight** (embeddings only as a reference point) | pure-Python BM25F + coverage scoring; **p50 0.4 ms, p95 < 1 ms** per query on 85 cards; embeddings used only in `eval.py` as the reference baseline |
| **Scalable** — easy to add / delete cards | catalog is one `products.json`; `PUT/DELETE /catalog/cards/{id}` or edit the file → index rebuilds in-process on the next request, no retrain, no redeploy |
| **Bulgarian & English** | one index over both languages of every page; per-script stemming; query-time transliteration (`kreditna karta`, `дск мобайл`) |
| Near-duplicate pages (individual / business / corporate) | collapsed into one card at build time (201 pages → 85 cards); all `document_id`s kept, first returned |
| Flagships drowning in generic tokens (*DSK Mobile* vs "mobile") | per-card `priority_boost`, curated `aliases`, `exact_match_pins`, phrase/head-of-name bonus; **flagship queries: 100 % Hit@1** |
| Mid-typing (every 10 chars) | prefix expansion of the last token; typo tolerance (Damerau-Levenshtein ≤ 1–2) |

**Evaluation** (73 bilingual queries + 10 out-of-catalog negatives, `python eval.py`):

| system | Hit@1 | Hit@3 | MRR | abstain on negatives | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| BM25F only | 82.2 % | 87.7 % | 0.857 | 80 % | 0.10 | 0.19 |
| + prefix / typo / transliteration | 91.8 % | 97.3 % | 0.944 | 80 % | 0.23 | 0.47 |
| **Ours** (+ curated aliases, pins, boosts) | **98.6 %** | **100 %** | **0.993** | 80 % | 0.35 | 0.76 |
| Multilingual embeddings (reference only) | see `eval_results_full.md` | | | | | |

Per-tag Hit@1 for ours: flagship 100 %, generic-token 100 %, product 100 %, prefix 100 %, cross-language 100 %, transliteration 100 %, typo 80 %, natural-language questions 100 %.

## Files

| file | purpose |
|---|---|
| `df_data.csv`, `df_mapping.csv` | provided raw data (201 scraped pages, BG↔EN mapping) |
| `build_catalog.py` | **offline**: pair BG/EN, collapse duplicates, clean markup, extract selling messages, categories, aliases, keywords, flagship config → `products.json` |
| `products.json` | the catalog — **single source of truth** consumed at runtime |
| `ranker.py` | `ProductCardRanker`: tokenisation, query understanding, BM25F + coverage scoring |
| `app.py` | FastAPI service: `/rank-product-cards`, `/catalog/cards` (add/delete), `/health`, hot-reload, cache, request log |
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
.venv/bin/uvicorn app:app --port 8000     # API
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
  model files, ~50 MB image. Index for 85 cards builds in ~50 ms, memory < 20 MB.
* **Latency:** p95 < 1 ms in-process; end-to-end on localhost single-digit ms.
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
