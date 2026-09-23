"""Generates demo.ipynb (then execute it with nbconvert). Kept in the repo so the
notebook is reproducible: `python make_notebook.py && jupyter nbconvert --execute --to notebook --inplace demo.ipynb`."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


code("""# Colab bootstrap: clones the repo and installs deps. Does nothing when run locally from the repo.
import os, sys
if 'google.colab' in sys.modules and not os.path.exists('ranker.py'):
    !git clone -q https://github.com/beckortikov/dsk-product-ranker
    %cd dsk-product-ranker
    %pip install -q -r requirements.txt
    print('ready')""")

md("""# DSK Bank — Product Cards Ranker · demo & evaluation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/beckortikov/dsk-product-ranker/blob/main/demo.ipynb) · [code on GitHub](https://github.com/beckortikov/dsk-product-ranker) · [live dashboard on Hugging Face](https://beckortikov-dsk-product-ranker.hf.space/dashboard/)

Lightweight bilingual (BG + EN) ranker for the Smart Search on dskbank.bg.
Zero ML at runtime, sub-millisecond, catalog is a hot-reloadable `products.json`.

0. The raw data: `df_data.csv` + `df_mapping.csv` — what is in there and what it implies
1. Building the catalog from the raw data (offline step) — one card per product
2. Query → ranked cards (+ interactive widget)
3. How a query is understood (prefix / typo / transliteration)
4. Evaluation: ours vs. lexical ablations vs. multilingual embeddings; dev set and held-out
5. Latency
6. Add / delete a product at runtime""")

code("""import json, sys, time, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path('.').resolve()))
pd.set_option('display.max_colwidth', 90)

df = pd.read_csv('df_data.csv')          # scraped product pages
mp = pd.read_csv('df_mapping.csv')       # BG page -> EN page
print('df_data.csv   ', df.shape, list(df.columns))
print('df_mapping.csv', mp.shape, list(mp.columns))
df.head(3)[['document_id', 'document_lang', 'product_name', 'document_url']]""")

md("## 0. The raw data\n\nBefore any model: what is actually in the two files, because every design decision below comes from here.")

code("""print('pages by language:', df.document_lang.value_counts().to_dict())
print('mapping status:    ', mp.match_status.value_counts().to_dict())
L = df.document_text.str.len()
print(f'page length (chars): median {int(L.median())}, p10 {int(L.quantile(.1))}, p90 {int(L.quantile(.9))}  -> long landing pages, not cards')
print(f'<img alt> tags per page: median {int(df.document_text.str.count("<img").median())}      -> markup noise to clean')
print(f'unique product_name: {df.product_name.nunique()} of {len(df)} pages          -> near-duplicates (client-type variants)')""")

md("**Bulgarian-only pages.** 9 pages have no English counterpart, and two of them are flagships from the spec (DSK Mobile, DSK Online). An English query must still find them, so BG and EN are indexed together in one card rather than as two indexes.")

code("""unmatched = mp[mp.match_status != 'MATCHED'].merge(df[['document_id', 'product_name']], left_on='bg_document_id', right_on='document_id')
unmatched[['bg_document_id', 'product_name']]""")

md("**Near-duplicates.** The same product is published for individual / business / corporate clients, up to six pages per product:")

code("""dup = df.groupby('product_name').document_id.agg(list)
dup[dup.str.len() > 2].to_frame('document_ids')""")

md("**The selling message is in the data after all.** Every page has a `#` title followed by a `##` hero tagline — the bank's own one-line pitch, exactly the `product_summary` the spec asks for. Raw page for DSK mToken:")

code("""raw = df.loc[df.document_id == 10575, 'document_text'].iloc[0]
print('\\n'.join(l for l in raw.split('\\n') if l.strip())[:700])""")

md("## 1. Building the catalog (offline)\n\n`build_catalog.py` turns the two CSVs into `products.json`: pairs BG↔EN via the mapping, collapses client-type variants into one card, strips markup and breadcrumbs, extracts the tagline, derives a category from the URL, adds curated aliases and TF-IDF keywords, and attaches the flagship config. Run it right here:")

code("""from build_catalog import build
t0 = time.perf_counter()
cards = build()
print(f'{len(df)} raw pages -> {len(cards)} product cards in {time.perf_counter()-t0:.1f} s')
print('cards merged from 3+ pages:', sum(1 for c in cards if len(c['document_ids']) > 2),
      '| BG-only cards:', sum(1 for c in cards if not c['product_name_en']),
      '| summary from hero tagline:', sum(1 for c in cards if c['summary_source'] == 'tagline'))
pd.DataFrame([{'card': c['canonical_key'], 'document_ids': c['document_ids'], 'summary_bg': c['summary_bg']}
              for c in cards if c['canonical_key'] in ('dsk mtoken', 'dsk mobile', 'физически пос терминал', 'кредитна карта galaxy')])""")

code("""# The runtime never sees the CSVs: it only reads this file.
Path('products.json').write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding='utf-8')
from ranker import ProductCardRanker
t0 = time.perf_counter(); ranker = ProductCardRanker(cards)
print(f'{len(ranker)} cards indexed in {(time.perf_counter()-t0)*1000:.0f} ms')""")

md("### What a card looks like\n\nOne card per product. Client-type variants are collapsed into one card that keeps every `document_id`; the first one is returned as the representative.")

code("""c = next(c for c in cards if c['canonical_key'] == 'dsk mtoken')
{k: (v if not isinstance(v, str) or len(v) < 120 else v[:120] + '…') for k, v in c.items() if k not in ('text_bg', 'text_en')}""")

md("## 2. Query → ranked cards\n\nResponse contract (per item): `document_id`, `product_name`, `product_summary`, `relevance`.")

code("""def show(q, k=5):
    print(f'\\n=== {q!r}')
    for r in ranker.rank(q, top_k=k):
        print(f\"  {r['relevance']:.2f}  [{r['document_id']:>5}]  {r['product_name']:<45}  {r['product_summary'][:60]}\")

for q in ['DSK Mobile', 'mobile', 'мобилно банкиране', 'DSK Smart', 'дск директ',
          'кредитна карта', 'ипотечен кредит', 'student loan', 'home insurance',
          'dsk mob',            # mid-typing
          'кредитна крата',     # typo
          'dsk mobail',         # transliteration-ish
          'такси за превод',    # weakly related → low relevance
          'зззз']:              # nothing
    show(q)""")

code("""try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    box = widgets.Text(description='query:', placeholder='type as a user would…', layout=widgets.Layout(width='600px'))
    out = widgets.Output()
    def on_change(change):
        with out:
            clear_output()
            for r in ranker.rank(change['new'], top_k=5):
                print(f\"{r['relevance']:.2f}  [{r['document_id']}]  {r['product_name']}  —  {r['product_summary'][:70]}\")
    box.observe(on_change, names='value')
    display(box, out)
except Exception as e:
    print('widget unavailable:', e)""")

md("## 3. How a query is understood\n\nEach token is mapped to catalog terms: exact → prefix (last token only) → transliteration → fuzzy (Damerau-Levenshtein ≤ 1–2). Match quality discounts the contribution.")

code("""for q in ['dsk mob', 'кредитна крата', 'kreditna karta', 'депозит', 'mtokn']:
    print(f'{q!r:20}', [[(m.term, m.quality) for m in g][:5] for g in ranker.analyze(q)])""")

md("""## 4. Evaluation

No query logs exist, so `eval.py` holds a hand-curated bilingual **dev set** (73 queries tagged
*flagship / generic / product / prefix / xlang / typo / translit / long* + 10 out-of-catalog negatives).

* **Hit@1 / Hit@3 / MRR** on positives; **abstain** = share of negatives returned empty or below threshold.

Systems: BM25F only → + query fallbacks → **ours** (+ curated aliases, pins, boosts) → multilingual sentence embeddings as the *reference point* the spec allows (not part of the final ranker).""")

code("""from eval import EVAL, NEGATIVES, evaluate, latency, lexical_ablation

rows = []
for name, r in [('BM25F only', lexical_ablation(cards, boosts=False, fuzzy=False)),
                ('BM25F + prefix/typo/translit', lexical_ablation(cards, boosts=False, fuzzy=True)),
                ('Ours', ProductCardRanker(cards))]:
    m = evaluate(lambda q, k: r.rank(q, k, min_relevance=0.0), cards)
    lat = latency(lambda q, k: r.rank(q, k), n=500)
    if hasattr(r, '_restore'): r._restore()
    rows.append({'system': name, 'Hit@1': m['hit@1'], 'Hit@3': m['hit@3'], 'MRR': m['mrr'], 'abstain': m['abstain'],
                 'p50 ms': lat['p50_ms'], 'p95 ms': lat['p95_ms'], **{f'{t}': v for t, v in m['by_tag'].items()}})
pd.DataFrame(rows).set_index('system').round(3)""")

md("**Held-out check.** The 73 queries above were written by the same person who tuned the ranker. `eval.py --holdout` generates 770 queries mechanically from the catalog (full names, 55 % prefixes, typos, transliterations, 2-word subsets, taglines) and is never tuned against — this is the honest number.")

code("""from eval import run_holdout
_ = run_holdout(cards)""")

code("""# Reference point: multilingual embeddings (downloads ~120 MB on first run; skip if offline)
try:
    from eval import EmbeddingRanker
    e = EmbeddingRanker(cards)
    m = evaluate(e.rank, cards, min_relevance=0.45)
    lat = latency(e.rank, n=100)
    print(f\"Embeddings  Hit@1 {m['hit@1']:.1%}  Hit@3 {m['hit@3']:.1%}  MRR {m['mrr']:.3f}  p50 {lat['p50_ms']:.1f} ms\")
    print('by tag:', {t: f'{v:.0%}' for t, v in m['by_tag'].items()})
except Exception as exc:
    print('embedding baseline skipped:', exc)""")

md("## 5. Latency")

code("""import statistics
qs = [q for q, _, _ in EVAL] * 30
ts = []
for q in qs:
    t0 = time.perf_counter(); ranker.rank(q, 5); ts.append((time.perf_counter() - t0) * 1000)
ts.sort()
print(f'p50 {statistics.median(ts):.2f} ms | p95 {ts[int(.95*len(ts))]:.2f} ms | p99 {ts[int(.99*len(ts))]:.2f} ms  over {len(ts)} queries, {len(ranker)} cards')""")

md("## 6. Add / delete a product at runtime\n\nA card is a dict. No retraining: the index rebuilds in-process in ~0.1 s. In production the same happens through `PUT/DELETE /catalog/cards/{id}` or by editing `products.json`.")

code("""new = {'card_id': 'зелена_ипотека', 'canonical_key': 'зелена ипотека', 'document_ids': [999999],
       'product_name_bg': 'Зелена ипотека', 'product_name_en': 'Green mortgage',
       'name_variants': ['Зелена ипотека', 'Green mortgage'], 'aliases': ['еко кредит', 'енергийно ефективен дом'],
       'summary_bg': 'По-ниска лихва за енергийно ефективен дом', 'summary_en': 'Lower rate for an energy-efficient home'}
t0 = time.perf_counter(); r2 = ProductCardRanker(cards + [new]); build_ms = (time.perf_counter() - t0) * 1000
print(f'rebuilt {len(r2)} cards in {build_ms:.0f} ms')
for q in ['зелена ипотека', 'green mortgage', 'еко кредит']:
    print(q, '→', r2.rank(q, 1)[0]['product_name'], r2.rank(q, 1)[0]['relevance'])""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("demo.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote demo.ipynb")
