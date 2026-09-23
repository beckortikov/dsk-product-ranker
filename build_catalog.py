"""Offline catalog builder for the DSK Bank product cards ranker.

Reads df_data.csv + df_mapping.csv and emits products.json — the single
source of truth that the runtime ranker consumes.

Offline is where the heavy / slow / non-deterministic work is allowed:
    1. Pair the BG and EN versions of every page via df_mapping.
    2. Collapse near-duplicate pages (individual / business / corporate
       variants of the same product) into one canonical card.
    3. Clean the scraped markdown (img tags, breadcrumbs, boilerplate).
    4. Extract a *selling message* per card: the page's hero tagline
       (the `##` right under the `#` title) — that is literally the bank's
       own marketing one-liner.  Optional `--llm` flag rewrites it with
       Claude for cards where no tagline exists.
    5. Derive a category from the URL breadcrumb (e.g. "кредитиране /
       жилищни и ипотечни кредити") — cheap extra recall for generic queries.
    6. Curated aliases (name variants + hand-written synonyms for flagships)
       and TF-IDF keywords (auto, lower-weighted at runtime).
    7. priority_boost + exact_match_pins for the bank's flagship apps.

Run:
    python build_catalog.py            # → products.json
    python build_catalog.py --llm      # + Claude-written summaries (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "df_data.csv"
MAPPING_CSV = ROOT / "df_mapping.csv"
OUT_JSON = ROOT / "products.json"

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

_HTML_TAG = re.compile(r"<[^>]+>")            # <img alt="…"> and friends — dropped entirely
_MD_HEADER = re.compile(r"^#+\s*", flags=re.MULTILINE)

# Site-chrome lines that appear on every page and carry no product signal.
_BOILERPLATE_LINES = {
    "индивидуални клиенти", "бизнес клиенти", "корпоративни клиенти",
    "корпоративни клиенти - продукти", "моят бизнес", "individual clients",
    "business clients", "corporate clients", "corporate clients - products",
    "my business", "вече съм клиент", "искам да стана клиент",
    "i am already a client", "i want to become a client",
}
# Tokens leaking from markup / icon names that TF-IDF otherwise picks up.
_KEYWORD_STOP = {
    "img", "alt", "accordion", "tick", "icon", "pdf", "yes", "no", "png",
    "svg", "jpg", "step", "стъпка", "the", "and", "your", "you", "can",
    "може", "можете", "можеш", "който", "която", "които", "това", "този",
    "тази", "при", "или", "към", "след", "като", "само", "още", "също",
}


def clean_text(text: str) -> str:
    """Scraped markdown → plain prose. Keeps sentence structure."""
    if not isinstance(text, str) or not text:
        return ""
    s = _HTML_TAG.sub(" ", text)
    s = _MD_HEADER.sub("", s)
    lines = []
    for ln in s.split("\n"):
        ln = re.sub(r"\s+", " ", ln).strip()
        if not ln or ln.lower() in _BOILERPLATE_LINES:
            continue
        lines.append(ln)
    return "\n".join(lines)


def extract_tagline(text: str) -> str:
    """The `##` immediately under the page `#` title — the hero selling line."""
    if not isinstance(text, str):
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            for nxt in lines[i + 1 : i + 4]:
                if nxt.startswith("## "):
                    tag = _HTML_TAG.sub("", nxt[3:]).strip()
                    if 8 <= len(tag) <= 160:
                        return tag
            break
    return ""


_WHAT_IS = re.compile(
    r"(какво представлява|какво е|what is|what does it)", flags=re.IGNORECASE
)


def extract_first_sentence(clean: str, max_chars: int = 180) -> str:
    """Fallback selling message: first sentence of the 'What is it?' section."""
    if not clean:
        return ""
    body = clean
    m = _WHAT_IS.search(clean)
    if m:
        body = clean[m.end():]
    sentences = re.split(r"(?<=[.!?])\s+|\n", body)
    pick = next((x.strip() for x in sentences if len(x.strip()) >= 40), "")
    if not pick:
        return ""
    if len(pick) > max_chars:
        pick = pick[:max_chars].rsplit(" ", 1)[0] + "…"
    return pick


# --------------------------------------------------------------------------- #
# Names / URLs
# --------------------------------------------------------------------------- #

CLIENT_TYPE_SUFFIXES = [
    r"\s*for individuals?\b", r"\s*for individual clients?\b",
    r"\s*for business clients?\b", r"\s*for business\b",
    r"\s*for corporate clients?\b", r"\s*for legal entities\b", r"\s*for sme\b",
    r"\s*за физически лица\b", r"\s*за индивидуални клиенти\b",
    r"\s*за бизнес клиенти\b", r"\s*за бизнеса\b",
    r"\s*за корпоративни клиенти\b", r"\s*за юридически лица\b",
]


def strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def canonicalize_name(name: str) -> str:
    """Lowercase, strip punctuation + client-type suffixes → dedupe key."""
    s = str(name).strip().lower()
    for pat in CLIENT_TYPE_SUFFIXES:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def stem_key(canonical: str) -> str:
    """Stemmed dedupe key — same tokenizer the runtime uses, so
    'физически пос терминал' and 'физически пос терминали' collide."""
    from ranker import tokenize  # local import: build script depends on runtime, not vice versa

    return " ".join(tokenize(canonical))


_CATEGORY_SKIP = {
    "индивидуални-клиенти", "бизнес-клиенти", "корпоративни-клиенти", "моят-бизнес",
    "individual-clients", "business-clients", "corporate-clients", "my-business",
    "индивидуални-продукти", "бизнес-продукти", "корпоративни-продукти",
    "individual-products", "business-products", "corporate-products", "en",
}


def category_from_url(url: str) -> str:
    """'…/кредитиране/жилищни-и-ипотечни-кредити/детайли-…/<slug>' →
    'кредитиране жилищни и ипотечни кредити'. Drops client-type + detail segments."""
    if not isinstance(url, str):
        return ""
    path = unquote(url.split("dskbank.bg", 1)[-1]).strip("/")
    segs = [s for s in path.split("/") if s]
    segs = segs[:-1]  # last segment is the product slug itself (already in name)
    keep = [
        s.replace("-", " ")
        for s in segs
        if s not in _CATEGORY_SKIP and not s.startswith(("детайли", "detail"))
    ]
    return " ".join(keep)


# --------------------------------------------------------------------------- #
# Flagship config — curated by the bank, lives in the catalog, hot-reloadable.
# Keys are canonical names (see canonicalize_name).
# --------------------------------------------------------------------------- #

PRIORITY_CONFIG: dict[str, dict] = {
    "dsk mobile": {
        "priority_boost": 1.35,
        "aliases": [
            "мобилно банкиране", "mobile banking", "ново мобилно банкиране",
            "мобилно приложение", "mobile app", "дск мобайл", "мобайл",
        ],
        "exact_match_pins": [
            "dsk mobile", "dsk mobile app", "dsk mobile banking", "dsk mobail",
            "дск мобайл", "dsk мобайл", "мобилно банкиране", "mobile banking",
            "мобилно банкиране дск", "mobile app",
        ],
    },
    "dsk smart": {
        "priority_boost": 1.35,
        "aliases": ["смарт", "dsk smart app", "мобилно банкиране dsk smart", "smart banking"],
        "exact_match_pins": ["dsk smart", "dsk smart app", "дск смарт", "smart"],
    },
    "dsk online": {
        "priority_boost": 1.35,
        "aliases": ["онлайн банкиране", "online banking", "интернет банкиране", "internet banking", "дск онлайн"],
        "exact_match_pins": ["dsk online", "дск онлайн", "dsk онлайн", "онлайн банкиране", "online banking"],
    },
    "дск директ": {
        "priority_boost": 1.30,
        "aliases": ["dsk direct", "онлайн банкиране за бизнес", "online banking for business", "интернет банкиране за бизнес"],
        "exact_match_pins": ["dsk direct", "дск директ", "direct"],
    },
    "dsk business": {
        "priority_boost": 1.25,
        "aliases": ["дск бизнес", "мобилно банкиране за бизнес", "business mobile banking", "business app"],
        "exact_match_pins": ["dsk business", "дск бизнес", "dsk business app"],
    },
    "dsk mtoken": {
        "priority_boost": 1.15,
        "aliases": ["мтокен", "токен", "token", "потвърждаване на преводи", "софтуерен токен"],
        "exact_match_pins": ["dsk mtoken", "дск мтокен", "mtoken", "мтокен"],
    },
}

# --------------------------------------------------------------------------- #
# Optional: Claude-written selling messages (offline only, never at runtime)
# --------------------------------------------------------------------------- #

_LLM_SYSTEM = (
    "You write one-line product taglines for a Bulgarian bank's website search. "
    "Given a product page, return JSON {\"bg\": ..., \"en\": ...}: a short selling "
    "message (max 90 characters each) in Bulgarian and in English that says what "
    "the product gives the customer. No product name repetition, no marketing fluff, "
    "no exclamation marks."
)


def llm_summaries(cards: list[dict]) -> None:
    """Rewrite summaries for every card with Claude. Requires `pip install anthropic`
    and ANTHROPIC_API_KEY. Deterministic taglines are kept where the model fails."""
    import anthropic  # local import: runtime never depends on it

    client = anthropic.Anthropic()
    for c in cards:
        page = (c["text_bg"] or c["text_en"])[:6000]
        try:
            resp = client.messages.create(
                model="claude-opus-5",
                max_tokens=300,
                system=_LLM_SYSTEM,
                messages=[{"role": "user", "content": f"Product: {c['product_name_bg'] or c['product_name_en']}\n\n{page}"}],
            )
            if resp.stop_reason == "refusal":
                continue
            text = next(b.text for b in resp.content if b.type == "text")
            data = json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))
            if data.get("bg"):
                c["summary_bg"] = data["bg"].strip()
            if data.get("en"):
                c["summary_en"] = data["en"].strip()
            c["summary_source"] = "llm"
        except Exception as exc:  # noqa: BLE001 — offline tool, keep going
            print(f"  llm summary failed for {c['card_id']}: {exc}")


# --------------------------------------------------------------------------- #
# Catalog build
# --------------------------------------------------------------------------- #

def _top_terms(corpus: list[str], n: int) -> list[list[str]]:
    idx = [i for i, t in enumerate(corpus) if t.strip()]
    out: list[list[str]] = [[] for _ in corpus]
    if len(idx) < 2:
        return out
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_df=0.5, min_df=2, sublinear_tf=True,
        token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
        stop_words=list(_KEYWORD_STOP),
    )
    X = vec.fit_transform([corpus[i] for i in idx])
    terms = vec.get_feature_names_out()
    for row_pos, doc_i in enumerate(idx):
        row = X.getrow(row_pos).toarray().ravel()
        top = row.argsort()[::-1][: n * 2]
        picked = []
        for j in top:
            if row[j] <= 0:
                break
            term = terms[j]
            if any(w in _KEYWORD_STOP for w in term.split()):
                continue
            picked.append(term)
            if len(picked) >= n:
                break
        out[doc_i] = picked
    return out


def build(use_llm: bool = False) -> list[dict]:
    df = pd.read_csv(DATA_CSV)
    mapping = pd.read_csv(MAPPING_CSV)

    # 1. BG-id ↔ EN-id (MATCHED rows only).
    bg2en = dict(
        mapping.loc[mapping.match_status == "MATCHED", ["bg_document_id", "en_document_id"]]
        .dropna().astype(int).itertuples(index=False, name=None)
    )
    en2bg = {v: k for k, v in bg2en.items()}
    by_id = df.set_index("document_id").to_dict(orient="index")

    # 2. Pair BG+EN docs into doc-units; unmatched pages travel alone.
    visited: set[int] = set()
    units: list[dict] = []
    for doc_id, row in by_id.items():
        if doc_id in visited:
            continue
        partner = bg2en.get(doc_id) or en2bg.get(doc_id)
        if partner and partner in by_id:
            bg_id, en_id = (doc_id, partner) if row["document_lang"] == "bg" else (partner, doc_id)
            units.append({
                "ids": [int(bg_id), int(en_id)],
                "bg": by_id[bg_id], "en": by_id[en_id],
            })
            visited.update({doc_id, partner})
        else:
            visited.add(doc_id)
            units.append({"ids": [int(doc_id)], row["document_lang"]: row})

    # 3. Collapse client-type duplicates. Two doc-units are the same product if
    #    their stemmed BG names match ("ПОС терминал" ~ "ПОС терминали") OR
    #    their EN names match ("Бинарна опция" & "Бинарна валутна опция" are
    #    both "Binary FX Option"). Union-find over both keys.
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    unit_keys: list[str] = []
    bg_tokens_by_en: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for u in units:
        bg_key = stem_key(canonicalize_name((u.get("bg") or u.get("en"))["product_name"]))
        unit_keys.append(bg_key)
        if u.get("en") is not None:
            en_key = stem_key(canonicalize_name(u["en"]["product_name"]))
            bg_tokens_by_en[en_key].append((bg_key, set(bg_key.split())))
    # Same EN name is only evidence of sameness when the BG names nest
    # (one token set ⊆ the other). Two different "Loan Protection Insurance"
    # products (mortgage vs consumer) stay separate.
    for pairs in bg_tokens_by_en.values():
        for i, (ka, ta) in enumerate(pairs):
            for kb, tb in pairs[i + 1:]:
                if ta <= tb or tb <= ta:
                    union(ka, kb)

    groups: dict[str, list[dict]] = defaultdict(list)
    for u, k in zip(units, unit_keys):
        groups[find(k)].append(u)
    # Human-readable canonical key: the shortest BG name in the group.
    groups = {
        canonicalize_name(min(
            [(x.get("bg") or x.get("en"))["product_name"] for x in us], key=len
        )): us
        for us in groups.values()
    }

    # 4. Cards.
    cards: list[dict] = []
    for key, us in groups.items():
        ids, names_bg, names_en, raw_bg, raw_en, urls, cats = [], [], [], [], [], [], []
        for u in us:
            ids.extend(u["ids"])
            for lang, names, raws in (("bg", names_bg, raw_bg), ("en", names_en, raw_en)):
                r = u.get(lang)
                if r is None:
                    continue
                names.append(str(r["product_name"]))
                raws.append(str(r["document_text"]))
                urls.append(r["document_url"])
                cats.append(category_from_url(r["document_url"]))
        text_bg = "\n\n".join(clean_text(t) for t in raw_bg).strip()
        text_en = "\n\n".join(clean_text(t) for t in raw_en).strip()
        tag_bg = next((t for t in map(extract_tagline, raw_bg) if t), "")
        tag_en = next((t for t in map(extract_tagline, raw_en) if t), "")
        cards.append({
            "card_id": key.replace(" ", "_") or f"card_{ids[0]}",
            "canonical_key": key,
            "document_ids": sorted(set(ids)),
            "product_name_bg": names_bg[0] if names_bg else "",
            "product_name_en": names_en[0] if names_en else "",
            "name_variants": sorted(set(names_bg + names_en)),
            "category": " ".join(sorted(set(c for c in cats if c))),
            "summary_bg": tag_bg or extract_first_sentence(text_bg),
            "summary_en": tag_en or extract_first_sentence(text_en),
            "summary_source": "tagline" if (tag_bg or tag_en) else "first_sentence",
            "text_bg": text_bg,
            "text_en": text_en,
            "urls": sorted(set(u for u in urls if isinstance(u, str))),
        })

    # 5. Keywords (auto) — bilingual TF-IDF over cleaned text.
    kw_bg = _top_terms([c["text_bg"] for c in cards], n=10)
    kw_en = _top_terms([c["text_en"] for c in cards], n=10)
    for c, a, b in zip(cards, kw_bg, kw_en):
        c["keywords"] = sorted(set(a) | set(b))

    # 6. Curated aliases + flagship config.
    for c in cards:
        cfg = PRIORITY_CONFIG.get(c["canonical_key"], {})
        aliases = {v.lower() for v in c["name_variants"]}
        aliases.update(a.lower() for a in cfg.get("aliases", []))
        c["aliases"] = sorted(aliases)
        c["priority_boost"] = cfg.get("priority_boost", 1.0)
        c["exact_match_pins"] = cfg.get("exact_match_pins", [])

    if use_llm:
        llm_summaries(cards)

    cards.sort(key=lambda c: c["canonical_key"])
    return cards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="rewrite summaries with Claude (offline)")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    cards = build(use_llm=args.llm)
    Path(args.out).write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    n_docs = sum(len(c["document_ids"]) for c in cards)
    print(f"Wrote {len(cards)} cards ({n_docs} raw docs) -> {args.out}")
    print("Flagships:", [c["canonical_key"] for c in cards if c["priority_boost"] > 1.0])
    print("Summary source:", pd.Series([c["summary_source"] for c in cards]).value_counts().to_dict())
    missing = [c["canonical_key"] for c in PRIORITY_CONFIG if c not in {x["canonical_key"] for x in cards}]
    if missing:
        print("WARNING: PRIORITY_CONFIG keys without a card:", missing)


if __name__ == "__main__":
    main()
