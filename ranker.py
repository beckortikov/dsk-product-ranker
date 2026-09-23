"""Lightweight bilingual product-cards ranker for DSK Bank Smart Search.

Runtime design (no ML, no network, pure Python, sub-millisecond):

    query ──► tokenize (BG/EN by script, stem)
          ──► match every token against the catalog vocabulary
                exact ► prefix (last token, user is mid-typing)
                      ► transliteration (latin↔cyrillic)
                      ► fuzzy (Levenshtein ≤ 1-2 via rapidfuzz)
          ──► score every card
                relevance  = field-coverage score  ∈ [0, 1]   (calibrated, comparable across queries)
                tie-break  = BM25F score                       (tf/idf/length-normalised)
                × priority_boost   + phrase bonus   + exact-match pin
          ──► drop cards below `min_relevance`, return top_k

Why two scores?  BM25 is a great *ranking* signal but its absolute value
tells the frontend nothing ("is this card actually about the query, or did
the word merely appear once in a 2 000-word page?").  The frontend polls
every 10 keystrokes and needs an absolute "show / don't show" signal, so
`relevance` is a coverage score: 1.0 = every query term is in the product
name, ~0.2 = terms only found deep in the page body.  BM25F breaks ties
between cards with equal coverage.

Catalog (products.json) is hot-swappable: adding / deleting a card is a
file edit + `.reload()`; the index rebuilds in well under a second.

Public API:
    ranker = ProductCardRanker.from_json("products.json")
    ranker.rank("DSK Mobile", top_k=5)
    # -> [{"document_id": int, "product_name": str,
    #      "product_summary": str, "relevance": float}, ...]
"""
from __future__ import annotations

import bisect
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import snowballstemmer

try:  # optional — typo tolerance degrades gracefully without it
    from rapidfuzz import process as _rf_process
    from rapidfuzz.distance import DamerauLevenshtein as _rf_lev  # transposition = 1 edit
except ImportError:  # pragma: no cover
    _rf_process = None
    _rf_lev = None

# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

_EN_STEM = snowballstemmer.stemmer("english")

# Snowball has no Bulgarian algorithm; this is a conservative suffix stripper.
# We'd rather under-stem than merge unrelated words.
_BG_SUFFIXES = (
    "ането", "ението", "ания", "ение",
    "ите", "ята", "ове", "еве",
    "ия", "ът", "ьт",
    "та", "то", "те",
    "ва", "не",
    "и", "а", "о", "е", "я", "ъ", "ь",
)


class _BgStemmer:
    @staticmethod
    def stemWord(word: str) -> str:
        if len(word) <= 4:
            return word
        for suf in _BG_SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                return word[: -len(suf)]
        return word


_BG_STEM = _BgStemmer()

_STOP_BG = {
    "и", "или", "на", "за", "от", "до", "по", "е", "са", "ли", "че", "като",
    "при", "след", "преди", "ще", "не", "във", "със", "с", "в",
    "та", "тази", "това", "тези", "този", "те", "то", "ти", "си", "се", "ми",
    "да", "как", "какво", "кой", "коя", "кое", "кои", "има", "искам", "мога",
}
_STOP_EN = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at",
    "by", "with", "is", "are", "be", "as", "from", "that", "this", "it",
    "you", "your", "we", "our", "i", "my", "me", "how", "what", "do", "can",
    "want", "need", "get",
}

_TOKEN_RE = re.compile(r"[\wа-яА-ЯёЁ]+", flags=re.UNICODE)


def _is_cyrillic_token(tok: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in tok)


def _strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_text(text: str) -> str:
    """Lowercase, strip diacritics + punctuation, collapse whitespace.
    Used for `exact_match_pins` and the phrase bonus."""
    s = _strip_diacritics(text.lower())
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _stem(tok: str) -> str:
    return _BG_STEM.stemWord(tok) if _is_cyrillic_token(tok) else _EN_STEM.stemWord(tok)


def tokenize(text: str, *, drop_stop: bool = True) -> list[str]:
    """Bilingual tokenizer: lowercase, drop stop-words, stem by script."""
    if not text:
        return []
    out: list[str] = []
    for t in (t.lower() for t in _TOKEN_RE.findall(text)):
        if drop_stop and (t in _STOP_BG or t in _STOP_EN):
            continue
        out.append(_stem(t))
    return out


# --------------------------------------------------------------------------- #
# Transliteration (BG official latin ↔ cyrillic), used only as a query fallback
# --------------------------------------------------------------------------- #

_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sht", "ъ": "a",
    "ь": "y", "ю": "yu", "я": "ya",
}
_LAT2CYR = sorted(
    [
        ("sht", "щ"), ("zh", "ж"), ("ch", "ч"), ("sh", "ш"), ("ts", "ц"),
        ("yu", "ю"), ("ya", "я"), ("ph", "ф"), ("kh", "х"), ("th", "т"),
        ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"),
        ("g", "г"), ("h", "х"), ("i", "и"), ("j", "дж"), ("k", "к"), ("l", "л"),
        ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("q", "к"), ("r", "р"),
        ("s", "с"), ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"), ("x", "кс"),
        ("y", "й"), ("z", "з"),
    ],
    key=lambda kv: -len(kv[0]),
)


def transliterate(tok: str) -> str:
    """cyrillic → latin, or latin → cyrillic (best effort)."""
    if _is_cyrillic_token(tok):
        return "".join(_CYR2LAT.get(ch, ch) for ch in tok)
    out, i = [], 0
    while i < len(tok):
        for src, dst in _LAT2CYR:
            if tok.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(tok[i])
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #

_FIELDS = ("name", "aliases", "category", "summary", "keywords", "body")

# Field weights: a hit in the product name is worth 5× a hit in the body.
#   name      — product_name in both languages (+ client-type variants)
#   aliases   — curated synonyms (catalog-managed, e.g. "мобилно банкиране")
#   category  — URL breadcrumb ("кредитиране жилищни и ипотечни кредити")
#   summary   — the selling message
#   keywords  — auto TF-IDF terms (noisy → low weight)
#   body      — full page text
_FIELD_WEIGHTS = {
    "name": 5.0, "aliases": 3.0, "category": 2.5, "summary": 2.0,
    "keywords": 1.5, "body": 1.0,
}
_MAX_FIELD_W = max(_FIELD_WEIGHTS.values())

# BM25 parameters (tie-breaker score).
_K1 = 1.4
_B = 0.6

# Query-term match quality: exact=1, prefix / translit / fuzzy are discounted.
_MATCH_EXACT = 1.0
_MATCH_PREFIX = 0.85
_MATCH_TRANSLIT = 0.75
_MATCH_FUZZY = 0.6

_PHRASE_BONUS = 0.15      # query is a substring of the product name
_HEAD_BONUS = 0.10        # …and the name *starts* with it ("Ипотечен кредит за…" ≻ "Застраховка … ипотечен кредит")
_SPECIFICITY = 0.1        # prefer the card whose name is mostly covered by the query
_ALIAS_EXACT_REL = 0.95   # query == one of the curated aliases
_BOOST_SCALE = 0.2        # priority_boost 1.35 → +0.07 relevance (nudge, not a hammer)
_UNMATCHED_PENALTY = 0.5  # weight of query tokens that hit nothing in the catalog
_PIN_RELEVANCE = 1.0      # exact_match_pins → hard #1
_PREFIX_MIN_LEN = 2
_FUZZY_MIN_LEN = 5        # shorter tokens produce too many false neighbours ("пица" → "лица")
_MAX_EXPANSIONS = 8


@dataclass
class _IndexedCard:
    card_id: str
    document_ids: list[int]
    product_name: str
    product_summary: str
    priority_boost: float
    exact_match_pins: list[str]
    aliases_normalized: set[str]
    name_normalized: str
    name_variants_normalized: list[str]
    name_terms: set[str] = field(default_factory=set)
    # term -> best (highest-weight) field weight it appears in
    best_field_w: dict[str, float] = field(default_factory=dict)
    # term -> field-weighted tf (BM25F)
    weighted_tf: dict[str, float] = field(default_factory=dict)
    weighted_len: float = 0.0


@dataclass
class _Match:
    term: str
    quality: float  # _MATCH_*


class ProductCardRanker:
    def __init__(self, cards: list[dict]) -> None:
        self._raw_cards = cards
        self._build()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, path: "str | Path") -> "ProductCardRanker":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def reload(self, path: "str | Path") -> None:
        self._raw_cards = json.loads(Path(path).read_text(encoding="utf-8"))
        self._build()

    def _build(self) -> None:
        indexed: list[_IndexedCard] = []
        core_vocab: set[str] = set()
        for c in self._raw_cards:
            name_display = c.get("product_name_bg") or c.get("product_name_en") or ""
            summary_display = c.get("summary_bg") or c.get("summary_en") or ""
            fields_text = {
                "name": " ".join(c.get("name_variants") or [name_display]),
                "aliases": " ".join(c.get("aliases", [])),
                "category": c.get("category", "") or "",
                "summary": " ".join(filter(None, [c.get("summary_bg"), c.get("summary_en")])),
                "keywords": " ".join(c.get("keywords", [])),
                "body": " ".join(filter(None, [c.get("text_bg"), c.get("text_en")])),
            }
            ic = _IndexedCard(
                card_id=c["card_id"],
                document_ids=list(c["document_ids"]),
                product_name=name_display,
                product_summary=summary_display,
                priority_boost=float(c.get("priority_boost", 1.0)),
                exact_match_pins=[normalize_text(p) for p in c.get("exact_match_pins", [])],
                aliases_normalized={normalize_text(a) for a in c.get("aliases", [])},
                name_normalized=normalize_text(fields_text["name"]),
                name_variants_normalized=[normalize_text(v) for v in (c.get("name_variants") or [name_display])],
            )
            for f in _FIELDS:
                w = _FIELD_WEIGHTS[f]
                toks = tokenize(fields_text[f])
                ic.weighted_len += w * len(toks)
                for t in toks:
                    ic.weighted_tf[t] = ic.weighted_tf.get(t, 0.0) + w
                    if w > ic.best_field_w.get(t, 0.0):
                        ic.best_field_w[t] = w
                if f == "name":
                    ic.name_terms = set(toks)
                if f in ("name", "aliases", "category", "summary"):
                    core_vocab.update(toks)
            indexed.append(ic)

        self._cards = indexed
        N = len(indexed)
        df: dict[str, int] = {}
        for ic in indexed:
            for t in ic.weighted_tf:
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        idfs = sorted(self._idf.values())
        self._median_idf = idfs[len(idfs) // 2] if idfs else 1.0
        self._avg_len = sum(ic.weighted_len for ic in indexed) / max(N, 1)
        # Prefix / fuzzy expansion runs against the *core* vocabulary
        # (name + aliases + summary) — body vocabulary is too noisy for it.
        self._core_set = core_vocab
        self._core_vocab = sorted(core_vocab, key=lambda t: -df.get(t, 0))  # for fuzzy
        self._core_sorted = sorted(core_vocab)                              # for prefix bisect
        self._body_sorted = sorted(set(df) - core_vocab)

    # ------------------------------------------------------------------ #
    # Query understanding
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prefix_range(sorted_vocab: list[str], stem: str) -> list[str]:
        lo = bisect.bisect_left(sorted_vocab, stem)
        hi = bisect.bisect_left(sorted_vocab, stem + "￿")
        return sorted_vocab[lo:hi]

    def _prefix_matches(self, stem: str) -> list[str]:
        if len(stem) < _PREFIX_MIN_LEN:
            return []
        hits = self._prefix_range(self._core_sorted, stem)
        if not hits and len(stem) >= 3:
            hits = self._prefix_range(self._body_sorted, stem)
        # most frequent terms first — they are the likeliest completions
        hits.sort(key=lambda t: -self._df.get(t, 0))
        return hits[:_MAX_EXPANSIONS]

    def _fuzzy_matches(self, stem: str) -> list[str]:
        if _rf_process is None or len(stem) < _FUZZY_MIN_LEN:
            return []
        max_d = 1 if len(stem) < 7 else 2
        hits = _rf_process.extract(
            stem, self._core_vocab, scorer=_rf_lev.distance,
            score_cutoff=max_d, limit=_MAX_EXPANSIONS * 2,
        )
        # Typos rarely hit the first letter — requiring it removes most junk.
        return [h[0] for h in hits if h[0][0] == stem[0]][:_MAX_EXPANSIONS]

    def _match_token(self, raw: str, *, is_last: bool) -> list[_Match]:
        """Map one raw query token to catalog terms.

        Strategies, best first: exact → prefix (last token only) →
        transliteration → fuzzy. An exact hit that exists only deep in some
        page body ("депозит") does not stop us from also trying the
        cross-script / fuzzy route ("deposit" in a product name)."""
        stem = _stem(raw)
        out: list[_Match] = []
        seen: set[str] = set()

        def add(terms: Iterable[str], quality: float) -> None:
            for t in terms:
                if t not in seen:
                    seen.add(t)
                    out.append(_Match(t, quality))

        exact = stem in self._idf
        if exact:
            add([stem], _MATCH_EXACT)
        if is_last:
            add(self._prefix_matches(stem), _MATCH_PREFIX)
        if not exact or stem not in self._core_set:
            tr = _stem(transliterate(raw))
            translit_hit = False
            if tr in self._idf:
                add([tr], _MATCH_TRANSLIT)
                translit_hit = True
            elif is_last:
                pre = self._prefix_matches(tr)
                add(pre, _MATCH_TRANSLIT)
                translit_hit = bool(pre)
            if not translit_hit:
                add(self._fuzzy_matches(stem) or self._fuzzy_matches(tr), _MATCH_FUZZY)
        return out

    def analyze(self, query: str) -> list[list[_Match]]:
        """Debug helper: how each query token was interpreted."""
        raws = [t.lower() for t in _TOKEN_RE.findall(query)]
        raws = [t for t in raws if t not in _STOP_BG and t not in _STOP_EN] or raws
        return [self._match_token(r, is_last=(i == len(raws) - 1)) for i, r in enumerate(raws)]

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def rank(self, query: str, top_k: int = 5, min_relevance: float = 0.1) -> list[dict]:
        if not query or not query.strip():
            return []
        q_norm = normalize_text(query)
        all_groups = self.analyze(query)
        groups = [g for g in all_groups if g]
        if not groups:
            return []

        # Per query-token weight: idf of its best candidate — rare words dominate.
        # Tokens that matched nothing still count (at half weight) so that
        # "credit crad" cannot score as high as "credit card".
        tok_idf = [max(self._idf.get(m.term, 0.0) for m in g) for g in groups]
        n_unmatched = len(all_groups) - len(groups)
        idf_total = (sum(tok_idf) + n_unmatched * self._median_idf * _UNMATCHED_PENALTY) or 1.0

        scored: list[tuple[float, float, _IndexedCard]] = []
        for ic in self._cards:
            coverage = 0.0
            bm25 = 0.0
            name_hits: set[str] = set()
            for g, w_q in zip(groups, tok_idf):
                best = 0.0
                for m in g:
                    fw = ic.best_field_w.get(m.term)
                    if fw is None:
                        continue
                    tf = ic.weighted_tf[m.term]
                    denom = tf + _K1 * (1 - _B + _B * ic.weighted_len / max(self._avg_len, 1e-9))
                    bm25 += self._idf[m.term] * tf * (_K1 + 1) / denom * m.quality
                    best = max(best, m.quality * fw / _MAX_FIELD_W)
                    if m.term in ic.name_terms:
                        name_hits.add(m.term)
                coverage += w_q * best
            if coverage <= 0.0:
                continue
            rel = coverage / idf_total
            # "кредитна карта" → the card *named* "Кредитна карта Galaxy" beats
            # "Застраховка … към кредитна карта": more of its name is covered.
            if ic.name_terms:
                rel *= 1.0 - _SPECIFICITY * (1.0 - len(name_hits) / len(ic.name_terms))
            if q_norm in ic.aliases_normalized:
                rel = max(rel, _ALIAS_EXACT_REL)
            if len(q_norm) >= 3 and q_norm in ic.name_normalized:
                rel += _PHRASE_BONUS
                if any(v.startswith(q_norm) for v in ic.name_variants_normalized):
                    rel += _HEAD_BONUS
            rel += (ic.priority_boost - 1.0) * _BOOST_SCALE
            if q_norm in ic.exact_match_pins:
                rel = max(rel, _PIN_RELEVANCE) + 1.0  # sort above everything, clamp later
            scored.append((rel, bm25 * ic.priority_boost, ic))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out = []
        for rel, _bm, ic in scored:
            rel = min(rel, 1.0)
            if rel < min_relevance:
                break
            out.append(
                {
                    # a card may cover several client-type variants; the first
                    # document_id is the representative one.
                    "document_id": ic.document_ids[0],
                    "product_name": ic.product_name,
                    "product_summary": ic.product_summary,
                    "relevance": round(float(rel), 4),
                }
            )
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._cards)

    def card_ids(self) -> Iterable[str]:
        return (c.card_id for c in self._cards)


__all__ = ["ProductCardRanker", "tokenize", "normalize_text", "transliterate"]
