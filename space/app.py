"""Hugging Face Space entry point (Gradio SDK, free tier).

The Space itself holds only this file + requirements.txt. On start it clones
the public GitHub repo (single source of truth) into a writable temp dir and
serves the real FastAPI app from there: dashboard at `/`, API at
`/rank-product-cards`, Swagger at `/docs`. A tiny Gradio UI is mounted at
`/gradio` as well. Restarting the Space = fresh clone = latest commit.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

REPO = os.environ.get("DSK_REPO", "https://github.com/beckortikov/dsk-product-ranker")
DST = os.environ.get("DSK_DIR", "/tmp/dsk-product-ranker")
PORT = int(os.environ.get("PORT", "7860"))

if not os.path.exists(os.path.join(DST, "ranker.py")):
    subprocess.run(["git", "clone", "--depth", "1", REPO, DST], check=True)

os.chdir(DST)
sys.path.insert(0, DST)
os.environ.setdefault("CATALOG_PATH", os.path.join(DST, "products.json"))

# The repo's module is also called app.py -> load it under another name.
_spec = importlib.util.spec_from_file_location("dsk_app", os.path.join(DST, "app.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dsk_app"] = _mod
_spec.loader.exec_module(_mod)
api = _mod.app

try:  # optional: a plain Gradio search box at /gradio
    import gradio as gr

    from ranker import ProductCardRanker  # noqa: E402  (path inserted above)

    _ranker = ProductCardRanker.from_json(os.environ["CATALOG_PATH"])

    def _search(q: str):
        rows = _ranker.rank(q or "", top_k=8)
        return [[r["relevance"], r["document_id"], r["product_name"], r["product_summary"]] for r in rows]

    with gr.Blocks(title="DSK Product Cards Ranker") as demo:
        gr.Markdown("## DSK Bank Product Cards Ranker\nFull dashboard: [/](../) · API docs: [/docs](../docs)")
        q = gr.Textbox(label="query (BG / EN, ranks on every keystroke)", placeholder="мобилно банкиране")
        out = gr.Dataframe(headers=["relevance", "document_id", "product_name", "product_summary"], interactive=False)
        q.change(_search, q, out)
    api = gr.mount_gradio_app(api, demo, path="/gradio")
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=PORT)
