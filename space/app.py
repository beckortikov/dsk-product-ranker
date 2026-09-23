"""Hugging Face Space entry point (Gradio SDK, ZeroGPU free tier).

The Space holds only this file + requirements.txt. On start it clones the
public GitHub repo (single source of truth) into a writable temp dir and
serves the real FastAPI app from there, mounted *inside* Gradio's server:

    /               Gradio search box (what the Space iframe shows) + links
    /dashboard/     the full live dashboard
    /dashboard/rank-product-cards, /dashboard/docs, ...   the API

Why inside Gradio: on ZeroGPU hardware the `spaces` runtime owns port 7860
and only hands traffic to a server started through `Blocks.launch()`, so a
bare uvicorn on 7860 fails with "address already in use".
Restarting the Space = fresh clone = latest commit.
"""
from __future__ import annotations

# ZeroGPU requires `import spaces` before gradio and one @spaces.GPU function.
try:
    import spaces  # noqa: F401
except ImportError:  # local run / CPU hardware
    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(fn=None, **_):
            return fn if fn else (lambda f: f)

import importlib.util
import os
import subprocess
import sys

import gradio as gr
from fastapi.responses import RedirectResponse
from starlette.routing import Mount, Route

REPO = os.environ.get("DSK_REPO", "https://github.com/beckortikov/dsk-product-ranker")
DST = os.environ.get("DSK_DIR", "/tmp/dsk-product-ranker")

if not os.path.exists(os.path.join(DST, "ranker.py")):
    subprocess.run(["git", "clone", "--depth", "1", REPO, DST], check=True)

os.chdir(DST)
sys.path.insert(0, DST)
os.environ.setdefault("CATALOG_PATH", os.path.join(DST, "products.json"))


@spaces.GPU
def _zerogpu_placeholder() -> str:
    """Satisfies the ZeroGPU runtime check; never invoked (CPU-only ranker)."""
    return "cpu-only ranker"


# The repo's module is also called app.py -> load it under another name.
_spec = importlib.util.spec_from_file_location("dsk_app", os.path.join(DST, "app.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dsk_app"] = _mod
_spec.loader.exec_module(_mod)
api = _mod.app

from ranker import ProductCardRanker  # noqa: E402  (path inserted above)

_ranker = ProductCardRanker.from_json(os.environ["CATALOG_PATH"])


def _search(q: str):
    rows = _ranker.rank(q or "", top_k=8)
    return [[r["relevance"], r["document_id"], r["product_name"], r["product_summary"]] for r in rows]


with gr.Blocks(title="DSK Product Cards Ranker") as demo:
    gr.Markdown(
        "## DSK Bank — Product Cards Ranker\n"
        "**[Open the full live dashboard →](dashboard/)** (relevance chart, query analysis, eval, latency) · "
        "[API docs](dashboard/docs) · [code on GitHub](https://github.com/beckortikov/dsk-product-ranker)\n\n"
        "Quick try below: ranks on every keystroke, Bulgarian or English."
    )
    q = gr.Textbox(label="query", placeholder="мобилно банкиране · kreditna karta · student loan · dsk mob")
    out = gr.Dataframe(headers=["relevance", "document_id", "product_name", "product_summary"], interactive=False)
    q.change(_search, q, out)
    gr.Examples(["DSK Mobile", "mobile", "мобилно банкиране", "ипотечен кре", "kreditna karta", "student loan"], q)


if __name__ == "__main__":
    # Gradio (patched by `spaces` on ZeroGPU) owns the port; we attach our
    # FastAPI app to its server after it is up. ssr_mode=False: with SSR on,
    # a Node proxy sits on the public port and answers *every* path with the
    # Gradio index, so /dashboard would never reach Python.
    demo.launch(server_name="0.0.0.0", prevent_thread_lock=True, ssr_mode=False)
    server = getattr(demo, "server_app", None) or getattr(demo, "app", None)
    # Gradio ends its route table with a catch-all that serves its index for
    # any path, so our routes must go *in front* of it, not be appended.
    server.router.routes.insert(0, Mount("/dashboard", app=api))
    server.router.routes.insert(0, Route("/dashboard", endpoint=lambda request: RedirectResponse(url="/dashboard/")))
    print("mounted ranker app at /dashboard (front of route table)", flush=True)
    demo.block_thread()
