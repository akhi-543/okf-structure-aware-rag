from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from okf_rag.ingest.load_pg import get_conn
from okf_web_bench.okf_web_bench_compare import (
    ALLOWED_CORPORA,
    ARM_FLAT,
    run_compare,
)
from okf_web_bench.okf_web_bench_models import BgeQueryEncoder, QwenGenerator, health_payload
from okf_web_bench.okf_web_bench_store import WebBenchStore, default_dsn, probe_postgres

_STATIC_DIR = Path(__file__).resolve().parent / "okf_web_bench_static"
# Only the flat lane embeds the question; the structured lane is metadata BM25.
_DENSE_ARMS = frozenset({ARM_FLAT})


def _load_qwen_runtime() -> tuple[object | None, bool, object | None]:
    if os.environ.get("OKF_WEB_BENCH_SKIP_QWEN") == "1":
        return QwenGenerator.load_tokenizer(), False, None
    qwen = QwenGenerator.from_pretrained()
    generate_fn = qwen.generate if qwen.available else None
    return qwen.tokenizer, qwen.available, generate_fn


@dataclass
class WebBenchRuntime:
    store: WebBenchStore
    encoder_available: bool
    qwen_available: bool
    tokenizer: object
    generate_fn: object | None
    cuda: bool
    cuda_index: int | None
    postgres: bool


class CompareRequest(BaseModel):
    question: str
    corpus: str
    lane_a: str
    lane_b: str
    top_k: int
    gpu: bool = False


_runtime: WebBenchRuntime | None = None


def _cuda_status() -> tuple[bool, int | None]:
    try:
        import torch

        cuda = torch.cuda.is_available()
        index = torch.cuda.current_device() if cuda else None
        return cuda, index
    except Exception:
        return False, None


def get_runtime() -> WebBenchRuntime:
    global _runtime
    if _runtime is None:
        dsn = default_dsn()
        conn_factory = lambda: get_conn(dsn)
        postgres = probe_postgres(conn_factory)
        encoder = BgeQueryEncoder.from_pretrained()
        cuda, cuda_index = _cuda_status()
        tokenizer, qwen_available, generate_fn = _load_qwen_runtime()
        store = WebBenchStore(conn_factory, encoder)
        _runtime = WebBenchRuntime(
            store=store,
            encoder_available=encoder.available,
            qwen_available=qwen_available,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
            cuda=cuda,
            cuda_index=cuda_index,
            postgres=postgres,
        )
    return _runtime


app = FastAPI()
app.mount(
    "/okf-web-bench-static",
    StaticFiles(directory=_STATIC_DIR),
    name="okf-web-bench-static",
)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "okf_web_bench_console.html")


@app.get("/api/okf-web-bench/health")
def health(runtime: WebBenchRuntime = Depends(get_runtime)) -> dict[str, Any]:
    payload = health_payload(
        postgres=runtime.postgres,
        bge=runtime.encoder_available,
        qwen=runtime.qwen_available,
        cuda=runtime.cuda,
        cuda_index=runtime.cuda_index,
    )
    payload["arms"] = runtime.store.available_arms()
    return payload


@app.get("/api/okf-web-bench/arms")
def arms(runtime: WebBenchRuntime = Depends(get_runtime)) -> dict[str, Any]:
    return {"arms": runtime.store.available_arms()}


@app.post("/api/okf-web-bench/compare")
def compare(
    body: CompareRequest,
    runtime: WebBenchRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    if not runtime.postgres:
        raise HTTPException(status_code=503, detail="postgres unavailable")
    if (
        body.lane_a in _DENSE_ARMS or body.lane_b in _DENSE_ARMS
    ) and not runtime.encoder_available:
        raise HTTPException(status_code=503, detail="bge unavailable")

    try:
        return run_compare(
            question=body.question,
            corpus=body.corpus,
            lane_a=body.lane_a,
            lane_b=body.lane_b,
            top_k=body.top_k,
            tokenizer=runtime.tokenizer,
            retrieve_fn=runtime.store.retrieve,
            generate_fn=runtime.generate_fn,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/okf-web-bench/corpus-map")
def corpus_map(
    corpus: str = Query(...),
    runtime: WebBenchRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    if not runtime.postgres:
        raise HTTPException(status_code=503, detail="postgres unavailable")
    if corpus not in ALLOWED_CORPORA:
        raise HTTPException(
            status_code=400,
            detail=f"corpus must be one of {sorted(ALLOWED_CORPORA)}",
        )
    return {"corpus": corpus, "documents": runtime.store.list_documents(corpus)}


@app.get("/api/okf-web-bench/corpus-map/doc")
def corpus_map_doc(
    corpus: str = Query(...),
    path: str = Query(...),
    runtime: WebBenchRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    if not runtime.postgres:
        raise HTTPException(status_code=503, detail="postgres unavailable")
    if corpus not in ALLOWED_CORPORA:
        raise HTTPException(
            status_code=400,
            detail=f"corpus must be one of {sorted(ALLOWED_CORPORA)}",
        )
    try:
        return runtime.store.load_document_map(corpus, path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
