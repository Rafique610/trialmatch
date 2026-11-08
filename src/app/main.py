"""Creates the app, loads the model once, maps errors to safe messages, includes the router.
Run: uv run uvicorn src.app.main:app --port 8000   (or: task app)   ->  http://localhost:8000/docs
"""

import logging
import threading

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.app.routers.screening import router
from src.app.screening import InputError

log = logging.getLogger("trialmatch")


class EngineNotReady(RuntimeError):
    pass


def create_app(engine=None, load_on_startup: bool = True) -> FastAPI:
    """`engine` lets tests inject a stand-in; otherwise the real model loads in a background thread at startup."""
    holder = {"engine": engine, "error": None}

    def load():
        try:
            from src.app.engine import Engine

            holder["engine"] = Engine()
        except Exception as e:  # reported by /health and 503s, details only in the server log
            log.exception("model failed to load")
            holder["error"] = type(e).__name__

    async def lifespan(app: FastAPI):
        if holder["engine"] is None and load_on_startup:
            threading.Thread(target=load, daemon=True).start()
        yield

    app = FastAPI(title="TrialMatch API", version="1.0",
                  description="Criterion-level clinical-trial screening with evidence. Educational prototype, not for clinical decisions.",
                  lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

    def get():
        if holder["engine"] is None:
            raise EngineNotReady(holder["error"] or "loading")
        return holder["engine"]

    app.state.engine = get
    app.state.engine_or_none = lambda: holder["engine"]

    @app.exception_handler(InputError)
    async def bad_input(_: Request, e: InputError):
        return JSONResponse(status_code=422, content={"detail": str(e)})

    @app.exception_handler(EngineNotReady)
    async def not_ready(_: Request, e: EngineNotReady):
        msg = "The model is still loading; try again in a minute." if str(e) == "loading" else "The model failed to load; check the server log."
        return JSONResponse(status_code=503, content={"detail": msg})

    @app.exception_handler(torch.OutOfMemoryError)
    async def oom(_: Request, e):
        torch.cuda.empty_cache()
        return JSONResponse(status_code=503, content={"detail": "The GPU ran out of memory for this input. Try a shorter note or fewer criteria."})

    @app.exception_handler(Exception)
    async def unexpected(_: Request, e: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal error; see the server log."})

    app.include_router(router)
    return app


app = create_app()
