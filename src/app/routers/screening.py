"""Thin routers: each endpoint wires a schema to one function."""

from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.app import bench, examples, report, screening
from src.app.schemas.screen import (AggregateRequest, AttentionBenchRequest, BenchRequest, Overall, ReportRequest, ScreenRequest,
                                    ScreenResponse)

router = APIRouter()


def engine(request: Request):
    return request.app.state.engine()


@router.get("/health")
def health(request: Request) -> dict:
    e = request.app.state.engine_or_none()
    return {"status": "ok" if e else "loading", "model_loaded": e is not None,
            "draft_loaded": bool(e and e.draft_loaded), "provenance": e.provenance if e else None}


@router.get("/examples")
def list_examples() -> list[dict]:
    """Validation-split notes with their expert-annotated criteria (the UI's "load example")."""
    return examples.examples()


@router.post("/screen", response_model=ScreenResponse)
def screen(body: ScreenRequest, request: Request) -> dict:
    return screening.screen(engine(request), body.note, [c.model_dump() for c in body.criteria], body.mode, body.gamma)


@router.post("/aggregate", response_model=Overall)
def aggregate(body: AggregateRequest) -> dict:
    """The ADR 0002 outcome rule over labels screened one criterion at a time (the UI's row-by-row run)."""
    return screening.aggregate([r.model_dump() for r in body.results])


@router.post("/benchmark")
def benchmark(body: BenchRequest, request: Request) -> dict:
    """The same note + criterion through no-cache, KV-cache and speculative decoding, measured now."""
    return bench.decoders(engine(request), body.note, body.criterion.type, body.criterion.text, body.gamma)


@router.post("/attention-bench")
def attention_bench(body: AttentionBenchRequest, request: Request) -> dict:
    """Small live run: reference vs tiled vs torch SDPA attention (12 heads, d = 128, fp16, causal)."""
    return bench.attention_live(engine(request), body.seq_lengths, body.block)


@router.post("/report")
def make_report(body: ReportRequest) -> Response:
    result = body.result.model_dump()
    if body.format == "markdown":
        return Response(report.markdown(result), media_type="text/markdown",
                        headers={"Content-Disposition": 'attachment; filename="trialmatch-report.md"'})
    return Response(report.to_html(result), media_type="text/html",
                    headers={"Content-Disposition": 'attachment; filename="trialmatch-report.html"'})
