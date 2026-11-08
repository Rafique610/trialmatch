"""Request and response models. Constraints here give clear 422s and a typed /docs page."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Note = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
CriterionText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2_000)]
Mode = Literal["no_cache", "kv_cache", "speculative"]
Label = Literal["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]


class Criterion(BaseModel):
    type: Literal["inclusion", "exclusion"]
    text: CriterionText


class ScreenRequest(BaseModel):
    note: Note = Field(description="Any patient note. Plain text is split into sentences; a note already numbered '0. …' is used as numbered.")
    criteria: list[Criterion] = Field(min_length=1, max_length=30)
    mode: Mode = "kv_cache"
    gamma: int = Field(2, ge=1, le=8, description="Draft tokens per round (speculative mode only)")

    model_config = {"json_schema_extra": {"examples": [{
        "note": "A 58-year-old man with type 2 diabetes presents with chest pain. His HbA1c is 8.1%. He has no history of stroke.",
        "criteria": [{"type": "inclusion", "text": "Adults aged 18 to 75 years"},
                     {"type": "exclusion", "text": "History of stroke"},
                     {"type": "inclusion", "text": "HbA1c between 7% and 10%"}],
        "mode": "kv_cache", "gamma": 2}]}}


class Sentence(BaseModel):
    id: int
    text: str


class Timing(BaseModel):
    prompt_tokens: int
    new_tokens: int
    prefill_ms: float
    decode_ms: float
    tok_s: float
    peak_gpu_gb: float
    gamma: int | None = None
    rounds: int | None = None
    acceptance_rate: float | None = None


class CriterionResult(BaseModel):
    type: Literal["inclusion", "exclusion"]
    text: str
    label: Label | None = Field(description="None when the model's answer could not be parsed")
    evidence_ids: list[int]
    evidence: list[Sentence]
    explanation: str
    missing_information: bool
    parse_error: str | None
    raw_output: str
    warnings: list[str]
    timing: Timing


class Overall(BaseModel):
    outcome: Literal["likely eligible", "likely ineligible", "needs review"]
    reason: str
    driving_criteria: list[int] = Field(description="Indexes into results")
    rule: str
    disclaimer: str


class Provenance(BaseModel):
    base_model: str
    adapter: str
    adapter_sha256: str | None
    draft_model: str
    precision: str
    decoding: str
    gpu: str


class ScreenResponse(BaseModel):
    sentences: list[Sentence]
    results: list[CriterionResult]
    overall: Overall
    mode: Mode
    gamma: int | None
    provenance: Provenance


class BenchRequest(BaseModel):
    note: Note
    criterion: Criterion
    gamma: int = Field(2, ge=1, le=8)


class AttentionBenchRequest(BaseModel):
    seq_lengths: list[Literal[256, 512, 1024, 2048, 4096]] = Field([256, 1024, 2048], min_length=1, max_length=5)
    block: Literal[32, 64, 128, 256] = 128


class ReportRequest(BaseModel):
    result: ScreenResponse
    format: Literal["markdown", "html"] = "html"
