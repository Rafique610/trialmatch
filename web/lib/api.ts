/** Typed client for the FastAPI backend (src/app). Every number the UI shows comes through here. */

export type CriterionType = "inclusion" | "exclusion";
export type Label = "included" | "not included" | "excluded" | "not excluded" | "not enough information" | "not applicable";
export type Mode = "no_cache" | "kv_cache" | "speculative";

export const LABELS: Record<CriterionType, Label[]> = {
  inclusion: ["included", "not included", "not enough information", "not applicable"],
  exclusion: ["excluded", "not excluded", "not enough information", "not applicable"],
};

export interface Criterion { type: CriterionType; text: string; expert_label?: Label; expert_evidence_ids?: number[] }
export interface Sentence { id: number; text: string }
export interface Timing {
  prompt_tokens: number; new_tokens: number; prefill_ms: number; decode_ms: number; tok_s: number; peak_gpu_gb: number;
  gamma?: number | null; rounds?: number | null; acceptance_rate?: number | null;
}
export interface CriterionResult {
  type: CriterionType; text: string; label: Label | null; evidence_ids: number[]; evidence: Sentence[]; explanation: string;
  missing_information: boolean; parse_error: string | null; raw_output: string; warnings: string[]; timing: Timing;
}
export interface Overall {
  outcome: "likely eligible" | "likely ineligible" | "needs review"; reason: string; driving_criteria: number[]; rule: string; disclaimer: string;
}
export interface Provenance {
  base_model: string; adapter: string; adapter_sha256: string | null; draft_model: string; precision: string; decoding: string; gpu: string;
}
export interface ScreenResponse {
  sentences: Sentence[]; results: CriterionResult[]; overall: Overall; mode: Mode; gamma: number | null; provenance: Provenance;
}
export interface Health { status: "ok" | "loading"; model_loaded: boolean; draft_loaded: boolean; provenance: Provenance | null }
export interface Example { id: string; patient_id: string; trial_id: string; trial_title: string; note: string; criteria: Criterion[] }

export interface DecoderRun extends Timing {
  mode: Mode; same_tokens_as_kv_cache: boolean; answer: string; speedup_vs_no_cache: number; new_tokens: number;
}
export interface DecoderBench { prompt_tokens: number; gamma: number; runs: DecoderRun[]; provenance: Provenance }
export interface Measure { ms?: number; ms_std?: number; extra_mb?: number; error?: string }
export interface AttentionRow { seq: number; max_abs_error_tiled_vs_reference: number; reference: Measure; tiled: Measure; torch_sdpa: Measure }
export interface AttentionBench { heads: number; head_dim: number; dtype: string; block: number; repeats: number; rows: AttentionRow[] }

/** A failed request, already phrased for the person using the page. */
export class ApiError extends Error {
  constructor(message: string, public status: number, public recovery: string) { super(message); }
}

function explain(status: number, detail: unknown): ApiError {
  if (status === 0 || status === 502 || status === 504)
    return new ApiError("The TrialMatch backend is not reachable.", status, "Start it with `task app` (or `task api`) and try again.");
  if (status === 503)
    return new ApiError(typeof detail === "string" ? detail : "The model is not ready.", status, "Wait for the status light to turn green, then retry.");
  if (status === 422) {
    const msg = Array.isArray(detail)
      ? detail.map((d: { loc?: (string | number)[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(" › ")}: ${d.msg}`).join("; ")
      : String(detail);
    return new ApiError(msg, status, "Fix the input above and run it again.");
  }
  return new ApiError(typeof detail === "string" ? detail : "Something went wrong on the server.", status, "Try again; if it repeats, check the backend log.");
}

/** The backend always answers errors in JSON; a non-JSON error came from the Next proxy failing to reach it. */
async function failure(res: Response): Promise<ApiError> {
  if (!(res.headers.get("content-type") ?? "").includes("application/json")) return explain(0, null);
  return explain(res.status, await res.json().then((j) => j.detail, () => null));
}

async function call<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      method: body === undefined ? "GET" : "POST", signal, cache: "no-store",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw explain(0, null);
  }
  if (!res.ok) throw await failure(res);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => call<Health>("/health"),
  examples: () => call<Example[]>("/examples"),
  screenOne: (note: string, c: Criterion, mode: Mode, gamma: number, signal?: AbortSignal) =>
    call<ScreenResponse>("/screen", { note, criteria: [{ type: c.type, text: c.text }], mode, gamma }, signal),
  aggregate: (results: { type: CriterionType; label: Label | null }[]) => call<Overall>("/aggregate", { results }),
  benchmark: (note: string, c: Criterion, gamma: number) =>
    call<DecoderBench>("/benchmark", { note, criterion: { type: c.type, text: c.text }, gamma }),
  attention: (seq_lengths: number[], block: number) => call<AttentionBench>("/attention-bench", { seq_lengths, block }),
  async report(result: ScreenResponse, format: "html" | "markdown") {
    const res = await fetch("/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ result, format }) });
    if (!res.ok) throw await failure(res);
    return res.blob();
  },
};

export const MODE_NAMES: Record<Mode, string> = { no_cache: "No cache", kv_cache: "KV cache", speculative: "Speculative" };
