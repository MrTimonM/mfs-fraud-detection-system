import type { Analysis, Decision, RiskLevel, TransactionInput } from "./domain";
// Response of the Python scoring service (mfs-guard-ml/backend, LightGBM).
export interface ModelEvidence {
  fraud_probability: number;
  risk_level: RiskLevel;
  decision: Decision;
  model: string;
  model_version: string;
  rule_risk_score: number;
  triggered_rules: string[];
  anomaly_score: number | null;
  device_risk: number;
  recipient_risk: number;
  graph_risk: number;
  top_risk_factors: string[];
  latency_ms: number;
  system_mode: string;
  requires_human_review: boolean;
  missing_optional_fields?: string[];
  profile_source?: string;
}
export type SystemMode = "ML_ACTIVE" | "DEGRADED";
export interface ModelOutcome {
  evidence: ModelEvidence | null;
  system_mode: SystemMode;
  error?: string;
  round_trip_ms?: number;
}
const decisions: Decision[] = [
  "APPROVE",
  "APPROVE_AND_MONITOR",
  "STEP_UP_AUTH",
  "TEMPORARY_HOLD",
  "REJECT_AND_FREEZE",
];
export const severity = (d: Decision) => decisions.indexOf(d);
export function modelServiceUrl() {
  // On by default: the local scoring service from mfs-guard-ml/backend.
  return (
    process.env.MFS_ML_SERVICE_URL?.replace(/\/$/, "") ||
    "http://127.0.0.1:8000"
  );
}
// Scores before authorization. Any failure returns DEGRADED so rules decide.
export async function scoreWithModel(
  payload: TransactionInput,
  timeoutMs = Number(process.env.MFS_ML_TIMEOUT_MS ?? 3000),
): Promise<ModelOutcome> {
  const url = modelServiceUrl();
  if (!url)
    return {
      evidence: null,
      system_mode: "DEGRADED",
      error: "MFS_ML_SERVICE_URL is not configured",
    };
  const started = performance.now();
  try {
    const res = await fetch(`${url}/api/v1/transactions/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(timeoutMs),
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`Model service returned ${res.status}`);
    const evidence = (await res.json()) as ModelEvidence;
    if (
      typeof evidence.fraud_probability !== "number" ||
      !decisions.includes(evidence.decision)
    )
      throw new Error("Model service returned an invalid response");
    return {
      evidence,
      system_mode: evidence.system_mode === "DEGRADED" ? "DEGRADED" : "ML_ACTIVE",
      round_trip_ms: performance.now() - started,
    };
  } catch (e) {
    return {
      evidence: null,
      system_mode: "DEGRADED",
      error: e instanceof Error ? e.message : "Model service unavailable",
    };
  }
}
// The AI model is the decision-maker. Without a model score the transaction
// is held for analyst review; it is never approved by default.
export function applyModel(a: Analysis, outcome: ModelOutcome): Analysis {
  a.rule_decision = a.decision;
  a.system_mode = outcome.system_mode;
  a.model_error = outcome.error;
  const ml = outcome.evidence;
  if (!ml) {
    a.decision = "TEMPORARY_HOLD";
    a.risk_level = "HIGH";
    return a;
  }
  a.ml = { ...ml, round_trip_ms: outcome.round_trip_ms };
  a.decision = ml.decision;
  a.risk_level = ml.risk_level;
  a.risk_score = Math.round(ml.fraud_probability * 100);
  return a;
}
