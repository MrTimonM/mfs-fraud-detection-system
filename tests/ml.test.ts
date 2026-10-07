import test from "node:test";
import assert from "node:assert/strict";
import { emptyState, scenario } from "../src/lib/scenarios";
import { ingest } from "../src/lib/service";
import { scoreWithModel, type ModelEvidence } from "../src/lib/ml";
const now = "2026-10-07T10:00:00.000Z";
const evidence = (over: Partial<ModelEvidence> = {}): ModelEvidence => ({
  fraud_probability: 0.5,
  risk_level: "CRITICAL",
  decision: "TEMPORARY_HOLD",
  model: "LightGBM",
  model_version: "LightGBM_200k-v1",
  rule_risk_score: 10,
  triggered_rules: [],
  anomaly_score: null,
  device_risk: 0,
  recipient_risk: 0,
  graph_risk: 0,
  top_risk_factors: ["new device"],
  latency_ms: 12,
  system_mode: "NORMAL",
  requires_human_review: true,
  ...over,
});
test("model decision replaces the rule decision when the service responds", () => {
  const a = ingest(emptyState(), scenario("normal", now), {
    evidence: evidence(),
    system_mode: "ML_ACTIVE",
  });
  assert.equal(a.rule_decision, "APPROVE");
  assert.equal(a.decision, "TEMPORARY_HOLD");
  assert.equal(a.system_mode, "ML_ACTIVE");
  assert.equal(a.ml?.model_version, "LightGBM_200k-v1");
});
test("model approval is final even when legacy rules would flag", () => {
  const a = ingest(emptyState(), scenario("fraud", now), {
    evidence: evidence({ decision: "APPROVE", risk_level: "LOW", fraud_probability: 0.03 }),
    system_mode: "ML_ACTIVE",
  });
  assert.equal(a.decision, "APPROVE");
  assert.equal(a.risk_score, 3);
});
test("model outage holds for review and is marked degraded", async () => {
  const saved = process.env.MFS_ML_SERVICE_URL;
  process.env.MFS_ML_SERVICE_URL = "http://127.0.0.1:9";
  const outcome = await scoreWithModel(scenario("fraud", now), 500);
  process.env.MFS_ML_SERVICE_URL = saved;
  assert.equal(outcome.system_mode, "DEGRADED");
  const a = ingest(emptyState(), scenario("normal", now), outcome);
  assert.equal(a.decision, "TEMPORARY_HOLD");
  assert.equal(a.ml, undefined);
});
