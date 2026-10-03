import test from "node:test";
import assert from "node:assert/strict";
import { analyze } from "../src/lib/engine";
import { averagePathLength } from "../src/lib/anomaly";
import { impactSummary, investigationBrief } from "../src/lib/impact";
import { emptyState, scenario } from "../src/lib/scenarios";
import { ingest, review } from "../src/lib/service";

const now = "2026-10-04T06:00:00.000Z";
function history() {
  const s = emptyState();
  for (let i = 0; i < 48; i++) {
    const t = scenario(
      "normal",
      new Date(Date.parse(now) - (48 - i) * 43200000).toISOString(),
    );
    t.amount = 800 + ((i * 137) % 800);
    const a = analyze(t, s);
    a.id = `baseline-${i}`;
    s.transactions.push(a);
  }
  return s;
}
test("isolation path normalization has exact small-sample values", () => {
  assert.equal(averagePathLength(1), 0);
  assert.equal(averagePathLength(2), 1);
  assert.ok(Math.abs(averagePathLength(3) - 5 / 3) < 1e-12);
});
test("cold start abstains; established behavior detects a rule-approved anomaly", () => {
  assert.equal(
    analyze(scenario("normal", now), emptyState()).anomaly?.score,
    null,
  );
  const s = history();
  const normal = analyze(scenario("normal", now), s);
  const anomalous = ingest(s, { ...scenario("normal", now), amount: 9000 });
  assert.equal(normal.anomaly?.status, "READY");
  assert.equal(normal.anomaly?.review_recommended, false);
  assert.equal(anomalous.decision, "APPROVE");
  assert.equal(anomalous.risk_score, 0);
  assert.equal(
    anomalous.anomaly?.review_recommended,
    true,
    `score: ${anomalous.anomaly?.score}`,
  );
  assert.equal(s.cases.length, 1);
  assert.match(investigationBrief(anomalous).next_action, /investigation only/);
  const saved = structuredClone(anomalous.anomaly);
  assert.equal(ingest(s, anomalous.payload).id, anomalous.id);
  review(s, s.cases[0].id, {
    action: "MARK_FALSE_POSITIVE",
    note: "Customer verified this unusual purchase",
  });
  assert.deepEqual(anomalous.anomaly, saved);
});
test("training excludes future, same-time, stale, other-account, rejected, and other-type data", () => {
  const s = history();
  const before = analyze(scenario("normal", now), s).anomaly;
  for (const change of [
    { timestamp: "2026-10-05T06:00:00.000Z" },
    { timestamp: now },
    { timestamp: "2026-08-01T06:00:00.000Z" },
    { user_id: "another-account" },
    { transaction_type: "CASH_IN" as const },
  ])
    s.transactions.push(
      analyze(
        { ...scenario("normal", "2026-10-03T06:00:00.000Z"), ...change },
        emptyState(),
      ),
    );
  s.transactions.push(
    analyze(
      {
        ...scenario("fraud", "2026-10-03T07:00:00.000Z"),
        user_id: "demo-normal",
        transaction_type: "SEND_MONEY",
      },
      emptyState(),
    ),
  );
  const after = analyze(scenario("normal", now), s).anomaly!;
  assert.equal(after.baseline_digest, before?.baseline_digest);
  assert.deepEqual(after.baseline_ids, before?.baseline_ids);
  // Same-time attempts affect live velocity, but never enter model training.
  assert.equal(after.observations[2].observed, 2);
  s.transactions.reverse();
  assert.deepEqual(analyze(scenario("normal", now), s).anomaly, after);
});
test("constant history abstains without fabricating model confidence", () => {
  const s = history();
  s.transactions.forEach((a) => {
    a.payload.amount = 1000;
    a.features.depletion_ratio = 1 / 15;
  });
  assert.equal(
    analyze(scenario("normal", now), s).anomaly?.status,
    "INSUFFICIENT_VARIATION",
  );
});
test("impact labels survive case closure and unknown labels do not become false positives", () => {
  const s = emptyState();
  const a = ingest(s, scenario("fraud", now));
  assert.equal(impactSummary(s).reviewed_alert_precision, null);
  review(s, s.cases[0].id, {
    action: "CONFIRM_FRAUD",
    note: "Confirmed synthetic fraud label",
  });
  review(s, s.cases[0].id, {
    action: "RELEASE_HOLD",
    note: "Closing simulated investigation",
  });
  let m = impactSummary(s);
  assert.equal(m.confirmed_fraud, 1);
  assert.equal(m.confirmed_fraud_exposure_bdt, a.payload.amount);
  assert.equal(m.reviewed_alert_precision, 1);
  review(s, s.cases[0].id, {
    action: "MARK_FALSE_POSITIVE",
    note: "Corrected synthetic ground truth",
  });
  m = impactSummary(s);
  assert.equal(m.false_positive, 1);
  assert.equal(m.confirmed_fraud_exposure_bdt, 0);
  assert.equal(m.reviewed_alert_precision, 0);
});
