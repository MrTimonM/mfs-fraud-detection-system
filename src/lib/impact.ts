import type { Analysis, State } from "./domain";

export function investigationBrief(t: Analysis) {
  const reasons = t.triggered_rules.map(
    (r) => `${r.code} v${r.version}: ${r.reason}`,
  );
  if (t.anomaly?.review_recommended)
    reasons.push(
      `Behavior anomaly score ${t.anomaly.score?.toFixed(3)} exceeds ${t.anomaly.threshold}, using ${t.anomaly.baseline_count} earlier policy-approved transactions.`,
    );
  return {
    what_happened: `${t.payload.user_id} submitted a ${t.payload.transaction_type.replaceAll("_", " ").toLowerCase()} of BDT ${t.payload.amount.toFixed(2)} via ${t.payload.channel} to ${t.payload.receiver_id}.`,
    why_risky: reasons.length
      ? reasons
      : [
          "No enabled policy rule or available anomaly model recommended intervention. This is not proof that the transaction is legitimate.",
        ],
    next_action:
      t.decision === "REJECT_AND_FREEZE"
        ? "Prioritize analyst investigation and verify the customer through a trusted channel before any account restriction. The freeze is a recommendation only."
        : t.decision === "STEP_UP_AUTH"
          ? "Request identity verification through a trusted channel and review the recorded signals."
          : t.anomaly?.review_recommended
            ? "Review the unusual behavior with the customer. The policy approved this transaction; the model recommends investigation only."
            : "Follow the current approval policy and continue monitoring. No money movement is performed by this prototype.",
    provenance:
      "Template summary of saved evidence; no LLM-generated claims. Anomaly scores are not fraud probabilities. Observed deviations are not causal feature attributions.",
  };
}

export function impactSummary(s: State) {
  const transactions = new Map(s.transactions.map((t) => [t.id, t]));
  const labels = s.cases.flatMap((c) => {
    const t = transactions.get(c.transaction_id);
    const label = c.actions
      .slice()
      .reverse()
      .find((a) => ["CONFIRM_FRAUD", "MARK_FALSE_POSITIVE"].includes(a.action));
    if (!t || !label) return [];
    return [{ t, fraud: label.action === "CONFIRM_FRAUD" }];
  });
  const fraud = labels.filter((l) => l.fraud);
  const reviewedIds = new Set(labels.map((l) => l.t.id));
  const queuedIds = new Set(s.cases.map((c) => c.transaction_id));
  const firstReviewSeconds = s.cases
    .flatMap((c) => {
      const first = c.actions.find((a) => a.action !== "STATUS_CHANGED");
      const seconds = first
        ? (Date.parse(first.timestamp) - Date.parse(c.created_at)) / 1000
        : NaN;
      return Number.isFinite(seconds) && seconds >= 0 ? [seconds] : [];
    })
    .sort((a, b) => a - b);
  const medianIndex = Math.floor(firstReviewSeconds.length / 2);
  const median = firstReviewSeconds.length
    ? firstReviewSeconds.length % 2
      ? firstReviewSeconds[medianIndex]
      : (firstReviewSeconds[medianIndex - 1] +
          firstReviewSeconds[medianIndex]) /
        2
    : null;
  return {
    screened: s.transactions.length,
    queued: queuedIds.size,
    reviewed: labels.length,
    confirmed_fraud: fraud.length,
    false_positive: labels.length - fraud.length,
    reviewed_alert_precision: labels.length
      ? fraud.length / labels.length
      : null,
    review_coverage: queuedIds.size ? labels.length / queuedIds.size : null,
    confirmed_fraud_exposure_bdt: fraud.reduce(
      (sum, l) => sum + l.t.payload.amount,
      0,
    ),
    median_first_review_seconds: median,
    first_review_sample_size: firstReviewSeconds.length,
    anomaly_only_cases: s.transactions.filter(
      (t) => t.decision === "APPROVE" && t.anomaly?.review_recommended,
    ).length,
    channels: (["APP", "USSD", "AGENT"] as const).map((channel) => {
      const tx = s.transactions.filter((t) => t.payload.channel === channel);
      return {
        channel,
        screened: tx.length,
        queued: tx.filter((t) => queuedIds.has(t.id)).length,
        reviewed: tx.filter((t) => reviewedIds.has(t.id)).length,
      };
    }),
  };
}
