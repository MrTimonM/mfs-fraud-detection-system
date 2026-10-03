import { analyze } from "../src/lib/engine";
import { emptyState, scenario } from "../src/lib/scenarios";
import { anomalyPolicy } from "../src/lib/anomaly";

// Entirely synthetic account patterns. Labels come from the generator, not rules.
// Validation and test have disjoint accounts; neither candidate set enters training.
const now = "2026-10-04T06:00:00.000Z";
function evaluate(split: "validation" | "test") {
  const outcomes: {
    channel: string;
    fraud: boolean;
    rule_flag: boolean;
    combined_flag: boolean;
    score: number | null;
  }[] = [];
  for (let wallet = 0; wallet < 12; wallet++) {
    const s = emptyState();
    const user = `${split}-wallet-${wallet}`;
    const channel = (["APP", "USSD", "AGENT"] as const)[wallet % 3];
    const center = wallet % 2 ? 2400 : 1200;
    for (let i = 0; i < 48; i++) {
      const t = scenario(
        "normal",
        new Date(Date.parse(now) - (48 - i) * 43200000).toISOString(),
      );
      Object.assign(t, {
        user_id: user,
        channel,
        balance_before: 60000,
        amount:
          center *
          (0.7 +
            ((i * (split === "validation" ? 137 : 173) + wallet * 71) % 600) /
              1000),
      });
      const a = analyze(t, s);
      a.id = `${user}-history-${i}`;
      s.transactions.push(a);
    }
    for (let i = 0; i < 10; i++) {
      const fraud = i >= 5;
      const t = scenario("normal", now);
      Object.assign(t, {
        transaction_id: `${user}-candidate-${i}`,
        user_id: user,
        channel,
        balance_before: 60000,
        amount: fraud ? center * (6 + i / 10) : center * (0.8 + i * 0.075),
      });
      const a = analyze(t, s);
      outcomes.push({
        channel,
        fraud,
        rule_flag: a.decision !== "APPROVE",
        combined_flag:
          a.decision !== "APPROVE" || !!a.anomaly?.review_recommended,
        score: a.anomaly?.score ?? null,
      });
    }
  }
  const metrics = (
    items: typeof outcomes,
    key: "rule_flag" | "combined_flag",
  ) => {
    const tp = items.filter((x) => x.fraud && x[key]).length;
    const fp = items.filter((x) => !x.fraud && x[key]).length;
    const fn = items.filter((x) => x.fraud && !x[key]).length;
    const tn = items.filter((x) => !x.fraud && !x[key]).length;
    return {
      sample_size: items.length,
      tp,
      fp,
      fn,
      tn,
      precision: tp + fp ? tp / (tp + fp) : null,
      recall: tp + fn ? tp / (tp + fn) : null,
      false_positive_rate: fp + tn ? fp / (fp + tn) : null,
      review_rate: (tp + fp) / items.length,
    };
  };
  return {
    split,
    rule_only: metrics(outcomes, "rule_flag"),
    rules_plus_anomaly: metrics(outcomes, "combined_flag"),
    channels: ["APP", "USSD", "AGENT"].map((channel) => ({
      channel,
      ...metrics(
        outcomes.filter((x) => x.channel === channel),
        "combined_flag",
      ),
    })),
    score_ranges: Object.fromEntries(
      [false, true].map((fraud) => {
        const scores = outcomes
          .filter((x) => x.fraud === fraud && x.score !== null)
          .map((x) => x.score!);
        return [
          fraud ? "injected" : "normal",
          { min: Math.min(...scores), max: Math.max(...scores) },
        ];
      }),
    ),
  };
}
console.log(
  JSON.stringify(
    {
      synthetic_only: true,
      model: anomalyPolicy,
      assumptions:
        "Regular account amounts with injected 6.5-6.9x transfers; balanced labels, not real fraud prevalence. No production efficacy claim.",
      results: [
        evaluate("validation"),
        ...(process.argv.includes("--validation-only")
          ? []
          : [evaluate("test")]),
      ],
    },
    null,
    2,
  ),
);
