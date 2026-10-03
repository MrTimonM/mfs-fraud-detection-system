import { createHash } from "node:crypto";
import type { Analysis, Features, State, TransactionInput } from "./domain";

export const anomalyPolicy = {
  version: "isolation-forest-v1",
  minimum_history: 20,
  maximum_history: 256,
  window_days: 30,
  trees: 64,
  sample_size: 128,
  review_threshold: 0.58,
} as const;

export interface AnomalyEvidence {
  model_version: string;
  status: "READY" | "INSUFFICIENT_HISTORY" | "INSUFFICIENT_VARIATION";
  score: number | null;
  threshold: number;
  review_recommended: boolean;
  baseline_count: number;
  baseline_ids: string[];
  baseline_digest: string;
  window_days: number;
  observations: {
    feature: string;
    observed: number;
    baseline_median: number;
  }[];
}
type Tree = {
  size: number;
  feature?: number;
  split?: number;
  left?: Tree;
  right?: Tree;
};
const vector = (t: TransactionInput, f: Features) => [
  Math.log1p(t.amount),
  f.depletion_ratio,
  f.tx_count_5m,
];
const names = ["Log amount", "Balance depletion ratio", "Five-minute attempts"];
function median(values: number[]) {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
// Expected path length for an unsuccessful search in a binary search tree.
export function averagePathLength(n: number): number {
  if (n <= 1) return 0;
  let harmonic = 0;
  for (let i = 1; i < n; i++) harmonic += 1 / i;
  return 2 * harmonic - (2 * (n - 1)) / n;
}
function randomSource(seed: number) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}
function build(
  rows: number[][],
  depth: number,
  maxDepth: number,
  random: () => number,
): Tree {
  const node: Tree = { size: rows.length };
  if (rows.length <= 1 || depth >= maxDepth) return node;
  const ranges = names
    .map((_, feature) => {
      const values = rows.map((row) => row[feature]);
      return { feature, min: Math.min(...values), max: Math.max(...values) };
    })
    .filter((r) => r.max > r.min);
  if (!ranges.length) return node;
  const range = ranges[Math.floor(random() * ranges.length)];
  const split = range.min + random() * (range.max - range.min);
  const left = rows.filter((row) => row[range.feature] <= split);
  const right = rows.filter((row) => row[range.feature] > split);
  if (!left.length || !right.length) return node;
  return {
    ...node,
    feature: range.feature,
    split,
    left: build(left, depth + 1, maxDepth, random),
    right: build(right, depth + 1, maxDepth, random),
  };
}
function pathLength(row: number[], tree: Tree, depth = 0): number {
  if (tree.feature === undefined) return depth + averagePathLength(tree.size);
  return pathLength(
    row,
    row[tree.feature] <= tree.split! ? tree.left! : tree.right!,
    depth + 1,
  );
}

export function detectAnomaly(
  t: TransactionInput,
  f: Features,
  s: State,
): AnomalyEvidence {
  const now = Date.parse(t.timestamp);
  // Strict event-time cutoff prevents current/future observations entering training.
  // Historical rule approvals are a proxy for normality, not verified clean labels.
  const baseline = s.transactions
    .filter(
      (a) =>
        a.payload.user_id === t.user_id &&
        a.payload.transaction_type === t.transaction_type &&
        a.payload.transaction_id !== t.transaction_id &&
        a.decision === "APPROVE" &&
        !a.anomaly?.review_recommended &&
        Date.parse(a.payload.timestamp) < now &&
        now - Date.parse(a.payload.timestamp) <=
          anomalyPolicy.window_days * 86400000,
    )
    .sort(
      (a, b) =>
        Date.parse(b.payload.timestamp) - Date.parse(a.payload.timestamp) ||
        a.id.localeCompare(b.id),
    )
    .slice(0, anomalyPolicy.maximum_history);
  const rows = baseline.map((a: Analysis) => vector(a.payload, a.features));
  const digest = createHash("sha256")
    .update(JSON.stringify(baseline.map((a, i) => [a.id, rows[i]])))
    .digest("hex");
  const evidence: AnomalyEvidence = {
    model_version: anomalyPolicy.version,
    status: "INSUFFICIENT_HISTORY",
    score: null,
    threshold: anomalyPolicy.review_threshold,
    review_recommended: false,
    baseline_count: baseline.length,
    baseline_ids: baseline.map((a) => a.id),
    baseline_digest: digest,
    window_days: anomalyPolicy.window_days,
    observations: [],
  };
  if (baseline.length < anomalyPolicy.minimum_history) return evidence;
  if (rows.every((row) => row.every((value, i) => value === rows[0][i])))
    return { ...evidence, status: "INSUFFICIENT_VARIATION" };
  const random = randomSource(parseInt(digest.slice(0, 8), 16));
  const size = Math.min(anomalyPolicy.sample_size, rows.length);
  const current = vector(t, f);
  let totalPath = 0;
  for (let i = 0; i < anomalyPolicy.trees; i++) {
    const sample = rows.slice();
    for (let j = sample.length - 1; j > 0; j--) {
      const k = Math.floor(random() * (j + 1));
      [sample[j], sample[k]] = [sample[k], sample[j]];
    }
    totalPath += pathLength(
      current,
      build(sample.slice(0, size), 0, Math.ceil(Math.log2(size)), random),
    );
  }
  const score =
    2 ** (-(totalPath / anomalyPolicy.trees) / averagePathLength(size));
  return {
    ...evidence,
    status: "READY",
    score,
    review_recommended: score >= evidence.threshold,
    observations: names.map((feature, i) => ({
      feature,
      observed: current[i],
      baseline_median: median(rows.map((row) => row[i])),
    })),
  };
}
