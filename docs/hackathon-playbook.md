# MFS Guard: hackathon product and validation playbook

Aligned with the supplied DIU CPC × upay AI Hackathon 2026 Student Project Guideline, especially Track 01 (pages 3–4), the idea framework (page 8), architecture (page 9), responsible AI (page 10), and judging criteria (page 11). The guideline is a reference for this project, not a claim of upay integration or endorsement.

## User, problem, and value

For a fraud operations analyst, reviewing isolated risk scores makes unusual transfers difficult to prioritize and explain. MFS Guard combines a transparent rule policy with learned account behavior, a saved evidence brief, and an analyst feedback loop. Success should be measured by additional confirmed fraud found per review, legitimate customers interrupted, and investigation time.

Customer benefit: unusual activity can be investigated before a consequential recommendation is acted on, with clear reasons and a way to correct false positives. No wallets are frozen and no money is moved by this app.

Problem frequency, current investigation time, fraud prevalence, and upay's operating costs have not been measured. Interview analysts and customers to validate these assumptions before asserting savings.

## What the intelligence does

The rule engine produces the existing policy score and decision. A separate TypeScript Isolation Forest fits 64 random isolation trees to an account's earlier policy-approved transactions of the same type, excluding previously model-flagged transactions. Shorter average paths indicate unusual behavior. This is a small independent implementation, not scikit-learn running inside the app; see the [Isolation Forest reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) for the underlying method.

Inputs: log(1 + amount), outgoing balance depletion ratio, and five-minute attempt count. Account identifiers select history but are not predictive features. No phone numbers, location, demographics, or customer text enter the model.

Training is limited to the most recent 256 eligible transactions in the preceding 30 days; each tree samples at most 128. A strict event-time cutoff excludes current, equal-time, and future transactions. Inference abstains with fewer than 20 eligible records or no feature variation. The prototype review threshold is 0.58, chosen using synthetic validation evidence before running the test split. Higher scores are more anomalous, not a probability of fraud.

Each transaction saves model version, score, threshold, baseline IDs/digest, baseline count, and observed features against baseline medians. A deterministic seed derived from the baseline digest makes repeated fitting on the same saved history repeatable. Model-only anomalies open an analyst case without altering the rule decision. Existing records without model evidence are shown as historical records; they are not rescored.

The investigation brief is a deterministic template grounded in these saved observations and rule traces. It does not call an LLM. Median comparisons are descriptive context, not SHAP values or causal explanations.

## Synthetic assumptions and benchmark

Run `npm run evaluate -- --validation-only` to inspect validation and `npm run evaluate` to reproduce both splits. The script does not write to storage. Each split contains 12 separate accounts, 48 earlier training transactions per account, and 120 candidate transfers: 60 normal and 60 injected anomalies. Training timestamps precede candidate timestamps. Candidate transfers are never added to model history.

Synthetic amounts vary around BDT 1,200 or 2,400. Injected positive examples are 6.5–6.9 times the usual amount while staying below rule depletion and new-device thresholds. Validation and test use separate account IDs and different amount sequences. Channel groups contain 40 candidates each. Labels come from injection assumptions, not from rule outcomes.

| Held-out synthetic test | Rules only | Rules plus anomaly review |
| --- | ---: | ---: |
| Detected injected anomalies | 0 / 60 | 60 / 60 |
| Normal transfers sent to review | 0 / 60 | 1 / 60 |
| Precision | Undefined (no alerts) | 98.36% |
| Recall of injected anomalies | 0% | 100% |
| False-positive rate | 0% | 1.67% |
| Review rate | 0% | 50.83% |

These are results on a deliberately narrow synthetic behavioral challenge, not real fraud performance. Repeated candidates within an account are correlated, labels are balanced, and all injected positives are deliberately large deviations. The benchmark does not establish performance on scams, natural legitimate outliers, unseen attacks, production prevalence, or upay customers. The test's APP group has one false positive (5% of its 20 negative examples); USSD and AGENT have zero in similarly small groups. This is a diagnostic check, not a fairness guarantee.

## Operational measures and evidence limits

The Impact & validation screen and authenticated `GET /api/v1/impact` report:

- Precision among explicitly reviewed alerts, using the latest fraud/false-positive disposition even after closure.
- Review coverage and false-positive review count, to expose workload and customer friction.
- Median elapsed time from case creation to first recorded substantive review, with sample count.
- Confirmed fraud transaction value, labeled exposure rather than prevented loss.
- Additional model-only cases and channel-specific queue rates.

Unreviewed alerts have no ground-truth label. Reviewing alerts alone cannot measure missed fraud, population false-positive rate, causal loss prevention, or time saved. Queue rates can vary because of risk mix and sampling. For a credible business case, also sample approved transactions and compare matched review workflows.

## Controlled validation and future scale

1. Interview analysts to establish a baseline and confirm the three-part brief reduces investigation effort. Compare median review time and agreement with and without the brief. An initial experiment target is 20% shorter review time; this is a hypothesis, not an achieved result.
2. Expand synthetic testing with legitimate large purchases, variable salary cycles, cold starts, time-boundary cases, account takeover, mule networks, and history poisoning. Reserve fresh test cohorts before changing features or thresholds.
3. If suitable governed data becomes available, run shadow inference with anonymized identifiers and event-time feature snapshots. Split chronologically, select thresholds on validation only, and hold out customer cohorts. Require documented data access, purpose, retention, and deletion controls.
4. Compare rules, anomaly review, and a supervised LightGBM model. Measure PR-AUC, precision, recall, legitimate customer interruption rate, alert yield, and review load at realistic prevalence. Evaluate channel/device/cohort groups with sample sizes and uncertainty. No supervised LightGBM or SHAP model is shipped today.
5. Let analysts review consequential actions; introduce individual accounts, roles, governed appeals, independent fraud labels, drift monitoring, feature integrity checks, and model version approvals before a pilot.
6. Replace whole-state loads and per-request training with indexed history queries, versioned cached per-account models, partitioned ingestion, and paginated APIs before high-volume use. Store inference snapshots so model refreshes cannot rewrite history.

Rule approvals are only a proxy for clean training data. Uncaught fraud can contaminate the baseline; excluding model flags can also reinforce blind spots. Constant historical features cannot be split by an Isolation Forest, so changes in such a feature may be missed. Caller-supplied device and behavior context requires trusted server-side verification in a real integration.
