# MFS Guard

**Spot suspicious transfers. Explain every decision. Act with confidence.**



A Mobile Financial Services fraud detection workspace built with Next.js, React, TypeScript, and PostgreSQL.

Built for **Track 01: Trust & Risk Intelligence** of DIU CPC × upay AI Hackathon 2026. Helps fraud analysts investigate unusual transfers with saved evidence and human review.

## Intelligence and measurable impact

- **Learned behavior:** a TypeScript Isolation Forest learns account-specific amount, depletion, and velocity patterns from earlier policy-approved transactions. Model-only alerts create review cases while preserving the rule decision.
- **Investigation brief:** explains what happened, why to investigate, and the next analyst action using saved evidence. Summaries are templates, not LLM-generated reasoning.
- **Impact & validation:** tracks reviewed alert precision, false-positive reviews, review coverage, first-review time, confirmed fraud exposure, and channel review rates.
- **Reproducible evidence:** `npm run evaluate` compares rules with rules plus anomaly review on separate synthetic validation/test accounts. See the [hackathon playbook](docs/hackathon-playbook.md) for assumptions, measured results, and the controlled-validation plan.

Anomaly scores are not fraud probabilities. New accounts need sufficient history; consequential actions remain analyst recommendations.

## Phase 2: smarter detection with LightGBM

**Next up:** add supervised LightGBM classification alongside the current rules and unsupervised anomaly model. LightGBM remains planned and has not been trained.

- **Training data:** use analyst-confirmed fraud and false-positive labels with transaction, velocity, balance, device, and recipient features. Keep only information available at screening time and split training, validation, and test data chronologically.
- **Improve LightGBM:** tune leaf count, tree depth, learning rate, and minimum samples per leaf; use early stopping and validate class weighting for imbalanced fraud data. See the [LightGBM tuning guide](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html).
- **Measure results:** compare against the rule engine using PR-AUC, precision, recall, F1, and false-positive rate. Choose decision thresholds on validation data and report final results on the held-out test set.
- **Explain and integrate:** add SHAP explanations, versioned models and feature schemas, and validated model-plus-rule decisions. Monitor drift and retrain with reviewed cases.

## AI fraud model (LightGBM)

The selected LightGBM model (see `mfs-guard-ml/reports/final_model_selection.md`) scores every transaction **before authorization** through the Python scoring service in `mfs-guard-ml/backend`. Its decision (`APPROVE`, `APPROVE_AND_MONITOR`, `STEP_UP_AUTH`, `TEMPORARY_HOLD`, `REJECT_AND_FREEZE`) is the final decision. The model is on by default at `http://127.0.0.1:8000` (override with `MFS_ML_SERVICE_URL`).

```sh
cd mfs-guard-ml
.venv/Scripts/python -m uvicorn backend.app.main:app --port 8000
# in the repo root, separate terminal
npm run dev
```

If the scoring service is unreachable, the transaction is marked `DEGRADED` and held (`TEMPORARY_HOLD`) for analyst review; it is never silently approved. The public Vercel demo cannot reach a local service, so it needs a hosted service URL.

## What works today

- 18 configurable fraud rules, custom rules, and transparent risk scores from 0–100.
- Decisions: `APPROVE`, `STEP_UP_AUTH`, or `REJECT_AND_FREEZE`.
- Dashboard, transaction simulator, alerts, case reviews, entity history, and CSV export.
- Saved decision evidence, policy snapshots, analyst notes, and audit trails.

Demo records are synthetic; freeze and release actions record recommendations.

## Quick start

Requires Node.js 22+.

```sh
npm ci
npm run dev
```

Open [localhost:3000](http://localhost:3000).

Public demo mode is enabled in `deployment.config.json`. No `.env` file, database, password, or session secret is required for this demo. Vercel keeps synthetic activity in server memory; records can reset or differ between instances.

## API

JSON endpoints use the analyst session cookie. Workflow routes start at `/api/v1`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST / DELETE | `/api/auth` | Sign in / sign out |
| POST | `/api/v1/transactions/analyze` | Screen and save a transaction |
| GET | `/api/v1/transactions`, `/api/v1/transactions/{id}` | Browse history and decision evidence |
| GET / PATCH | `/api/v1/alerts/{id}` | View or update an alert |
| POST | `/api/v1/cases/{id}/review` | Record a review and analyst notes |
| GET / POST | `/api/v1/rules` | List or create rules |
| PATCH | `/api/v1/rules/{code}` | Update a rule |
| GET | `/api/v1/dashboard/summary`, `/api/v1/audit` | Dashboard metrics and audit history |

See the [API reference](docs/api.md) for payloads, responses, and all endpoints.

## Docs

- [Architecture](docs/architecture.md) — scoring, storage, and authentication.
- [Demo scenarios](docs/demo-scenarios.md) — sample transactions and expected decisions.
- [Implementation plan](IMPLEMENTATION_PLAN.md) — system design and roadmap.
