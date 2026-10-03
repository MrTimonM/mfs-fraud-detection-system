# MFS Guard

A rule-based Mobile Financial Services fraud detection prototype built with Next.js, React, and TypeScript. The frontend and API run in one application. PostgreSQL provides durable storage; no Python server is required.

## What it does

- Evaluates 18 configurable fraud rules, plus custom numeric feature rules.
- Calculates balance depletion, velocity, fund turnaround, device drift, location travel speed, and recipient risk from transaction history.
- Produces a transparent score from 0 to 100 and an `APPROVE`, `STEP_UP_AUTH`, or `REJECT_AND_FREEZE` recommendation.
- Saves transaction inputs, calculated features, rule versions, individual triggers, and the complete policy snapshot used for each decision.
- Provides an operations dashboard, simulator, transaction history, alert queue, case reviews, rule editor, entity history, and audit log.
- Supports filtering, CSV export, deterministic demo presets, analyst notes, and fraud/false-positive dispositions.

This is a Phase 1 prototype. Freeze and release actions record recommendations; they do not interact with wallets or move money. All included demo records are synthetic.

## Run locally

Use Node.js 22 or newer.

```sh
npm ci
npm run dev
```

Open http://localhost:3000. Without a database, the app creates 48 synthetic records and persists local changes in `.data/state.json`. That directory is ignored by Git. Local demo mode does not require a password unless you configure one.

## Deploy on Vercel with PostgreSQL

1. Create a PostgreSQL database using Neon, Supabase, or another provider. Copy its **pooled connection URL**, including the provider's SSL parameters.
2. Copy `.env.example` to `.env.local` and set:

   ```dotenv
   DATABASE_URL=your-postgresql-connection-url
   ANALYST_PASSWORD=your-workspace-password
   SESSION_SECRET=your-random-secret-at-least-32-characters
   DEMO_MODE=false
   ```

   Generate a session secret locally with `node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"`. Never commit `.env.local` or share the secret.

3. Initialize the database:

   ```sh
   npm run db:migrate
   # Optional: seed synthetic records in an empty database
   npm run db:seed
   ```

   Migration is repeatable and preserves edited rules. Seed refuses to overwrite a database that already contains transactions. Migration is a separate setup step; builds never write to the database.

4. Import [this GitHub repository](https://github.com/MrTimonM/mfs-fraud-detection-system) into Vercel. Select **Next.js**, leave the root directory at the repository root, use `npm ci` for installation, and `npm run build` for the build command. Use Node.js 22 or newer and the default output directory.
5. Add the same environment variables to the Vercel project for each environment that uses the database, then deploy. The app requires an analyst password and a secret of at least 32 characters for persistent Vercel mode.
6. Sign in with `ANALYST_PASSWORD`, submit a simulator transaction, and confirm that its saved record survives a page reload.

Use separate databases for preview and production if you want to keep their activity isolated. Use a dedicated database role with only the permissions the application needs.

Official references: [Next.js route handlers](https://nextjs.org/docs/app/getting-started/route-handlers), [Vercel Node.js functions](https://vercel.com/docs/functions/runtimes/node-js), [Vercel Git deployments](https://vercel.com/docs/git).

### Public demonstration deployment

To deploy a synthetic demo without a database, leave `DATABASE_URL` unset and explicitly set `DEMO_MODE=true` in Vercel. The UI labels this mode. Data is held in server memory, can reset, and can differ between instances. Use PostgreSQL for a reliable shared workflow.

## Verification

```sh
npm test
npm run typecheck
npm run build
npm run test:e2e
```

The browser suite starts a development server when needed. Windows uses installed Google Chrome; on Linux/macOS install the test browser with `npx playwright install --with-deps chromium`. Set `TEST_BASE_URL` to test another running local instance. Run end-to-end tests against an isolated demo workspace: the suite creates synthetic transactions and review events.

The unit suite checks every initial rule, score boundaries, validation, history windows, idempotency, immutable policy snapshots, custom rules, and signed sessions. Browser tests exercise the actual API and UI, including scenario decisions, notes, filters, exports, rule changes, and responsive routes. PostgreSQL integration can be checked against an empty disposable database with `TEST_DATABASE_URL=... npm run test:db`; it refuses to run against a database with transaction data.

## Design and implementation notes

- [Original implementation plan](IMPLEMENTATION_PLAN.md)
- [Architecture and prototype limits](docs/architecture.md)
- [API reference](docs/api.md)
- [Demo scenarios](docs/demo-scenarios.md)

`ANALYST_PASSWORD` is shared workspace authentication, not individual user accounts. Audit actors are `analyst`, `engine`, or `system`. Login throttling is instance-local. A real operations rollout should replace shared authentication with individual accounts, roles, distributed throttling, and retention policies.

## Phase 2: machine learning

The feature engine and rule engine are separate modules. Every transaction stores its original features and rule evidence; case review actions provide analyst labels without rewriting the original screening decision.

Phase 2 can export confirmed fraud and false-positive cases into a reviewed training dataset, build time-based train/validation/test splits, train XGBoost or LightGBM, and compare precision, recall, F1, ROC-AUC, and PR-AUC. Add a versioned inference adapter and SHAP explanations before combining validated model predictions with deterministic rules. The current app does not claim to contain a trained model or a measured fraud-detection accuracy.
