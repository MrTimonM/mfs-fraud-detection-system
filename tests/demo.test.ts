import test from "node:test";
import assert from "node:assert/strict";
import { withState, storageMode } from "../src/lib/store";
import { ingest } from "../src/lib/service";
import { scenario } from "../src/lib/scenarios";

test("public Vercel demo works without custom environment variables or a database", async () => {
  const keys = ["VERCEL", "DATABASE_URL", "DEMO_MODE"] as const;
  const before = keys.map((key) => process.env[key]);
  try {
    process.env.VERCEL = "1";
    delete process.env.DATABASE_URL;
    delete process.env.DEMO_MODE;
    assert.equal(storageMode(), "ephemeral-demo");
    const payload = {
      ...scenario("normal"),
      transaction_id: "PUBLIC-DEMO-TEST",
    };
    const saved = await withState(true, (s) => ingest(s, payload));
    const reloaded = await withState(false, (s) =>
      s.transactions.find((t) => t.id === saved.id),
    );
    assert.equal(reloaded?.payload.transaction_id, payload.transaction_id);
  } finally {
    keys.forEach((key, i) => {
      if (before[i] === undefined) delete process.env[key];
      else process.env[key] = before[i];
    });
  }
});
