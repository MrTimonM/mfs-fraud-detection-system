import { test, expect } from "@playwright/test";
import { scenario } from "../../src/lib/scenarios";
test("API validates requests and preserves idempotent fraud evidence", async ({
  request,
}) => {
  const t = scenario("fraud");
  t.transaction_id = "E2E-" + crypto.randomUUID();
  t.user_id = "e2e-" + crypto.randomUUID();
  const bad = await request.post("/api/v1/transactions/analyze", {
    data: { ...t, amount: 90000 },
  });
  expect(bad.status()).toBe(422);
  const response = await request.post("/api/v1/transactions/analyze", {
    data: t,
  });
  expect(response.status()).toBe(200);
  const a = await response.json();
  expect(a.decision).toBe("REJECT_AND_FREEZE");
  expect(a.risk_score).toBe(100);
  expect(a.rule_snapshot.length).toBeGreaterThanOrEqual(18);
  const again = await request.post("/api/v1/transactions/analyze", { data: t });
  expect((await again.json()).id).toBe(a.id);
  const conflict = await request.post("/api/v1/transactions/analyze", {
    data: { ...t, amount: 48000 },
  });
  expect(conflict.status()).toBe(409);
  const cases = await (await request.get("/api/v1/cases")).json();
  expect(
    cases.filter((c: { transaction_id: string }) => c.transaction_id === a.id),
  ).toHaveLength(1);
  const c = cases.find(
    (c: { transaction_id: string }) => c.transaction_id === a.id,
  );
  const reviewed = await request.post(`/api/v1/cases/${c.id}/review`, {
    data: { action: "CONFIRM_FRAUD", note: "Synthetic end-to-end test review" },
  });
  expect((await reviewed.json()).status).toBe("CONFIRMED_FRAUD");
  expect(
    (await (await request.get(`/api/v1/transactions/${a.id}`)).json()).decision,
  ).toBe("REJECT_AND_FREEZE");
  const audit = await (await request.get("/api/v1/audit")).json();
  expect(
    audit.some(
      (x: { action: string; entity_id: string }) =>
        x.action === "CASE_REVIEWED" && x.entity_id === c.id,
    ),
  ).toBe(true);
});
test("simulator submits all three scenarios and opens saved evidence", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/simulator");
  for (const [button, decision] of [
    ["Normal transaction", "Approve"],
    ["Suspicious activity", "Step up auth"],
    ["Fraud-like cash-out", "Reject and freeze"],
  ]) {
    await page.getByRole("button", { name: button }).click();
    await page
      .getByLabel("User ID", { exact: true })
      .fill("e2e-" + crypto.randomUUID());
    await page
      .getByRole("button", { name: "Analyze transaction", exact: true })
      .click();
    await expect(page.locator("#analysis-result .result-banner h2")).toHaveText(
      decision,
    );
  }
  await page.getByRole("button", { name: "Open saved record" }).click();
  await expect(
    page.getByRole("heading", { name: "Investigation detail" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Triggered rule evidence" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});
test("filters, empty states, exports, and rule changes work", async ({
  page,
  request,
}) => {
  await page.goto("/transactions");
  await page
    .getByRole("textbox", { name: "Search records" })
    .fill("DOES-NOT-EXIST");
  await expect(
    page.getByRole("heading", { name: "No records match these filters." }),
  ).toBeVisible();
  await page.getByRole("textbox", { name: "Search records" }).fill("");
  await page
    .getByRole("combobox", { name: "Decision", exact: true })
    .selectOption("REJECT_AND_FREEZE");
  await expect(page.locator("tbody tr").first()).toContainText(
    "Reject and freeze",
  );
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export CSV" }).click();
  expect((await download).suggestedFilename()).toBe("mfs-transactions.csv");
  const rules = await (await request.get("/api/v1/rules")).json();
  const original = rules.find((r: { code: string }) => r.code === "R003");
  try {
    await page.goto("/rules");
    await page
      .locator("tr")
      .filter({ hasText: "Failed PIN attempts" })
      .getByRole("button", { name: "Edit" })
      .click();
    await page.getByLabel("Weight", { exact: true }).fill("17");
    await page.getByRole("button", { name: "Save rule" }).click();
    await expect(page.getByRole("dialog")).not.toBeVisible();
    const updated = await (await request.get("/api/v1/rules")).json();
    expect(
      updated.find((r: { code: string }) => r.code === "R003").weight,
    ).toBe(17);
  } finally {
    await request.patch("/api/v1/rules/R003", {
      data: { weight: original.weight },
    });
  }
});
test("case notes are saved and visible after reload", async ({
  page,
  request,
}) => {
  const t = scenario("suspicious");
  t.transaction_id = "E2E-" + crypto.randomUUID();
  t.user_id = "e2e-" + crypto.randomUUID();
  const a = await (
    await request.post("/api/v1/transactions/analyze", { data: t })
  ).json();
  const cases = await (await request.get("/api/v1/cases")).json();
  const c = cases.find(
    (c: { transaction_id: string }) => c.transaction_id === a.id,
  );
  await page.goto("/cases/" + c.id);
  await page.getByLabel("Review action").selectOption("MARK_FALSE_POSITIVE");
  await page
    .getByLabel("Analyst notes")
    .fill("Known customer confirmed the synthetic transaction.");
  await page.getByRole("button", { name: "Save review" }).click();
  await expect(
    page.getByText("Review recorded. The screening decision is preserved."),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByText("Known customer confirmed the synthetic transaction."),
  ).toBeVisible();
});
test("custom numeric rules can be created in the editor", async ({
  page,
  request,
}) => {
  const code =
    "CUSTOM_E2E_" +
    crypto.randomUUID().replaceAll("-", "").slice(0, 12).toUpperCase();
  try {
    await page.goto("/rules");
    await page.getByRole("button", { name: "Add rule" }).click();
    await page.getByLabel("Rule code", { exact: true }).fill(code);
    await page
      .getByLabel("Rule name", { exact: true })
      .fill("Synthetic test rule");
    await page.getByLabel("Weight", { exact: true }).fill("0");
    await page.getByRole("button", { name: "Save rule" }).click();
    await expect(page.getByRole("dialog")).not.toBeVisible();
    await expect(page.locator("tr").filter({ hasText: code })).toBeVisible();
    const rules = await (await request.get("/api/v1/rules")).json();
    expect(
      rules.find((r: { code: string }) => r.code === code).condition.feature,
    ).toBe("tx_count_5m");
  } finally {
    await request.patch("/api/v1/rules/" + code, { data: { enabled: false } });
  }
});
test("desktop and mobile routes have no overflow or browser errors", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const route of [
      "/",
      "/transactions",
      "/alerts",
      "/cases",
      "/rules",
      "/simulator",
      "/profiles",
      "/impact",
      "/audit",
    ]) {
      await page.goto(route);
      await expect(page.locator("main h1")).toBeVisible();
      await expect(page.locator(".loading")).not.toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      ).toBe(true);
    }
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Recent fraud alerts" }),
    ).toBeVisible();
    await page.screenshot({
      path: `.impeccable/review/verified-${width}.png`,
      fullPage: true,
    });
  }
  expect(errors).toEqual([]);
});

test("behavior-only case preserves policy approval and exposes review impact", async ({
  page,
  request,
}) => {
  test.setTimeout(90000);
  const user = "e2e-behavior-" + crypto.randomUUID();
  const now = Date.now();
  for (let i = 0; i < 48; i++) {
    const t = scenario(
      "normal",
      new Date(now - (48 - i) * 43200000).toISOString(),
    );
    Object.assign(t, {
      user_id: user,
      transaction_id: `${user}-${i}`,
      amount: 800 + ((i * 137) % 800),
    });
    const response = await request.post("/api/v1/transactions/analyze", {
      data: t,
    });
    expect(response.ok()).toBe(true);
  }
  const t = scenario("normal", new Date(now).toISOString());
  Object.assign(t, {
    user_id: user,
    transaction_id: `${user}-anomaly`,
    amount: 9000,
  });
  const response = await request.post("/api/v1/transactions/analyze", {
    data: t,
  });
  expect(response.ok()).toBe(true);
  const a = await response.json();
  expect(a.decision).toBe("APPROVE");
  expect(a.anomaly.status).toBe("READY");
  expect(a.anomaly.review_recommended).toBe(true);
  const cases = await (await request.get("/api/v1/cases")).json();
  const c = cases.find(
    (c: { transaction_id: string }) => c.transaction_id === a.id,
  );
  expect(c).toBeTruthy();
  await page.goto(`/cases/${c.id}`);
  await expect(
    page.getByText(
      "Rule policy approves; behavior model recommends analyst review.",
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Investigation brief" }),
  ).toBeVisible();
  await page.getByLabel("Review action").selectOption("MARK_FALSE_POSITIVE");
  await page
    .getByLabel("Analyst notes")
    .fill("Synthetic customer verified an unusual transfer.");
  await page.getByRole("button", { name: "Save review" }).click();
  await expect(
    page.getByText("Review recorded. The screening decision is preserved."),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByText("Synthetic customer verified an unusual transfer."),
  ).toBeVisible();
  await page.screenshot({
    path: ".impeccable/review/behavior-evidence.png",
    fullPage: true,
  });
  await page.goto("/impact");
  await expect(
    page.getByRole("heading", { name: "Channel review comparison" }),
  ).toBeVisible();
  await page.screenshot({
    path: ".impeccable/review/impact-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 900 });
  await page.screenshot({
    path: ".impeccable/review/impact-mobile.png",
    fullPage: true,
  });
  const metrics = await (await request.get("/api/v1/impact")).json();
  expect(metrics.anomaly_only_cases).toBeGreaterThan(0);
  expect(metrics.false_positive).toBeGreaterThan(0);
});
