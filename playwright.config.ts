import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/e2e",
  workers: 1,
  fullyParallel: false,
  use: {
    baseURL: process.env.TEST_BASE_URL ?? "http://localhost:3000",
    browserName: "chromium",
    channel: "chrome",
    headless: true,
  },
  reporter: "list",
  timeout: 45000,
});
