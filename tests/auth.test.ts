import test from "node:test";
import assert from "node:assert/strict";
import { sessionValid, signSession } from "../src/lib/auth";
test("signed sessions reject tampering and expired tokens", () => {
  const oldPassword = process.env.ANALYST_PASSWORD;
  const oldSecret = process.env.SESSION_SECRET;
  process.env.ANALYST_PASSWORD = "test-only-password";
  process.env.SESSION_SECRET = "test-only-secret-at-least-32-characters";
  try {
    const token = signSession();
    assert.equal(sessionValid(token), true);
    assert.equal(sessionValid(token + "x"), false);
    assert.equal(sessionValid("0." + token.split(".")[1]), false);
    assert.equal(sessionValid(undefined), false);
  } finally {
    if (oldPassword === undefined) delete process.env.ANALYST_PASSWORD;
    else process.env.ANALYST_PASSWORD = oldPassword;
    if (oldSecret === undefined) delete process.env.SESSION_SECRET;
    else process.env.SESSION_SECRET = oldSecret;
  }
});
