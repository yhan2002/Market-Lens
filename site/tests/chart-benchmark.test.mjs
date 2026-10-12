import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";
import { trailingDifference } from "../lib/roro.ts";
import { benchmarkPrices, nasdaqClose, nasdaqDailyReturn, nasdaqAcceleration } from "../lib/chart-benchmark.ts";

const point = (day, close, previousDay, previousClose) => ({
  snapshot: { as_of: day }, benchmark_ticker: "^NDX", benchmark_close: close,
  benchmark_previous_as_of: previousDay, benchmark_previous_close: previousClose,
});
const price = { as_of: "2026-09-14", close: 102.3, previous_as_of: "2026-09-11", previous_close: 102.3 / .9918 };

test("daily -0.82% is distinct from cumulative +2.30%; untagged NQ is not NDX", () => {
  const last = point(price.as_of, price.close, price.previous_as_of, price.previous_close);
  assert.equal(nasdaqDailyReturn(last, price.as_of).toFixed(2), "-0.82");
  assert.equal(((nasdaqClose(last) / 100 - 1) * 100).toFixed(2), "2.30");
  assert.equal(nasdaqClose({ benchmark_close: 30000 }), null);
  assert.equal(nasdaqDailyReturn({ ...last, benchmark_ticker: "NQ=F" }, price.as_of), null);
  assert.equal(nasdaqDailyReturn({ ...last, benchmark_previous_close: null }, price.as_of), null);
});

test("daily returns use previous index session and acceleration does not bridge a missing session", () => {
  const thursday = point("2026-09-10", 100, "2026-09-09", 100);
  const friday = point("2026-09-11", 101, "2026-09-10", 100);
  const monday = point("2026-09-14", 101, "2026-09-11", 101);
  const accelerations = nasdaqAcceleration([thursday, friday, monday]);
  assert.equal(accelerations[0], null);
  assert.ok(Math.abs(accelerations[2] + 1) < 1e-10);
  assert.equal(nasdaqDailyReturn(monday, "2026-09-14"), 0);
  assert.deepEqual(nasdaqAcceleration([thursday, monday]), [null, null]);
});

test("upload contract rejects legacy, invalid, duplicate and future-dated prices", () => {
  const envelope = { benchmark_ticker: "^NDX", benchmark_prices: [price] };
  assert.deepEqual(benchmarkPrices(envelope), [price]);
  assert.equal(benchmarkPrices({ benchmark_prices: [price] }), null);
  assert.equal(benchmarkPrices({ ...envelope, benchmark_ticker: "NQ=F" }), null);
  assert.equal(benchmarkPrices({ ...envelope, benchmark_prices: [price, price] }), null);
  for (const bad of [{ ...price, close: NaN }, { ...price, close: 0 }, { ...price, close: true },
    { ...price, as_of: "2026-02-30" }, { ...price, previous_as_of: "2026-09-15" }, { ...price, previous_close: null }]) {
    assert.equal(benchmarkPrices({ ...envelope, benchmark_prices: [bad] }), null);
  }
});

test("cash-index migration and route upsert leave legacy NQ and model snapshots untouched", () => {
  const db = new DatabaseSync(":memory:");
  try {
    db.exec("CREATE TABLE roro_shadow_snapshots (as_of_key INTEGER, benchmark_close REAL, payload TEXT)");
    db.exec("INSERT INTO roro_shadow_snapshots VALUES (20260914, 30000, 'immutable-risk')");
    db.exec(readFileSync(new URL("../drizzle/0012_nasdaq100_chart_prices.sql", import.meta.url), "utf8"));
    const route = readFileSync(new URL("../app/api/roro/route.ts", import.meta.url), "utf8");
    const sql = route.match(/INSERT INTO nasdaq100_chart_prices[\s\S]+?(?=`)/)[0];
    const statement = db.prepare(sql);
    statement.run(20260914, 102.3, "2026-09-11", 103, "test");
    statement.run(20260914, 102.4, "2026-09-11", 103, "test");
    assert.equal(db.prepare("SELECT close FROM nasdaq100_chart_prices").get().close, 102.4);
    assert.equal(db.prepare("SELECT benchmark_close FROM roro_shadow_snapshots").get().benchmark_close, 30000);
    assert.equal(db.prepare("SELECT payload FROM roro_shadow_snapshots").get().payload, "immutable-risk");
  } finally { db.close(); }
});

test("actual chart components render daily and cumulative labels independently", () => {
  const source = readFileSync(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  const charts = source.slice(source.indexOf("function RiskScoreHistory("), source.indexOf("function ModelPerspectives("));
  const { outputText } = ts.transpileModule(`${charts}\nexport { RiskScoreHistory, RiskAccelerationHistory };`, {
    compilerOptions: { jsx: ts.JsxEmit.React, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  });
  const module = { exports: {} };
  new Function("React", "module", "exports", "nasdaqClose", "nasdaqDailyReturn", "nasdaqAcceleration", "trailingDifference", outputText)(
    React, module, module.exports, nasdaqClose, nasdaqDailyReturn, nasdaqAcceleration, trailingDifference,
  );
  const points = [point("2026-09-10", 100, "2026-09-09", 100),
    point("2026-09-11", price.previous_close, "2026-09-10", 100),
    point(price.as_of, price.close, price.previous_as_of, price.previous_close)]
    .map((item) => ({ ...item, snapshot: { ...item.snapshot, series_version: "test", payload: { market_risk_score: 50 } } }));
  const html = renderToStaticMarkup(React.createElement(module.exports.RiskScoreHistory, { points }));
  assert.match(html, /나스닥100 · 전일 대비/);
  assert.match(html, /-0\.82%/);
  assert.match(html, /차트 기간 누적/);
  assert.match(html, /\+2\.30%/);
  assert.match(html, /미국 종가 2026-09-14/);
  const legacy = renderToStaticMarkup(React.createElement(module.exports.RiskScoreHistory, {
    points: points.map((item) => ({ ...item, benchmark_ticker: undefined })),
  }));
  assert.doesNotMatch(legacy, /\+2\.30%|-0\.82%/);
  assert.match(legacy, /데이터 없음/);
  assert.match(renderToStaticMarkup(React.createElement(module.exports.RiskAccelerationHistory, { points })), /위험·나스닥100 변화 가속도/);
});
