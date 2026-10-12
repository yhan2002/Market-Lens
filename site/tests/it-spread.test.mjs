import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ts from 'typescript';
import * as internals from '../lib/it-internals.ts';
import { itSpreadHistory, pairRelationship, pairRotationRows, pairSpreadPercentile, isItInternalsHistory, IT_INTERNALS_SERIES, SECTOR_PAIR_NAMES, sectorPairPrices, widestSectorPair, rankedSectorPairs } from '../lib/it-internals.ts';
const rows = Array.from({length:80}, (_,i) => [new Date(Date.UTC(2025,0,i+1)).toISOString().slice(0,10),0,0,0,0,0,0,i/100,null]);
const result = itSpreadHistory(rows, rows[63][0]).at(-1);
assert.equal(result.mean,31);
assert.ok(Math.abs(result.sd - Math.sqrt(336)) < 1e-10);
assert.ok(Math.abs(result.z - 32/Math.sqrt(336)) < 1e-10);
assert.deepEqual(itSpreadHistory(rows,rows[63][0]), itSpreadHistory(rows.slice(0,64),rows[63][0]));
assert.equal(itSpreadHistory(rows,rows[62][0]).at(-1).z,null);
assert.equal(itSpreadHistory(rows.map(r => [...r.slice(0,7),0,null]),rows[79][0]).at(-1).z,null);
console.log('Spread statistics: sample SD, lag, cutoff, warm-up and zero variance passed.');
const longRows = Array.from({length:600}, (_,i) => [new Date(Date.UTC(2024,0,i+1)).toISOString().slice(0,10),0,0,0,0,0,0,i/100,null]);
const extended = itSpreadHistory(longRows,longRows[599][0],504);
assert.equal(extended.length,504);
assert.deepEqual(extended.slice(-63),itSpreadHistory(longRows,longRows[599][0]));

const prices = [["2025-01-01", 100, 100, 100]];
for (let i = 1; i < 180; i++) {
  const x = Math.sin(i) * 0.008, last = prices.at(-1);
  prices.push([new Date(Date.UTC(2025,0,i+1)).toISOString().slice(0,10), last[1]*(1+0.001+2*x), last[2]*(1+x), last[3]*1.0004]);
}
const cutoff = prices[120][0];
const relation = pairRelationship(prices, cutoff);
assert.ok(Math.abs(relation.correlation - 1) < 1e-10);
assert.ok(Math.abs(relation.beta - 2) < 1e-10);
assert.ok(Math.abs(relation.r2 - 1) < 1e-10);
assert.deepEqual(relation, pairRelationship(prices.slice(0,121), cutoff));
assert.equal(pairRelationship(prices, prices[62][0]).beta, null);
assert.equal(pairRelationship(prices.map(p => [p[0],100,100,100]), cutoff).correlation, null);
const rotations = pairRotationRows(prices, cutoff);
assert.deepEqual(rotations, pairRotationRows(prices.slice(0,121), cutoff));
assert.equal(rotations.length, 61);
const latest = rotations.at(-1), base = prices[100], end = prices[120];
assert.ok(Math.abs(latest[7] - (end[1]/base[1] - end[2]/base[2])) < 1e-10);
assert.ok(Math.abs(latest[3] - (end[1]/base[1] - end[3]/base[3])) < 1e-10);
assert.equal(latest[8], null);
assert.deepEqual(pairRotationRows(prices, "2000-01-01"), []);
const spreadRows = Array.from({length:1261}, (_,i) => [new Date(Date.UTC(2020,0,i+1)).toISOString().slice(0,10),0,0,0,0,0,0,i / 1260,null]);
assert.deepEqual(pairSpreadPercentile(spreadRows, spreadRows.at(-2)[0]), {n:1260, percentile:1});
assert.deepEqual(pairSpreadPercentile(spreadRows, spreadRows[1258][0]), {n:1259, percentile:null});
assert.deepEqual(pairRelationship(prices, cutoff, 126), pairRelationship(prices.slice(0,121), cutoff, 126));
const legacy = {schema_version:2,series_version:IT_INTERNALS_SERIES,generated_at:"2025-05-01T00:00:00Z",start:rotations[0][0],as_of:cutoff,cadence:"DAILY",benchmark:"XLK",research_only:true,proxies:{semiconductor:"SOXX",software:"IGV"},rows:rotations};
assert.ok(isItInternalsHistory(legacy));
const payload = {...legacy,daily_pairs:{hw_sw:prices.slice(0,121),comm_energy:prices.slice(0,121)}};
assert.ok(isItInternalsHistory(payload));
for (const invalid of [prices, prices.slice(0,120), [...prices.slice(0,120), prices[119]], [...prices.slice(0,120), [cutoff,NaN,100,100]], [...prices.slice(0,120),[cutoff,0,100,100]]]) {
  assert.equal(isItInternalsHistory({...payload,daily_pairs:{...payload.daily_pairs,comm_energy:invalid}}),false);
}
console.log('Daily pairs: OLS direction/intercept, cutoff, warm-up, zero variance, return spread and payload validation passed.');

const sectors = {
  tickers: Object.keys(SECTOR_PAIR_NAMES),
  rows: prices.map((p, i) => [p[0], ...Object.keys(SECTOR_PAIR_NAMES).map((_, j) =>
    100 * Math.exp(i * 0.0002 * j + Math.sin(i / (j + 3)) * 0.03)), p[3]]),
};
const date = sectors.rows[150][0];
const ranking = rankedSectorPairs(sectors, date);
const widest = widestSectorPair(sectors, date);
const returns20 = sectors.tickers.map((_, i) => sectors.rows[150][i + 1] / sectors.rows[130][i + 1] - 1);
assert.equal(widest.first, sectors.tickers[returns20.indexOf(Math.max(...returns20))]);
assert.equal(widest.second, sectors.tickers[returns20.indexOf(Math.min(...returns20))]);
assert.ok(Math.abs(widest.spread - Math.max(...returns20) + Math.min(...returns20)) < 1e-10);
assert.deepEqual(widest, widestSectorPair({...sectors, rows: sectors.rows.slice(0, 151)}, date));
assert.equal(widestSectorPair(sectors, sectors.rows[19][0]), null);
assert.ok(widestSectorPair(sectors, sectors.rows[20][0]));
assert.equal(widestSectorPair(sectors, '2000-01-01'), null);
const tiedPair = widestSectorPair({...sectors, rows: sectors.rows.map(p => [p[0], ...Array(12).fill(100)])}, date);
assert.equal(tiedPair.spread, 0);
assert.notEqual(tiedPair.first, tiedPair.second);
const widestDetail = pairRotationRows(sectorPairPrices(sectors, widest.first, widest.second), date).at(-1);
assert.ok(Math.abs(widest.spread - widestDetail[7]) < 1e-10);
assert.equal(ranking.length, 55);
assert.equal(new Set(ranking.map(p => [p.first, p.second].sort().join('/'))).size, 55);
assert.ok(ranking.every((p, i) => p.date === date && (!i || Math.abs(ranking[i - 1].z) >= Math.abs(p.z))));
assert.deepEqual(ranking, rankedSectorPairs({...sectors, rows: sectors.rows.slice(0, 151)}, date));
for (const pair of ranking) {
  const detail = itSpreadHistory(pairRotationRows(sectorPairPrices(sectors, pair.first, pair.second), date), date, 1).at(-1);
  assert.ok(Math.abs(pair.z - detail.z) < 1e-10);
  const reverse = itSpreadHistory(pairRotationRows(sectorPairPrices(sectors, pair.second, pair.first), date), date, 1).at(-1);
  assert.ok(Math.abs(pair.z + reverse.z) < 1e-10);
}
assert.deepEqual(sectorPairPrices(sectors, 'XLC', 'XLC'), []);
assert.deepEqual(rankedSectorPairs(sectors, sectors.rows[122][0]), []);
assert.deepEqual(rankedSectorPairs({...sectors, rows: sectors.rows.map(p => [p[0], ...Array(12).fill(100)])}, date), []);
const sectorPayload = {...payload, sector_prices: {...sectors, rows: sectors.rows.slice(0,121)}};
assert.ok(isItInternalsHistory(sectorPayload));
for (const invalid of [
  {...sectorPayload.sector_prices, tickers: Array(11).fill('XLC')},
  {...sectorPayload.sector_prices, rows: sectors.rows},
  {...sectorPayload.sector_prices, rows: [[cutoff, ...Array(12).fill(0)]]},
]) assert.equal(isItInternalsHistory({...sectorPayload, sector_prices: invalid}), false);
console.log('11 sectors: 55 unique pairs, absolute ranking, chart parity, cutoff, reverse sign, warm-up and validation passed.');

const panelSource = readFileSync(new URL('../app/SectorPairPanel.tsx', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(panelSource, {
  compilerOptions: { jsx: ts.JsxEmit.React, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const panelModule = { exports: {} };
new Function('React', 'require', 'module', 'exports', outputText)(React,
  (path) => path === 'react' ? React : path === '../lib/it-internals' ? internals : { default: () => null },
  panelModule, panelModule.exports);
const renderPanel = (props) => renderToStaticMarkup(React.createElement(panelModule.exports.default, props));
const panelHtml = renderPanel({ sectors, selectedDate: date });
assert.deepEqual([...panelHtml.matchAll(/<option value="([^"]+)" selected="">/g)].map(m => m[1]), [widest.first, widest.second]);
assert.match(panelHtml, /자동 선택/);
assert.match(panelHtml, /최대 스프레드 자동 선택/);
assert.match(renderPanel({ prices, selectedDate: cutoff }), /기존 XLC\/XLE 비교/);
console.log('Sector pair default: actual panel renders the widest 20-day pair and retains the legacy fallback.');
