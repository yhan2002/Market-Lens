import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { isRoroNarrative } from '../lib/roro-narrative.ts';
import { BREADTH_BASE_SERIES } from '../lib/roro.ts';

const narrative = { as_of: '2026-09-08', series_version: 'test', provenance_hash: 'a'.repeat(64),
  summary: '위험이 완화됐습니다.', model: 'google/gemini-2.5-flash-lite', generated_at: '2026-09-09T00:00:00Z' };
assert.ok(isRoroNarrative(narrative));
for (const change of [{ as_of: '2026-02-30' }, { summary: '' }, { summary: 'x'.repeat(181) },
  { summary: 'a\nb' }, { provenance_hash: 'bad' }, { extra: 1 }]) {
  assert.equal(isRoroNarrative({ ...narrative, ...change }), false);
}
const snapshot = JSON.stringify({ provenance_hash: narrative.provenance_hash });
let overlay = null;
const env = { UPLOAD_TOKEN: 'test-only', ACTIVE_RORO_SERIES: 'test', DB: {
  prepare(sql) { return { bind(...args) { return {
    async first() { return sql.includes('roro_shadow_snapshots') ? { payload: snapshot } : overlay; },
    async run() {
      assert.ok(sql.startsWith('INSERT INTO roro_narratives'));
      overlay = { payload: args[2] };
    },
  }; }, async run() { assert.ok(sql.includes('CREATE')); } }; },
} };
const rows = [{ id: 'test:20260908', payload: snapshot }];
const chain = { select() { return this; }, from() { return this; }, where() { return this; },
  orderBy() { return this; }, async limit() { return rows; } };
const source = readFileSync(new URL('../app/api/roro/route.ts', import.meta.url), 'utf8')
  .replace(/^import .*;\r?\n/gm, '');
const context = { exports: {}, env, isRoroNarrative, BREADTH_BASE_SERIES, Request, Response, URL,
  getDb: () => chain, roroShadowSnapshots: {}, eq() {}, desc() {},
  roroHistoryFallbackSeries: () => [] };
vm.runInNewContext(ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
} }).outputText, context);
const post = (value, token = 'test-only') => context.exports.POST(new Request('https://test/api/roro', {
  method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: JSON.stringify({ narrative: value }),
}));
assert.equal((await post(narrative, 'wrong')).status, 401);
assert.equal((await post({ ...narrative, summary: '' })).status, 400);
assert.equal((await post({ ...narrative, provenance_hash: 'b'.repeat(64) })).status, 409);
assert.equal(overlay, null);
assert.equal((await post(narrative)).status, 200);
const get = query => context.exports.GET(new Request(`https://test/api/roro${query}`, {
  headers: { Authorization: 'Bearer test-only' },
}));
assert.equal(await (await get('')).text(), snapshot);
const displayed = await (await get('?include_narrative=1')).json();
assert.deepEqual(displayed.narrative, narrative);
assert.deepEqual(displayed.snapshot, JSON.parse(snapshot));
console.log('Narrative validation, authorization, provenance and immutable GET checks passed.');
