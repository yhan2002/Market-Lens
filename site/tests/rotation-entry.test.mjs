import assert from 'node:assert/strict';
import test from 'node:test';
import {isRotationEntry, ROTATION_ENTRY_SERIES, ROTATION_SECTORS} from '../lib/rotation-entry.ts';

test('entry contract preserves nominee, cooldown, baseline and source-data exclusion', () => {
  const ranks = Object.fromEntries(ROTATION_SECTORS.map((t,i) => [t,i+1]));
  const changes = Object.fromEntries(ROTATION_SECTORS.map(t => [t,0]));
  changes.XLF=6; changes.XLV=-6;
  const row = {series_version:ROTATION_ENTRY_SERIES,as_of:'2026-09-09',recorded_at:'2026-09-10T00:00:00Z',phase:'ACTIVE',alert:true,threshold:6,nominated:'XLF',nearest4:'XLF',score:6,ranks,rank_change5:changes,days_since_last_alert:null};
  assert.equal(isRotationEntry(row),true);
  assert.equal(isRotationEntry({...row,nominated:'XLV'}),false);
  assert.equal(isRotationEntry({...row,days_since_last_alert:9}),false);
  assert.equal(isRotationEntry({...row,days_since_last_alert:9,alert:false}),true);
  assert.equal(isRotationEntry({...row,phase:'BASELINE',as_of:'2026-09-08',alert:false}),true);
  assert.equal(isRotationEntry({...row,phase:'BASELINE'}),false);
  assert.equal(isRotationEntry({...row,phase:'HISTORICAL_REPLAY',as_of:'2026-06-01'}),true);
  assert.equal(isRotationEntry({...row,phase:'HISTORICAL_REPLAY',as_of:'2026-05-29'}),false);
  assert.equal(isRotationEntry({...row,input_prices:{}}),false);
  assert.equal(isRotationEntry({...row,as_of:'2026-02-30'}),false);
  assert.equal(isRotationEntry({...row,as_of:'2027-01-04',phase:'EXPIRED_STATE_ONLY',alert:false}),true);
});
