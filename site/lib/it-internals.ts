export const IT_INTERNALS_SCHEMA = 2 as const;
export const IT_INTERNALS_SERIES = "it-internals-shadow-v3-soxx-igv-xlk-daily-r20-r60-f20";

export const IT_STATES = [
  "BROAD_STRENGTH", "BROAD_WEAKNESS", "SEMICONDUCTOR_LEAD", "SOFTWARE_LEAD",
] as const;
export const IT_WINNERS = ["SEMICONDUCTOR", "SOFTWARE", "TIE"] as const;

export type ItState = (typeof IT_STATES)[number];
export type ItWinner = (typeof IT_WINNERS)[number];
export type ItStateCode = 0 | 1 | 2 | 3;

export type ItInternalsRow = [
  decisionDate: string, state20: ItStateCode, state60: ItStateCode,
  soxx20: number, igv20: number, soxx60: number, igv60: number,
  pairSpread20: number, pairForward20: number | null,
];

export type PairPriceRow = [date: string, first: number, second: number, benchmark: number];
export type SectorPrices = { tickers: string[]; rows: [string, ...number[]][] };
export const SECTOR_PAIR_NAMES: Record<string, string> = {
  XLB: "Materials", XLC: "Communication Services", XLE: "Energy", XLF: "Financials",
  XLI: "Industrials", XLK: "Information Technology", XLP: "Consumer Staples",
  XLRE: "Real Estate", XLU: "Utilities", XLV: "Health Care", XLY: "Consumer Discretionary",
};

export function sectorPairPrices(data: SectorPrices, first: string, second: string): PairPriceRow[] {
  const a = data.tickers.indexOf(first), b = data.tickers.indexOf(second);
  if (a < 0 || b < 0 || a === b) return [];
  return data.rows.map(([date, ...prices]) => [date, prices[a], prices[b], prices[data.tickers.length]]);
}

export function widestSectorPair(data: SectorPrices, cutoff: string) {
  const sample = data.rows.filter((row) => row[0] <= cutoff).slice(-21);
  if (sample.length < 21 || data.tickers.length < 2) return null;
  const [date, ...latest] = sample[20], [, ...start] = sample[0];
  const ranked = data.tickers.map((ticker, i) => ({ ticker, value: latest[i] / start[i] - 1 }))
    .sort((a, b) => b.value - a.value || a.ticker.localeCompare(b.ticker));
  return { first: ranked[0].ticker, second: ranked.at(-1)!.ticker, date, spread: ranked[0].value - ranked.at(-1)!.value };
}

export function rankedSectorPairs(data: SectorPrices, cutoff: string) {
  // Only the latest 124 prices are needed for 60D warm-up + 63 prior spreads + today.
  const sample = { ...data, rows: data.rows.filter((row) => row[0] <= cutoff).slice(-124) };
  return data.tickers.flatMap((first, i) => data.tickers.slice(i + 1).flatMap((second) => {
    const rows = pairRotationRows(sectorPairPrices(sample, first, second), cutoff);
    const last = itSpreadHistory(rows, cutoff, 1).at(-1);
    return last?.z !== null && last?.z !== undefined && Number.isFinite(last.z)
      ? [{ first, second, ...last, z: last.z }] : [];
  })).sort((a, b) => Math.abs(b.z) - Math.abs(a.z) || a.first.localeCompare(b.first) || a.second.localeCompare(b.second));
}

export type ItInternalsHistory = {
  schema_version: typeof IT_INTERNALS_SCHEMA;
  series_version: typeof IT_INTERNALS_SERIES;
  generated_at: string;
  start: string;
  as_of: string;
  cadence: "DAILY";
  benchmark: "XLK";
  research_only: true;
  proxies: {
    semiconductor: "SOXX";
    software: "IGV";
  };
  rows: ItInternalsRow[];
  daily_pairs?: { hw_sw: PairPriceRow[]; comm_energy: PairPriceRow[] };
  sector_prices?: SectorPrices;
};

// OLS: first daily return = intercept + beta * second daily return.
export function pairRelationship(rows: PairPriceRow[], cutoff: string, window = 63) {
  const prices = rows.filter((row) => row[0] <= cutoff).slice(-window - 1);
  const returns = prices.slice(1).map((row, i) => [row[1] / prices[i][1] - 1, row[2] / prices[i][2] - 1]);
  const n = returns.length;
  const empty = { n, correlation: null, beta: null, r2: null };
  if (n !== window) return empty;
  const mean = [0, 1].map((j) => returns.reduce((sum, row) => sum + row[j], 0) / n);
  const variance = [0, 1].map((j) => returns.reduce((sum, row) => sum + (row[j] - mean[j]) ** 2, 0));
  if (variance.some((value) => value <= 1e-20)) return empty;
  const covariance = returns.reduce((sum, row) => sum + (row[0] - mean[0]) * (row[1] - mean[1]), 0);
  const correlation = Math.max(-1, Math.min(1, covariance / Math.sqrt(variance[0] * variance[1])));
  return { n, correlation, beta: covariance / variance[1], r2: correlation ** 2 };
}

// Position of today's 20-day spread in the trailing observation distribution.
export function pairSpreadPercentile(rows: ItInternalsRow[], cutoff: string, window = 1_260) {
  const available = rows.filter((row) => row[0] <= cutoff);
  const sample = available.slice(-window).map((row) => row[7]);
  if (sample.length !== window) return { n: sample.length, percentile: null };
  const current = sample.at(-1)!;
  return { n: sample.length, percentile: sample.filter((spread) => spread <= current).length / sample.length };
}

export function pairRotationRows(prices: PairPriceRow[], cutoff: string): ItInternalsRow[] {
  return prices.filter((row) => row[0] <= cutoff).flatMap((row, i, available) => {
    if (i < 60) return [];
    const r = (j: 1 | 2 | 3, window: number) => row[j] / available[i - window][j] - 1;
    const relative = (j: 1 | 2, window: number) => r(j, window) - r(3, window);
    const state = (a: number, b: number): ItStateCode => a >= 0 ? (b >= 0 ? 0 : 2) : (b < 0 ? 1 : 3);
    const a20 = relative(1, 20), b20 = relative(2, 20), a60 = relative(1, 60), b60 = relative(2, 60);
    return [[row[0], state(a20, b20), state(a60, b60), a20, b20, a60, b60, r(1, 20) - r(2, 20), null]];
  });
}

// Compare each observation with the preceding 63 sessions, excluding that day.
export function itSpreadHistory(rows: ItInternalsRow[], cutoff: string, displaySessions = 63) {
  const available = rows.filter((row) => row[0] <= cutoff);
  return available.map((row, index) => {
    const prior = available.slice(Math.max(0, index - 63), index).map((item) => item[7] * 100);
    const spread = row[7] * 100;
    const mean = prior.length === 63 ? prior.reduce((a, b) => a + b, 0) / 63 : null;
    const sd = mean === null ? null : Math.sqrt(prior.reduce((sum, value) => sum + (value - mean) ** 2, 0) / 62);
    return { date: row[0], spread, mean, sd, z: mean !== null && sd !== null && sd > 1e-10 ? (spread - mean) / sd : null };
  }).slice(-displaySessions);
}

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function onlyKeys(value: Record<string, unknown>, keys: readonly string[]) {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function stateCode(value: unknown): value is ItStateCode {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value < IT_STATES.length;
}

function isoDate(value: unknown): value is string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

function validRow(value: unknown): value is ItInternalsRow {
  if (!Array.isArray(value) || value.length !== 9) return false;
  const [date, state20, state60, soxx20, igv20, soxx60, igv60, spread20, forward20] = value;
  return isoDate(date) && date >= "2001-01-01" && stateCode(state20) && stateCode(state60) &&
    [soxx20, igv20, soxx60, igv60, spread20].every(finite) && (forward20 === null || finite(forward20));
}

export function isItInternalsHistory(value: unknown): value is ItInternalsHistory {
  if (!object(value) || !onlyKeys(value, [
    "schema_version", "series_version", "generated_at", "start", "as_of", "cadence",
    "benchmark", "research_only", "proxies", "rows",
    ...(value.daily_pairs === undefined ? [] : ["daily_pairs"]),
    ...(value.sector_prices === undefined ? [] : ["sector_prices"]),
  ]) || value.schema_version !== IT_INTERNALS_SCHEMA || value.series_version !== IT_INTERNALS_SERIES ||
      !isoDate(value.start) || !isoDate(value.as_of) || value.start > value.as_of || value.cadence !== "DAILY" ||
      value.benchmark !== "XLK" || value.research_only !== true ||
      typeof value.generated_at !== "string" || Number.isNaN(new Date(value.generated_at).valueOf()) ||
      !object(value.proxies) || !onlyKeys(value.proxies, ["semiconductor", "software"]) ||
      value.proxies.semiconductor !== "SOXX" || value.proxies.software !== "IGV" ||
      !Array.isArray(value.rows) || value.rows.length === 0 || value.rows.length > 8_000 || !value.rows.every(validRow)) return false;
  if (value.daily_pairs !== undefined) {
    const asOf = value.as_of;
    const pairs = value.daily_pairs;
    if (!object(pairs) || !onlyKeys(pairs, ["hw_sw", "comm_energy"])) return false;
    for (const prices of Object.values(pairs)) {
      if (!Array.isArray(prices) || prices.length === 0 || prices.length > 8_000 || !prices.every((row, i) =>
        Array.isArray(row) && row.length === 4 && isoDate(row[0]) && row[0] <= asOf &&
        row.slice(1).every((p) => finite(p) && p > 0) && (i === 0 || row[0] > prices[i - 1][0]))) return false;
      if (prices.at(-1)[0] !== value.as_of) return false;
    }
  }
  const rows = value.rows as ItInternalsRow[];
  if (value.sector_prices !== undefined) {
    const asOf = value.as_of;
    const data = value.sector_prices;
    if (!object(data) || !onlyKeys(data, ["tickers", "rows"]) || !Array.isArray(data.tickers) ||
        data.tickers.length !== 11 || new Set(data.tickers).size !== 11 ||
        !data.tickers.every((ticker) => typeof ticker === "string" && Object.hasOwn(SECTOR_PAIR_NAMES, ticker)) ||
        !Array.isArray(data.rows) || data.rows.length === 0 || data.rows.length > 1_384) return false;
    const prices = data.rows;
    if (!prices.every((row, i) => Array.isArray(row) && row.length === 13 && isoDate(row[0]) &&
        row[0] <= asOf && row.slice(1).every((p) => finite(p) && p > 0) &&
        (i === 0 || row[0] > prices[i - 1][0])) || prices.at(-1)[0] !== value.as_of) return false;
  }
  const dates = rows.map((row) => row[0]);
  return dates[0] === value.start && dates.at(-1) === value.as_of &&
    dates.every((day, index) => index === 0 || day > dates[index - 1]);
}
