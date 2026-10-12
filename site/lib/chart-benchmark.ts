export const CHART_BENCHMARK = "^NDX";

export type ChartBenchmark = {
  benchmark_ticker?: string;
  benchmark_close: number | null;
  benchmark_previous_as_of?: string | null;
  benchmark_previous_close?: number | null;
};

const positivePrice = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value > 0 && value < 1_000_000_000;

function isoDate(value: unknown): value is string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isFinite(date.valueOf()) && date.toISOString().slice(0, 10) === value;
}

export function nasdaqClose(point: ChartBenchmark): number | null {
  // Untagged legacy benchmark_close values are NQ futures, never cash-index data.
  return point.benchmark_ticker === CHART_BENCHMARK && positivePrice(point.benchmark_close)
    ? point.benchmark_close : null;
}

export function nasdaqDailyReturn(point: ChartBenchmark, asOf: string): number | null {
  const close = nasdaqClose(point);
  return close !== null && positivePrice(point.benchmark_previous_close) &&
    isoDate(point.benchmark_previous_as_of) && isoDate(asOf) && point.benchmark_previous_as_of < asOf
    ? (close / point.benchmark_previous_close - 1) * 100 : null;
}

export function nasdaqAcceleration(points: (ChartBenchmark & { snapshot: { as_of: string } })[]) {
  const returns = points.map((point) => nasdaqDailyReturn(point, point.snapshot.as_of));
  return returns.map((value, index) => {
    const previous = returns[index - 1];
    return value != null && previous != null &&
      points[index].benchmark_previous_as_of === points[index - 1].snapshot.as_of
      ? value - previous : null;
  });
}

type BenchmarkPrice = { as_of: string; close: number; previous_as_of: string | null; previous_close: number | null };

export function benchmarkPrices(value: unknown): BenchmarkPrice[] | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const candidate = value as Record<string, unknown>;
  if (Object.keys(candidate).length !== 2 || candidate.benchmark_ticker !== CHART_BENCHMARK ||
      !Array.isArray(candidate.benchmark_prices) || candidate.benchmark_prices.length < 1 || candidate.benchmark_prices.length > 40) return null;
  const dates = new Set<string>();
  const prices: BenchmarkPrice[] = [];
  for (const item of candidate.benchmark_prices) {
    if (!item || typeof item !== "object" || Array.isArray(item)) return null;
    const price = item as Record<string, unknown>;
    if (Object.keys(price).length !== 4 || !isoDate(price.as_of) || !positivePrice(price.close) || dates.has(price.as_of)) return null;
    if (!(price.previous_as_of === null && price.previous_close === null) &&
        !(isoDate(price.previous_as_of) && price.previous_as_of < price.as_of && positivePrice(price.previous_close))) return null;
    dates.add(price.as_of);
    prices.push(price as BenchmarkPrice);
  }
  return prices;
}
