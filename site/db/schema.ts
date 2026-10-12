import { primaryKey, index, integer, real, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

export const newsSnapshots = sqliteTable("news_snapshots", {
  hours: integer("hours").notNull(),
  part: integer("part").notNull(),
  body: text("body").notNull(),
  generatedAt: text("generated_at").notNull(),
}, table => [primaryKey({columns: [table.hours, table.part]})]);

export const newsArticles = sqliteTable("news_articles", {
  id: text("id").primaryKey(),
  publishedAt: integer("published_at").notNull(),
  firstCollectedAt: text("first_collected_at"),
  analysisUpdatedAt: text("analysis_updated_at"),
  updatedAt: text("updated_at").notNull(),
  body: text("body").notNull(),
}, table => [index("news_articles_time_id").on(table.publishedAt, table.id)]);

export const newsCatalog = sqliteTable("news_catalog", {
  hours: integer("hours").primaryKey(),
  updatedAt: text("updated_at").notNull(),
  body: text("body").notNull(),
});
export const newsLoginRate = sqliteTable("news_login_rate", {
  id: integer("id").primaryKey(),
  bucket: integer("bucket").notNull(),
  count: integer("count").notNull(),
});

export const dashboardSnapshots = sqliteTable("dashboard_snapshots", {
  id: integer("id").primaryKey(),
  payload: text("payload").notNull(),
  generatedAt: text("generated_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

export const dashboardPolicySnapshots = sqliteTable(
  "dashboard_policy_snapshots",
  {
    id: text("id").primaryKey(),
    seriesVersion: text("series_version").notNull(),
    asOfKey: integer("as_of_key").notNull(),
    payload: text("payload").notNull(),
    generatedAt: text("generated_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (table) => [
    uniqueIndex("uq_dashboard_policy_series_date_v1").on(table.seriesVersion, table.asOfKey),
  ],
);

export const dashboardRefreshRequests = sqliteTable(
  "dashboard_refresh_requests",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    status: text("status").notNull(),
    requestedAt: text("requested_at").notNull(),
    startedAt: text("started_at"),
    finishedAt: text("finished_at"),
    asOf: text("as_of"),
    detail: text("detail"),
  },
  (table) => [
    index("idx_dashboard_refresh_requests_status_id").on(table.status, table.id),
  ],
);

export const roroShadowSnapshots = sqliteTable(
  "roro_shadow_snapshots",
  {
    id: text("id").primaryKey(),
    seriesVersion: text("series_version").notNull(),
    asOfKey: integer("as_of_key").notNull(),
    payload: text("payload").notNull(),
    benchmarkClose: real("benchmark_close"),
    generatedAt: text("generated_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (table) => [
    uniqueIndex("uq_roro_shadow_series_date_v1").on(table.seriesVersion, table.asOfKey),
  ],
);

// Display-only cash index: kept separate from immutable NQ-based model history.
export const nasdaq100ChartPrices = sqliteTable("nasdaq100_chart_prices", {
  asOfKey: integer("as_of_key").primaryKey(),
  close: real("close").notNull(),
  previousAsOf: text("previous_as_of"),
  previousClose: real("previous_close"),
  updatedAt: text("updated_at").notNull(),
});

export const sectorRoroShadowSnapshots = sqliteTable(
  "sector_roro_shadow_snapshots",
  {
    id: text("id").primaryKey(),
    seriesVersion: text("series_version").notNull(),
    asOfKey: integer("as_of_key").notNull(),
    payload: text("payload").notNull(),
    generatedAt: text("generated_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (table) => [
    uniqueIndex("uq_sector_roro_shadow_series_date_v1").on(table.seriesVersion, table.asOfKey),
  ],
);

export const sectorTurningGaugeSnapshots = sqliteTable(
  "sector_turning_gauge_snapshots",
  {
    id: text("id").primaryKey(),
    seriesVersion: text("series_version").notNull(),
    asOfKey: integer("as_of_key").notNull(),
    payload: text("payload").notNull(),
    generatedAt: text("generated_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (table) => [
    uniqueIndex("uq_sector_turning_gauge_series_date_v1").on(table.seriesVersion, table.asOfKey),
  ],
);

export const sectorResearchHistory = sqliteTable("sector_research_history", {
  id: integer("id").primaryKey(),
  payload: text("payload").notNull(),
  generatedAt: text("generated_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

export const itInternalsHistory = sqliteTable("it_internals_history", {
  id: integer("id").primaryKey(),
  payload: text("payload").notNull(),
  generatedAt: text("generated_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});
export const roroNarratives = sqliteTable("roro_narratives", {
  id: text("id").primaryKey(),
  provenanceHash: text("provenance_hash").notNull(),
  payload: text("payload").notNull(),
});

export const rotationEntrySnapshots = sqliteTable("rotation_entry_snapshots", {
  asOf: text("as_of").primaryKey(),
  payload: text("payload").notNull(),
});

export const sectorTransitionSnapshots = sqliteTable("sector_transition_snapshots", {
  asOf: text("as_of").primaryKey(),
  payload: text("payload").notNull(),
});

export const sectorHybridSnapshots = sqliteTable("sector_hybrid_snapshots", {
  asOf: text("as_of").primaryKey(),
  payload: text("payload").notNull(),
});
