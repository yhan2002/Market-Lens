interface Fetcher {
  fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response>;
}

interface D1PreparedStatement {
  bind(...values: unknown[]): D1PreparedStatement;
  run(): Promise<{ meta?: { changes?: number } }>;
  all<T = unknown>(): Promise<{ results: T[] }>;
  first<T = unknown>(): Promise<T | null>;
}

interface D1Database {
  prepare(query: string): D1PreparedStatement;
  batch<T = unknown>(statements: D1PreparedStatement[]): Promise<T[]>;
  exec(query: string): Promise<unknown>;
}

declare module "cloudflare:workers" {
  export const env: {
    DB: D1Database;
    UPLOAD_TOKEN?: string;
    VIEW_TOKEN?: string;
    ACTIVE_POLICY_SERIES?: string;
    LEGACY_POLICY_SERIES?: string;
    ACTIVE_RORO_SERIES?: string;
    ACTIVE_SECTOR_RORO_SERIES?: string;
    ACTIVE_SECTOR_TURNING_SERIES?: string;
  };
}
