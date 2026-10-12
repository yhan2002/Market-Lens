CREATE TABLE `roro_shadow_snapshots` (
	`id` text PRIMARY KEY NOT NULL,
	`series_version` text NOT NULL,
	`as_of_key` integer NOT NULL,
	`payload` text NOT NULL,
	`generated_at` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `uq_roro_shadow_series_date_v1` ON `roro_shadow_snapshots` (`series_version`,`as_of_key`);