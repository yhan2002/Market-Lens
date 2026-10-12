CREATE TABLE `sector_turning_gauge_snapshots` (
	`id` text PRIMARY KEY NOT NULL,
	`series_version` text NOT NULL,
	`as_of_key` integer NOT NULL,
	`payload` text NOT NULL,
	`generated_at` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `uq_sector_turning_gauge_series_date_v1` ON `sector_turning_gauge_snapshots` (`series_version`,`as_of_key`);