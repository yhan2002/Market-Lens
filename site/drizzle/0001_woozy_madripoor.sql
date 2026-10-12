CREATE TABLE `dashboard_policy_snapshots` (
	`id` text PRIMARY KEY NOT NULL,
	`series_version` text NOT NULL,
	`as_of_key` integer NOT NULL,
	`payload` text NOT NULL,
	`generated_at` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `uq_dashboard_policy_series_date_v1` ON `dashboard_policy_snapshots` (`series_version`,`as_of_key`);
--> statement-breakpoint
PRAGMA optimize;
