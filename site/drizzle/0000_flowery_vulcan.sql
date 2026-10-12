CREATE TABLE `dashboard_snapshots` (
	`id` integer PRIMARY KEY NOT NULL,
	`payload` text NOT NULL,
	`generated_at` text NOT NULL,
	`updated_at` text NOT NULL
);
