CREATE TABLE `dashboard_refresh_requests` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`status` text NOT NULL,
	`requested_at` text NOT NULL,
	`started_at` text,
	`finished_at` text,
	`as_of` text,
	`detail` text
);
--> statement-breakpoint
CREATE INDEX `idx_dashboard_refresh_requests_status_id` ON `dashboard_refresh_requests` (`status`,`id`);
--> statement-breakpoint
PRAGMA optimize;
