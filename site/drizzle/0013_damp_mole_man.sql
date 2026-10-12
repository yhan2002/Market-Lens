CREATE TABLE `news_articles` (
	`id` text PRIMARY KEY NOT NULL,
	`published_at` integer NOT NULL,
	`first_collected_at` text,
	`analysis_updated_at` text,
	`updated_at` text NOT NULL,
	`body` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `news_articles_time_id` ON `news_articles` (`published_at`,`id`);--> statement-breakpoint
CREATE TABLE `news_catalog` (
	`hours` integer PRIMARY KEY NOT NULL,
	`updated_at` text NOT NULL,
	`body` text NOT NULL
);
