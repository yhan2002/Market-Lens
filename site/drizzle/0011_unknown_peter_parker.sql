CREATE TABLE `news_login_rate` (
	`id` integer PRIMARY KEY NOT NULL,
	`bucket` integer NOT NULL,
	`count` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `news_snapshots` (
	`hours` integer NOT NULL,
	`part` integer NOT NULL,
	`body` text NOT NULL,
	`generated_at` text NOT NULL,
	PRIMARY KEY(`hours`, `part`)
);
