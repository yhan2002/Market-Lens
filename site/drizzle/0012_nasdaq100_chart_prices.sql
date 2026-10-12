CREATE TABLE `nasdaq100_chart_prices` (
	`as_of_key` integer PRIMARY KEY NOT NULL,
	`close` real NOT NULL,
	`previous_as_of` text,
	`previous_close` real,
	`updated_at` text NOT NULL
);
