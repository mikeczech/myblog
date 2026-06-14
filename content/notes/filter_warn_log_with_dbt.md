+++
date = 2026-05-23
draft = false
tags = ["data-science", "data quality"]
+++

Recently, I've been working with dbt again and came across `dbt test --store-failures`, which automatically writes failing rows to tables. This is really useful for a warn-log-filter pattern: raise a warning when rows look invalid or suspicious, log them to a table for later analysis, and optionally filter them out without stopping the whole pipeline.

```
{{ config(severity = 'warn', store_failures = true) }}

select ...

```
