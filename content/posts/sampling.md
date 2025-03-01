+++
title = 'Optional Sampling for Better Feedback Loops'
date = 2023-10-04
draft = false
+++

The main difference between machine learning applications and regular software is the dependence on a larger amount of (historical) data. With regards to speeding up the development process, a key insight is that it is usually not necessary to use all the data at every step. For example, testing if your data and machine learning pipelines work might only require a small, representative sample of your data. **Hence, it is often a good idea to add an option for sampling.** Only later in the development process, we run the pipelines on a larger amount of data, e.g. to ensure scalability, covering very rare edge cases, and - of course - to create the most powerful models. Note that sampling also enables developers to work on their local machines for a longer period, as the memory and compute requirements are kept low, potentially saving cloud costs and improving tooling.

Depending on how you access your data, data sampling (or taking a subset) might not be trivial though: A query like the following reliably produces a random sample, but still reads all the data in the background and might produce a high cloud bill.
```sql
SELECT * FROM table ORDER BY rand() LIMIT 1000
```
But note that for simple code testing, it might be good enough to skip randomization and just select the first N rows. This would be cost-efficient again and probably the most simple and flexible way to get a subset of your data:
```sql
SELECT * FROM table LIMIT 1000
```
If you are lucky, your data processing engine explicitly supports sampling from a table, e.g. [as it is the case for BigQuery](https://cloud.google.com/bigquery/docs/table-sampling) or [Trino](https://trino.io/docs/current/sql/select.html#tablesample)):
```sql
SELECT * FROM dataset.my_table TABLESAMPLE SYSTEM (10 PERCENT)
```
This comes with different limitations though, as is described in the engine's documentation (e.g. [data sampling might not be supported from views](https://cloud.google.com/bigquery/docs/table-sampling#limitations)).

If your data is partitioned (e.g. by date) another door for selecting a subset opens: Instead of a larger time period (e.g. the last year), one might just select a single partition while developing, increasing the time period step by step later on:
```sql
SELECT * FROM table WHERE dt = '20240601'
```
This only reads a single partition, but also assumes that it is small enough to enable a faster development. That assumption might not be true for very large datasets, e.g. tracking data from a popular online shop. But we might combine this technique with the previous ideas to both decrease the amount of processed data and getting a small, random sample:
```sql
SELECT * FROM table WHERE dt = '20240601' ORDER BY rand() LIMIT 1000
```

Note that for your own sanity it is important to make sure that your sample is always reproducible. Depending on your data processing engine, you might want to consider setting a seed ([e.g. in Polars](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.sample.html)) or resort [to more sophisticated approaches if a seed is not an option (e.g. for some query engines)](https://stackoverflow.com/questions/46019624/how-to-do-repeatable-sampling-in-bigquery-standard-sql).
