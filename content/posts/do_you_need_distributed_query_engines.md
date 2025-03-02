+++
title = 'Do you really need a distributed query engine?'
date = 2024-10-11
draft = false
+++

At first glance, using distributed computing sounds like a great idea. What's better than using a fast computer? Using two or more fast computers at once! This is the realm of data processing engines like [Spark](https://spark.apache.org/), [Trino](https://trino.io/), or [Snowflake](https://www.snowflake.com/en/product/features/horizon/). The downside is that overuse of distributed computing can significantly slow you down and unnecessarily inflate cloud expenses.

Distributed computing always comes with a certain overhead (because your job and data need to be *distributed*) and sometimes adds complexity, making it not always worth it. Nowadays, this is more often the case as the definition of "Big Data" is shifting: With the advent of widely available, powerful consumer CPUs (e.g., Apple's M series), large-RAM cloud machines ([yourdatafitsinram.net](https://yourdatafitsinram.net)), and multi-threaded query engines like Polars or DuckDB, it is now possible to process hundreds of gigabytes or even terabytes (*currently, [AWS supports up to 24 TiB of memory on a single machine](https://aws.amazon.com/ec2/instance-types/high-memory/))* of data on a single machine (locally or in the cloud)—of course, this statement depends somewhat on what your data looks like and the queries you might want to execute.

The result is **better feedback loops and (sometimes) less complicated tooling** compared to what you would normally get with distributed query engines. Looking back, I have seen many use cases that wouldn't have required a distributed engine if other best practices like columnar storage, table partitioning, or the smallest-fitting data types had been in place.

In the case of unavoidable, **very large amounts of data**, I recommend first crunching the data into manageable pieces with a distributed engine like Spark (e.g., by aggregation) and then proceeding with Polars or DuckDB to iterate quickly and cheaply. The latter is often called "last-mile" data processing.

However, especially in large organizations, it is not unreasonable to have a distributed engine like Snowflake, BigQuery, or Athena (Trino) in place to support fast ad-hoc queries. This approach might save costs on data transfer ([which can be huge](https://www.reddit.com/r/aws/comments/xtq63m/why_isnt_there_more_outrage_over_aws_absolutely/)) from the cloud to local machines, does not have any hardware requirements for the user, and might be inexpensive on small datasets anyway due to the billing model (e.g., based on the amount of processed data, $5 per TB for BigQuery).
