+++
title = 'Why Machine Learning Projects Slow Down'
date = 2024-05-10
draft = true
+++

In my experience with machine learning projects, I’ve noticed a consistent pattern: everything starts off fast and exciting. But as time passes, progress slows. Eventually, even the best developers, including data scientists and engineers, start to hesitate before making even small changes to the code. These tasks feel increasingly daunting—often rightly so. This reluctance to tweak things means small but important improvements get ignored, halting the kind of iterative progress essential for creating exceptional products.

Why does this slowdown occur? Often, the focus is primarily on the non-functional requirements of model services, such as achieving low latency, while neglecting the performance of the development environment. As more data is added and automated testing expands, feedback loops become progressively slower, especially during tasks like executing data queries or training models. Increasing on-demand cloud costs can also contribute to the problem.

This post aims to provide practical tips for improving feedback loops, drawing on common scenarios I've encountered over the years. Please note that it specifically addresses technical challenges. Business-related challenges, which may be equally significant, are not covered here.

## Data Management

If you encounter slow SQL queries or data processing in general, there might be a problem with the data and its storage. Even if the delay is only a few minutes, the iterative and explorative nature of working with data can cripple productivity (and sanity). Interestingly, I often observed that people do not question the lack of performance here, because they just assume that processing larger amounts of data has to take some time. This can be the case, but not always!

### Wrong Data Format

There are various ways for storing data and each has its trade-offs in terms of performance. In case of query engines like Trino (distributed) or DuckDB (local), the first choice usually boils down to either choosing a row-based format (e.g. AVRO) or a column-based one (e.g. Parquet or Apache Iceberg). In most ML settings, the latter turned out to be preferable, as we usually deal with many columns, but only require a subset at a time. This ensures that query engines are able to only read the required columns, lowering the amount of data to be processed. This implies both a better performance and sometimes decreased costs if an engine like Athena is used (as the billing is based on the amount of processed bytes for a query). In contrast, row-based format are more suitable for OLTP use cases (Online Transaction Processing).

- no / wrong partitioning / clustering
- wrong datatypes (e.g. float64 instead of float16)
- dense vs. sparse data
- communication: data engineers and scientists

### Data-related Issues

- unexpected duplicates after a join
- sampling capabilities
- something about unstructured data (images, text?)

## Hardware and Resource Optimization

- Using distributed computing everywhere
- No GPU (too obvious?)
- Misconfigured data loader (+ batch size)

## Code Efficiency

- Choosing the wrong algorithm and/or algorithm implementation
- Loops instead of vectorization (e.g., Python vs. NumPy, pandas vs. polars)
- No parallelization
- testing and debugging practices

