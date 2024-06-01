+++
title = 'Why Machine Learning Projects Slow Down'
date = 2024-05-10
draft = true
+++

In my experience with machine learning projects, I’ve noticed a consistent pattern: everything starts off fast and exciting. But as time passes, progress slows. Eventually, even the best developers, including data scientists and engineers, start to hesitate before making even small changes to the code. These tasks feel increasingly daunting—often rightly so. This reluctance to tweak things means small but important improvements get ignored, halting the kind of iterative progress essential for creating exceptional products.

Why does this slowdown occur? Often, the focus is primarily on the non-functional requirements of model services, such as achieving low latency, while neglecting the performance of the development environment. As more data is added and automated testing expands, feedback loops become progressively slower, especially during tasks like executing data queries or training models. Increasing on-demand cloud costs can also contribute to the problem.

This post aims to provide practical tips for improving feedback loops, drawing on common scenarios I've encountered over the years. Please note that it specifically addresses technical challenges. Business-related challenges, which may be equally significant, are not covered here.

## Data Management

- no sampling
- wrong table formats (e.g. row-based vs. column-based), no / wrong partitioning
- wrong datatypes (e.g. float64 instead of float16)
- unexpected duplicates after a join

## Hardware and Resource Optimization

- Using distributed computing everywhere
- No GPU (too obvious?)
- Misconfigured data loader (+ batch size)

## Code Efficiency

- Choosing the wrong algorithm and/or algorithm implementation
- Loops instead of vectorization (e.g., Python vs. NumPy, pandas vs. polars)
- testing and debugging practices

