+++
title = 'Polars is_in vs. inner join'
date = 2025-03-03
draft = false
tags = ["data-science", "polars", "python"]
+++

I’m currently working with large Polars dataframes (20M+ rows) and have noticed that using an inner join to filter on a categorical column can be significantly slower than using [is_in](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.is_in.html). There’s a [good example on GitHub that demonstrates this issue](https://github.com/pola-rs/polars/issues/8927):

```python
In [156]: N = 10

In [157]: df = pl.DataFrame({"x": pl.Series(range(N))})

In [158]: %timeit -n10 -r10 df.filter(pl.col("x").is_in(df.select("x").to_series() + 1))
120 µs ± 12.3 µs per loop (mean ± std. dev. of 10 runs, 10 loops each)

In [159]: %timeit -n10 -r10 df.join(df.select("x") + 1, on="x")
865 µs ± 101 µs per loop (mean ± std. dev. of 10 runs, 10 loops each)
```

The somewhat obvious lesson here is to avoid using significantly more expensive operations when a much simpler and more appropriate alternative is available.
