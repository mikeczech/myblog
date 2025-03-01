+++
title = 'Reconsidering Data Types for Efficiency'
date = 2022-09-20
draft = false
+++

### 

You notice that queries and data operations take a long time and sometimes blow up (local) memory. Then it's time to have a closer look at the used data types for each column in case of tabular data. I've observed that people often use **too big data types**. For example, a feature column might be stored with a much higher precision than necessary, like float64.

A practical remedy is to opt for smaller data types like float32 or float16, which roughly halve the memory needed with each reduction. In some cases, more specialized techniques like embedding [quantization](https://huggingface.co/blog/embedding-quantization#improving-scalability) or [Matryoshka embedding models](https://huggingface.co/blog/matryoshka) can vastly reduce memory footprint without significantly harming performance.

However, lowering precision can lead to accuracy issues, depending on the domain and use case. My advice is to start with higher precision (e.g., float64) and then evaluate whether using float32 or float16 impacts your results. This is especially easy to experiment with in libraries like **Polars**, where data type conversion is efficient, and you can leverage functions like `.cast(pl.Float32)` to test smaller types.

To measure memory usage in Polars, you can use the built-in method to calculate memory footprint per DataFrame:
```python
import polars as pl
from polars import DataFrame

df = DataFrame({"feature": [1.0] * 10_000_000})

# Float64 (default)
float64_memory = df.estimated_size(unit='mb')

# Convert to Float32 and measure again
df_float32 = df.with_columns(df["feature"].cast(pl.Float32))
float32_memory = df_float32.estimated_size(unit='mb')

print(f"Float64 Memory: {float64_memory} MB")
print(f"Float32 Memory: {float32_memory} MB")
```
Output:
```
Float64 Memory: 76.29 MB
Float32 Memory: 38.15 MB
```
