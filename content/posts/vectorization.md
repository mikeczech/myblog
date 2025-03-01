+++
title = 'Enhancing Code Performance with Vectorization'
date = 2022-01-22
draft = false
+++

Python has the reputation of being slow and often rightly so. One of the reasons why this language is still so popular for a compute-heavy field like machine learning is that most libraries are actually written in a faster language like C, C++, or Rust. Moreover, libraries like NumPy or Polars make heavy use of so-called *vectorized computing*, where operations are applied on entire arrays or vectors at once, without explicit loops in the code. These operations are often mapped to highly efficient CPU instructions, allowing for parallel processing and using modern hardware features like [SIMD (Single Instruction, Multiple Data)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data). If you have ever asked yourself why [Polars can be so fast](https://pola.rs/posts/i-wrote-one-of-the-fastest-dataframe-libraries/), vectorized computing with SIMD is one of the reasons.

In contrast, classic for-loops in Python are inherently slower because they execute at the interpreter level, where each iteration has overhead associated with Python’s dynamic typing, function calls, and memory management.

For example, let's compare squaring each element in a large array using a for-loop versus vectorization:

```python
data = [i for i in range(1000000)]
result = []
for x in data:
    result.append(x ** 2)
```

Here is the vectorized version of the same logic:

```python
import numpy as np
data = np.arange(1000000)
result = data ** 2
```

When measuring the execution time, the vectorized version is dramatically faster. In a test run with [timeit](https://docs.python.org/3/library/timeit.html):

- **For-loop time:** 186 ms ± 562 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
- **Vectorized time:** 1.24 ms ± 28.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

That's a **100x speedup** with just a small change in code!

The good news is that in data-heavy code, it is often possible to re-write a for-loop into a vectorized computation using libraries like NumPy, usually achieving a massive speedup depending on your hardware.

Finally, to give you a somewhat less-known example, iOS also [supports vector-processing in Swift and Objective-C](https://developer.apple.com/accelerate/), which made a tremendous performance difference for an on-device machine learning app I recently developed. Keep in mind though, that it often makes the code a bit harder to read.
