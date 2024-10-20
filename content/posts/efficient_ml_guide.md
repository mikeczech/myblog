+++
title = 'A Concise Guide to Efficient ML Development'
date = 2024-10-20
draft = false
+++

In my experience with machine learning projects, I’ve noticed a consistent pattern: everything starts off fast and exciting. But as time passes, progress slows. Eventually, even the best developers start to hesitate before making even small changes to the code. These tasks feel increasingly daunting—often rightly so. This reluctance to tweak things means small but important improvements get ignored, halting the kind of iterative progress essential for creating exceptional products.

Why does this slowdown occur? Often, the focus is primarily on the non-functional requirements of model services, such as achieving low latency, while neglecting the performance of the development environment. As more data is added and automated testing expands, feedback loops often become progressively slower, especially during tasks like executing data queries or training models. Increasing on-demand cloud costs can also contribute to the problem.

This post is a braindump of my thoughts and experiences, and it might grow in the future as I continue to explore these issues. It aims to provide practical tips for improving development performance, drawing on common scenarios I've encountered over the years. Please note that it specifically addresses technical challenges; business-related challenges, which may be equally significant, are not covered here.

#### Table of contents

1. [Data Management](#data-management)
    * [Optimizing Data Processing Performance with the Right Format](#optimizing-data-processing-performance-with-the-right-format)
    * [Table Partitioning and Clustering Strategies](#table-partitioning-and-clustering-strategies)
    * [Selecting Appropriate Data Types](#selecting-appropriate-data-types)
    * [Handling Sparse vs. Dense Data](#handling-sparse-vs-dense-data)
    * [The Importance of Effective Communication](#the-importance-of-effective-communication)
2. [Data Preprocessing and Retrieval](#data-preprocessing-and-retrieval)
    * [Evaluating the Need for Distributed Query Engines](#evaluating-the-need-for-distributed-query-engines)
    * [Implementing Sampling Techniques](#implementing-sampling-techniques)
    * [Optimizing Data Loading and GPU Usage](#optimizing-data-loading-and-gpu-usage)
3. [Code Efficiency](#code-efficiency)
    * [Using Profiling Tools](#using-profiling-tools)
    * [Enhancing Performance with Vectorization](#enhancing-performance-with-vectorization)
    * [Applying Mixed Precision Training](#applying-mixed-precision-training)
    * [Best Practices in Testing and Debugging](#best-practices-in-testing-and-debugging)
4. [Summary](#summary)

## Data Management

If you encounter slow SQL queries or data processing in general, it might indicate a problem with your data and how it's stored. Even if the delay is only a few minutes, the iterative and exploratory nature of working with data can cripple productivity (and sanity). I've often observed that people don't question this lack of performance because they assume that processing larger amounts of data inevitably takes time. While that can indeed be the case, it's not always true!

### Optimizing Data Processing Performance with the Right Format

There are various ways to store data, and each comes with its own trade-offs in terms of performance. With query engines like [Trino](https://trino.io/) (distributed) or [DuckDB](https://duckdb.org/) (local), the initial choice usually boils down to either a **row-based format** (e.g., [Avro](https://avro.apache.org/)) or a **column-based one** (e.g., [Parquet](https://parquet.apache.org/)). Note that this most likely does not apply to cases where a [data warehouse solution](https://cloud.google.com/learn/what-is-a-data-warehouse?hl=en) like [BigQuery](https://cloud.google.com/bigquery?hl=en) is used, as such systems automatically convert the ingested data to a suitable format.

In most ML settings, a column-based format is often preferable because we usually deal with many columns but require only a subset at a time. This ensures that query engines can read only the required columns, reducing I/O and speeding up queries. More generally, columnar formats enable [predicate pushdown](https://www.dremio.com/wiki/predicate-pushdown/), allowing filters to be applied at the storage level (see, for example, the [Athena user guide](https://docs.aws.amazon.com/athena/latest/ug/columnar-storage.html)). For query engines like Athena or BigQuery, this can also lead to decreased cloud costs, as billing is based on the amount of data processed per query. Furthermore, columnar formats [support efficient compression algorithms](https://parquet.apache.org/docs/file-format/data-pages/compression/) that reduce data size without significantly impacting decompression speed. Smaller data sizes mean less data to read from disk, which directly improves query performance.

In contrast, row-based formats are more suitable for [OLTP (Online Transaction Processing)](https://en.wikipedia.org/wiki/Online_transaction_processing) and streaming use cases, such as when processing and writing data with [Apache Flink](https://flink.apache.org/). If you are ingesting data via streaming, it might be preferable to initially use a row-based format for efficient data processing. However, it's often advantageous to later convert this data into a columnar format that is more suitable for data analysis.

### Table Partitioning and Clustering Strategies

Another performance optimization again relates to typical usage patterns of the data: Like requiring only a subset of columns, we often want to only use a subset of rows. For example, we might only consider a given time period like all the rows from the last three months. This is where [**table partitioning**](https://cloud.google.com/bigquery/docs/partitioned-tables) is helpful to divide the data into more manageable blocks. The result is faster (and likely cheaper) queries if the queries filter by a given partitioning scheme. Note that your data processing engine and storage determine how the definition of partitions is implemented. For example, [*Hive-style partitioning*](https://delta.io/blog/pros-cons-hive-style-partionining/) separates data in folders:

```sql
purchases/date=20240801/
    fileA.parquet
    fileB.parquet
    fileC.parquet

purchases/date=20240802/
    fileA.parquet
    fileB.parquet
    fileC.parquet
```

Suppose you now want to run a query like `SELECT count(*) FROM purchases WHERE date='20240802'`, then the query engine only needs to read the files from `purchases/date=20240802`, skipping all the other data files. Hive-style partitioning works well, but also comes with several limitations with respect to exacerbating the [small file problem](https://blog.cloudera.com/the-small-files-problem/), processing overhead (file listings), and operational complexity (e.g. when altering partitions). In contrast, modern data formats like [Apache Iceberg](https://iceberg.apache.org/docs/1.4.0/partitioning/#partitioning-in-hive) and [Delta Lake](https://delta.io/) provide more sophisticated partitioning techniques, mitigating some of the drawbacks. Besides performance-related aspects, the concept of immutable partitioning itself leads to [interesting opportunities for building robust and reproducible data pipelines](https://maximebeauchemin.medium.com/functional-data-engineering-a-modern-paradigm-for-batch-data-processing-2327ec32c42a).

In addition to table partitioning, [**table clustering**](http://add/link) (or sometimens called bucketing) sorts related rows (according to a given set of columns) *within* a partition to align the order to common query usage patterns, enhancing performance. It's often advisable to consider both partitioning and clustering for achieving maximum performance, though the former is certainly more critical for all workloads that process larger amounts of data.

### Selecting Appropriate Data Types

So you made sure that the table format is correct, and partitioning is in place, but queries and data operations still take a long time and sometimes blow up (local) memory. Then it's time to have a closer look at the used data types for each column in case of tabular data. I've observed that people often use **too big data types**. For example, a feature column might be stored with a much higher precision than necessary, like float64.

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

### Handling Sparse vs. Dense Data

Besides selecting the most suitable data type, recognizing **sparse vs. dense data** can vastly reduce memory requirements. Sparse data refers to data where most entries are zero, while dense data contains a higher proportion of non-zero values.

A dense matrix with many non-zero elements requires significantly more memory compared to a sparse matrix, which contains mostly zeros. Efficiently handling sparse data is crucial to avoid memory issues and improve processing speed.

To manage sparse data effectively, it is best stored as a [sparse matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html). For instance, TF-IDF vectorizers often produce large matrices where each column represents the presence of a specific word in a document. These matrices are typically sparse since each document contains only a fraction of all possible words. This efficiency allows us to train logistic regression models locally on a MacBook, even with thousands of TF-IDF features. On the other hand, OpenAI's embeddings have "only" 1536 dense dimensions, making their representation more compact but not sparse. [^4]

Libraries like scikit-learn often handle sparse matrices [automatically](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer.transform), making it easy to work with high-dimensional but sparse data. However, for custom data, it’s up to the user to choose between dense or sparse representations.

Here is a code example illustrating how to work with sparse data using scikit-learn:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
documents = [
    "Machine learning is amazing",
    "Deep learning is a branch of machine learning",
    "Sparse data can be highly efficient"
]

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("Shape of TF-IDF Matrix:", tfidf_matrix.shape)
print("Type of TF-IDF Matrix:", type(tfidf_matrix))
```

The `tfidf_matrix` generated above is a sparse matrix of type `<class 'scipy.sparse.csr.csr_matrix'>`, allowing scikit-learn to efficiently manage high-dimensional data without excessive memory use.



### The Importance of Effective Communication

Considering all the previous observations, one key insight becomes clear: since many performance optimizations depend on how the data is actually used, establishing good communication between data providers (e.g., data engineers) and data consumers (e.g., data scientists or analysts) is of utmost importance. For example, if data scientists notice that certain filters are consistently applied—such as selecting data by date—they should communicate this to the data engineers. This way, the data engineers can incorporate these patterns into the table partitioning scheme, helping to keep performance high and costs low.


## Data Preprocessing and Retrieval

You've got fast access to all the data you need, and you're ready to do interesting things with it. Now, it's up to the tooling and its user to make the most of it. The tooling includes the choice of query engine, hardware, and its configuration. Let's discuss a few bottlenecks that I have observed on several occasions.

### Evaluating the Need for Distributed Query Engines

At first glance, using distributed computing sounds like a great idea. What's better than using a fast computer? Using two or more fast computers at once! This is the realm of data processing engines like Spark, Trino, or Snowflake. The downside is that overuse of distributed computing can really slow you down.

Distributed computing always comes with a certain overhead (because your job and data need to be *distributed*) and sometimes adds complexity, making it not always worth it. Nowadays, this is more often the case as the definition of "Big Data" is shifting. With the advent of widely available, powerful consumer CPUs (e.g., Apple's M series), large-RAM cloud machines ([yourdatafitsinram.net](https://yourdatafitsinram.net)), and multi-threaded query engines like Polars or DuckDB, it is now possible to process hundreds of gigabytes or even terabytes (*currently, [AWS supports up to 24 TiB of memory on a single machine](https://aws.amazon.com/ec2/instance-types/high-memory/))* of data on a single machine (locally or in the cloud)—of course, this statement depends somewhat on what your data looks like and the queries you might want to execute. The result is better feedback loops and less complicated tooling compared to what you would normally get with distributed query engines. Looking back, I have seen many use cases that wouldn't have required a distributed engine if other best practices like columnar storage, table partitioning, or the smallest-fitting data types had been in place.

In the case of unavoidable, **very large amounts of data**, I recommend first crunching the data into manageable pieces with a distributed engine like Spark (e.g., by aggregation) and then proceeding with Polars or DuckDB to iterate quickly and cheaply. The latter is often called "last-mile" data processing.

However, especially in large organizations, it is not unreasonable to have a distributed engine like Snowflake, BigQuery, or Athena (Trino) in place to support fast ad-hoc queries. This approach might save costs on data transfer ([which can be huge](https://www.reddit.com/r/aws/comments/xtq63m/why_isnt_there_more_outrage_over_aws_absolutely/)) from the cloud to local machines, does not have any hardware requirements for the user, and might be inexpensive on small datasets anyway due to the billing model (e.g., based on the amount of processed data, $5 per TB for BigQuery).


### Implementing Sampling Techniques

The main difference between machine learning applications and regular software is the dependence on a larger amount of (historical) data. With regards to speeding up the development process, another key insight is that it is usually not necessary to use all the data at every step. For example, testing if your data and machine learning pipelines work might only require a small, representative sample of your data. Hence, it is often a good idea to add an option for sampling. Only later in the development process, we run the pipelines on a larger amount of data, e.g. to ensure scalability, covering very rare edge cases, and - of course - to create the most powerful models. Note that sampling also enables developers to work on their local machines for a longer period, as the memory and compute requirements are kept low, potentially saving cloud costs and improving tooling.

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


### Optimizing Data Loading and GPU Usage

Today, it's not too uncommon to have data processing tasks—including training neural networks—that require a GPU to finish in a reasonable amount of time. Acquiring one is fairly straightforward thanks to the abundance of cloud providers like AWS, GCP, Paperspace, and so on. However, it's up to the user to squeeze as much compute as possible out of the GPU, which can be non-trivial.

One common pattern is observing the GPU sporadically jumping to 100% utilization and then dropping back to 0%, with this cycle repeating over time. This hints at a bottleneck in the interplay between data loading and GPU processing. In libraries like PyTorch or TensorFlow, a first step would be to check the configuration of the data loader. A data loader is responsible for loading data in the background and providing it to the GPU. Common parameters include the number of worker processes, batch size, pre-fetch factor, etc. The optimal values depend on the available compute resources (e.g., number of CPU cores, CPU and GPU memory) and the nature of the problem to solve ([some ML tasks like contrastive learning require large batch sizes to perform well](https://lilianweng.github.io/posts/2021-05-31-contrastive/#large-batch-size)).

For example, a typical configuration for the PyTorch data loader might look like this:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

In this setup:

- `batch_size=128` defines the number of samples processed before the model is updated.
- `num_workers=4` uses four subprocesses for data loading.
- `pin_memory=True` enables faster data transfer to the GPU.
- `prefetch_factor=2` allows the loader to prefetch batches in advance.

Furthermore, disk utilization can also be a bottleneck if reading the data is much slower than processing it. This can be mitigated by replacing a regular HDD with an SSD.[^6] I experienced this problem when working with image data.

Common CLI tools for bottleneck investigation are `nvidia-smi` for GPU monitoring and `htop` for CPU monitoring. To monitor disk utilization, tools like `iotop` or `iostat` are useful. For example, you can use `iotop` to see real-time disk I/O usage:

```bash
sudo iotop
```

This command displays a list of processes performing I/O operations, helping you identify if disk I/O is the bottleneck.


Also, profiling tools like [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) can provide insights into performance issues by analyzing the time spent on data loading versus computation. Here's how you might use it:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for batch in train_loader:
        # Your training loop here
        pass

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

By examining the profiler output, you can identify bottlenecks in your code and adjust accordingly.



## Code Efficiency

Your development infrastructure is set up to maximize available hardware—whether it's a machine with 192 vCPUs, an NVIDIA H100 GPU, both, or an entire cluster. Now it's time to retrieve data and write code to make the most of it.

This section focuses on what happens after data preprocessing and retrieval, especially when you need to write custom code beyond what a query engine can handle. What bottlenecks might you encounter here?

### Using Profiling Tools

To identify and resolve performance bottlenecks in your Python code, consider these tools:

- [cProfile](https://docs.python.org/3/library/profile.html): Built-in profiler for measuring function execution times.
- [line_profiler](https://pypi.org/project/line_profiler/): Profiles execution time of individual code lines.
- [memory_profiler](https://pypi.org/project/memory_profiler/): Monitors memory usage over time.
- [Py-Spy](https://github.com/benfred/py-spy): Sampling profiler for running programs without code modification.
- [Scalene](https://github.com/plasma-umass/scalene): High-performance CPU and memory profiler.

By integrating these tools, you can efficiently identify and address performance issues in your code.

### Enhancing Performance with Vectorization

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


### Applying Mixed Precision Training

If your GPU utilization is almost always at 100%, it is now time to consider speeding up the actual computation that happens there. Note that most operations on the GPU involve dealing with floating point variables (activations, gradients, and so on), each having a certain precision -- most commonly float32 (or full precision). The idea of *mixed precision training* is to optimize computational efficiency by utilizing lower-precision numerical formats for a subset of the variables. A popular choice here is float16 (half precision) which usually both improves training speed and reduces the overall memory usage, i.e. also making it possible to use larger batch sizes. [^2] Note that no task-specific accuracy is lost compared to full precision training, as the GPU automatically identifies the steps that still require full precision [^1]. Thus, it is almost always a good idea to enable mixed precision training if your hardware and use case supports this.

Mixed precision training was introduced with the [NVIDIA Pascal architecture (2016)](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/). For example, popular GPU-accelerated instance types like [G4 / NVIDIA T4 (Turing, 2018)](https://aws.amazon.com/ec2/instance-types/g4/), [G5 / NVIDIA A10G (Ampere, 2020)](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [NVIDIA L4 instances on GCP (Lovelace, 2023)](https://cloud.google.com/blog/products/compute/introducing-g2-vms-with-nvidia-l4-gpus) all support mixed precision training (while the outdated [NVIDIA K80 (Tesla, 2014)](https://www.nvidia.com/en-gb/data-center/tesla-k80/) does not).

So what speed improvement can we expect here? NVIDIA mentions an up to 3x speedup for training [^1]. From practice, I can mostly confirm this statement: Enabling mixed precision training usually at least halved the training time on typical classification tasks based on [Distilbert](https://huggingface.co/docs/transformers/en/model_doc/distilbert) fine-tuning. In addition, I was often able to double the batch size without running into out-of-memory issues. An excellent resource for mixed precision training is also the official [NVIDIA FAQ](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#faq).

### Best Practices in Testing and Debugging

In contrast to hardware-specific aspects like SIMD, a typical bottleneck might also lie in not using good practices for software development in general. I observed this especially for data scientists who used to only write code in Jupyter notebooks and then were expected to build production services. People often just moved their code to workflow engines like [Apache Airflow](https://airflow.apache.org) or [Metaflow](https://metaflow.org), resulting in a single, large file which both contains the workflow definition and the code for model training. In order to test that a change to the code didn't break anything, the whole workflow has to be run, often implying a painfully slow - and sometimes expensive - feedback loop until things work out as expected. Sometimes this is somewhat mitigated by  running the workflow on a data sample instead of the whole dataset, but it will still often feel slow to work on these things - because a workflow usually does a lot of things: data loading, validation, preprocessing, and so on.

To speed things up here, I encourage people to always split their workflow into more manageable modules and write [automated (unit-) tests](https://docs.pytest.org/en/stable/)[^3]. This not only makes later changes much easier (due to automatically testing a set of assumptions that have been made during development), but also enables a much more focused development / debugging flow where only the parts of the codebase affected by a change have to be executed at a time. Hence, in contrast to what the name at first suggests, automatic testing is not only about ensuring correctness of your code, but also about improving feedback loops.

Key here is also that most of the tests do not depend on other, complex, or even external components like databases, as this would again slow down the execution of such tests. I recommend to understand the idea behind the [Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html) to achieve a good trade-off between testing accuracy and speed.


## Summary

We’ve explored various methods to accelerate your code when building large-scale data-driven solutions. Optimization strategies depend on your specific tech stack, available infrastructure (such as high-memory cloud machines), budget, and the size of the data you’re processing—there’s no one-size-fits-all approach. Consulting technology-specific optimization guides, like those for [Athena](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html) or [BigQuery](https://cloud.google.com/bigquery/docs/best-practices-performance-overview), can also enhance performance.

**This post offers a broad overview rather than detailed explanations of each approach, aiming to enable you to conduct more in-depth research.** While this guide covers several key strategies, many additional ideas also remain unexplored. I aim to expand upon them over time as opportunities allow.


[^1]: For more information on mixed-precision training, see [NVIDIA’s documentation](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html).

[^2]: There are now additional data type options like BF16 or TF32 available from the NVIDIA Ampere architecture onwards. See [this NVIDIA blog post](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/) for more details.

[^3]: If appropriate, [Test-Driven Development](https://martinfowler.com/bliki/TestDrivenDevelopment.html) might also be a good idea.

[^4]: While OpenAI embeddings are still more powerful for many natural language understanding tasks, it’s often beneficial to consider simpler text representations as a baseline (e.g., TF-IDF or FastText with SIF). You may find that these already perform well enough for your use case, allowing you to save significantly on OpenAI API costs.

[^6]: Major cloud providers like AWS or GCP make it easy to attach an additional fast SSD to a virtual machine.

