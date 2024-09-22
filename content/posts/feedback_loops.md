+++
title = 'Technical Reasons for Slow ML Projects'
date = 2024-05-10
draft = true
+++

In my experience with machine learning projects, I’ve noticed a consistent pattern: everything starts off fast and exciting. But as time passes, progress slows. Eventually, even the best developers, including data scientists and engineers, start to hesitate before making even small changes to the code. These tasks feel increasingly daunting—often rightly so. This reluctance to tweak things means small but important improvements get ignored, halting the kind of iterative progress essential for creating exceptional products.

Why does this slowdown occur? Often, the focus is primarily on the non-functional requirements of model services, such as achieving low latency, while neglecting the performance of the development environment. As more data is added and automated testing expands, feedback loops become progressively slower, especially during tasks like executing data queries or training models. Increasing on-demand cloud costs can also contribute to the problem.

This post aims to provide practical tips for improving feedback loops, drawing on common scenarios I've encountered over the years. Please note that it specifically addresses technical challenges. Business-related challenges, which may be equally significant, are not covered here.

**Mention that all the advice depends on the amount of data you have**

#### Table of contents

1. [Data Management](#data-management)
    * [Optimizing Data Processing Performance with the Right Format](#optimizing-data-processing-performance-with-the-right-format)
    * [Table Partitioning and Clustering](#table-partitioning-and-clustering)
    * [Data Types](#data-types)
    * [Sparse vs. Dense Data](#sparse-vs-dense-data)
    * [Communication is Key](#communication-is-key)
2. [Data Preprocessing and Retrieval](#data-preprocessing-and-retrieval)
    * [Do You Really Need a Distributed Query Engine?](#do-you-really-need-a-distributed-query-engine)
    * [Sampling](#sampling)
    * [Interplay Between Data Loading and GPU](#interplay-between-data-loading-and-gpu)
3. [Code Efficiency](#code-efficiency)
    * [Loops Instead of Vectorization](#loops-instead-of-vectorization)
    * [Mixed Precision Training](#mixed-precision-training)
    * [Testing and Debugging Practices](#testing-and-debugging-practices)

## Data Management

If you encounter slow SQL queries or data processing in general, it might indicate a problem with your data and how it's stored. Even if the delay is only a few minutes, the iterative and exploratory nature of working with data can cripple productivity (and sanity). I've often observed that people don't question this lack of performance because they assume that processing larger amounts of data inevitably takes time. While that can sometimes be the case, it's not always true!

### Optimizing Data Processing Performance with the Right Format

There are various ways to store data, and each comes with its own trade-offs in terms of performance. With query engines like [Trino](https://trino.io/) (distributed) or [DuckDB](https://duckdb.org/) (local), the initial choice usually boils down to either a **row-based format** (e.g., [Avro](https://avro.apache.org/)) or a **column-based one** (e.g., [Parquet](https://parquet.apache.org/)). Note that this most likely does not apply to cases where a [data warehouse solution](https://cloud.google.com/learn/what-is-a-data-warehouse?hl=en) like [BigQuery](https://cloud.google.com/bigquery?hl=en) is used, as such systems automatically convert the ingested data to a suitable format.

In most ML settings, a column-based format is often preferable because we usually deal with many columns but require only a subset at a time. This ensures that query engines can read only the required columns, reducing I/O and speeding up queries. More generally, columnar formats enable [predicate pushdown](https://www.dremio.com/wiki/predicate-pushdown/), allowing filters to be applied at the storage level (see, for example, the [Athena user guide](https://docs.aws.amazon.com/athena/latest/ug/columnar-storage.html)). For query engines like Athena or BigQuery, this can also lead to decreased cloud costs, as billing is based on the amount of data processed per query. Furthermore, columnar formats [support efficient compression algorithms](https://parquet.apache.org/docs/file-format/data-pages/compression/) that reduce data size without significantly impacting decompression speed. Smaller data sizes mean less data to read from disk, which directly improves query performance.

In contrast, row-based formats are more suitable for [OLTP (Online Transaction Processing)](https://en.wikipedia.org/wiki/Online_transaction_processing) and streaming use cases, such as when processing and writing data with [Apache Flink](https://flink.apache.org/). If you are ingesting data via streaming, it might be preferable to initially use a row-based format for efficient data processing. However, it's often advantageous to later convert this data into a columnar format that is more suitable for data analysis.

### Table Partitioning and Clustering

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

Suppose you now want to run a query like `SELECT count(*) FROM purchases WHERE date='20240802'`, then the query engine only needs to read the files from `purchases/date=20240802`, skipping all the other data files. Hive-style partitioning works well, but also comes with several limitations with respect to exacerbating the [small file problem](https://blog.cloudera.com/the-small-files-problem/), processing overhead (file listings), and operational complexity (e.g. when altering partitions). In contrast, modern data formats like [Apache Iceberg](https://iceberg.apache.org/docs/1.4.0/partitioning/#partitioning-in-hive) and [Delta Lake](https://delta.io/) provide more sophisticated partitioning techniques, mitigating some of the drawbacks. Besides performance-related aspects, the concept of immutable partitioning itself leads to [interesting opportunities for building robust and reproducible data pipelines](http://link/to/the/functional/data/engineer).

In addition to table partitioning, [**table clustering**](http://add/link) (or sometimens called bucketing) sorts related rows (according to a given set of columns) *within* a partition to align the order to common query usage patterns, enhancing performance. It's often advisable to consider both partitioning and clustering for achieving maximum performance, though the former is certainly more critical for all workloads that process larger amounts of data.

### Data Types

So you made sure that the table format is correct and partitoning and/or is in place to, but queries / data operations still take a long time and sometimes blow up (local) memory. Then it's time to have a closer look at the used data types for each column in case of tabular data. I oberved that people often use **too big data types**. For example, a feature column might be stored with a much higher precision than necessary, e.g. float64.

As a remedy, smaller data types like float32 or float16 might me much more reasonable and roughly halve the memory necessary with each step down. In case of embeddings, [quantization](https://huggingface.co/blog/embedding-quantization#improving-scalability) or [Matryoshka embedding models](https://huggingface.co/blog/matryoshka) help to vastly reduce the memory footprint while preserving much of the performance.

But be careful: In some cases, a lower precision can come with bad consequences, e.g. with respect to model output. It depends a lot on the domain, but I usually start with bigger data types until I can evaluate the impact of using a lower precision and make changes to the data types accordingly.

* show CLI for measuring memory usage?
* show graph that compares the memory usage for different data types?

### Sparse vs. Dense Data

Besides selecting the most-suitable data type, it is also useful to recognize the presence **of sparse (vs. dense) data** in order to vastly reduce the memory requirements. We speak of sparse data if most entries are zero.

* show comparision between sparse and dense data

To keep this type of data manageable, it is paramount to store it as a [sparse matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html). As an example, TFIDF vectorizers produce extremely large matrices where each column represents the presence of a word (or token) in a text. That's why we are able to train a logistic regression locally on a Macbook -- with many thousands of TF-IDF matrix dimensions. In contrast, the OpenAI Embeddings "only" have 1536 (dense) dimensions. Fascinating, isn't it? [^4]

Libraries like scikit-learn often do this [automatically in the background](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer.transform), making it a breeze to work with high-dimensional, but sparse data. But for custom data sources it is up to the user to store it either as a dense or sparse representation.

* Add code examples for scikit-learn and scipy sparse matrices


### Communication is Key

Considering all the previous observations, one key insight becomes clear: Since many performance optimizations depend on *how the data is acually used*, it is of utmost importance to establish a good communication between data providers (e.g. data engineers) and consumers (e.g. data scientists or analysts). For example, data scientists might notice that certain filters are almost always used (e.g. selecting by date) and this should then be incorporated as the table partitioning in order to keep performance high and costs low.


## Data Preprocessing and Retrieval

You've got fast access to all the data you need and you are ready to do interesting things with it. Now it is up to the tooling and its user to make the most of it. The tooling includes the choice of query engine, hardware, and its configuration. Let's discuss a few bottlenecks that I observed a couple of times.

### Do You Really Need a Distributed Query Engine?

At first, using distributed computing sounds like a great idea. What's better than using a fast computer? Using two or more fast computers at once! This is the realm of data processing engines like Spark, Trino, or Snowflake. It might also help making your CV even more shiny, because relying on distributed computing proves that you are able to work with huge amounts of data and that's what only the best data scientists do. The downside is that its overuse can really slow you down.

Distributed computing always comes with a certain overhead (because your job and data needs to be *distributed*) and (sometimes) complexity (even if just in the form of more dependencies to additional services), making it sometimes not worth it. Nowadays the latter appears to be much more frequently the case as the the definition of "Big Data" is shifting. With the advent of widely available, strong consumer CPUs (e.g. Apples M series), [large-RAM cloud machines](https://yourdatafitsinram.net) and multi-threaded query engines like Polars or DuckDB it is now possible to process hundreds of gigabytes or even terrabytes (* currently, [AWS supports up to 24 TiB of memory on a single machine](https://aws.amazon.com/ec2/instance-types/high-memory/#:~:text=AWS%20offers%20a%20wide%20range,memory%20databases%20like%20SAP%20HANA.)) of data on a single machine (locally or in the cloud) - of course, this statements depends somewhat on how your data looks like and the queries you might want to execute. The result is better feedback loops and (sometimes) less complicated tooling as you would normally get with distributed query engines. Looking back, I have rarely seen use cases that really required a distributed engine if other best practices like columnar storage, table partitioning or the smallest-fitting data types have been in place. 

In case of unavoidable, very large amounts of data (giving a number here is surprisingly difficult and would be outdated fast), I recommend to first crunch the data into managable pieces with a distributed engine like Spark (e.g. by aggregation) and then proceed with Polars or DuckDB to iterate fast and cheaply. But especially in large organizations, it is also not unreasonable to have a distributed engine like Snowflake, Big Query or Athena (Trino) in place to at least support fast ad-hoc queries: This might save costs for data transfer (which might be huge) from the cloud to local machines, does not have any hardware requirements for the user, and might be cheap on small data anyway due to the billing model (e.g. based on the amount of processed data, 1TB/5$ for BigQuery). Life is a trade-off.

### Sampling

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


### Interplay Between Data Loading and GPU

Today it is not too uncommon to have at least some data processing (and that includes training a neural network) that requires a GPU for finishing in a reasonable amount of time. Getting one is not too difficult due to the abundance of cloud providers like AWS, GCP, Paperspace, and so on. It is then up to the user to squeeze as much compute as possible out of the GPU, which can be non-trivial.

One common pattern is that the GPU sporadically jumps to 100% (e.g. according to nvidia-smi) and then goes back to 0% for a while until the cycle repeats. This hints to a bottleneck regarding the interplay between data loading and GPU processing of that data. In case of libraries like PyTorch or Tensorflow, a first step would be therefore to check the configuration of the dataloader. A dataloader is responsible for loading the data in the background and providing it to the GPU. Common parameters include the number of worker processes, batch size, pre-fetch factor etc. Its values depend on the available compute resources (e.g. number of CPU cores, CPU and GPU memory) and the nature of the problem to solve (some ML tasks like contrastive learning require large batch sizes to perform well). Furthermore, disk utilization can also be a bottleneck (if reading the data is much slower than processing it) and can be e.g. mitigated by replacing a regular HDD with an SSD. Common CLI tools for bottleneck investigation are nvidia-smi, htop, [some disk utilization tool TODO, also point to pytorch profiling].



## Code Efficiency

Your development infrastructure and tooling is configured to make the most of the available hardware for your use case, let it be a machine with 192 vCPUs, an NVIDIA H100 GPU, both at once, or even a whole cluster. Now it's time to retrieve the data and write some code to do some cool things with it.

This section is all about what comes after your data preprocessing and retrieval. It's about the phase where a query engine reaches its limits and one starts to write code outside of it. What bottlenecks are to be encountered here?

**TODO: Profiling Tools**

### Loops Instead of Vectorization

Python has the reputation of being slow and often rightly so. One of the reasons why this language is still so popular for a compute-heavy field like machine learning is that most libraries are actually written in a faster language like C, C++, or Rust. Moreover, libraries like NumPy or Polars make heavy use of so-called *vectorized computing*, where operations are applied on entire arrays or vectors at once, without explicit loops in the code. These operations are often mapped to highly efficient CPU instructions, allowing for parallel processing and using modern hardware features like [SIMD (Single Instruction, Multiple Data)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data). If you have ever asked yourself why [Polars can be so fast](https://pola.rs/posts/i-wrote-one-of-the-fastest-dataframe-libraries/), vectorized computing with SIMD is one of the reasons.

In contrast, classic For-loops in Python are inherently slower, because they execute at the interpreter level, where each iteration has overhead associated with Python’s dynamic typing, function calls, and memory management. The good message is that in data-heavy code, it is often possible to re-write a For-loop into a vectorized computation, e.g. using NumPy, usually achieving a massive speed up depending on your hardware. Finally, to give you a somewhat less-known example, iOS also [supports vector-processing in Swift and Objective C](https://developer.apple.com/accelerate/), which made a tremendous performance difference for an on-device machine learning app I recently developed. Amazing! Keep in mind though, that it often makes the code a bit harder to read.


### Mixed Precision Training

If your GPU utilization is almost always at 100%, it is now time to consider speeding up the actual computation that happens there. Note that most operations on the GPU involve dealing with floating point variables (activations, gradients, and so on), each having a certain precision -- most commonly float32 (or full precision). The idea of *mixed precision training* is to optimize computational efficiency by utilizing lower-precision numerical formats for a subset of the variables. A popular choice here is float16 (half precision) which usually both improves training speed and reduces the overall memory usage, i.e. also making it possible to use larger batch sizes. Note that no task-specific accuracy is lost compared to full precision training, as the GPU automatically identifies the steps that still require full precision [^1][^2]. Thus, it is almost always a good idea to enable mixed precision training if your hardware and use case supports this.

Mixed precision training was introduced with the [NVIDIA Pascal architecture (2016)](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/). For example, popular GPU-accelerated instance types like [G4 / NVIDIA T4 (Turing, 2018)](https://aws.amazon.com/ec2/instance-types/g4/), [G5 / NVIDIA A10G (Ampere, 2020)](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [NVIDIA L4 instances on GCP (Lovelace, 2023)](https://cloud.google.com/blog/products/compute/introducing-g2-vms-with-nvidia-l4-gpus) all support mixed precision training (while the outdated [NVIDIA K80 (Tesla, 2014)](https://www.nvidia.com/en-gb/data-center/tesla-k80/) does not).

So what speed improvement can we expect here? NVIDIA mentions an up to 3x speedup for training [^1]. From practice, I can mostly confirm this statement: Enabling mixed precision training usually at least halved the training time on typical classification tasks based on [Distilbert](https://huggingface.co/docs/transformers/en/model_doc/distilbert) fine-tuning. In addition, I was often able to double the batch size without running into out-of-memory issues. An excellent resource for mixed precision training is also the official [NVIDIA FAQ](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#faq).

### Testing and Debugging Practices

In contrast to hardware-specific aspects like SIMD, a typical bottleneck might also lie in not using good practices for software development in general. I observed this especially for data scientists who used to only write code in Jupyter notebooks and then were expected to build production services. People often just moved their code to workflow engines like [Apache Airflow](https://airflow.apache.org) or [Metaflow](https://metaflow.org), resulting in a single, large file which both contains the workflow definition and the code for model training. In order to test that a change to the code didn't break anything, the whole workflow has to be run, often implying a painfully slow - and sometimes expensive - feedback loop until things work out as expected. Sometimes this is somewhat mitigated by  running the workflow on a data sample instead of the whole dataset, but it will still often feel slow to work on these things - because a workflow usually does a lot of things: data loading, validation, preprocessing, and so on.

To speed things up here, I encourage people to always split their workflow into more manageable modules and write [automated (unit-) tests](https://docs.pytest.org/en/stable/)[^3]. This not only makes later changes much easier (due to automatically testing a set of assumptions that have been made during development), but also enables a much more focused development / debugging flow where only the parts of the codebase affected by a change have to be executed at a time. Hence, in contrast to what the name at first suggests, automatic testing is not only about ensuring correctness of your code, but also about improving feedback loops.

Key here is also that most of the tests do not depend on other, complex, or even external components like databases, as this would again slow down the execution of such tests. I recommend to understand the idea behind the [Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html) to achieve a good trade-off between testing accuracy and speed.


- Choosing the wrong algorithm and/or algorithm implementation
- No parallelization
- data validation
- optimize resources later

[^1]: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
[^2]: Nowadays, there also exist more data type options like BF16 or TF32 from the NVIDIA Ampere architecture onwards, see https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
[^3]: If reasonable, [Test Driven Development](https://martinfowler.com/bliki/TestDrivenDevelopment.html) might also be a good idea.
[^4]: Of course, OpenAI embeddings are still much more powerful for many natural language understanding tasks. It's often a good idea though to at least consider more simple text representations as a baseline (e.g. TF-IDF or FastText with SIF). Maybe you will learn that this already performs well enough for your use case and you save a ton of OpenAI API costs.
