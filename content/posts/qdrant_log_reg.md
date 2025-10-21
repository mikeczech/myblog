+++
title = 'Optimizing for Precision and Recall in Qdrant'
date = 2025-09-06
draft = false
tags = ["python", "sql", "data-science", "vector-db"]
[params]
  math = true
+++

When building a recommendation system, vector search is often used to generate a set of promising candidates. For example, one might retrieve the top-M videos according to past click-behavior of a user, using a shared user-video embedding space. Here, one usually optimizes for *recall* - i.e. we want to have as many relevant videos among the top-M as possible. This is the realm where vector databases like Qdrant excel at.

The next step is then to re-rank such candidates in order to improve *precision* - i.e. we optimize the list of candidates such that the most relevant appear on the top. This is often done using addtional data like certain user affinities towards certain topics. This two-step procedure is basically needed due to the sheer amount of recommendable items: We first use a cheap operation (i.e. vector search) to reduce the number of candidates in order to subsequently perform a much more expensive operation (re-ranking) to determine the most promising items to recommend. 

Traditionally, we would implement this with a bunch of different technologies: e.g. Qdrant for candidate generation and then additionally something like XGBoost or Deep Learning-based models to re-rank the candidates. This leads to complexity during model serving: The serving layer must be able to handle the amount of candidates returned by the candidate genration, which can be challenging when you scale the service to thousands of requests per second. In addition, there must be a way to integrate an additional model next to candidate generation, which may be difficult if e.g. the backend is written in Scala and the model is originally written in Python - one usually integrates the model via ab additional microservive, only serving the re-ranking model. While this gives one the most flexibility, it feels a bit too much for simple MVPs where iteration speed is much more important than using the best model possible.

To address this complexity and have something simple to later iterate on, one might move the re-ranking into the Qdrant query itself, doing both candidate generation and re-ranking in one-go. This can be done by e.g. implementing a logistic regression with Qdrant's hybrid queries on top of the vector search.

Recall the definition for the probability function of a logistic regression model:

\[
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
\]

where P(y = 1 | x) models the relevance for candidate x. This can be written via the Qdrant API as follows:


```python
from qdrant_client import models

w = [0.1, 0.5, 0.4] # example weights
b = 0.2 # example bias

# z = w^T x + b
z_linear = models.SumExpression(sum=[
    models.MultExpression(mult=[w[0], "$score"]),
    models.MultExpression(mult=[w[1], "x1"]),
    models.MultExpression(mult=[w[2], "x2"]),
    b,
])

# σ(z) = 1 / (1 + exp(-z))
logistic_prob = models.DivExpression(
    div=models.DivParams(
        left=1.0,
        right=models.SumExpression(sum=[
            1.0,
            models.ExpExpression(exp=models.MultExpression(mult=[-1.0, z_linear]))
        ])
    )
)
```

and then be used as part of a Qdrant formula query:

```python
client = QdrantClient(url="http://localhost:6333")

result = client.query_points(
    collection_name="mycollection",
    prefetch=models.Prefetch(
        query=[0.2, 0.8, 0.6],
        limit=50
    ),
    query=models.FormulaQuery(
        formula=logistic_prob
    )
)
```

That's it! This query now combines both candidate generation (prefetch) with re-ranking. Note that the weights of the logistic regression must still be learned -- e.g. with the implementation from scikit-learn.

