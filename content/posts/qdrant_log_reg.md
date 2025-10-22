+++
title = 'Lightweight Re-Ranking in Qdrant Using Logistic Regression'
date = 2025-09-06
draft = false
tags = ["python", "data-science", "vector-db"]
[params]
  math = true
+++

I’ve worked a little more with [Qdrant’s hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) and noticed that they are useful beyond [what I described in my last article](/posts/qdrant_geo). When building a recommendation system, we usually start with vector search to retrieve promising candidates — for example, the top-M videos for a user based on their past behavior in a shared embedding space. This stage focuses on [recall](https://en.wikipedia.org/wiki/Precision_and_recall), ensuring that relevant items make it into the shortlist. The next step, re-ranking, then improves [precision](https://en.wikipedia.org/wiki/Precision_and_recall) by surfacing the most relevant items using richer signals, such as user affinities toward certain item types.

This often involves multiple systems — a vector database like Qdrant for candidate generation and a separate model (such as [XGBoost](https://xgboost.readthedocs.io/en/stable/) or a neural network) for re-ranking — leading to additional complexity.

It turns out that you can use Qdrant’s hybrid queries to push *model-based re-ranking* into the vector database itself!  One can implement lightweight models, such as [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), directly inside the query, performing candidate generation and re-ranking through a single API call. **It's useful in early-stage projects where iteration speed and going live quickly are more important than using the most sophisticated model possible.** Let’s take a look at an example.

Recall the definition of the probability function of a logistic regression model:

\[
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
\]

where P(y = 1 | x) models the relevance for candidate x. This can be expressed via the Qdrant API as follows:


```python
from qdrant_client import models

w = [0.1, 0.5, 0.3, 0.1] # example weights
b = 0.2 # example bias
user_features = {...}

# z = w^T x + b
z_linear = models.SumExpression(sum=[
    models.MultExpression(mult=[w[0], "$score"]),
    models.MultExpression(mult=[w[1], "x1"]), # from Qdrant payload
    models.MultExpression(mult=[w[2], "x2"]), # from Qdrant payload
    models.MultExpression(mult=[w[3], user_features[user_id]["item_affinity"]]),
    b,
])
```

Note that we can drop the sigmoid term as it does not influence the ranking. This formula can then be used as part of a Qdrant query:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

result = client.query_points(
    collection_name="mycollection",
    prefetch=models.Prefetch(
        query=[0.2, 0.8, 0.6],
        limit=50
    ),
    query=models.FormulaQuery(
        formula=z_linear
    )
)
```

That’s it! This query now combines both candidate generation (via prefetch) and re-ranking. Note that the weights of the logistic regression model must still be learned — for example, [using an implementation from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). This approach isn’t meant to replace a full-fledged ranking system though, but I think it could be an excellent starting point for rapid experimentation and early-stage deployments.

