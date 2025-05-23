+++
title = 'Combining Vector Search with Business Logic in Qdrant'
date = 2025-05-17
draft = false
tags = ["python", "sql", "data-science", "vector-db"]
+++

I've been using [Qdrant](http://qdrant.tech/) for vector similarity search in a [recommendation system](https://www.nvidia.com/en-us/glossary/recommendation-system/), and one challenge that keeps coming up is how to incorporate additional information beyond just the vectors themselves.

Let's say you're building a social network that helps people connect based on shared interests. Your user vectors capture relationships like "User A loves dogs" or "User B is into cats" — learned through collaborative filtering, contrastive learning, or similar techniques. Of course, in real applications, these vectors often represent much more abstract patterns. On the other side, users might also want to meet people who live nearby for occasional coffee dates or hangouts. If your vectors don't encode location data (like which city someone lives in), you can't rely on simple similarity search alone to find the best matches. You need to blend vector-based similarity with domain-specific business logic.

The [latest version of Qdrant introduces a Score-Boosting Reranker](https://qdrant.tech/blog/qdrant-1.14.x/) that makes this whole process much cleaner. The idea is to take your standard semantic or distance-based ranking and apply a rescoring step on top. Going back to our social network example, you'd first run a vector similarity search on user interest vectors, then adjust those scores to factor in how close people live to each other.

Let's look at some code. First, upload your points along with their geographical coordinates (latitude and longitude):


```python
from qdrant_client import QdrantClient
from qdrant_client import models

client = QdrantClient(url="http://localhost:6333")

client.upsert(
    collection_name="mycollection",
    points=models.Batch(
        ids=[42],
        payloads=[
            {"location": models.GeoPoint(lat=53.5413, lon=9.9845)},
        ],
        vectors=[
            [0.9, 0.1, 0.1],
        ],
    ),
)
```

Then you can query using both vector similarity and geographical proximity:

```python
result = client.query_points(
    collection_name="mycollection",
    prefetch=models.Prefetch(
        query=[0.2, 0.8, 0.6],
        limit=50
    ),
    query=models.FormulaQuery(
        formula=models.SumExpression(sum=[
            "$score",
            models.GaussDecayExpression(
                gauss_decay=models.DecayParamsExpression(
                    x=models.GeoDistance(
                        geo_distance=models.GeoDistanceParams(
                            origin=models.GeoPoint(
                                lat=52.504043,
                                lon=13.393236
                            ),  # Berlin
                            to="location"
                        )
                    ),
                    scale=5000  # 5km
                )
            )
        ]),
        defaults={"location": models.GeoPoint(lat=48.137154, lon=11.576124)}  # Munich
    )
)
```

In real-world scenarios, you'll want to fine-tune the parameters ([target, scale, and midpoint](https://qdrant.tech/documentation/concepts/hybrid-queries/#decay-functions)) based on your specific use case. It's also worth considering adding weights using a [MultExpression](https://qdrant.tech/documentation/concepts/hybrid-queries/#score-boosting) to balance the importance of vector similarity against geographical proximity — after all, living nearby might not always be the deciding factor for every recommendation.
