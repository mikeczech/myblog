+++
title = 'A Lightweight Vector DB with DuckDB and Cloud Run'
date = 2025-03-02
draft = false
tags = ["data-engineering", "cloud", "gcp", "python", "duckdb", "vector-db"]
+++

While working on a project, I recently needed a lightweight, serverless vector database solution. The vectors needed to be stored alongside additional metadata, and the database itself would only be updated daily—i.e., **read-only for most users**. Additionally, the dataset was relatively small.

I initially considered using SQLite with Alex Garcia's [sqlite-vec](https://github.com/asg017/sqlite-vec) extension. However, I realized that my application was primarily generating *ranked product feeds*, where an embedded [OLAP](https://en.wikipedia.org/wiki/Online_analytical_processing#:~:text=In%20computing%2C%20online%20analytical%20processing,online%20transaction%20processing%20(OLTP)) database would likely be a better fit. So, I decided to try [DuckDB's](https://duckdb.org/) [VSS extension](https://duckdb.org/docs/stable/extensions/vss.html).

The application ingests data daily as a [Parquet file](https://parquet.apache.org/) and loads it into a DuckDB database using the following script. This script also installs the VSS extension and creates an [HNSW index](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) for the *embedding column* (a 256-dimensional vector) to enable fast approximate nearest neighbor search using [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

```python
# /// script
# dependencies = [
#   "polars==1.22.0",
#   "pyarrow==19.0.0",
#   "duckdb==1.2.0"
# ]
# ///
import sys
import polars as pl
import duckdb

def main(parquet_file, duckdb_file):
    df = pl.read_parquet(parquet_file)

    con = duckdb.connect(duckdb_file)
    con.execute("DROP TABLE IF EXISTS articles;")
    con.register("df_temp", df.to_arrow())
    con.execute("""
        INSTALL vss;
        LOAD vss;

        SET hnsw_enable_experimental_persistence = true;

        CREATE TABLE articles AS
        SELECT
            * EXCLUDE(embedding),
            CAST(embedding AS FLOAT[256]) AS embedding
        FROM df_temp;

        CREATE INDEX embedding_hnsw_index
        ON articles
        USING HNSW (embedding)
        WITH (metric = 'cosine');
    """)

    con.unregister("df_temp")

    con.close()
    print(f"Created single DuckDB table 'articles' in {duckdb_file}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_parquet_file> <output_duckdb_file>")
        sys.exit(1)

    parquet_file = sys.argv[1]
    duckdb_file  = sys.argv[2]

    main(parquet_file, duckdb_file)
```

To expose this database via a REST API, I created a serverless service using FastAPI, Docker, and Cloud Run. The database is copied into the Docker image—this is feasible because it’s read-only and (still) small. When the data is updated, I simply redeploy the service with the latest dataset.

```docker
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inventory.duckdb .
RUN python -c "\
import duckdb; \
conn = duckdb.connect('data.duckdb'); \
conn.execute('INSTALL vss;'); \
conn.close()"

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The endpoint code for querying the vector database is as follows:

```python
from typing import List, Dict, Any

import duckdb
import uvicorn
from fastapi import FastAPI, Query, Body
from pydantic import BaseModel

app = FastAPI()

DATABASE_PATH = "data.duckdb"

class Request(BaseModel):
    offset: int = 0
    limit: int = 20
    embedding: List[float]

@app.post("/feed")
def get_feed(
    request: Request
) -> List[Dict[str, Any]]:
    offset = request.offset
    limit = request.limit
    embedding = request.embedding

    query = """
    LOAD vss;

    SELECT
        article_id,
        main_name,
        retail_price
    FROM articles
    ORDER BY array_distance(embedding, $emb::FLOAT[256])
    LIMIT $limit OFFSET $offset
    """

    with duckdb.connect(DATABASE_PATH) as conn:
        cursor = conn.execute(query, {"offset": offset, "limit": limit, "emb": embedding})
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    return [dict(zip(columns, row)) for row in rows]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
I can’t yet speak to its real-world performance, but initial results look promising. While scaling to larger datasets might introduce new challenges, the simplicity, cost-efficiency, and ease of deployment make this approach an excellent fit for smaller projects. If performance bottlenecks arise, exploring optimizations like external storage, query tuning, or alternative vector search libraries / databases could be the next steps (e.g. [qdrant](https://qdrant.tech/)).

