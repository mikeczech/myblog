+++
title = 'BigQuery SQL Query Testing with DuckDB and SQLGlot'
date = 2023-01-12T22:32:58+01:00
draft = false
tags = ["testing", "duckdb", "sql"]
+++

SQL plays a pivotal role in data management, yet its testing phase often encounters challenges, particularly due to dependencies on large-scale systems like BigQuery or Hadoop. This dependency not only complicates the process but also slows it down.

Previously, we developed [BQuest, a simple Python library that streamlined testing for BigQuery queries](https://github.com/ottogroup/bquest). It worked by converting data from Pandas dataframes into BigQuery tables, running the queries, and then converting the results back into dataframes. This process was beneficial but had its drawbacks, including slow feedback times and heavy reliance on BigQuery.

Now, we're shifting towards a more efficient approach with SQLGlot and DuckDB. SQLGlot allows us to adapt SQL queries written for various platforms (like BigQuery) to a format compatible with DuckDB. DuckDB, an SQL database engine, then executes these adapted queries. This method is not only faster but also reduces dependency on external services like BigQuery.

## Example of Using SQLGlot and DuckDB

Here's a simplified example of how to use SQLGlot and DuckDB for testing SQL queries:

```python
import pytest
import sqlglot
import duckdb
import pandas as pd

def test_sql_query():
    # given
    data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
    df = pd.DataFrame(data)

    bigquery_sql = "SELECT col1 FROM test_table WHERE col2 > 4;"
    duckdb_sql = sqlglot.transpile(bigquery_sql, read='bigquery', write='duckdb')[0]

    # Setup DuckDB connection and create a table
    conn = duckdb.connect()
    conn.execute("CREATE TABLE test_table AS SELECT * FROM df;")

    # when
    result = conn.execute(duckdb_sql).fetchdf()

    # then
    expected = pd.DataFrame({'col1': [2, 3]})
    pd.testing.assert_frame_equal(result, expected)
```

In this example, we create a Pandas dataframe, write an SQL query for BigQuery, transpile it to DuckDB's SQL format using SQLGlot, and then execute it with DuckDB.

But it's important to also note some limitations. SQLGlot might not support certain specialized functions, such as User-Defined Functions (UDFs), in some SQL dialects. In such cases, a workaround (e.g. mockups) might be necessary, which might not be ideal but generally works.

## Conclusion

The integration of SQLGlot and DuckDB offers a more streamlined and efficient method for SQL query testing, significantly reducing reliance on large-scale systems and speeding up the testing process. Though it may not be a perfect solution for every scenario, it represents a significant step forward in simplifying SQL testing processes.
