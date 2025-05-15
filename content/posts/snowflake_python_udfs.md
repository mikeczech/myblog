+++
title = 'A First Glance at Python UDTFs in Snowflake'
date = 2025-04-11
draft = false
tags = ["python", "sql", "data-science"]
+++

Recently, I've been working more with [dbt](https://www.getdbt.com/) / [Snowflake](https://www.snowflake.com/en/emea/) and needed to utilize a multi-output regression model from SQL. The model was implemented in Python using scikit-learn and [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html). A natural approach for integrating such models with SQL is to [create a user-defined table function (UDTF)](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-tabular-vectorized#create-a-udtf-with-a-vectorized-end-partition-method). [^1] I specifically chose a table function because both the model's input and output consisted of multiple values. Here are the lessons I've learned so far:

### Performance is Good Enough, Memory is Limited

Thanks to [support for vectorization](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-tabular-vectorized#create-a-udtf-with-a-vectorized-end-partition-method), performance has been surprisingly good for a non-trivial regression model—a pipeline with multiple complex pre-processing steps and a final LightGBM layer. For millions of datapoints, predictions via the UDTF are generated in approximately 5 minutes:


```sql
CREATE OR REPLACE FUNCTION regression_model_udtf(feat1 FLOAT, feat2 FLOAT)
RETURNS TABLE(prediction_1 FLOAT, prediction_2 FLOAT, prediction_3 FLOAT)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.12'
HANDLER = 'Handler'
PACKAGES = (...)
AS $$
import pandas as pd
...

model = load_model()

class Handler:
    @vectorized(input=pd.DataFrame)
    def process(self, df):
        predictions = model.predict(df)

        return pd.DataFrame({
            'prediction_1': predictions[:, 0],
            'prediction_2': predictions[:, 1], 
            'prediction_3': predictions[:, 2]
        })
$$;
```

On the other hand, memory is quite limited when using Snowflake's default warehouses. Initially, I encountered out-of-memory issues, but managed to overcome these with a simple batching mechanism.

```sql
WITH batched_data AS (
    SELECT 
        FLOOR((ROW_NUMBER() OVER (ORDER BY id)) / 10000) AS batch_id,
        *
    FROM input_table
)
SELECT
    *
FROM batched_data b,
     TABLE(regression_model_udtf(feat1, feat2) PARTITION BY batch_id)
```

### Dependency Management Works, but Has Its Shortcomings

By default, Snowflake UDTFs allow you to install additional Python packages from Anaconda. An important (and somewhat hidden) detail is that these packages must come from a [dedicated Snowflake channel](https://repo.anaconda.com/pkgs/snowflake). While this is generally sufficient for most use cases, I encountered several outdated packages during implementation.

```sql
CREATE OR REPLACE FUNCTION regression_model_udtf(features ARRAY)
...
PACKAGES = (
    'scikit-learn==1.1.3',
    'lightgbm==3.3.2', 
    'pandas==1.4.4', 
    'numpy==1.23.3',
    'joblib==1.1.0'
)
AS $$
...
```

Furthermore, it appears impossible to freeze a Python environment completely—you can only pin high-level dependencies. This makes production usage of Snowflake UD(T)Fs somewhat questionable, as transitive dependencies might change in the background after redeployment, potentially breaking the UDTF or, worse, causing unpredictable behavior.

What would be preferable is the ability to use an existing pyproject.toml with an accompanying lock-file for a UDTF. This approach would align more closely with standard Python ecosystem practices and would eliminate redundancy when specifying dependencies (since the Python code typically already comes with a pyproject.toml or at least a requirements.txt).

### Integrating Custom Python Packages is Easy

In addition to using standard Python packages, our model depended on custom code that needed to be available within the UDTF (such as custom transformers for data preprocessing). Adding a custom Python package to the UDTF proved remarkably straightforward: you simply upload the zipped package to a [Snowflake Stage](https://docs.snowflake.com/en/sql-reference/sql/create-stage) and reference it. The only limitation is that such packages cannot contain native code (e.g., Rust or C/C++).

```sql
CREATE OR REPLACE STAGE my_packages;
PUT file://path/to/my_custom_package.zip @my_packages;

CREATE OR REPLACE FUNCTION regression_model_udtf(features ARRAY)
...
IMPORTS = ('@my_packages/my_custom_package.zip')
...
AS $$
...
```

### There Is a Way to Get Around Most Limitations

Finally, it's worth noting that most of the previously mentioned shortcomings relate to my use of default warehouses for UDTF execution. The trade-off here is between ease-of-use and the technical limitations this approach entails.

If you have the resources to self-manage some infrastructure, [Snowpark Container Services](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview) might provide more flexibility in terms of performance (e.g., using GPUs) and how the compute environment is configured (it supports custom Docker images). In my case, however, I can tolerate the limitations for now and have therefore opted for the simpler alternative.

In summary, I'm satisfied with Snowflake UDTFs for less complex models, though I expect to learn more about their advantages and disadvantages in the near future. 

[^1]: An alternative here is to use the model registry of Snowflake. Though I haven't had a look at it in detail, a UDTF seemed to be the simpler alternative as the model was already versioned using MLFlow and required some Snowflake-specific business logic on top.
