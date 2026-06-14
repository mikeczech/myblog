+++
date = 2026-05-23
draft = false
tags = ["data-science", "data quality"]
+++

Recently, I've been working more with [dbt](https://www.getdbt.com/) again and came across a useful way to handle questionable rows: [configure a test to warn and store its failures](https://docs.getdbt.com/reference/resource-configs/store_failures?version=2.0&name=Fusion).

```sql
{{ config(
    severity = 'warn',
    store_failures = true
) }}

select *
from {{ ref('some_model') }}
where ...
```

With [dbt test](https://docs.getdbt.com/docs/build/data-tests?version=2.0&name=Fusion), this writes the failing rows to a table instead of stopping the whole pipeline. That makes it a handy way to flag invalid or suspicious records, keep them available for investigation, and optionally exclude them from downstream models.
