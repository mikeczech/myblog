+++
title = 'Using PDB in Metaflow'
date = 2025-03-23
draft = false
tags = ["python", "mlops"]
+++

For some projects, I use Netflix's [Metaflow](https://github.com/Netflix/metaflow) to build machine learning pipelines. I generally enjoy using it because it allows me to run pipelines both locally and remotely on [AWS Batch](https://aws.amazon.com/batch/), depending on the resource requirements of a pipeline. However, what I struggled with was using [PDB](https://docs.python.org/3/library/pdb.html) within pipeline steps. Since Metaflow starts multiple processes in the background, [breakpoints cause pipelines to simply get stuck](https://github.com/Netflix/metaflow/issues/89).

As a solution, you can use [web-pdb](https://github.com/romanvm/python-web-pdb), which works excellently with Metaflow:

```bash
pip install web-pdb
```

To use it in Metaflow steps:

```python
from metaflow import FlowSpec, step

class Flow(FlowSpec):
    @step
    def start(self):
        self.next(self.a)
        
    @step
    def a(self):
        import web_pdb; web_pdb.set_trace()  # set a breakpoint here
        self.next(self.end)
        
    @step
    def end(self):
        print('success')
        
if __name__ == '__main__':
    Flow()
```

This creates a web interface at http://\<your Python machine hostname or IP\>:5555, which provides you with the standard pdb interface.
