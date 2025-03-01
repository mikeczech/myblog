+++
title = 'Optimizing Data Loading and GPU Usage in PyTorch'
date = 2024-03-20
draft = false
+++

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

Furthermore, disk utilization can also be a bottleneck if reading the data is much slower than processing it. This can be mitigated by replacing a regular HDD with an SSD.[^1] I experienced this problem when working with image data.

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

[^1]: Major cloud providers like AWS or GCP make it easy to attach an additional fast SSD to a virtual machine.
