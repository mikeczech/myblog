+++
title = 'Enhance Training Speed with Mixed Precision Training'
date = 2023-12-20
draft = false
tags = ["machine-learning", "neural-networks", "performance-tuning"]
+++

If your data loading is efficient and GPU utilization is almost always at 100%, it is time to consider speeding up the actual computation that happens there. Note that most operations on the GPU involve dealing with floating point variables (activations, gradients, and so on), each having a certain precision -- most commonly float32 (or full precision). The idea of *mixed precision training* is to optimize computational efficiency by utilizing lower-precision numerical formats for a subset of the variables.

A popular choice here is float16 (half precision) which usually both improves training speed and reduces the overall memory usage, i.e. also making it possible to use larger batch sizes. [^2] Note that no task-specific accuracy is lost compared to full precision training, as the GPU automatically identifies the steps that still require full precision [^1]. Thus, it is almost always a good idea to enable mixed precision training if your hardware and use case supports this.

Mixed precision training was introduced with the [NVIDIA Pascal architecture (2016)](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/). For example, popular GPU-accelerated instance types like [G4 / NVIDIA T4 (Turing, 2018)](https://aws.amazon.com/ec2/instance-types/g4/), [G5 / NVIDIA A10G (Ampere, 2020)](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [NVIDIA L4 instances on GCP (Lovelace, 2023)](https://cloud.google.com/blog/products/compute/introducing-g2-vms-with-nvidia-l4-gpus) all support mixed precision training (while the outdated [NVIDIA K80 (Tesla, 2014)](https://www.nvidia.com/en-gb/data-center/tesla-k80/) does not).

So what speed improvement can we expect here? NVIDIA mentions an up to 3x speedup for training [^1]. From practice, I can mostly confirm this statement: Enabling mixed precision training usually at least halved the training time on typical classification tasks based on [Distilbert](https://huggingface.co/docs/transformers/en/model_doc/distilbert) fine-tuning. In addition, I was often able to double the batch size without running into out-of-memory issues. An excellent resource for mixed precision training is also the official [NVIDIA FAQ](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#faq).

Here are a few resources on how to enable mixed precision training with

- [PyTorch](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)

Hugging Face also provides [a great guide on how to speed up training on a single GPU](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one).

[^1]: For more information on mixed-precision training, see [NVIDIA’s documentation](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html).

[^2]: There are now additional data type options like BF16 or TF32 available from the NVIDIA Ampere architecture onwards. See [this NVIDIA blog post](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/) for more details.
