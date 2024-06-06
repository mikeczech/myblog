+++
title = 'Modern Ways for Getting Ground Truth Data'
date = 2024-05-10
draft = true
+++

- LLMs are great, but slow and expensive on scale. It's also challenging to estimate the expected performance.
- Often, it is sufficient to have a very small, specific model instead, but that requires ground truth data.
- Traditionally, this involves manual labeling - usually a tedious process.
- Manual labeling limits the agility of a team, because making changes sometimes requires starting from scratch again.
- Are there better strategies for getting ground truth data?

## Strategies

1) Maybe a ground truth is already somewhere hidden in your data (e.g. work experiences of profiles on a jobs platform). If there is not a ground truth, there might still be a good proxy for it to simplify things.
2) Use LLMs for ground truth generation
    - Challenge: How good is the generated ground truth?
    - Use logprobs to separate high-confidence from low-confidence labels. Then manually check the latter. This at least mitigates the labeling effort. But requires that the logprobs are actually meaningful.
    - Employ existing ground truth data for prompt evaluation. Manually label a small dataset, then build a prompt that reliably re-produces the labels. If there is ground truth data for a different task, this might also be useful for checking the prompt.
    - Sample multipe times from the LLM (using a corresponding temperature) as another way for estimating confidence in labels. This might also be useful for a multi-label setting where mutliple labels might apply for a datapoint.
    - Let it quote and then check the quote against the real text to counteract hallucinations.
