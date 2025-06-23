+++
title = 'Local LLMs on the Rise'
date = 2025-06-11
draft = false
tags = ["llm", "duckdb"]
+++

After tinkering with [Qwen3-30b](https://github.com/QwenLM/Qwen3) and [Gemma3-12b](https://blog.google/technology/developers/gemma-3/) via [Ollama](https://ollama.com/) for a week, I have the feeling that we're at a turning point with local LLMs.

Until now I've been quite hesitant to use local LLMs. The quality gap (and risk of encountering hallucinations) has just been too high to justify not simply using the rather cheap APIs from OpenAI, Google, and others -- assuming that other aspects like data privacy don't matter. But I'm starting to change my mind. 

Here's what I've been able to do somewhat reliably:

* keyword generation from complex search queries
* reliable [structured output](https://ollama.com/blog/structured-outputs) and [tool calling](https://ollama.com/blog/tool-support) (unfortunately, not with Gemma 3)
* generating SQL for [DuckDB](https://duckdb.org/)
* extracting specific parts from documents (e.g. the required skills in job postings)
* summarizing a list of retrieved documents

That list alone covers an endless number of potential use cases! I haven't done any objective comparisons, but the subjective quality without extensive prompt engineering looks good. Qwen3-30b excelled at more analytical tasks like code generation, while Gemma 3 seemed to lead to more natural conversations with the chat assistant -- so I ended up using both. The speed on my MacBook M4 Pro (48GB RAM) isn't great, but it's definitely something you can work with.

I'm still not exactly sure what the role of local LLMs will be in the future, but it's nice to throw anything at an LLM without having to think about token usage and API limitations (and circumventing the potentially tedious process of getting an API key in a big company). [Besides, it seems that OpenAI is planning to release their first open model sometime later this year](https://www.youtube.com/watch?v=V979Wd1gmTU)
