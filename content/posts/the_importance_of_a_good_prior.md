+++
title = 'Artificial Intelligence Beyond Text or Vision'
date = 2025-10-28
draft = true
tags = ["machine-learning", "neural-networks"]
+++

I've recently listened to the podcasts of Dwarkesh Patel with Andrej Karpathy, Sergey Levine, and Richard Sutton. The point that resonated with me the most is **the need for a good prior to develop an efficient artificial intelligence**. I believe we will see many AI applications in which building a strong domain-specific prior is essential, extending far beyond text and vision. Contrary to the widely held belief that AI will soon hit a ceiling because we’re running out of training data, I think the real opportunity lies in tapping into the vast reservoirs of company-specific data that never make it into the public domain.

A prior is essentially the basic knowledge a model possesses before it begins learning the specifics. It's one of the key ingredients that led us to efficient applications of reinforcement learning -- with AlphaZero being one of the first breakthroughs. Nowadays, the most popular example is certainly training a next-token predictor like GPT - the prior - and then using reinforcement learning from human feedback (RLHF) to eventually get a useful chat assistant out of it. A third example I like is how such a prior makes a difference in robotics development: If a prior already provides a robot or autonomous vehicle with a good (though imperfect) understanding of how the world works, further exploration and learning turns out to be much simpler.

I’ve been successfully using the same pattern (prior + fine-tuning) in various industry settings for quite some time now -- often under a different name. For example, candidate generation (also named retrieval) in recommender systems serves as the prior, while ranking models or bandit algorithms play the role of fine-tuning. In practice, this usually involves connecting multiple data sources or representations of data, as I described in a previous blog post about linking a language model (the prior) with a GBDT model.

One key insight is that priors need not be limited to text or vision; they can also arise from business-specific modalities. Such modalities include user-product interactions in e-commerce, supply chain events in logistics, or career transitions on platforms like LinkedIn. But how does that work in practice? In case of user-product interactions, models are trained on vast amounts of tracking events. Each of these is a small, structured data point:

\[
(\text{timestamp}, \text{user\_id}, \text{product\_id}, \text{event\_type})
\]

where event_type can be a click, bookmark, purchase, and so on. Such data helps an artificial intelligence to learn answers to questions like

* What products are usually bookmarked or purchased together? (**product-focused**)
* What users share similar interests based on the products they bought in the past? (**user-focused**)
* What products most likely fall in the area of interest for a given user based on past behavior - or vice-versa? (**user-product-focused**)

From a more technical perspective, this was first achieved using matrix factorization-based approaches like collaborative filtering, which is still going strong today. Later, it was also discovered that Word2Vec can create representations of items from interaction sequences. The key idea here is to treat such sequences of IDs as a "sentence" and train a language model on that, resulting in embeddings for items (e.g. product IDs). However, one key disadvantage here is that traditional collaborative filtering and Word2Vec do not support using heterogeneous data - like a mix of item IDs, text or images (think of user profiles or images of products along with their IDs). This is where nowadays neural networks shine. Two-tower architectures, in particular, can ingest multiple modalities and project them into a shared embedding space, producing a far more expressive prior over users, items, and their relationships. See the following figure for an illustration or have a look at this simple toy implementation in Pytorch. The CLIP paper is also worth reading here.

I think it's fascinating that - in theory - this approach allows us to learn embeddings on almost arbitrary relationships we find in the real world. While LLMs / VLMs are great for a ton of real-world tasks, I assume that there are many applications, which quickly reach the limits of languages in general. For example, there might be a countless number of hard-to-describe intricacies that make a user of your online shop purchase one specific item. It's possible to use text here, but I suppose that this puts an artificial ceiling on what's achievable with AI here (though, in many cases, simply using an LLM first might make a great baseline!). In addition, there might be operational benefits of using domain-specific AI models over LLMs: Maybe the use of language (or vision) isn't the most efficient way to approach a problem. A domain-specific AI would then be more lightweight, impliying less operational overhead -- e.g. faster feedback loops and a smaller cloud bill.

On the other hand, it's still requires a lot of research to apply AI to other domains. The reason why AI is so popular today is mainly due to the fact that most technical breakthroughs happened for text and vision so far -- e.g. the advent of the transformer architecture in combination with stronger GPUs and the vast amounts of available text and image data on the public internet. In general, it's not trivial to find useful encodings for arbitrary modalities - for text it's tokens, for vision it's pixels, but what about events for a supply chain in logistics? Furthermore, there's more to what makes someone purchase an item in an online shop than the related product description and image: Availability, similar options beyond the online shop, and so on. Moreover, there might be much more noise in other domains than for text and image data. Given a prefix of a sentence, it's relatively predictable what comes next: "This is a cute little ..." likely results in words like "cat" or "dog", but certainly not in "giant". Given a sequence of purchased products of a user, it's much more difficult to predict the next one.

I'm really looking forward to seeing this immense potential of AI being transferred to other domains as well. It's already happening and already started actually before ChatGPT & Co. became so popular. But there are still challenges ahead. So what's a good way to learn about AI in domains beyond text and vision? I think an excellent resource here is Kaggle as there are (past) competitions from a lot of different domains available: From autonomous driving over e-commerce (as we've seen before) to areas like drug discovery. I usually recommend to try to build one meaningful solution for a competition by yourself to get familiar with basic challenges -- even if the competition is already over. In addition, I really enjoy looking at the most successful solutions of a competiton (1st, 2nd, and 2rd place), which are usually described in the discussions sections shortly after a competition has finished. Here's a small list of competetions I've found helpful in the past:

- TODO
- TODO
- TOOD

Have fun!



---

* Important insight: tracking data contains much more noise than text data
* https://arxiv.org/abs/1907.06902 Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches
* https://youtu.be/IByC2keY3vo?si=syHZa7sx4x7hxoML&t=625
* Does a single embedding suffice?
* A user can be projected to text by looking on what they clicked on
* You don't need more control, you just need better data scientists
* Item2Vec: Neural Item Embedding for Collaborative Filtering
* Kaggle is a great place to learn about different applications of AI
