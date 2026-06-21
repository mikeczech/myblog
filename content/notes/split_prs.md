+++
date = 2026-06-21
draft = false
tags = ["ai", "engineering"]
+++

I’ve noticed that with AI-assisted coding, it’s becoming even more common to end up with large PRs. This is problematic for several reasons:

* bugs are harder to spot
* merge conflicts increase
* reviews take longer 

The interesting thing is that AI coding agents are also really good at splitting large PRs into smaller, more manageable pieces! I’ll often ask an agent to identify independent changes, separate refactors from functional changes, and suggest a sequence of smaller PRs.

To me, this shows that **AI-assisted coding isn’t just about producing more code in less time. It can also help reinforce good engineering practices**, which are necessary if you want to scale development over time.
