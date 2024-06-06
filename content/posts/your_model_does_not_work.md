+++
title = 'Why Your ML Model Does not Work'
date = 2024-05-10
draft = true
+++


In machine learning it is surprisingly easy to build something that looks like iit is working but acutally isn't. That is problematic on many levels. First, you might be rolling out broken software with all the consequences: A bad user experience or even causing serious harm to someone. Delegating the business into a wrong direction, burning millions of dollars (being a data-driven company is risky if your data is wrong). One the more subtle end, we see promotions to people who produced great, but misleading numbers (due to ignoring best practices, hurray). 

We see such issues often in products where it is not obvious that they are working (e.g. recommender systems). Of course, there are AB tests, but here we can also have subtle bugs.

Here is a list of possible reasons why your great-looking model acutally does not work:

- target leakage in your features
- duplicates, leaking to the test set
- bad train-test splitting
- imbalanced classes
