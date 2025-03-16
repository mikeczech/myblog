+++
title = 'Finding the Maximal Rectangle in Augmented Reality'
date = 2025-03-16
draft = false
tags = ["algorithms", "swift", "augmented-reality", "llm"]
+++

A few weeks ago, I had the task of determining a reasonable location for placing virtual objects on detected walls in [Apple's ARKit](https://developer.apple.com/augmented-reality/arkit/). This was challenging because real objects like doors, shelves, and so on can interfere with the detected wall. Moreover, it was important to detect the available space for virtual objects (e.g. 100 x 100 cm).

As a first step, I used [raycasting](https://developer.apple.com/documentation/arkit/arraycastquery) over a coarse grid to determine any interference points on the detected wall. This basically gave me a two-dimensional matrix of 0s (free space) and 1s (occupied space). Then I realized that I was actually interested in finding the maximal rectangle from this matrix. Interestingly, [I recalled that I had once dealt with this problem on LeetCode](https://leetcode.com/problems/maximal-rectangle/)! But instead of using my old, ugly code, I had ChatGPT generate a quick solution, and it [worked on the first attempt](https://gist.github.com/mikeczech/8255057dec8e82619d1354cce21fbe05) [^1].

Below on the left, you can see the augmented reality view of this data. On the right, you can see the matrix including the maximal rectangle that was derived from this view.

{{< img-side "/raycast.png" "/raycast_max_rectangle.png" "Raycast" "Max Rectangle" >}}

At this point, I was still experimenting with the algorithm in Python and [Jupyter Lab](https://jupyter.org/), which was especially useful for visualization. However, I eventually needed to implement this algorithm in Swift to deploy it on iOS. Here again, ChatGPT proved to be quite useful for [transpiling the code from Python to Swift](https://gist.github.com/mikeczech/4d43aa4497700631b0d7c52969859898). It's fascinating that LLMs enable us to switch so easily between programming languages. What a time to be alive!


[^1]: This is actually not surprising as LLMs usually work best on well-known problems where a ton of data is available on the public internet.
