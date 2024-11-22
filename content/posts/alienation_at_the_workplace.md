+++
title = 'Alienation at the Workplace'
date = 2024-11-11
draft = true
+++

* In big companies there is often a big gap between the actual work and value produced ([if there is any](https://www.amazon.de/Bullshit-Jobs-A-Theory/dp/B079YXRZ7Q/ref=sr_1_2?__mk_de_DE=ÅMÅŽÕÑ&crid=18C9G1DU5GU7H&dib=eyJ2IjoiMSJ9.s6U2Zz9mxFw77-7nJoRGVdfwEjO5cwPy0e5sTivJUbJ0oTA4xPlgDxb-BknpdXi7IoeCYSibTeFqK8gaaXnFmuyLQrV2N7ts1ufsOVnb05DibX3p8W6l1_g16RQ622YfbbDGOp-n5Rdptj7D4YEARXY6e6JmdqBTGYYfGEadW4lpwWFZQBhdGbk_bo8H_rQUjBuQlF95P7FFY3KSuicAi0R5XXtEQ4tjO-kGae6Vx_g.I-GWCcjNFMsZpKAzMKC9QMKJEqsiq3QbPLASEz1hGDA&dib_tag=se&keywords=bullshit+jobs&qid=1725785630&sprefix=bullshit+jobs%2Caps%2C113&sr=8-2)), leading to alienation.
* That's actually an idea from Marx, referring to the industrial revolution in the  1700s and 1800s.
* In the past, the value of work has been much more obvious - there used to be a stronger connection beween the work and the people who benefited from it. For example, if someone is harvesting potatoes all day to feed their family, the produced value is much more immediate and clear.
* Why feedback is not only important to provide more value, but is also about the psychological benefits of experiencing the value that your employees create.
* The alienation in the workforce in larger societies also had interesting cultural implications: The idea of a judging god is pretty new and only occurs is large populations. If there's no immediate feedback, then people apparently require a supernatural being to reward or punish them. In large companies, that being might be the upper management.
* The idea of firing someone in small-scale societies / tribes is ridiculous.
* **Correctly assigned ownership of software artifacts is important**. If the owning person or team is too far away from the value proposition of the artifact (= business impact, in the best case), motivation and quality declines.
* What does "too far away" mean here? That's a non-trivial question. One option could be the **ability and degree to measure the business impact** of what is built.
    - Let's say a team is building a recommendation engine for an online shop. They are able to measure the impact of their work by conducting AB tests and, hence, correctly own the engine. The impact is still a bit distant and abstract in the sense that customers usually won't thank the team for recommending good products, but much better than not able to measure at all.
    - Let's say there is another team that is producing data that is later used as model features for the recommendation engine. They are able to measure the quality of the data they produce, but that's pretty much it. If they are lucky, they are indirectly notified of their feature's impact by the recommender's team latest AB test (which is part of a separate priorization process).

* What's the solution here? One option could be to move the ownership and development of the model features to the recommendation engine team. This is IMHO the cleanest option if the recommendation engine is the only consumer of the features. This might make it necessary to stock up the team's resources though and the danger of creating a team that has too many responsibilities.

* If the features produced by the other team is used in many places (e.g. for a search engine too) things get more complicated. A less drastic approach would be to enable the other team to measure their impact on the recommendation engine autonomously. It's also important that they are always able to deploy their changes without depending on other team's priorizaton processes (e.g. by strict versioning of their services, i.e. they are able to provdide a new version without breaking the consuming services). That's true for data as well.
