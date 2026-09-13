---
layout: default
title: Chapter 1
---

# Chapter 1

## Maximum likelihood

Suppose

$$
x_1,\ldots,x_n \sim p_\theta(x).
$$

The maximum likelihood estimator is

$$
\hat{\theta}
=
\arg\max_\theta
\sum_{i=1}^{n}\log p_\theta(x_i).
$$

## Code

```python
import torch

x = torch.randn(32, 128)
y = model(x)
