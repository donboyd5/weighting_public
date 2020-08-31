# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 06:36:35 2020
Playing around with automatic differentiation and numerical differentiation

@author: donbo
"""

# conda install -c conda-forge jax

import jax.numpy as jnp
from jax import grad, jit, vmap


def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)
        return outputs


def logprob_fun(params, inputs, targets):
    preds = predict(params, inputs)
    return jnp.sum((preds - targets)**2)


grad_fun = jit(grad(logprob_fun))  # compiled gradient evaluation function
# fast per-example grads
perex_grads = jit(vmap(grad_fun, in_axes=(None, 0, 0)))


