"""Flax finite-budget composite critic."""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class Critic(nn.Module):
    hidden: tuple = (512, 512, 512)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        logit = nn.Dense(1)(x)
        return jnp.squeeze(logit, axis=-1)
