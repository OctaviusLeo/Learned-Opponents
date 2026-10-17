"""The small array API that the pure rule modules use, implemented with torch.

``obs_action_spec``, ``obstacles``, and ``rules`` are written against an
``xp`` module (numpy in CI); passing ``torch_xp`` runs the identical code on
GPU tensors inside the environment.
"""

from __future__ import annotations

import torch


def asarray(x, dtype=None):
    if dtype is bool:
        dtype = torch.bool
    return torch.as_tensor(x, dtype=dtype)


def where(cond, a, b):
    return torch.where(cond, a, b)


def maximum(a, b):
    return torch.clamp_min(a, b) if not torch.is_tensor(b) else torch.maximum(a, b)


def minimum(a, b):
    return torch.clamp_max(a, b) if not torch.is_tensor(b) else torch.minimum(a, b)


def min(x, axis=None):  # noqa: A001 - mirrors numpy's name
    return torch.amin(x, dim=axis)


def stack(xs, axis=0):
    return torch.stack(list(xs), dim=axis)


def concatenate(xs, axis=0):
    return torch.cat(list(xs), dim=axis)


isfinite = torch.isfinite
clip = torch.clip
cos = torch.cos
sin = torch.sin
sqrt = torch.sqrt
abs = torch.abs  # noqa: A001
