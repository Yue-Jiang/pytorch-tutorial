"""Day 9 — Devices & performance."""
import time
from collections.abc import Callable

import torch


def best_device() -> torch.device:
    """cuda if available, else mps if available, else cpu.
    Return a torch.device object."""
    raise NotImplementedError


def move_tree(obj, device: torch.device):
    """Recursively move every tensor inside obj to device.

    obj may be: a Tensor, a dict (values moved), a list, a tuple
    (type preserved!), or any nesting of those; non-tensor leaves
    (ints, strings, None...) pass through unchanged.
    """
    raise NotImplementedError


def time_op(fn: Callable[[], object], iters: int = 20, warmup: int = 3) -> float:
    """Median wall-clock seconds per call of fn().

    Call fn `warmup` times untimed first, then time each of `iters` calls
    individually (time.perf_counter) and return the MEDIAN duration.
    """
    raise NotImplementedError
