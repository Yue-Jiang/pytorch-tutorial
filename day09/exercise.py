"""Day 9 — Devices & performance."""
import time
import statistics
from collections.abc import Callable

import torch


def best_device() -> torch.device:
    """Return the best available compute device as a torch.device OBJECT
    (not a string).

    The resolution order is: if CUDA is available, return the cuda
    device; otherwise, if Apple's Metal backend is available (checked
    with torch.backends.mps.is_available()), return the mps device;
    otherwise fall back to the cpu device. On your Mac, this function
    should come back with mps — your GPU, finally in play.
    """
    if torch.cuda.is_available():
        return torch.device("gpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def move_tree(obj, device: torch.device):
    """Recursively move every tensor found anywhere inside `obj` onto the
    given device, and return a structure of the same shape.

    The argument may be any of the following, nested arbitrarily deep:
    a Tensor (move it with .to(device)); a dict (return a new dict with
    the same keys and every value processed recursively); a list (return
    a new list with every item processed recursively); or a tuple
    (process every item recursively, and make sure the result is still a
    TUPLE — accidentally turning tuples into lists is the classic bug
    here). Anything else — an int, a string, None, and so on — is not a
    tensor container and should be returned unchanged.

    Why this function exists: checkpoints and data batches are often
    dicts or lists of tensors, and moving such a structure between
    devices requires exactly this kind of small recursive walker.
    """
    if torch.is_tensor(obj):
        obj = obj.to(device)
        return obj
    if isinstance(obj, dict):
        new_obj = dict()
        for k, v in obj.items():
            new_obj[k] = move_tree(v, device)
        return new_obj
    if isinstance(obj, list):
        new_obj = [None] * len(obj)
        for i, v in enumerate(obj):
            new_obj[i] = move_tree(v, device)
        return new_obj
    if isinstance(obj, tuple):
        new_obj = [None] * len(obj)
        for i, v in enumerate(obj):
            new_obj[i] = move_tree(v, device)
        return tuple(new_obj)
    return obj


def time_op(fn: Callable[[], object], iters: int = 20, warmup: int = 3) -> float:
    """Measure how long one call of fn() takes, honestly, in seconds.

    The procedure has three parts. First, call fn() `warmup` times
    WITHOUT timing anything — the first calls of an operation often pay
    one-time costs that should not be counted. Second, time each of
    `iters` further calls INDIVIDUALLY: for each call, record
    time.perf_counter() before and after, and keep the difference.
    (Use time.perf_counter, not time.time — perf_counter is the
    high-resolution monotonic clock that exists for exactly this job.)
    Third, return the MEDIAN of the recorded durations, as a plain
    float. One of the tests deliberately makes a single call much slower
    than the rest and requires your result to shrug it off — the median
    is what makes that possible.
    """
    for i in range(warmup):
        fn()
    times = [None] * iters
    for i in range(iters):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times[i] = end - start
    return statistics.median(times)
