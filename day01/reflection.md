# Day 1 reflection — tensor anatomy (2026-07-15)

## Mistakes I made (and what they taught)

1. **Bare `eye` and `full`** — creation ops live in the `torch` namespace.
   Same species as course #1's bare `d_model` (names live somewhere;
   Python won't search modules I didn't name).
2. **Float-step `arange` for an inclusive range** — `arange(0, 1.1, 0.1)`
   guesses the endpoint AND accumulates rounding error step by step
   (last element ≈ 1.0000001, not 1.0). The tool for "N points, both ends
   exact" is `linspace` — each point computed independently. Rule:
   **arange for integers, linspace for floats.**
3. **Carried arange's mental model into linspace** — its third argument is
   `steps` (the COUNT, an int), not the stride. The two think in opposite
   directions: arange takes stride, derives count (excludes stop);
   linspace takes count, derives stride = (end−start)/(steps−1) (includes
   both ends). The TypeError's signature list said it plainly: read the
   signatures in the error.
4. **`ref.dtype()` / `ref.device()`** — those are attributes, not methods.
   No rule predicts which (numel()/size()/element_size() are methods;
   dtype/device/shape are properties — historical accident); the error
   text ("object is not callable") is the tell.

## New knowledge

- **`x.to(ref)`** — `.to()` accepts a TENSOR and copies its dtype and
  device in one call. The idiom for "make this match that."
- **`linspace`** and the arange/linspace division of labor.
- The `_like` family (zeros_like etc.) = shape+dtype+device copied.
- bytes = `numel() × element_size()` — compute from the tensor's own
  properties, never a hardcoded dtype table.

## Carry into day 2

Storage & strides. Quiz pending: bytes of `x.t()` vs `x.t().contiguous()`
— which allocates? (Also promotion: `arange(5) + 0.5` → dtype?)
