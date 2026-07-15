# Day 6 — Autograd (~35 min)

Course #1 used the tape; today you open it. Leaf tensors, `grad_fn`,
`detach`, numeric gradient checking, and one custom `autograd.Function`
with a hand-derived backward.

## Concepts

- **Leaves** are tensors you created (params, inputs with
  `requires_grad_()`); everything computed from them carries a `grad_fn`.
  `backward()` fills `.grad` on leaves only.
- **`detach()`** returns a view that the tape treats as a constant —
  gradients stop there. (`.data` looks similar and is the dangerous
  legacy spelling — it skips the version-counter safety.)
- **Finite differences**: `df/dx ≈ (f(x+ε) − f(x−ε)) / 2ε` — the universal
  autograd lie-detector (`torch.autograd.gradcheck` industrializes it).
- **Custom Function**: subclass `torch.autograd.Function`, write
  `forward(ctx, x)` and `backward(ctx, grad_out)`, save what backward
  needs with `ctx.save_for_backward`.

## Your task

In `day06/exercise.py`: `grad_of`, `numeric_grad`, `half_stopped`,
and the `MyCube` custom Function.

## Check

```bash
python check.py day06
```

## Hints

<details><summary>Hint — grad_of</summary>
Clone-and-detach the input, `requires_grad_(True)` it, run f, backward,
return `.grad`. Detach first so the caller's tensor stays untouched.
</details>

<details><summary>Hint — MyCube backward</summary>
d(x³)/dx = 3x². Save x in forward (`ctx.save_for_backward(x)`), and
backward returns `grad_out * 3 * x**2` — chain rule: incoming gradient
times local derivative.
</details>
