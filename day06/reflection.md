# Day 6 reflection — autograd (2026-07-15)

## The model, from the numeric walkthrough (w=3, b=1, x=2)

Forward stores INGREDIENTS, not derivatives: parameters (leaves, empty
.grad mailboxes) + a tape of stations, each saving only what its slope
formula needs. **Nothing is stored "at neurons" — ever.** backward() is a
transient MESSAGE relay: start v=1.0 at the scalar loss; each station
multiplies the incoming message by its local slope and passes it upstream;
messages are discarded on consumption; the only durable writes are
deposits into LEAF mailboxes (w.grad=28, b.grad=14 — "bump w by ε, loss
rises 28ε"). step() then reads mailboxes and walks downhill (w ← 3 −
0.1·28 = 0.2). Neurons don't exist in the machine: stations are OPS (one
matmul = a whole layer); connections = weight-matrix entries; wiring =
the forward code; the tape = one execution trace per pass.

## VJP, precisely

- The name is literal arithmetic: **V**ector–**J**acobian **P**roduct =
  vᵀJ. J holds this station's local slopes (∂out_i/∂in_j); v carries NO
  derivatives — it's the care-weights from downstream (v_i = ∂L/∂out_i).
  The product IS the chain rule: (vᵀJ)_j = Σ_i ∂L/∂out_i · ∂out_i/∂in_j =
  ∂L/∂in_j. Day-4 frame: multiply along the path, sum over outputs.
- **grad_out naming**: "out" = which variable the derivative is ABOUT
  (your output), not the travel direction. Vocab map: grad_out =
  grad_output = incoming message = v = ∂L/∂output = cotangent (JAX).
- **Scalar-root rule = missing default, not missing capability.** The
  engine only ever computes vᵀJ; non-scalar roots work fine with
  backward(gradient=v). The error says it exactly: "grad can be
  *implicitly created* only for scalar outputs" — for m outputs there's
  no canonical v, so torch won't guess. `y.sum().backward()` ≡
  `y.backward(torch.ones_like(y))` — the sum trick IS choosing v = ones.
  Full Jacobian = m VJP sweeps, never done implicitly
  (torch.autograd.functional.jacobian loops them). Training asks ONE
  question (does the loss go down) → one row → one sweep.
- Engine internals: C++ (ATen kernels + dispatcher; autograd wrappers
  code-generated from tools/autograd/derivatives.yaml; multithreaded C++
  backward engine). Python is the front desk. Define-by-run lineage:
  Chainer / HIPS autograd → PyTorch 2016.

## The incantation, now with receipts

`x.detach().clone().requires_grad_(True)` — the 2×2 (data × history):

| | keeps tape | cuts tape |
|---|---|---|
| shares boxes | any view | detach() |
| new boxes | clone() | detach().clone() ← the corner |

...then requires_grad_(True) plants a fresh leaf+mailbox at the corner.
Demonstrated omissions: no clone → shared boxes (caller's edit bled in:
777); no detach → clone rides the CALLER's tape and my backward filled
base.grad (gradient leakage into a stranger's parameters); no flag →
grad_fn=None, nothing recorded. **Detach cuts the copy's UPSTREAM link**
(input→x side): data still flows forward; gradients dead-end; x itself
keeps all its own links. Default detach() has requires_grad=False (branch
not even taped); detach().requires_grad_(True) = "deliver ∂L/∂x to a
mailbox at the cut but never past it" = grad_of's exact trick.

## Mistakes I made (and what they taught)

1. `requires_grad(True)` — **property to read, underscore-method to
   write** (requires_grad vs requires_grad_). "bool is not callable."
2. Missing `return x.grad` — the club's third-course chapter.
3. **numeric_grad on scalar probes** — I pictured x as a LIST OF CANDIDATE
   scalar points (sapply-world); it's ONE point in ℝⁿ, and the loop is
   over DIRECTIONS from that point, not places. f eats the whole vector;
   the probe is x + ε·e_i — full vector present, one coordinate nudged
   (e_i = one-hot basis vector; in code: clone, bump one entry). My
   version coincides with truth exactly for SEPARABLE f (n independent
   1-D problems glued) — and the checker's functions were all separable:
   **third checker gap**, patched with a coupled (Σt)² test. Central
   difference: probe both sides, O(ε²). Clone-per-probe > perturb/restore
   (float restore (a+ε)−ε ≠ a isn't guaranteed; mutation window on
   borrowed tensor). Training = ONE point in ℝ^200k, one gradient arrow,
   one step — never 200k candidate models.
4. **Called detach, dropped the result** — out-of-place scar, autograd
   edition: bare `b.detach()` severs a copy that instantly dies; the
   product taped the live b (b.grad = a's values told the story:
   ∂(ab)/∂b = a). Fix: use the severed copy IN the expression.
5. MyCube: missing unstash ("name 'x' is not defined" — backward has NO
   access to forward's world except the drawer), then missing DEPOSIT
   ("expected 1, got 0" — empty drawer). The two halves fail
   independently; forward computes fine without saving (test_forward
   passed!), the crime surfaces only at withdrawal.

## ctx, demystified (built it myself in 8 lines)

ctx = the station's drawer: `save_for_backward(*tensors)` = deposit
METHOD (parens; *args collects into a tuple — Python's version of R's
`...`; symmetric star scatters in calls: f(*tup), f(**dct));
`saved_tensors` = withdrawal PROPERTY (no parens; @property). DIY class
with those two members, driven by hand, reproduced autograd's [12, 3]
exactly. Real autograd adds logistics (apply manufactures one ctx per
call, injects it positionally — `ctx` is convention like self/cls, NOT
reserved; proved by renaming it `drawer`) and armor (version counters on
saved tensors). Never call MyCube.forward()/MyCube() directly — apply is
the tape-registering entrance. Drawer freed after backward (why keeping
losses keeps graphs).

## Also learned

- **half_stopped = stop-gradient**, the sg[·] of papers: RL target
  networks, BYOL/DINO teachers, VQ-VAE, GAN D-steps. `.sum()` is just the
  scalar-izer (v = 1 entrance); expected grad exactly b's values since
  ∂Σa·b/∂a = b.

## Carry into day 7

Modules with state: Dropout and BatchNorm from scratch. Door question:
inverted dropout scales survivors by 1/(1−p) at TRAIN time — what
expectation is being preserved, and what would silently break at eval
time if you skipped the scaling?
