# Day 7 — The nn.Module system (~35 min)

Course #1 taught init-vs-forward and Parameters. Today: the two module
features it never needed — **`self.training`** (train/eval behavior) and
**buffers** (state that's saved but not trained) — by building the two
classic modules that use them.

## Concepts

- **`module.train()` / `.eval()`** flip `self.training` on the module and
  every submodule. Your transformer ignored it; Dropout and BatchNorm are
  the reason it exists.
- **Dropout**: in training, zero each element with prob p and scale
  survivors by `1/(1-p)` (so the expected value is unchanged — "inverted
  dropout"); in eval, identity. The scaling is the classic interview trap.
- **Buffers**: `self.register_buffer("name", tensor)` — travels with the
  module (state_dict, .to(device)) but invisible to the optimizer.
  BatchNorm's running mean/var are THE canonical buffers.
- **BatchNorm1d**: train mode normalizes by the BATCH's stats and updates
  running estimates; eval mode normalizes by the RUNNING stats. (You wrote
  the "why transformers don't use this" essay on course #1 day 5 — now
  build the thing itself and touch its sharp edges.)

## Your task

In `day07/exercise.py`: `MyDropout`, `MyBatchNorm1d`.

## Check

```bash
python check.py day07
```

## Hints

<details><summary>Hint — dropout mask</summary>
`(torch.rand_like(x) > self.p)` is a keep-mask; multiply and scale. Use
`self.training` to branch.
</details>

<details><summary>Hint — batchnorm running update</summary>
`running = (1 - momentum) * running + momentum * batch_stat`, done under
no-grad semantics (use `.detach()` on the batch stats or in-place ops on
buffers). Match nn.BatchNorm1d: batch var for NORMALIZING is biased,
but the var stored into running stats is UNBIASED.
</details>
