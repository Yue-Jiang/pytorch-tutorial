# Day 10 — Capstone: bug forensics (~40 min)

Every other day gave you blank functions. This one gives you **six working-
looking, broken functions** — each containing exactly one bug from a class
you've personally been bitten by. Your job: run the checker, read each
failure, diagnose, and repair *in place*.

All six parts start **red** (not blank) — that's the format. No hints in
the code; the test failures are your only clues, and by now that's enough.

The six bug classes are old friends (in scrambled order): a numerical
stability omission, an inverted mask, a memory-layout crash, a wrong-axis
reduction that only square inputs would hide, an autograd-rules violation,
and a semantic misuse of a loss API.

## Rules

- Fix each function with the SMALLEST correct change — don't rewrite.
- Diagnose before editing: for each one, say (to yourself, or to Claude
  for the review) *which bug class it is* and *what symptom the test
  showed*. The naming is the skill being tested.

## Check

```bash
python check.py day10
```

When all six are green, bring your six one-line diagnoses to the review —
that list IS the final exam of both courses.
