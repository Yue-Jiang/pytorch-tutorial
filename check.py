#!/usr/bin/env python
"""Friendly checker — tests one part at a time, in the intended order.

Usage (from this folder, with the transformer-tutorial env active):

    python check.py day01               # status of every part of day 1
    python check.py 1                   # same thing
    python check.py day01 dtypes        # verbose details for one part
    python check.py day01 --all         # don't stop at the first failure
    python check.py reset day01         # restore the blank skeleton (asks first)

Plain `pytest day01 -v` still works if you prefer raw output.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# (label, test class) in the order you should implement them
COMPONENTS = {
    "day01": [
        ("creation", "TestCreation"),
        ("dtypes", "TestDtypes"),
        ("bytes", "TestBytes"),
    ],
    "day02": [
        ("view_detective", "TestViewDetective"),
        ("stride_prediction", "TestStridePrediction"),
        ("expand_vs_repeat", "TestExpandVsRepeat"),
        ("windows", "TestWindows"),
    ],
    "day03": [
        ("gather", "TestGather"),
        ("row_picks", "TestRowPicks"),
        ("topk_mask", "TestTopkMask"),
        ("scatter_histogram", "TestScatterHistogram"),
    ],
    "day04": [
        ("scores", "TestScores"),
        ("outer", "TestOuter"),
        ("trace", "TestTrace"),
        ("weighted_sum", "TestWeightedSum"),
        ("bilinear", "TestBilinear"),
    ],
    "day05": [
        ("logsumexp", "TestLogsumexp"),
        ("masked_mean", "TestMaskedMean"),
        ("standardize", "TestStandardize"),
        ("pairwise", "TestPairwise"),
    ],
    "day06": [
        ("grad_plumbing", "TestGradPlumbing"),
        ("numeric_grad", "TestNumericGrad"),
        ("detach_semantics", "TestDetachSemantics"),
        ("custom_function", "TestCustomFunction"),
    ],
    "day07": [
        ("dropout", "TestDropout"),
        ("batchnorm", "TestBatchNorm"),
    ],
    "day08": [
        ("collate", "TestCollate"),
        ("sgd_momentum", "TestSGDMomentum"),
        ("clip", "TestClip"),
    ],
    "day09": [
        ("device", "TestDevice"),
        ("tree_move", "TestTreeMove"),
        ("timing", "TestTiming"),
    ],
    "day10": [
        ("softmax_bug", "TestSoftmaxBug"),
        ("mask_bug", "TestMaskBug"),
        ("view_bug", "TestViewBug"),
        ("mean_bug", "TestMeanBug"),
        ("update_bug", "TestUpdateBug"),
        ("loss_bug", "TestLossBug"),
    ],
}

GREEN, RED, YELLOW, DIM, BOLD, END = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m",
)


def normalize_day(arg: str) -> str:
    day = arg.lower().removeprefix("day").lstrip("0") or "0"
    return f"day{int(day):02d}"


def run_component(day: str, cls: str, verbose: bool) -> tuple[str, str]:
    """Returns (status, output) where status is 'pass'|'todo'|'fail'."""
    cmd = [
        sys.executable, "-m", "pytest",
        f"{day}/test_{day}.py::{cls}",
        "-v" if verbose else "-q",
        "--tb=short", "--no-header", "-x",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        return "pass", out
    if "NotImplementedError" in out:
        return "todo", out
    return "fail", out


def reset_day(day: str) -> int:
    src = ROOT / "skeletons" / f"{day}_exercise.py"
    dst = ROOT / day / "exercise.py"
    if not src.exists():
        print(f"No skeleton found at {src}")
        return 1
    if src.read_text() == dst.read_text():
        print(f"{day}/exercise.py is already the blank skeleton — nothing to do.")
        return 0
    answer = input(
        f"This overwrites {day}/exercise.py with the blank skeleton, "
        f"discarding what you typed. Continue? [y/N] "
    )
    if answer.strip().lower() != "y":
        print("Aborted.")
        return 1
    dst.write_text(src.read_text())
    print(f"{day}/exercise.py reset to the blank skeleton.")
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--all"]
    stop_at_failure = "--all" not in sys.argv
    if not args:
        print(__doc__)
        return 1

    if args[0] == "reset":
        if len(args) < 2:
            print("Usage: python check.py reset dayNN")
            return 1
        day = normalize_day(args[1])
        if day not in COMPONENTS:
            print(f"Unknown day '{args[1]}'. Days: day01 .. day10")
            return 1
        return reset_day(day)

    day = normalize_day(args[0])
    if day not in COMPONENTS:
        print(f"Unknown day '{args[0]}'. Days: day01 .. day10")
        return 1
    only = args[1].lower() if len(args) > 1 else None

    comps = COMPONENTS[day]
    if only:
        matches = [(l, c) for l, c in comps if only in l.lower()]
        if not matches:
            print(f"No part of {day} matches '{only}'. "
                  f"Parts: {', '.join(l for l, _ in comps)}")
            return 1
        label, cls = matches[0]
        status, out = run_component(day, cls, verbose=True)
        print(out)
        return 0 if status == "pass" else 1

    print(f"\n{BOLD}{day}{END}")
    all_pass = True
    for label, cls in comps:
        status, out = run_component(day, cls, verbose=False)
        if status == "pass":
            print(f"  {GREEN}✓ {label}{END}")
        elif status == "todo":
            all_pass = False
            print(f"  {YELLOW}○ {label}{END} {DIM}— not implemented yet{END}")
        else:
            all_pass = False
            print(f"  {RED}✗ {label}{END} — a check failed:\n")
            for line in out.splitlines():
                print(f"    {line}")
            print(f"\n  {DIM}Re-run just this part with:"
                  f"  python check.py {day} {label}{END}")
            if stop_at_failure:
                print(f"  {DIM}(fix this before moving on; use --all to "
                      f"run everything anyway){END}")
                remaining = comps[comps.index((label, cls)) + 1:]
                for l2, _ in remaining:
                    print(f"  {DIM}· {l2} — skipped{END}")
                break

    if all_pass:
        print(f"\n{GREEN}{BOLD}All of {day} passes! 🎉{END} "
              f"Tell Claude — time for code review.\n")
        return 0
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
