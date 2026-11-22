# Poke or Ask

A simulated robot pokes an unseen door to decide whether it is a slider, a hinge, or locked. When it is unsure, it asks a human.

## Questions

1. Which poke teaches the robot the most?
2. Does the robot's confidence tell it when to ask?

Earlier work on door opening from 2014 fixes the hidden parameters. Here they are randomized on every trial (hinge position, slide direction, stiffness), and the asking question is new. The citation is still to be added.

## Setup

- Python with numpy and matplotlib only. Type-checked with mypy strict, linted with ruff, tested with pytest.
- A poke is a position and a push direction: 3 x 3 spots times 8 directions, 72 in all.
- A stiff slider and a locked door both barely move, so they are the hard case.
- Fixed seeds, 300 or more trials per setting, bootstrap error bars.

## Steps

| Week | Step | Done when |
|---|---|---|
| 1 | Simulator with 3 door types, random hidden parameters, noisy motion per poke | Each type responds differently, and a test shows parameters change per trial |
| 2 | Baseline: random pokes and a probability over the 3 types | Accuracy at 1, 2, and 4 pokes, with error bars |
| 3 | Smart pokes (the one that cuts uncertainty most) against random | Accuracy against number of pokes |
| 4 | Ask a human when the top probability is below a threshold, against asking at random at the same rate | Accuracy against ask rate, and a calibration plot |
| 5 | README with question, method, two figures, failures, and how to rerun | Someone else can rerun it |

## Metrics

1. Accuracy against number of pokes.
2. Accuracy against ask rate.
3. Calibration: when it says 90%, how often it is right.
4. False quiets: answered without asking and wrong.
5. Ask rate on stiff sliders against locked doors.

## Possible extensions

- Vary the poke budget (3, 5, 8).
- Add a cost: a wrong answer costs 10, asking costs 1.
- Ask by the gap between the top two guesses instead of the top probability.

## Rules

- A result where smart pokes do not beat random gets reported as it is.
- No tuning on test trials.
- No new features before step 5 is done.
