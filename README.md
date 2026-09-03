# sepsis-timing — Phase 1

Implementation of Phase 1 (spec §10.1) of the timing-aware sepsis alarming
project. Phase 2 (the monotone/independent multi-horizon heads, the
multi-horizon loss, and the expected-utility decision layer) is **not**
implemented here — see the SCOPE CONTROL block at the top of the spec.
This repo builds the measurement instrument, the baseline, and the
evidence that the decision-rule gap is real, so that Phase 2 has something
concrete to improve on.

## Setup

```bash
pip install -r requirements.txt
```

## Running Phase 1 end to end

```bash
python -m src.data.download --out data/raw            # stage 1
python -m src.data.parse --raw data/raw --out data/processed
python -m src.data.verify_manifest                     # stage 1 gate
python -m src.experiments.eda                           # stage 2
pytest tests/ -q                                        # stages 3-5 gates
python -m src.experiments.run_grid                       # stage 6
python -m src.experiments.analyse --checkpoint ... --meta ...   # stage 7
```

## Design decisions and disclosed deviations

Recorded here because the spec explicitly asks several of these to be
documented and defensible in a viva, not just implicit in the code.

### §2.3 — six-hour label offset
`t_sepsis = first_positive_hour(SepsisLabel) + 6`. Verified two ways before
any preprocessing code was written: against the official PhysioNet
documentation (`SepsisLabel` is 1 for `t >= t_sepsis - 6`), and against the
official scorer's own internal derivation
(`third_party/physionet/evaluate_sepsis_score.py:compute_prediction_utility`,
`t_sepsis = np.argmax(labels) - dt_optimal` with `dt_optimal = -6`). Both
agree exactly with the spec. Covered by `tests/test_labels.py`, including a
randomised cross-check against the official formula on 20 synthetic
patients.

### §3.3 — training-hour selection: option (b) adopted
B1 trains against per-hour `SepsisLabel` only for hours `t <= t_sepsis` on
septic patients (all hours for non-septic), per the spec's own
recommendation — post-onset hours are a degenerate prediction task
(`SepsisLabel` trivially 1) that would dilute the gradient. Recorded in
`configs/base.yaml` (`training_hour_selection: "b"`). Evaluation always
covers every hour of every patient's stay regardless of this setting
(`src/eval/harness.py` never applies the training mask) — the spec is
explicit that evaluation must match what the official scorer does.

### §5.2 — B1's baseline decision rule uses `r[t]` directly, not `F[t,6]`
B1 has no multi-horizon head (that's P1/P2, Phase 2), so its scalar risk
score `r[t]` (the sigmoid head's output) is thresholded directly. The
`F[t,6]` framing in §5.2 applies once the monotone multi-horizon head
exists.

### qSOFA proxy (B0) — disclosed deviation from true qSOFA
True qSOFA is three criteria (respiratory rate >= 22, systolic BP <= 100
mmHg, altered mentation / GCS < 15), positive screen at score >= 2. This
dataset (spec §2.2's 41 columns) has no mental-status/GCS column, so the
third criterion cannot be computed. `src/decision/qsofa.py` implements a
2-criterion proxy: alarm at the first hour both available criteria are met
simultaneously — the only way to reach a qSOFA-equivalent score of 2 with
what this dataset contains. B0 is therefore a weaker "non-ML floor" than
true clinical qSOFA would be; this is disclosed, not hidden.

### "Provided example data" for harness validation (stage 4) — substituted
Checked directly: `physionetchallenges/evaluation-2019` and the
`*-example-2019` repositories do not ship a separate example
labels/predictions bundle for `evaluate_sepsis_score.py` (see
`third_party/physionet/PROVENANCE.md`). Validation instead runs the
harness's full write-files-then-score path against many synthetic patients
(4 prediction strategies) and a real downloaded PhysioNet slice, checked
against an *independent subprocess call* to the vendored script's
documented CLI, asserting exact agreement on all five output scores
(`tests/test_harness_validation.py`). This is arguably a stronger check
than agreement against one fixed example file would have been, since it
exercises every branch of the utility formula across many onset times.

### Utility function extraction (stage 5)
`src/decision/utility.py`'s `U(prediction, hour, t_sepsis)` is a
line-by-line transcription of the per-hour branch inside the vendored
`compute_prediction_utility`, with constants copied verbatim (not
recalled): `dt_early=-12, dt_optimal=-6, dt_late=3, max_u_tp=1,
min_u_fn=-2, u_fp=-0.05, u_tn=0`. Validated by summing `U()` over full
patient stays and checking exact agreement with the vendored function on
180 randomised synthetic patients across 4 prediction patterns
(`tests/test_utility.py`).

### The utility function does not appear in training
B1 is trained with plain BCE against `SepsisLabel`; nothing in
`src/experiments/train.py`'s loss touches `src/decision/utility.py`. This
mirrors the spec's design principle for the (Phase 2) multi-horizon loss
(§4.5): the model is not fitted to one hospital's cost structure, so
changing the utility function only requires changing the decision layer.

### Citations (spec §13)
The MASK/DELTA feature blocks (`src/data/preprocess.py`) are the GRU-D
lineage of missingness-aware recurrent features (Che et al., 2018,
*Scientific Reports* 8:6085) — implemented and used here, not claimed as a
contribution (spec §1.3). DeepAISE, Morrill, and CORAL/CORN citations apply
to Phase 2 components (the monotone multi-horizon head and survival-style
framing) not present in this phase.

## What's deliberately not here (Phase 2, spec §10.2)

- `src/models/heads.py` contains only the sigmoid head. No monotone or
  independent multi-horizon head, no stub, no `NotImplementedError`.
- No `src/models/losses.py` (multi-horizon loss).
- No `src/decision/stopping.py` (expected-utility decision layer).
- No `p1.yaml` / `p2.yaml` / `a1.yaml` configs.
- No `test_monotone.py` / `test_decision_toys.py`.

The encoder (`src/models/encoder.py`) takes no condition-specific
arguments and produces `h` of shape `(B, T, 64)` regardless of what head
consumes it — this is what "head-agnostic" means in practice. The
evaluation harness (`src/eval/harness.py`) accepts any
`patient_id -> alarm_hour | None` function; B0's qSOFA rule, B1's tuned
threshold, and (in Phase 2) the expected-utility decision layer are all
just instances of that same interface.
