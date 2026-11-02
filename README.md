# sepsis-timing

Early sepsis alarms from hourly ICU data (PhysioNet/CinC 2019 challenge data, Hospital A).

Most sepsis alarm systems turn a risk score into an alarm with one tuned threshold. The challenge's scoring rule rewards alarms that land a few hours before onset and penalizes very early and late ones. We tested whether choosing the alarm hour by expected utility (P2) beats a tuned threshold (P1), with the trained model held fixed.

## Result

It did not, on this data. P1 and P2 use the same trained weights and the same test patients (3,051 patients, 3 seeds). Normalized utility (0 = never alarm, 1 = perfect timing):

| Condition | Utility (mean ± sd over 3 seeds) |
|---|---|
| B0, qSOFA-style rule | 0.078 |
| B1, GRU + sigmoid + threshold | 0.345 ± 0.027 |
| P1, monotone multi-horizon head + threshold on F[t,6] | 0.345 ± 0.006 |
| P2, same weights + expected-utility rule | 0.335 ± 0.044 |

Paired bootstrap, P2 minus P1: -0.010, 95% CI [-0.041, +0.020]. The interval contains zero, so the hypothesis that the decision layer improves utility is not supported here. AUROC is the same for both (0.796 vs 0.795). At matched alarm rates P1 and P2 have the same sensitivity and lead time, so P2 sits on P1's own curve.

Why, as far as the saved numbers show: P2 behaves like a fixed cutoff of about 5% on F[t,12], while P1's tuned threshold moves with each model (0.03 to 0.08 across seeds). The seed whose model gives non-septic patients the most baseline risk is the one where the untuned cutoff over-alarms (P2 0.285 vs P1 0.339). The model is also overconfident when F is 0.1 or higher. A1 (unconstrained head) has only about 1% tiny monotonicity violations and no utility difference, so the evidence for the monotone constraint mattering is weak. Across the ablations (K, mask/delta, hidden size) P2 was never significantly better than P1; removing mask/delta or shrinking the GRU to 32 units made P2 worse.

Post hoc extension: giving P2 one parameter, a scale on the false-alarm cost chosen on validation like P1's threshold, closes the gap. On Hospital B, which no model or threshold had seen, it ties P1 (0.242 vs 0.244, paired difference -0.002, CI [-0.004, +0.002]) and beats the parameter-free P2 (0.210). It does not beat P1, and it is no longer parameter-free.


## Layout

```
src/data/         parsing, preprocessing, splits, leakage checks, cached tensors
src/models/       GRU encoder, heads (sigmoid, monotone, independent), masked loss
src/decision/     utility function, threshold rule, qSOFA rule, expected-utility rule (stopping.py)
src/eval/         scorer harness, fast utility tables, metrics, bootstrap
src/experiments/  training, evaluation, analysis, replay demo, report
tests/            311 tests
configs/          base.yaml, phase2.yaml (all conditions), one file per condition
third_party/      official PhysioNet scoring script, unmodified
```

## Running it

```
pip install -r requirements.txt
python -m src.data.download && python -m src.data.parse
python -m pytest tests -q
python -m src.experiments.run_phase2 prepare
python -m src.experiments.run_phase2 train P 0      # conditions and seeds: configs/phase2.yaml
python -m src.experiments.evaluate_runs baselines
python -m src.experiments.evaluate_runs runs
python -m src.experiments.analyse_phase2
python -m src.experiments.exploratory_followups      # post hoc recalibration and alternative reading
python -m src.experiments.extension_cost_scale       # post hoc cost scale and Hospital B (inference only, ~2 min)
python -m src.experiments.replay                     # results/phase2/replay/replay.html
python -m src.experiments.make_report
python -m src.experiments.make_extension_report
```

Each run takes roughly 10 to 100 minutes on CPU (the hidden-size-128 runs are the slow ones). Existing runs are not overwritten. Checkpoints and prediction arrays are git-ignored, so a fresh clone retrains from the configs and fixed seeds.

## Notes

- The replay page is a replay of historical, de-identified stays. It is not a live system.
- `SepsisLabel` turns positive six hours before onset, so onset is the first positive hour + 6. This is computed in one place (`src/data/parse.py`) and tested against the official scorer's own formula.
- The spec does not define `U(alarm at s, onset)`. We use the official utility of an alarm that persists from hour s, which matches how the scorer counts it. This was fixed before any test result was seen. Afterwards, as an exploratory check, we tried an event-level reading and validation-fitted recalibration; neither rescued P2 (best validation-chosen variant: -0.033 vs P1, CI [-0.053, -0.013]).
- qSOFA normally uses three criteria. The dataset has no mental-status column, so B0 uses respiration rate and systolic BP only.
- EtCO2 is never observed at Hospital A. The normalization code falls back to mean 0 and std 1 for it and prints a warning.
- No outlier handling is implemented. No model was trained on Hospital B or on a pooled cohort; the A-trained checkpoints were applied to Hospital B for inference only (see the extension below).

## References

- Reyna et al., Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019, Critical Care Medicine, 2020. Data: https://physionet.org/content/challenge-2019/1.0.0/
- Morrill et al., The signature-based model for early detection of sepsis from electronic health records in the intensive care unit, Computing in Cardiology, 2019
- Shashikumar et al., DeepAISE: an interpretable and recurrent neural survival model for early prediction of sepsis, Artificial Intelligence in Medicine, 2021
- Che et al., Recurrent neural networks for multivariate time series with missing values, Scientific Reports, 2018 (mask and time-delta features)
- Cao et al., Rank consistent ordinal regression for neural networks with application to age estimation, Pattern Recognition Letters, 2020 (CORAL)
- Shi et al., Deep neural networks for rank-consistent ordinal regression based on conditional probabilities, arXiv:2111.08851 (CORN)
- Cho et al., Learning phrase representations using RNN encoder-decoder for statistical machine translation, EMNLP 2014 (GRU)
