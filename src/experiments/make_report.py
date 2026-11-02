from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TAB = Path("results/phase2/tables")
OUT = Path("docs/REPORT.md")


def tbl(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / f"{name}.csv")


def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    def cell(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return str(int(v)) if float(v).is_integer() and abs(v) >= 100 else fmt.format(v)
        return str(v)
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(lines)


def row(df, col, val):
    return df[df[col] == val].iloc[0]


def parse_ci(s):
    a, b = s.strip("[]").split(",")
    return float(a), float(b)


def main():
    cfg = yaml.safe_load(Path("configs/phase2.yaml").read_text())
    summ = json.loads(Path("results/phase2/summary.json").read_text())
    prim, matched, allc = tbl("table1_primary"), tbl("table2_matched_operating_points"), tbl("table3_all_conditions")
    cmp_, cal, coh = tbl("table4_paired_bootstrap"), tbl("table5_calibration_ece"), tbl("table6_coherence_violations")
    dis, mech = tbl("table7_disagreement_summary"), tbl("table8_p2_implied_cutoff")
    official = json.loads((TAB / "official_scorer_check.json").read_text())
    max_off = max(v["abs_diff"] for v in official.values())

    primary = row(cmp_, "comparison (B - A)", "P1 -> P2 (primary, same weights)")
    lo, hi = parse_ci(primary["95% CI"])
    d = primary["difference"]
    if lo > 0:
        h1 = "H1 is supported: the paired CI is entirely above zero."
    elif hi < 0:
        h1 = "H1 is contradicted: P2 is worse than P1 and the paired CI is entirely below zero."
    else:
        h1 = "H1 is not supported: the paired CI contains zero, so by the criterion in the specification the hypothesis is falsified for this model and data."
    p1, p2 = row(prim, "condition", "P1"), row(prim, "condition", "P2")
    d_auroc = p2["AUROC"] - p1["AUROC"]
    h2 = (f"H2 holds: AUROC is {p1['AUROC']:.3f} for P1's score and {p2['AUROC']:.3f} for P2's decision margin (difference {d_auroc:+.4f}), "
          "so any utility change is a timing effect and not a discrimination change." if abs(d_auroc) < 0.01
          else f"AUROC differs by {d_auroc:+.4f} between P1 and P2, which the specification says to treat as a possible bug; see the checks section.")

    pm = matched[matched["seed"] == "mean"].iloc[0]
    b1p1 = row(cmp_, "comparison (B - A)", "B1 -> P1 (output representation)")
    b0b1 = row(cmp_, "comparison (B - A)", "B0 -> B1")
    ds = pm["P2 sensitivity"] - pm["P1 sensitivity at P2's alarm rate"]
    dl = pm["P2 lead (h)"] - pm["P1 lead (h) at P2's alarm rate"]
    same_curve = abs(ds) < 0.02 and abs(dl) < 1.5
    match_txt = ("At matched operating points P2 lies on P1's own sensitivity versus alarm-rate curve" if same_curve
                 else "At matched operating points P2 and P1 differ")
    l_b, h_b = parse_ci(b1p1["95% CI"]); l_0, h_0 = parse_ci(b0b1["95% CI"])
    rep_txt = ("Output representation made no measurable difference" if l_b <= 0 <= h_b else
               ("The multi-horizon output representation improved utility" if l_b > 0 else "The multi-horizon output representation reduced utility"))
    gru_txt = ("while the trained GRU clearly beats the qSOFA proxy" if l_0 > 0 else "and the trained GRU does not clearly beat the qSOFA proxy")

    # A1 / H3
    a1_txt = "A1 was not run."
    if (coh["condition"] == "A1").any():
        a = coh[coh["condition"] == "A1"]
        rate, size, big = a["violation rate (all test hours)"], a["mean size of a violation"], a["largest violation"]
        c1 = row(cmp_, "comparison (B - A)", "P2 -> A1 decision layer (H3)")
        l3, h3 = parse_ci(c1["95% CI"])
        equal = l3 <= 0 <= h3
        negligible = rate.mean() < 0.05 and size.mean() < 0.01     # judgement for this report, not from the spec
        a1_txt = (f"A1's independent head violates monotonicity in {100 * rate.mean():.2f}% of (t, k) pairs on average over seeds "
                  f"(range {100 * rate.min():.2f}% to {100 * rate.max():.2f}%); when it does, the drop is {size.mean():.4f} in probability on average "
                  f"(largest {big.max():.3f}). P1/P2's head violates it in 0% by construction. "
                  f"With the decision layer applied, A1 minus P2 is {c1['difference']:+.3f} in normalised utility (95% CI {c1['95% CI']}). ")
        if negligible and equal:
            a1_txt += ("H3 is not supported. The specification says H3 is falsified if the independent head shows negligible coherence violations and equal utility. "
                       "Here violations are rare and small (we call them negligible when under 5% of pairs with a mean drop under 0.01 in probability; "
                       "that threshold is a judgement made for this report) and the utility interval contains zero.")
        elif equal:
            a1_txt += ("Violations are not negligible, but utility is equal within noise, so the evidence for H3 is mixed: incoherence is measurable, "
                       "yet it did not produce a measurable utility loss.")
        else:
            a1_txt += "The interval excludes zero, so the monotone and unconstrained heads differ in utility under the decision layer."

    # ablations
    abl = cmp_[cmp_["comparison (B - A)"].str.contains("ablation")]
    abl_txt = "No ablation runs completed." if abl.empty else (
        "Ablations compared with the main configuration (all deltas are paired over the same test patients and seeds):\n\n"
        + md(abl[["comparison (B - A)", "difference", "95% CI", "excludes 0", "per-seed differences"]]))

    mp = mech[mech["condition"] == "P"].dropna(axis=1, how="all")
    never_later = bool(mp["P2 alarm never later than P1"].all())
    taus = mp["tau for P1 (validation)"]
    kcol = [c for c in mp.columns if c.startswith("smallest F[t,")][0]
    cut = mp[kcol]
    base_col = [c for c in mp.columns if c.startswith("mean F[t,")][0]
    same_seed = int(mp[base_col].idxmax()) == int(mp["P2 false-alarm share of non-septic"].idxmax())
    mech_txt = (
        ("P2 never alarms later than P1 in any seed, so everything that differs from P1 is an earlier alarm or an extra one. " if never_later
         else "P2 sometimes alarms later than P1; see the category table below. ")
        + f"In closed form P2 behaves like a near-fixed cutoff on F[t,K]: its hourly gain from alarming is the sum over k of p_k times the scorer's hourly gain for an onset k hours away "
          f"(0 at k = 12 rising to 1.6 at k = 1) minus 0.05 times p_none, which for a flat onset distribution puts the break-even near F[t,K] of 5 to 6%. "
          f"The saved numbers agree: the smallest F[t,K] at which P2 fires is {cut.min():.3f} to {cut.max():.3f} across seeds, while the validation-tuned P1 threshold on F[t,6] "
          f"is {taus.min():.3f} to {taus.max():.3f}. P1's threshold adapts to the model and P2's cutoff cannot. "
          + ("The seed whose model gives the highest baseline risk to non-septic patients is also the seed where the untuned cutoff flags the largest share of them."
             if same_seed else "")
    )

    p1p2 = cmp_[cmp_["comparison (B - A)"].str.contains("P1 -> P2")].copy()
    ci_pairs = [parse_ci(c) for c in p1p2["95% CI"]]
    n_pos = sum(lo_ > 0 for lo_, _ in ci_pairs); n_neg = sum(hi_ < 0 for _, hi_ in ci_pairs)
    sig_abl = abl[abl["excludes 0"] == "yes"]
    sig_txt = ("; ".join(f"{r['comparison (B - A)']} ({r['difference']:+.3f})" for _, r in sig_abl.iterrows()) or "none")
    abl_summary = (f"Across the {len(p1p2)} configurations with both rules (main, K = 6, 18, no mask/delta, hidden 32, hidden 128), P2 minus P1 was significantly positive in {n_pos}, "
                   f"significantly negative in {n_neg} and indistinguishable from zero in {len(p1p2) - n_pos - n_neg}. "
                   f"Ablation deltas whose interval excludes zero: {sig_txt}. "
                   f"These are {len(cmp_)} unadjusted 95% intervals in total, so a few exclusions are expected by chance and the ablation results are descriptive, not confirmatory.")
    tail = cal[(cal["horizon k"].astype(str) == "mean")]
    cal_main = tail[tail["condition"] == "P"].iloc[0]
    a1_tail = cal[(cal["condition"] == "A1") & (cal["horizon k"].astype(str) == "mean")]
    a1_tail_txt = (f" The unconstrained head's tail ECE is {a1_tail.iloc[0]['tail ECE, F>=0.1 (training hours)']:.3f} against {cal_main['tail ECE, F>=0.1 (training hours)']:.3f} for the monotone head, so the constraint did not buy better tail calibration here." if len(a1_tail) else "")


    ext = Path("results/phase2/extension")
    ext_sec = ""
    if (ext / "hospitalB_summary_blank_etco2.csv").exists():
        EB = pd.read_csv(ext / "hospitalB_summary_blank_etco2.csv").set_index("rule")
        EBB = pd.read_csv(ext / "hospitalB_bootstrap_blank_etco2.csv")
        lam = json.loads((ext / "summary.json").read_text())["lambda_star_per_seed"]
        d_t = EBB[(EBB["A"] == "P1") & (EBB["B"] == "P2 (lam*)")].iloc[0]; d_p = EBB[(EBB["A"] == "P1") & (EBB["B"] == "P2")].iloc[0]
        ext_sec = f"""
## 9. Post hoc extension: a one-parameter P2 and Hospital B (not part of the locked comparison)

Full tables are in `docs/EXTENSION.md` (generated from `results/phase2/extension/`). P2 was given one parameter, a scale lambda on the false-alarm cost, chosen per seed on the Hospital A validation split like P1's tau (chosen: {lam['0']:.2f}, {lam['1']:.2f}, {lam['2']:.2f}). Nothing was retrained. On Hospital B (20,000 stays, never used for any choice; end-tidal CO2 blanked as at training) the mean test utilities are P1 {EB.loc['P1','utility mean']:.3f}, parameter-free P2 {EB.loc['P2','utility mean']:.3f}, P2 with tuned lambda {EB.loc['P2 (lam*)','utility mean']:.3f}, B1 {EB.loc['B1','utility mean']:.3f}. Tuned P2 minus P1 is {d_t['difference B - A']:+.3f} (95% CI [{d_t['ci_low']:+.3f}, {d_t['ci_high']:+.3f}]), a tie; parameter-free P2 minus P1 is {d_p['difference B - A']:+.3f} (CI [{d_p['ci_low']:+.3f}, {d_p['ci_high']:+.3f}]). The adaptive version does not beat P1 and is no longer parameter-free; its Hospital A test gain is a second look at a used split and is not counted.
"""
    expl = Path("results/phase2/exploratory")
    has_expl = (expl / "variants_summary.csv").exists()
    expl_sec, expl_lim, expl_qa1, expl_qa2 = "", "", "", ""
    if has_expl:
        S = pd.read_csv(expl / "variants_summary.csv"); C = pd.read_csv(expl / "variants_paired_bootstrap.csv")
        CALX = pd.read_csv(expl / "calibration_effect.csv")
        chosen = json.loads((expl / "chosen_variant.json").read_text())["chosen_on_validation"]
        vsP1 = C[C["versus (A)"] == "P1 (locked)"]
        rescued = [r["variant (B)"] for _, r in vsP1.iterrows() if parse_ci(r["95% CI"])[0] > 0]
        ch = vsP1[vsP1["variant (B)"] == chosen].iloc[0]; ch2 = C[(C["variant (B)"] == chosen) & (C["versus (A)"] == "primary P2 (locked)")].iloc[0]
        pers = vsP1[vsP1["variant (B)"] == "persistent + shared Platt"].iloc[0]
        c0, c1 = CALX[CALX["calibration"] == "none"].iloc[0], CALX[CALX["calibration"] == "shared Platt"].iloc[0]
        ev0 = S[S["variant"] == "event-level + none"].iloc[0]; ev1 = S[S["variant"] == "event-level + shared Platt"].iloc[0]
        cp = json.loads((expl / "calibration_params.json").read_text())
        a_all = [v["shared Platt"]["a"][0] for v in cp.values()]; b_all = [v["shared Platt"]["b"][0] for v in cp.values()]
        a_rng, b_rng = (min(a_all), max(a_all)), (min(b_all), max(b_all))
        verdict = ("None of the six variants has a paired interval entirely above P1, so neither explanation rescues P2." if not rescued
                   else f"Variants whose paired interval versus P1 is entirely above zero: {', '.join(rescued)}. Because six variants were examined on the same test set, treat this as a hypothesis for a fresh split, not a result.")
        expl_lim = (f" Section 8 reports an exploratory check of an event-level reading and of validation-fitted recalibration; "
                    + ("it did not change the conclusion." if not rescued else "it found a variant worth testing on a new split."))
        expl_qa1 = (f"Partly tested, after the result was locked (section 8). Swapping in an event-level reading and recalibrating F on validation "
                    + ("did not rescue P2. " if not rescued else "produced a candidate that needs a fresh split. ")
                    + f"The variant chosen on validation ({chosen}) scored {ch['difference B - A']:+.3f} against P1 (CI {ch['95% CI']}). Other readings remain untried.")
        expl_qa2 = "Price a 'never alarm' candidate with the full-stay false-alarm cost; calibrate during training; train on Hospital B or a pooled cohort; repeat on more splits and seeds. Recalibration, the event-level reading (section 8), a validation-tuned cost scale and an inference-only Hospital B evaluation (section 9) are done."
        expl_sec = f"""
## 8. Exploratory follow-ups (post hoc, not part of the locked comparison)

After the primary result was locked, two untested explanations for the null remained: that the probabilities are too overconfident for an expected-utility calculation, and that the persistent-alarm reading of `U(alarm at s, onset)` is the wrong one. Both can be checked by inference alone, on the saved predictions of the primary model (condition P, 3 seeds, nothing retrained).

Variants: reading in {{persistent (the primary P2), event-level}} times calibration in {{none, one shared Platt scaling, per-horizon Platt scaling plus a running maximum over k}}. The event-level reading scores a positive prediction at hour s by the scorer's true-positive curve, `V1[t,s] = sum_k p[t,k] U(1, s, t+k) + p_none U(1, s, none)`, and alarms when now is the argmax over `s = t..t+K` and `V1[t,t] > 0`; it follows the spec's prose and toy cases more literally (peak six hours before onset) but ignores persistence and missed-onset costs. Calibration is fitted on the validation split only (same hours and horizons as training). The variant is chosen by seed-averaged validation utility; all six are shown on test for transparency. Hand-computed tests for the event-level rule are in `tests/test_event_reading.py`, and for the calibrators in `tests/test_calibrate.py`.

{md(S)}

Paired bootstrap on test (B minus A):

{md(C.drop(columns=['per-seed differences']))}

Effect of recalibration on the monotone model's test probabilities (mean over seeds):

{md(CALX, '{:.4f}')}

**Reading of the results.** {verdict} Recalibration lowered the tail ECE from {c0['tail ECE F>=0.1']:.3f} to {c1['tail ECE F>=0.1']:.3f} (shared Platt) but left the persistent-reading P2 where it was (shared Platt versus P1: {pers['difference B - A']:+.3f}, CI {pers['95% CI']}), which fits the mechanism in section 4.3: P2's cutoff sits at F of about 5%, in the low range where the model was already close to calibrated, while the overconfidence is in the tail. The event-level reading behaves very differently: uncalibrated it alarms on only {100 * ev0['alarmed share (test)']:.1f}% of test stays (test utility {ev0['test utility (mean)']:.3f}), with shared Platt scaling it alarms on {100 * ev1['alarmed share (test)']:.1f}% and scores {ev1['test utility (mean)']:.3f}, and with per-horizon scaling it almost never alarms ({100 * S[S['variant'] == 'event-level + per-horizon Platt'].iloc[0]['alarmed share (test)']:.1f}%). The shared fits have slopes {a_rng[0]:.2f} to {a_rng[1]:.2f} and offsets {b_rng[0]:+.2f} to {b_rng[1]:+.2f}, which pull high probabilities down and lift only the very lowest slightly. A change that small moving the alarm rate by an order of magnitude shows that this rule sits on a knife edge (it needs `V1[t,t] > 0` while the penalty for a false alarm is only 0.05); we did not isolate the cause. The variant chosen on validation, {chosen}, scores {ch['difference B - A']:+.3f} against P1 (CI {ch['95% CI']}) and {ch2['difference B - A']:+.3f} against the primary P2 (CI {ch2['95% CI']}).

Caveats: P1's validation utility is optimistic because its threshold was tuned on that same split, while the P2 variants have nothing tuned on it apart from the choice among six; the six variants are six extra looks at the same test set; and only one alternative reading was tried. Nothing in sections 2 to 7 was changed after these runs.
"""

    text = f"""# Timing-aware sepsis alarming: Phase 2 methods and results

Everything below is generated from the saved outputs in `results/phase2/` by `src/experiments/make_report.py`.
Test set: {summ['n_test_patients']} Hospital A patients ({summ['n_septic_test']} septic), seeds {summ['seeds']}.

## 1. Question

Does replacing a validation-tuned threshold on a scalar risk with an expected-utility stopping rule improve
timing-dependent utility, holding the encoder and the trained weights fixed? P1 and P2 are two decision rules
applied to the same trained monotone multi-horizon model, so any difference is the decision layer.

## 2. Headline result

**{h1}** On the test split, P2 minus P1 in normalised utility is {d:+.3f} (95% paired patient-bootstrap CI {primary['95% CI']}, seed-wise differences {primary['per-seed differences']}).
{h2}

{match_txt}: reading P1 at P2's alarm rate gives sensitivity {pm['P1 sensitivity at P2\'s alarm rate']:.3f} against P2's {pm['P2 sensitivity']:.3f}, and median lead time {pm['P1 lead (h) at P2\'s alarm rate']:.1f} h against {pm['P2 lead (h)']:.1f} h.
{rep_txt} (B1 to P1: {b1p1['difference']:+.3f}, CI {b1p1['95% CI']}), {gru_txt} (B0 to B1: {b0b1['difference']:+.3f}, CI {b0b1['95% CI']}).

This is a null result for the proposed decision rule and it is reported as such. Section 5 lists what the data say about why and what remains untested.

## 3. Methods

**Data and splits.** PhysioNet/CinC 2019, Hospital A, patient-level stratified 70/15/15 split (seed 0), identical to Phase 1.
Preprocessing statistics are fitted on the training patients only. `SepsisLabel` turns positive six hours before onset, so onset is `first positive hour + 6`.
Features are the 120-channel VALUES/MASK/DELTA representation (Che et al. 2018); the mask/delta ablation uses the 40 VALUES channels.

**Model.** The Phase 1 encoder: 2-layer unidirectional GRU, hidden size 64, dropout 0.2. The monotone head maps the encoder output to K logits as
`[base, base + cumsum(softplus(increments))]`, and `F[t,k] = sigmoid(logit_k)` is non-decreasing in k by construction (tested). The onset distribution is
`p[t,1] = F[t,1]`, `p[t,k] = F[t,k] - F[t,k-1]`, `p_none = 1 - F[t,K]`. A1 uses an unconstrained sigmoid head. Encoder initialisation is identical across heads for a given seed, as are
the batch order and the dropout stream.

**Targets and loss.** `y[t,k] = 1` if onset `<= t + k`, for k = 1..K (K = 12 unless ablated). Non-septic patients are censored where `t + k` reaches the first unobserved hour. Padded hours are masked and, following spec section 3.3 option (b), septic patients contribute only hours `t <= t_sepsis`.
The loss is the masked binary cross-entropy summed over horizons and divided by the number of active terms. The utility function is not in the loss.

**Training.** Adam, learning rate 1e-3, batch of 64 patients, at most 50 epochs, gradient clipping at norm 1.0, early stopping with patience 8 on validation utility. For the shared P1/P2 model the stopping utility is P1's (tuned threshold on `F[t,6]`), so checkpoint selection favours the baseline.

**Decision rules.** P1 alarms at the first hour with `F[t,6] > tau`; `tau` is tuned on the validation split only. P2 alarms at the first hour `t` where `V[t,t]` is the maximum over `s = t..t+K` of
`V[t,s] = sum_k p[t,k] U(alarm at s, onset t+k) + p_none U(alarm at s, no onset)`. The spec leaves `U(alarm at s, .)` open; it is taken as the official utility of "0 before s, 1 from s on", built from the validated hourly `U()`, because the alarm persists and the scorer sums every hour (a persisting alarm on a stay with no onset costs `u_fp` each hour). Ties go to the earlier hour. The rule is checked against hand-computed cases and against the vendored official scorer (`tests/test_decision_toys.py`).
For A1, `F` is replaced by its running maximum before deriving `p`, because an unconstrained head can give negative increments.

**Metrics.** Normalised challenge utility (primary), AUROC and AUPRC, alarms per patient-day and median lead time (P1 swept over `tau` and read at P2's operating point), reliability curves and ECE per horizon, coherence-violation rate, paired patient bootstrap (2000 resamples, same resample applied to both conditions and all seeds) with effect sizes.
Sensitivity is the share of septic patients whose alarm falls in [onset - 12, onset + 3]. Table utilities were checked against the vendored official scorer for B0, B1, P1, P2 and A1 (largest absolute difference {max_off:.1e}).

## 4. Results

### 4.1 Primary comparison

{md(prim)}

Matched operating points (P1 read off its `tau` sweep at P2's sensitivity or alarm rate):

{md(matched)}

![primary](../results/phase2/figures/fig_primary_utility.png)
![matched](../results/phase2/figures/fig_matched_operating_points.png)

### 4.2 Paired bootstrap (B minus A)

{md(cmp_)}

### 4.3 Why P2 and P1 differ

{mech_txt}

{md(mp)}

Patients where the two rules disagree:

{md(dis)}

![examples](../results/phase2/figures/fig_disagreement_examples.png)

### 4.4 Calibration

Mean ECE over horizons for the monotone model is {cal_main['ECE (training hours)']:.4f} on the hours the model was trained for, which looks small only because most terms have `F` near zero. The tail ECE over terms with `F >= 0.1` (added after the first reliability curves were inspected) is {cal_main['tail ECE, F>=0.1 (training hours)']:.3f}: the model is clearly overconfident in the range where an alarm decision is made.{a1_tail_txt}

{md(cal[cal['horizon k'].astype(str).isin(['1', '3', '6', '9', '12', 'mean'])], '{:.4f}')}

![calibration](../results/phase2/figures/fig_calibration.png)

### 4.5 A1 and coherence (H3)

{a1_txt}

{md(coh, '{:.4f}')}

### 4.6 Ablations

{abl_txt}

{abl_summary}

![ablations](../results/phase2/figures/fig_ablations.png)

### 4.7 All conditions

{md(allc)}

## 5. Limitations and deviations

- **Interpretation of the decision rule.** `U(alarm at s, onset)` is not defined in the spec; the persistent-alarm reading above was fixed before any run. A different reading could change P2; one alternative was checked only after the result was locked.{expl_lim}
- **Myopic approximation** (spec 5.4): the belief is held fixed at hour t, and the rule compares alarming now with alarming later inside the K window; it never prices the full-stay cost of a false alarm against never alarming.
- **No parameters in P2.** P1 adapts its threshold to each trained model; P2 cannot. Part of any P1 advantage is that adaptivity, and the spec requires P2 to be parameter-free.
- **Overconfidence.** The decision layer assumes meaningful probabilities; the reliability curves show overconfidence in the tail, and the cause was not isolated.
- **Spec notes.** The censoring mask in the spec is written for 1-indexed hours; the code uses 0-indexed rows (`t + k >= T`), and this corrected a Phase 1 off-by-one in an artifact that no Phase 1 result used. The spec says septic onset is inside the record; 15 of 2932 septic stays have onset after the last row (targets are still defined).
- **A1 repair.** A1's `F` is repaired with a running maximum before the decision layer; violation rates are measured on the raw output.
- **Single split.** One Hospital A split with three training seeds; seed-to-seed spread is large (P2 ranges widely), so the bootstrap CI over patients understates total uncertainty. No model was trained on Hospital B or on a pooled cohort (spec 11.1 allows dropping those splits); the A-trained checkpoints were applied to Hospital B for inference only in the post hoc extension (section 9).
- **Utility scale.** Normalised utility here is on the Hospital A test split and is not comparable to challenge leaderboard scores.
- **Early stopping noise.** Validation utility fluctuates epoch to epoch, so the stopping epoch (12 to 20) is partly luck.
- **No outlier handling** (carried from Phase 1).
- **Reruns and corrections, for the record.** The first launch of primary run P seed 0 was killed after about two minutes and restarted because its output was block-buffered; it had produced no result. A bug in a synthetic test cohort (onsets beyond the record) was fixed in `tests/test_fast_utility.py`, not in the library. The tail-ECE column and the violation-size columns were added after the first reliability curves and A1 numbers were inspected. `results/phase2/logs/analysis_primary.log` and `analysis_dry.log` are earlier analysis passes on fewer conditions; `analysis_final.log` is the pass the tables come from. No model was retrained or retuned after test results were seen.

## 6. Related work

Sepsis onset prediction on this dataset: Reyna et al. 2020 (challenge and utility); Morrill et al. 2019 (signature features with gradient boosting, first place); Shashikumar et al. 2021 (DeepAISE, a recurrent survival model that outputs hourly onset risk). Missingness-aware recurrent features: Che et al. 2018 (GRU-D), used as is. Rank-consistent ordinal outputs: Cao et al. 2020 (CORAL) and Shi et al. 2021 (CORN); the monotone head is in that family of ideas but is not either construction. None of these components is claimed as novel; the only thing tested here is the decision procedure.

## 7. Reproducing

```
python -m src.data.download && python -m src.data.parse
python -m src.experiments.run_phase2 prepare
python -m src.experiments.run_phase2 train P 0          # conditions and seeds in configs/phase2.yaml
python -m src.experiments.evaluate_runs baselines && python -m src.experiments.evaluate_runs runs
python -m src.experiments.analyse_phase2 && python -m src.experiments.exploratory_followups
python -m src.experiments.extension_cost_scale && python -m src.experiments.make_extension_report
python -m src.experiments.replay && python -m src.experiments.make_report
python -m pytest tests -q
```

{expl_sec}{ext_sec}
## Appendix: questions to be ready for

**What is the main finding?** {h1} P2 minus P1 is {d:+.3f} (CI {primary['95% CI']}). The decision layer did not buy utility over a tuned threshold on this model.

**Is the null caused by how you read `U(alarm at s, onset)`, or by the probabilities being overconfident?** {expl_qa1 if has_expl else 'Not known; not tested.'}\n\n**Why is P2 not tuned?** The specification makes the decision layer parameter-free so that a gain can only come from the cost model, not from fitting. The price is that it cannot adapt to a particular model's baseline risk, which P1's threshold does; section 4.3 shows that this is where the two diverge.

**How do you know P1 and P2 share weights?** They are derived from one array `F` in `src/experiments/evaluate_runs.py`; each checkpoint is reloaded, its SHA-256 is compared with the hash saved at training, and the reloaded model must reproduce the saved predictions. Both rules carry the same hash in the saved evaluation files.

**Why did you checkpoint on P1's utility?** P1 and P2 share weights, so one criterion is needed. Using P1's favours the baseline, which is the conservative choice for a claim that P2 is better.

**Phase 1 said a per-patient-optimal threshold could reach 0.88. Why does P2 not get close?** That ceiling picks each patient's threshold with hindsight and cannot be achieved by any deployed rule. It shows the headroom, not what a causal rule can reach, and P2 is causal.

**What does the calibration result mean for the decision layer?** The expected-utility calculation treats `F` as probabilities. In the tail (F of 0.1 or more) the model is overconfident, so the calculation is working with inflated onset probabilities exactly where it decides. Recalibrating `F` on the validation split before applying P2 was tried afterwards (section 8) and did not change the result, because P2's cutoff sits at F of about 5%, where the model was already close to calibrated.

**What would you do next?** {expl_qa2 if has_expl else 'Recalibrate F on validation and re-evaluate P2; study other readings of U on validation; run the cross-hospital split.'}\n\n**Why three seeds, and what do the intervals cover?** Seeds capture training variance, which is large here (P2 per-seed utility ranges widely). The bootstrap resamples patients with the seed set fixed, so together they cover test-patient sampling but not an independent re-split of the data.

## References

- Reyna, M. A., Josef, C. S., Jeter, R., et al. (2020). Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210-217.
- Morrill, J., Kormilitzin, A., Nevado-Holgado, A., Swaminathan, S., Howison, S., & Lyons, T. (2019). The signature-based model for early detection of sepsis from electronic health records in the intensive care unit. *Computing in Cardiology*, 46. doi:10.22489/CinC.2019.014
- Shashikumar, S. P., Josef, C. S., Sharma, A., & Nemati, S. (2021). DeepAISE: an interpretable and recurrent neural survival model for early prediction of sepsis. *Artificial Intelligence in Medicine*, 113, 102036.
- Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8, 6085.
- Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters*, 140, 325-331.
- Shi, X., Cao, W., & Raschka, S. (2021). Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. arXiv:2111.08851.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP 2014*.
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print("wrote", OUT, len(text), "chars")


if __name__ == "__main__":
    main()
