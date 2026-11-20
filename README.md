# cognitive-resonance

> **English** · [Italiano](README.it.md)

**When two minds err in opposite directions, their errors cancel. How far does the
pair go, and does it behave like a resonant circuit?**

This is the sequel to [turing-meter](https://github.com/paithon-it/turing-meter) and to
its first experiment, `angle-between-errors`, the measurements behind the paithon.it
article [*Intelligenza apparente*](https://paithon.it/intelligenza-apparente/) (in
Italian; doi:[10.5281/zenodo.24107649](https://doi.org/10.5281/zenodo.24107649)). There
the angle `phi` between the errors of a person and of a machine was measured between 59
and 81 degrees, and the region beyond 90 degrees was left unexplored. Here we go there.

> Status: **version 2, runs done** (19 November 2026). E0 to E5 are told in the paithon.it
> article [*Risonanza cognitiva*](https://paithon.it/risonanza-cognitiva/) (in Italian;
> doi:[10.5281/zenodo.24140355](https://doi.org/10.5281/zenodo.24140355)). Four new
> experiments, E6 to E9, were **registered before their runs**: protocol 2.1 was pushed to
> this repository before any training, on five seeds that no earlier run used; version 2.2
> added a reading of E6 before any of its files existed. Of their 29 criteria, **25 pass
> and 4 do not** (table below). On 20 November three readings made after those results became
> three more tests, E10 to E12, registered with protocol 2.3 before their run, on seeds 7 to
> 11: of their 15 criteria, **12 pass and 3 do not**. The protocol is in [`PROTOCOL.md`](PROTOCOL.md) (Italian
> original [`PROTOCOL.it.md`](PROTOCOL.it.md)); what changed after a run is in its
> "Deviations" section.

## The identity

Two judges decide between two classes, each with a Gaussian cue; their noises correlate
with `rho = cos(phi)`. The pair adds the cues with a weight `w` on the machine. In units
of the person, with `k = dM / dH`:

    Tu(w)^2 = (1 + w k)^2 / (1 + w^2 + 2 rho w)

For `k = 1`, writing `Omega = sqrt(w)`:

    Tu(w)^2 - 1 = tan^2(phi/2) / (1 + Q^2 (Omega - 1/Omega)^2),      Q = 1 / (2 cos(phi/2))

This is the response of a series RLC circuit, with `Omega` in place of the normalized
frequency. At the peak (`w = 1`) the pair is worth `2Q = 1/cos(phi/2)` turings: `sqrt(2)`
at 90 degrees, 2 at 120, 3.9 at 150, without bound as `phi` goes to 180. The band, where
`Tu^2 - 1` stays above half its peak, narrows as the angle opens: a sharper resonance.
Allowing weights of either sign, `Tu(w)^2` becomes a Fano profile, with a null at
`w0 = -1/k` where the two signals cancel and only the noise is left. The derivation is in
one page in [`docs/identity.md`](docs/identity.md).

## The quantities

- `d_a = sqrt(2) z(AUC)`: the sensitivity of a judge, from its whole cue, without a
  threshold. Yes/no `d'` at threshold zero as in turing-meter, as a secondary measure.
- `k = dM / dH`, the machine in the person's unit.
- `rho`, `phi = arccos(rho)`: within-class Pearson correlation of the standardized cues
  (primary), and the tetrachoric correlation of the correct answers (the turing-meter
  estimator), side by side.
- `Tu_app = sqrt(1 + k^2 - 2 k cos phi) / sin phi`: the apparent turings, the peak of the
  pair; `sqrt(1 + k^2)` at 90 degrees.
- **Tuned turing**: the weight chosen on half of the images, the turing read on the
  other half, and the two halves swapped. It is the measure; the maximum over the grid on
  the same data is optimistic and is not used.
- `Q = 1/(2 cos(phi/2))`, the quality factor of the pair; the band as `w_high / w_low`.
- The null `w0`, its width (the weights where `|Tu| < 1/2`), and `q = (1 - k cos phi) /
  (k sin phi)`, the Fano asymmetry, zero at the useful angle `arccos(1/k)`.

## The experiments

| | What | Data |
|---|---|---|
| E0 | the instrument on data from the model: bias and coverage | simulated |
| E1 | the bench: small CNNs on five CIFAR-10 class pairs, a frame whose brightness the "plus" and "minus" networks learn in opposite directions, a knob `mu` that sets the angle | GPU runs |
| E2 | rephasing: a machine trained to decorrelate from a biased network, without being told the bias | GPU runs |
| E2 bis | the partner trained on the error of the pair, and one that sees only the frame (exploratory, v1.3, after E2 and E3) | GPU runs |
| E2 ter | the soloist that learns the person: the best machine alone, fine-tuned on the error of the sum too, and the same machine with a branch that sees only the frame (exploratory, v1.4) | GPU runs |
| E3 | real people: CIFAR-10H votes against a ResNet-18, and versions fine-tuned toward antiphase | GPU runs, CIFAR-10H |
| E3 bis | the recipe of the sum with real people: the base machine fine-tuned on the people's cue plus its own logits (exploratory, v1.4) | GPU runs, CIFAR-10H |
| E4 | the pairs already measured, reread on this scale | published angles |
| E5 | the null, with weights of either sign (added in v1.2, before any analysis) | the files of E1 |
| E6 | trust made to measure: four partners, each built for a different weight of the machine in the sum. Does the weight at which the pair pays off follow the weight the partner was built for? (v2.0) | GPU runs |
| E7 | different views: judge and machine see different halves of the image, with everything, a quarter or nothing in common. What does a partner trained from scratch learn? (v2.0) | GPU runs |
| E8 | the bench of the chart: the judge sees the whole image, the machine all, half or a quarter of it. A theorem predicts the pair with the soloist, with the partner and with a blind corrector from measures taken on the judge and the soloist alone (v2.1) | GPU runs |
| E9 | the shape of the noise: the networks of E1 tested with a frame of the same variance and another shape (two values, uniform, heavy tails). Where does the formula stop holding for right answers? (v2.1) | GPU runs |
| E10 | the chart that tells the truth: at test the frame has a sensitivity of its own (0, 0.5, 1, 1.5) and the judge, trained where it was worth 2, overweights it. Does a branch trained on the error of the sum remove only the excess the theorem says is there? (v2.3) | GPU runs |
| E11 | on the same files: at a fixed weight, is a partner that removes the whole push worse than one that removes only the excess, and worse than no correction at all? (v2.3) | GPU runs |
| E12 | on the same files: how many yes-or-no answers of the judge does a correction need, against judgments with a graded cue? (v2.3) | analysis |

What came out of the four new experiments (25 units each for E6, 75 for the others; medians
over five tasks and five seeds):

| | Criteria | In one line |
|---|---|---|
| E6 | 7 of 7 | the pair peaks at the weight the partner is built for (0.54, 1.12, 2.43 for 0.5, 1, 2) and reaches the same height as with the soloist; at equal weights the partner wins in 25 units of 25, at weight 8 the soloist does |
| E6, design reading | 3 of 4 | the partner beats the soloist up to 0.96, 1.83 and 3.03 for the branches built for 0.5, 1 and 2 (predicted 1, 2, 4). **Not confirmed (G2)**: at 0.285 of the weight due the partner wins in 17 units of 25, one short of the 18 required, by 0.01 turing |
| E7 | 5 of 6 | where the views share nothing the pair with the partner is worth 2.92 against 2.70 with the soloist, in 5 tasks of 5. **Not confirmed (V4)**: a partner trained from scratch learns a third of the task the judge does not cover, not the 0.7 required |
| E8 | 7 of 7 | when the machine sees a quarter of the image the pair with the partner is worth 2.87 against 2.49 with the soloist, in 5 tasks of 5; the gain is 1.02 times the one the theorem predicts; the blind corrector beats the soloist in 25 units of 25 |
| E9 | 3 of 5 | with a Gaussian frame the formula holds again on new seeds; with a two-valued frame it promises 3.69 where the pair gives 2.29, and the pair itself does not depend on the shape. **Not confirmed**: heavy tails move the ratio to 0.95, not below 0.9 (S2), and the uniform frame goes to 0.945, just below its band (S4) |

What came out of the three tests of protocol 2.3 (50 files, seeds 7 to 11; medians over the units):

| | Criteria that pass | In short |
|---|---|---|
| E10 | 6 of 6 | with the machine seeing a quarter of the image the branch built for the level removes 77 and 46 per cent of the judge's push where the excess of the theorem is 75 and 46 (frame worth 0.5 and 1), and almost nothing where the judge's weight is almost right (1.5). With the whole view it removes the same shares where the theorem says 42 and -23: the mismatch seen on old data, written as an expectation, is confirmed |
| E11 | 5 of 5 | at face value, with a frame worth 1 the partner that removes only the excess beats the one that removes everything by 0.18, in 25 units of 25; with a frame worth 1.5 the one that removes everything is 0.38 below the soloist with no branch, and gives fewer right answers than the judge alone in 25 units of 25. At the tuned weight the two partners differ by less than 1 per cent |
| E12 | 1 of 4 | **the prediction was too pessimistic.** With yes-or-no answers 25 judgments keep 90 per cent of the gain, against 15 with the graded cue: 1.6 times as many, where the registered band was 3 to 9 (Y2); Y1 misses by one unit (19 of 25) and Y3 fails with it. A factor was missing in the prediction, found afterwards: the bias left is kappa / (n (1 + B)^2), which gives 1.8 |

Eleven readings of data already collected, none of them a test (the last seven were added after the results of E6 to E9; the last one after those of E10):

| | What | Result file |
|---|---|---|
| the identity | with sensitivity as a signal-to-noise ratio, the formula of the pair is an algebraic identity of the sample moments: E1 and E5 test how Gaussian the cues are, not the formula | `results/identity.json` |
| the peak moves | on the files of E2 ter, the pair at fixed weights: the partner does not raise the maximum, it moves the weight at which the pair reaches it | `results/e6_trust_from_e2ter.json` |
| the three machines | the theorem of E8 on the files of E2 bis and E2 ter, unit by unit | `results/chart_from_e2.json` |
| who a partner is worth it for | the design reading of E6 (protocol § 12 bis) on the one branch of E2 ter: it beats the soloist up to about twice the weight it is built for, and not for someone who gives the soloist a quarter of the weight due | `results/e6_design_from_e2ter.json` |
| the push of the branch | on the files of E6: the branch cancels the judge's push with no systematic error, and against the same machine without the branch the break-even falls at 1.02, 2.05, 4.25 and 8.96 for the weights 0.5, 1, 2 and 4 (predicted 1, 2, 4, 8); the earlier break-even against the control soloist comes from the control | `results/e6_push.json` |
| the theorem on the different views | the theorem of E8 on the files of E7: where the views share nothing it predicts 2.91 for the pair with the partner, measured 2.92, and the partner beats both the soloist and an ideal blind corrector in 25 units of 25 | `results/chart_from_e7.json` |
| how many judgments | on the files of E8: a correction learned from 20 judgments with a known answer keeps about 94 per cent of the gain, from 100 about 99; the bias left is 1 / n | `results/e8_judgments.json` |
| when the trace says something true | the theorem extended to an informative trace that the judge overweights: only the excess is a bias. On the files of E8, resampled, a branch trained on the error of the sum removes 68 per cent of the push where the excess is 68 per cent | `results/e8_informative.json` |
| the theorem on people | on the 74 pairs of people and machines of turing-meter: with a bias of the size documented on asylum judges (B between 0.002 and 0.01) the best pair would grow by less than 0.1 per cent in the median; if half of a person's noise were a readable bias, by about 14 per cent. The bias of the people is not measured | `results/e4_private.json` |
| the pairs at face value, on old data | the look at the resampled files of E8 that sized the expectations of E11 | `results/e8_informative_face.json` |
| what the branch leaves | on the results of E10: at both views the branch leaves the frame about the slope of its own log-likelihood ratio (0.38, 0.89, 1.51 against 0.36, 0.79, 1.33), whatever the machine sees | `results/e10_trace_left.json` |

Results, with the outcome of every criterion: [`results/RESULTS.md`](results/RESULTS.md).

## What is here

- `src/cognitive_resonance/`: the library. `model.py` (the formulas: `tu`, `tu_app`,
  `w_star`, `q_factor`, band edges, the Fano form and the null), `estimators.py`
  (standardization, `d_a` from ranks with integer bootstrap weights, correlations, the
  tuned turing, measured band and null, cluster bootstrap over images), `raw.py` (the
  readers of the raw files, which check the contract of § 11), `simulate.py` (data from
  the model, for E0), `synthetic.py` (fake raw files, only for tests and dry runs).
- `analysis/`: one script per experiment, each reading `results/raw/` and writing a JSON
  file with all the measures and the outcomes; `figure_data.py` gathers the data of the
  figures. For version 2: `fixed.py` (the pair at fixed weights, with paired bootstrap),
  `trust.py`, `views.py`, `chart.py`, `shape.py` (E6 to E9), `identity.py`, and
  `design.py`, the design reading of E6 (protocol § 12 bis): who a partner is worth it for, and the rule
  by which a machine is chosen.
- `src/cognitive_resonance/chart.py`: the theorem of the three machines (the pair with
  the soloist, with the partner, and the band in which a blind corrector wins).
- `gpu/`: the scripts that train the networks and write the raw cues.
- `tests/`: the identities, the estimators on data simulated from the model, the readers.
- `results/`: the derived results. The raw cues (`results/raw/**/*.npz`) are not tracked.

## Rerunning it

```bash
uv sync --extra dev
uv run pytest -q
uv run python analysis/e0_simulation.py        # results/e0_simulation.json
uv run python analysis/e4_reread.py            # results/e4_reread.json
```

With the raw files of the GPU runs in `results/raw/`:

```bash
uv run python analysis/bench.py                # E1, results/e1_bench.json
uv run python analysis/null.py                 # E5, results/e5_null.json
uv run python analysis/rephase.py              # E2, results/e2_rephase.json
uv run python analysis/rephase_bis.py          # E2 bis, results/e2bis_rephase.json
uv run python analysis/rephase_ter.py          # E2 ter, results/e2ter_rephase.json
uv run python analysis/download_cifar10h.py    # CIFAR-10H votes into data/ (not tracked)
uv run python analysis/people.py --turing-meter PATH_TO_TURING_METER_CLONE   # E3
uv run python analysis/people_bis.py --turing-meter PATH_TO_TURING_METER_CLONE   # E3 bis
uv run python analysis/figure_data.py          # results/figure_data.json
uv run python analysis/results_md.py           # results/RESULTS.md
```

Version 2. The readings of old data need the raw files of E1, E2 bis and E2 ter:

```bash
uv run python analysis/identity.py             # results/identity.json
uv run python analysis/trust.py --ter          # results/e6_trust_from_e2ter.json
uv run python analysis/chart.py --e2           # results/chart_from_e2.json
uv run python analysis/design.py --ter         # results/e6_design_from_e2ter.json
```

The new experiments, once their raw files are in `results/raw/{trust,views,chart,shape}`
(`gpu/run.sh trust|views|chart|shape` writes them):

```bash
uv run python analysis/trust.py                # E6, results/e6_trust.json
uv run python analysis/views.py                # E7, results/e7_views.json
uv run python analysis/chart.py                # E8, results/e8_chart.json
uv run python analysis/shape.py                # E9, results/e9_shape.json
uv run python analysis/design.py               # E6, design reading (after trust.py), results/e6_design.json
uv run python analysis/push.py                 # E6, the push of the branch (after trust.py), results/e6_push.json
uv run python analysis/chart.py --views        # the theorem on the files of E7, results/chart_from_e7.json
uv run python analysis/judgments.py            # how many judgments a correction needs, results/e8_judgments.json
uv run python analysis/informative.py          # when the trace says something true, results/e8_informative.json
uv run python analysis/private_people.py --turing-meter PATH_TO_TURING_METER_CLONE   # results/e4_private.json
uv run python analysis/informative_face.py     # the look at old data behind § 17, results/e8_informative_face.json
uv run python analysis/trace.py                # E10 and E11 (raw files in results/raw/trace, from gpu/run.sh trace), results/e10_trace.json
uv run python analysis/yesno.py                # E12, results/e12_yesno.json
uv run python analysis/trace_left.py           # what the branch leaves (after trace.py), results/e10_trace_left.json
```

E3 and E3 bis import `compare_votes` from a local clone of turing-meter, and write its
commit in `results/e3_people.json` and `results/e3bis_people.json`. The analyses use at
most three processes and fit in a few minutes each on four cores.

## How to cite

endjneer (2026). *Risonanza cognitiva*. Version 1. Zenodo. https://doi.org/10.5281/zenodo.24140357

All versions: https://doi.org/10.5281/zenodo.24140355

Code and data: https://github.com/paithon-it/cognitive-resonance

## License

- Code under the MIT License, © 2026 paithon.it.
- The protocol (`PROTOCOL*.md`), `docs/` and the derived results (`results/`) under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); what derives from CIFAR-10H
  (`results/e3_people.json`, `results/e3bis_people.json`) keeps its license,
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- The data of the sources (CIFAR-10, CIFAR-10H) are not redistributed: the scripts read
  them from their sources, under their authors' licenses.
