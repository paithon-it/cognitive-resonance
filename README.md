# cognitive-resonance

> **English** · [Italiano](README.it.md)

**A machine built on its user, one that knows their errors and errs the other way: is the
pair worth far more than the two? That was the hypothesis. It did not hold.**

Against a person's error a machine can do one thing: cancel the part of it that can be
read from a trace. Nothing measured here oscillates or is amplified, and the name of the
repository is the name of the hypothesis. What is left is a unit for how much there is to
cancel, a measure of what it costs to learn it, and a bench of small networks that checks
the algebra. These are the measurements behind the paithon.it article
[*Risonanza cognitiva*](https://paithon.it/risonanza-cognitiva/) (in Italian, with an
English translation; doi:[10.5281/zenodo.24140355](https://doi.org/10.5281/zenodo.24140355)).

It is the sequel to [turing-meter](https://github.com/paithon-it/turing-meter) and to its
first experiment, `angle-between-errors`, the measurements behind
[*Intelligenza apparente*](https://paithon.it/intelligenza-apparente/) (in Italian;
doi:[10.5281/zenodo.24107649](https://doi.org/10.5281/zenodo.24107649)). There the angle
`phi` between the errors of a person and of a machine lay between 59 and 81 degrees, and
the region beyond 90 degrees was left unexplored. The first version of this project went
there by reading the pair as a resonant circuit ("The identity", below). That reading is
exact as algebra; it does not give a machine that errs opposite to a person where it cannot
read the person's error.

> Status: **protocol 2.6** (21 November 2026). The expectations of every test are written
> and pushed to this repository before its run; the protocol is in
> [`PROTOCOL.md`](PROTOCOL.md) (Italian original [`PROTOCOL.it.md`](PROTOCOL.it.md)), and
> what changed after a run is in its "Deviations" section. They are written by the same
> system that then runs the tests, minutes before, and on the bench with thresholds sized
> on earlier runs of the same tasks: they keep a threshold from being chosen afterwards,
> and they are not preregistrations in the full sense. Code, analyses and reviews come from
> language models; no human reader has checked this work yet.
>
> | Tests | Protocol | Criteria that pass |
> |---|---|---|
> | E6 to E9, on the bench | 2.1 and 2.2, 19 November | 25 of 29 |
> | E10 to E12, on the bench | 2.3, 20 November | 12 of 15 |
> | E13, on people | 2.4, 20 November | 2 of 3 |
> | E14 and E15, on the bench | 2.5, 21 November | 3 of 3 and 3 of 3 |
> | E16, a language model as the judge | 2.6, 21 November | 2 of 3 |

**Names.** The article calls *generic machine* what the code and the older tables call the
soloist (`M`), *tailored machine* the partner (`Mramo`), *corrector* the blind corrector
(`Mblind`), and *judge* the network that stands in for the person.

## In short

- **From the history of choices, no measurable gain.** On 1,968 people of the Confidence
  Database the bias readable from the history of their own trials is worth 0.015 of the rest
  of the noise in the median. Removing it leaves the right answers where they were (74.7 per
  cent before, 74.8 after; they go down in 32 per cent of the people and up in 29). The
  graded judgment gains 1.1 per cent of sensitivity, an effect of a scale with few steps. It is
  a lower bound: one weak trace, laboratory tasks, no machine beside the person (E13; a
  second implementation written from the protocol alone gives the same numbers).
- **Knowing a person costs data.** On the people with at least 800 trials to learn from, a
  corrector learned from a person's own trials takes away 2.7 per cent of sensitivity
  with 50 trials, gives nothing with 100, and matches one learned from the other people only
  at 800 (`results/e13_people_more.json`, a reading added afterwards).
- **Where the bias is strong, correcting counts and is not enough.** A small open language
  model answering 6,000 two-option questions follows the opinion of whoever asks: B = 0.48,
  and right answers fall from 75.2 to 67.5 per cent. A corrector that knows only the
  opinion brings them back to 71.4 and gives back 0.49 of the sensitivity lost, where the
  expectation said at least half (E16; Phi-3.5 mini: B 0.63, share given back 0.40; Qwen2.5 7B: B 0.04, share given back 0.30).
  The sentence with the opinion costs from 2 to 8 per cent of sensitivity whichever opinion it carries, and a corrector
  that also knows the judgment it corrects gives back about as much as the linear one (0.48, 0.51 and 0.34;
  `results/e16_llm_judge_more.json`, readings added afterwards).
- **What a machine built on its user can give.** It gives back the part of what the person
  knows, and the machine does not, that the person's readable bias was covering, and nothing
  more: in squared sensitivity the pair is worth `d_m^2 + P` with a machine that also reads
  the bias and `d_m^2 + P (1 - r^2) / (1 - r^2 + B)` with one that does not
  (`cognitive_resonance.chart.pair_information`). It is the algebra of suppressor variables;
  the bench of small networks checks that training gets where the algebra says, on five
  tasks, with a planted bias of about 0.7.
- **Where the corrector stops winning.** Beside the judge a corrector that knows nothing of
  the task beats the generic machine until the machine sees between 14 and 19 columns
  of 28; the proposition puts the crossing a median of 0.2 columns away (E15). The condition is a band
  (`chart.blind_band`): when the two noises resemble each other more than the bias weighs (`r^2 > B`), a very weak
  machine, taken with a negative weight, beats the corrector again. That lower edge has not been tested.
- **What counts as a bias depends on the pair.** A trace that says something true deserves a
  weight that grows with what the machine sees; only the excess is a bias, and a machine that
  removes all of it does harm (E10, E11).
- **What a strong shortcut leaves.** A network trained with a shortcut learns less from the
  image. With the strongest shortcut, removing the frame from the output keeps 0.53 of the
  sensitivity of its twin trained without it; retraining the last layer keeps 0.66; training
  three times as long, 0.74. With a shortcut of strength 1 almost everything comes back
  (0.99) (E14).
- **Opposite errors.** Errors in antiphase are a hidden common cause entering with opposite
  signs: a machine can build them only on the part of the person's error it can read, and
  there building them is correcting.

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
| E13 | on people: how much of the judgment of a person can be read from the history of their own trials, and what does a corrector that sees no stimulus give back? The Confidence Database (CC0), 180 files split into exploration and confirmation by the hash of their name before any was opened (v2.4) | analysis, public data |
| E14 | what a shortcut leaves unlearned: the "plus" networks of E1 trained for 30 and 90 epochs, with the frame removed from the output and with the last layer trained again where the frame says nothing (v2.5) | GPU runs |
| E15 | the threshold crossed: the generic machine of E8 trained again at seven widths between half and the whole image, to see where it overtakes the corrector beside the judge, and where the proposition says it should (v2.5) | GPU runs |
| E16 | a judge with a bias nobody planted: an open language model on two-option questions of MMLU, with the opinion of whoever asks as the trace; the same question with no opinion gives the clean judgment (v2.6) | GPU runs, public data |

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
| E13 | 2 of 3 | **the readable bias of people is small.** On the 1,968 people of the 44 confirmation files the median B is 0.015 (0.011 to 0.018), where the bench had about 0.7 planted; with the trials shuffled it is zero, and it is above zero in 42 files of 44 (P1). A corrector learned on the first half of the trials of a person gives 1.1 per cent of sensitivity on the second half (1.006 to 1.016; P2 asked for less than 2), and no more than a corrector learned on the other people of the same experiment. P3 does not pass: the corrector gives a little more than the formula, because on a confidence scale of a few steps the trace reorders trials that fell on the same step. Counting the answers themselves, the corrector gives no more right answers (74.7 per cent before, 74.8 after). Checks added afterwards: a second implementation written from the protocol alone gives the same numbers; without the ten doubtful files the medians do not move (`results/e13_people_audit.json`) |

What came out of the three tests of protocols 2.5 and 2.6 (21 November):

| | Criteria that pass | In short |
|---|---|---|
| E14 | 3 of 3 | **part of what a shortcut seemed to leave unlearned comes back.** With the strongest shortcut the network keeps 0.53 of the sensitivity of its twin with the frame removed from the output, 0.66 with the last layer trained again (learned and not used), and 0.74 after a training three times as long (learned later): between 0.14 and 0.30 is still missing, depending on the task. At strength 1 almost nothing is missing after 90 epochs (0.99). The last layer trained again costs 0.03 of its own on the network with no shortcut |
| E15 | 3 of 3 | **the threshold is where the proposition puts it.** In every task the generic machine overtakes the corrector once, between 14.1 and 19.3 columns of 28; in 4 tasks of 5 the predicted crossing is at most 1.6 columns away (median 0.2); in the fifth the proposition has no crossing in the range |
| E16 | 2 of 3 | **a strong bias that is not an addend.** For the primary judge B = 0.48; right answers 75.2 with no opinion, 67.5 with it, 71.4 with a linear corrector, which gains 19 per cent of sensitivity where the proposition says 21. **Not confirmed (J2)**: it gives back 0.487 of the sensitivity lost, below the 0.5 written beforehand: the opinion costs more than what the trace lets one read, and the push is 1.7 times larger on the doubtful questions. Without criteria: Phi-3.5 mini B = 0.63, share given back 0.40; Qwen2.5 7B B = 0.04, share given back 0.30, with the push 8.5 times larger on the doubtful questions |

Thirteen readings of data already collected, none of them a test (the last nine were added after the results of E6 to E9; the last three after those of E10):

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
| the turing of the pair, one factor and two addends | the theorem divided by what the judge is worth alone: Tu^2 = (1 + B)(q^2 + p). The factor is the bias cancelled; inside the sum, what the machine knows of the task and the private part of the judge. In d'^2 the parts simply add up, D^2 = d_m^2 + P (`cognitive_resonance.chart.pair_information`). On the three benches it gives the measured turings (1.34 against 1.33 with the partner, 1.32 against 1.33 with the blind machine, where the machine sees a quarter of the image). The same script measures the other way of removing the bias: a judge trained without it, with the soloist, gives more right answers than the judge with the partner in 25 pairs of 25 at every view, at a turing between 1.01 and 1.04 | `results/pair_factors.json` |
| what a shortcut costs twice | on the files of E1: a network trained where the frame says something about the class, tested where it says nothing. Removing the frame from the output gives back what the bias covered, not everything the network lost while it leaned on the frame (E14 tells what was learned and not used, and what is learned later): with the frame removed it keeps 0.94 of the sensitivity of the same network trained without shortcut at mu = 1, 0.80 at 1.5, 0.53 at 2, falling with mu in 5 tasks of 5 | `results/shortcut.json` |
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
- For protocols 2.4 to 2.6: `src/cognitive_resonance/readable.py` and `analysis/people_bias.py` (E13, with
  `people_bias_second.py`, `people_audit.py`, `people_more.py` and `people_by_hand.py`, which prints every step for one
  person), `analysis/nonadditive.py`, `gpu/shortcut.py` and `analysis/unlearned.py` (E14), `gpu/crossing.py` and
  `analysis/crossing.py` (E15), `gpu/judge.py`, `src/cognitive_resonance/opinion.py` and `analysis/llm_judge.py` (E16).
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
uv run python analysis/factors.py              # the turing of the pair in its parts, on the files of E1, E8 and E10, results/pair_factors.json
uv run python analysis/shortcut.py             # what a shortcut costs twice, on the files of E1, results/shortcut.json
uv run python analysis/people_more.py          # E13, the curve of personalisation, the threshold and the map (after download_confidence_db.py)
uv run python analysis/people_by_hand.py       # E13, every step for one person, to redo it by hand
uv run python analysis/nonadditive.py          # a push that depends on uncertainty, simulated
uv run python analysis/manifest_later.py       # the manifests of E14 to E16 from the per-unit records
uv run python analysis/unlearned.py            # E14, results/e14_unlearned.json
uv run python analysis/crossing.py             # E15, results/e15_crossing.json
uv run python analysis/llm_judge.py test       # E16, results/e16_llm_judge.json
uv run python analysis/llm_judge_more.py       # E16, where the loss comes from and a corrector that knows more
uv run python analysis/download_confidence_db.py   # the Confidence Database into data/confidence_db (CC0, 255 MB)
uv run python analysis/people_bias.py          # E13, results/e13_people_bias.json
uv run python analysis/people_bias_second.py   # E13 again, written from the protocol alone, results/e13_second_implementation.json
uv run python analysis/people_audit.py         # E13, the files checked one by one, results/e13_people_audit.json
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
