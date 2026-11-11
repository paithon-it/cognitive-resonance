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

> Status: **private, in progress** (November 2026). The protocol was written before any
> run ([`PROTOCOL.md`](PROTOCOL.md), Italian original [`PROTOCOL.it.md`](PROTOCOL.it.md));
> what changed afterwards goes in its "Deviations" section.

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
  figures.
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

E3 and E3 bis import `compare_votes` from a local clone of turing-meter, and write its
commit in `results/e3_people.json` and `results/e3bis_people.json`. The analyses use at
most three processes and fit in a few minutes each on four cores.

## License

- Code under the MIT License, © 2026 paithon.it.
- The protocol (`PROTOCOL*.md`), `docs/` and the derived results (`results/`) under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); what derives from CIFAR-10H
  (`results/e3_people.json`, `results/e3bis_people.json`) keeps its license,
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- The data of the sources (CIFAR-10, CIFAR-10H) are not redistributed: the scripts read
  them from their sources, under their authors' licenses.
