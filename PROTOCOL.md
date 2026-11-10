# Cognitive resonance: protocol

Version 1.2, 10 November 2026. Written and recorded in the repository **before any run**:
hypotheses, estimators and criteria are below, and whatever changes afterwards goes in
the "Deviations" section at the end, with the date and the reason. Original language:
Italian (`PROTOCOL.md` is the translation).

## 1. What we want to know

This is the sequel to the `angle-between-errors` experiment of
[turing-meter](https://github.com/paithon-it/turing-meter). There the angle `phi` between
the errors of a person and of a machine, measured on four public collections, had a
median between 59 and 81 degrees; the ideal of the pair was restricted to non-negative
weights, and the region beyond 90 degrees was not explored. The formula for apparent
turings,

    Tu_app = sqrt(1 + k^2 - 2 k cos(phi)) / sin(phi),      k = dM / dH,

grows without bound as `phi` goes toward 180 degrees: errors in antiphase cancel out. We
call this condition **cognitive resonance**, and there are six questions.

- **Q1.** Does the promise of the formula beyond 90 degrees come true on real networks,
  with real images, or does it stay in the Gaussian model?
- **Q2.** As the weight given to the machine varies, does the turing have the shape and
  the width of a resonance curve?
- **Q3.** Can antiphase be built by training a machine on the errors of a mind
  ("rephasing")? And is the best machine for the pair the best one on its own?
- **Q4.** With real people, does the angle open?
- **Q5.** Where do the pairs already measured sit on this scale?
- **Q6.** Next to the peak, is there a null, a weight at which the pair is worth nothing? Under
  which conditions does it appear? (Added in version 1.2: see § 8 bis.)

## 2. The model

Two judges decide between two classes. Each has a graded cue: `x_H = s dH/2 + n_H`,
`x_M = s dM/2 + n_M`, with `s = +1` or `-1` according to the class, noises with unit
variance and correlation `rho = cos(phi)`. The pair adds the cues with a weight `w >= 0`
on the machine, `x_H + w x_M`. In units of the person:

    Tu(w)^2 = (1 + w k)^2 / (1 + w^2 + 2 rho w)

The maximum is `Tu_app`, at the weight `w* = (k - rho) / (1 - rho k)`.

**Resonance identity.** For `k = 1`, with `Omega = sqrt(w)`:

    Tu(w)^2 - 1 = tan^2(phi/2) / (1 + Q^2 (Omega - 1/Omega)^2),     Q = 1 / (2 cos(phi/2))

It is the resonance curve of a series RLC circuit, with `Omega` in place of the
normalized frequency. At the peak `Tu = 2 Q = 1 / cos(phi/2)`. We call `Q` the **quality
factor** of the pair, and we define it this way for every `k`. The **band** is the set of
weights in which `Tu^2 - 1` stays above half of its maximum; for `k = 1` the edges are
`Omega - 1/Omega = +-1/Q`, and the width is measured as the ratio `w_high / w_low`.

Other quantities, with the names of turing-meter: `k = dM/dH`; yield on the machine
`dS/dM`; useful angle `arccos(min(k, 1/k))`.

## 3. Estimators, fixed beforehand

They apply to judges whose whole cue is available (networks); for real people the
turing-meter instrument is used (§ 7).

- **Cue.** `z` = log-odds toward class 1. Each judge is standardized by dividing by the
  within-class standard deviation, estimated on the **validation** set.
- **Sensitivity.** Primary: `d_a = sqrt(2) z(AUC)`, without a threshold. Secondary, as in
  turing-meter: yes/no convention, `z(hit) + z(correct rejections)`, at threshold zero.
- **Correlation.** Primary: Pearson correlation of the standardized cues, within the
  true class, pooled over the two classes (residuals from the class mean). Bridge to
  turing-meter: tetrachoric correlation of the correct answers within class. Both are
  reported, and the distance between them is a result. `phi = arccos(rho)`.
- **Turing at weight w.** `Tu(w) = d_a(z_H + w z_M) / d_a(z_H)`, on a grid of 49 weights
  evenly spaced in logarithm between 1/64 and 64.
- **Tuned turing.** The weight is chosen on half of the images and the turing is read on
  the other half, then they are swapped (mean of the two). The maximum on the same grid
  and on the same data is optimistic, and is not used as a measure.
- **Prediction.** `Tu_pred(w)` of § 2 with the measured `k` and `rho`: no free parameter.
- **Measured band.** On the weights of the grid, by linear interpolation in `ln w`.
- **Uncertainty.** Bootstrap over images (all the presentations of the same image stay
  together), 1,000 resamples, 95 per cent percentile intervals. In each resample the
  tuning weight is chosen again, on a sparser grid (13 weights); if computing time is not
  enough, the one chosen on the whole data is kept, and this is declared in the results.

## 4. E0. The instrument on the model

Data generated from the model of § 2, with the size of the bench (2,000 images times 5
presentations), `k` in {0.7; 1; 1.5} and `rho` in {0.6; 0.3; 0; -0.3; -0.6; -0.8; -0.9;
-0.95}; 200 experiments per situation for the bias. Coverage costs a whole resampling per
experiment, and is checked on a reduced set: `k = 1`, `rho` in {0; -0.6; -0.9}, 100
experiments, 300 resamples.

- **Criterion E0a.** Bias of the tuned turing within 5 per cent of the true value for
  `rho >= -0.9`.
- **Criterion E0b.** On the reduced set, the 95 per cent interval contains the true value
  between 90 and 98 times out of a hundred.
- For `rho = -0.95` we report how far off the instrument is, without a criterion: there
  the pair errs so rarely that the sensitivity is poorly estimated.

## 5. E1. The bench: two opposite biases

Small networks, real images, and a knob that sets the angle.

- **Tasks.** Five pairs of CIFAR-10 classes that get confused: cat/dog, deer/horse,
  automobile/truck, airplane/ship, bird/frog. Each pair is a binary task: 10,000 training
  images (1,000 held out for validation), 2,000 test images.
- **Network.** The small convolutional network of turing-meter's `gpu/silicon.py`, with a
  binary output. Epochs and learning rate are fixed on a pilot at `mu = 0`, on the
  validation set, and then do not change.
- **The spurious cue.** Each image carries a two-pixel frame with a brightness
  proportional to a number `a`. In training `a` depends on the class: `a ~ N(+mu, 1)` for
  one class and `N(-mu, 1)` for the other. The "plus" networks see it one way, the
  "minus" networks the opposite way. In test and in validation `a` does not depend on the
  class: `a ~ N(0, 1 + mu^2)`, the same variance seen in training. The pilot checks that
  the frame can be read (a linear probe on the image recovers `a` with `R^2 > 0.9`).
- **The knob.** `mu` in {0; 0.25; 0.5; 0.75; 1; 1.5; 2}.
- **Judges.** For each task, `mu` and seed: a "plus" network (the person), a "minus" one
  (the machine), a second "plus" with another seed (another person with the same bias).
  Three seeds. **Opposite** pairs (plus, minus) and **equal** pairs (plus, plus).
- **Test.** Each test image is presented 5 times with different values of `a`, the same
  for the judges of a pair: 10,000 presentations per task.
- **Reference.** At `mu = 0` the networks have no bias: their sensitivity alone and in a
  pair is the absolute reference.

Hypotheses and criteria:

- **H1a, the knob works.** In the opposite pairs `rho` falls with `mu` (rank correlation
  between `mu` and `rho` not above -0.9, as a median over tasks and seeds), and at the two
  highest values of `mu` the interval of `phi` lies entirely above 90 degrees in at least
  4 tasks out of 5.
- **H1b, beyond Pythagoras.** Where the interval of `phi` lies above 90 degrees, the tuned
  turing exceeds `sqrt(1 + k^2)`, with the lower end of its interval above it, in at
  least 80 per cent of the pairs.
- **H1c, the law.** In the same pairs the ratio between the tuned turing and `Tu_app` has
  a median between 0.9 and 1.1, and the mean absolute relative error is smaller than
  that of the prediction at 90 degrees.
- **H1d, the tuning.** In the same pairs the logarithm of the measured bandwidth has a
  median within 30 per cent of the predicted one, and the width falls with `phi` (rank
  correlation not above -0.7).
- **H1e, the same bias.** In the equal pairs `rho` rises with `mu`, and at the highest
  `mu` the median tuned turing does not exceed 1.1.
- **H1f, nothing is created.** At every `mu` the sensitivity of the tuned opposite pair
  does not exceed by more than 5 per cent that of two networks without bias put
  together: resonance gives back what the bias was covering, it does not add cues.

## 6. E2. Rephasing

Given a mind with a bias, we train the machine that compensates for it by looking only
at the cues of that mind, without telling it what the bias is.

- **The person H.** A "plus" network with `mu = 1`, trained on half A of the training
  set.
- **The machines.** Trained on half B, with a frame that does **not** depend on the
  class. `M_0`: normal training (the best on its own). `M_lambda`: to the loss we add
  `lambda` times the Pearson correlation, within the batch, between the cue of M and that
  of H toward the true class, with the class means removed; H is frozen and looks at the
  same presentation. `lambda` in {0; 0.3; 1; 3; 10}.
- **Controls.** `H'`: another "plus" network with `mu = 1`, trained on half B (a similar
  person). Two seeds, five tasks.
- **Measures.** `phi(H, M_lambda)`, `k`, sensitivity of `M_lambda` alone, tuned turing
  and absolute sensitivity of the pair, slope of the cue of M on `a` within class (the
  bias M has learned; that of H is positive).

Hypotheses and criteria:

- **H2a.** `phi(H, M_lambda)` grows with `lambda` (rank correlation at least 0.9, as a
  median) and for at least one `lambda` its interval lies above 90 degrees, in at least
  3 tasks out of 5.
- **H2b, the best partner is not the best soloist.** There is a `lambda > 0` for which
  `M_lambda` alone is less sensitive than `M_0`, and the pair `(H, M_lambda)` is more
  sensitive than the pair `(H, M_0)`, with intervals of the two differences that exclude
  zero, in at least 3 tasks out of 5.
- **H2c.** Where H2a holds, the slope of the cue of `M_lambda` on `a` has the opposite
  sign to that of H.

If H2a does not pass, we report how far the angle has moved.

## 7. E3. Real people

- **Data.** CIFAR-10H (Peterson and colleagues, 2019): about 51 human votes for each of
  the 10,000 test images of CIFAR-10, ten classes. License CC BY-NC-SA 4.0: it is
  downloaded with a script, not redistributed.
- **Base machine.** A ResNet-18 for CIFAR trained on the 50,000 training images, at least
  93 per cent on the test set.
- **Natural angle.** turing-meter's `compare_votes` (pinned version), votes of the people
  against the verdict of the machine, `m = 10`.
- **Machine in antiphase.** The 10,000 images are split into two halves per class. On one
  half the base machine is fine-tuned with the normal loss plus `lambda` times the
  correlation between its cue toward the true class and that of the people,
  `logit((h + 0.5) / (n + 1))` with `h` correct votes out of `n`; `lambda` in {0; 1; 3;
  10}, where 0 is the control. It is measured on the other half, then they are swapped.
  Two seeds.
- **Measures.** Angle, accuracy of the machine, apparent turings, and a combiner
  (logistic regression on the vote share and the probability of the machine, with
  cross-validation) for the turing of the group.

Hypotheses:

- **H3a.** For some `lambda > 0` the angle on the unseen half is more open than with
  `lambda = 0`, with an interval of the difference (paired bootstrap over images) that
  excludes zero. The stated expectation is uncertain: with five thousand images and one
  person in twenty getting it wrong, it may well not open.
- **H3b.** The angle stays below 90 degrees for every `lambda`.

## 8. E4. The pairs already measured

From the angles published in "Intelligenza apparente" we compute `Q = 1/(2 cos(phi/2))`
and the apparent turing the same pair would have at equal sensitivity. It is a
rereading, without criteria: ChatBench 81 degrees, αNLI 64, HAIID text 65, HAIID images
78, objects 59; two people with each other 66 and 54; two networks with each other
between 22 and 43.

## 8 bis. E5. The null

Added on 10 November 2026 with version 1.2, at the author's request, with the GPU runs
already started and **before the analysis of the raw data was written or looked at**. It
needs no new runs: it uses the files of E1.

**The model.** If the weight can have any sign, the signed turing is
`Tu(w) = (1 + w k) / sqrt(1 + w^2 + 2 rho w)`, and its square is a Fano profile:

    Tu(w)^2 = k^2 (q + eps)^2 / (1 + eps^2)
    eps = (w + cos(phi)) / sin(phi),      q = (1 - k cos(phi)) / (k sin(phi))

- **The null** is at `w0 = -1/k` (`eps = -q`), for every angle: the two signals cancel
  and only the noise is left. Beyond the null the turing is negative: the pair errs
  systematically.
- **The peak** is at `eps = 1/q` and is worth `Tu_app`.
- `q = 0` at the useful angle, `cos(phi) = 1/k`: the peak disappears and only the null
  is left.
- **The width of the null** is the interval of weights in which `|Tu| < 1/2`. For
  `k = 1` it grows with the angle, the opposite of the band of the peak.
- **Two conditions.** With `k > 0` (both above chance) the null requires a negative
  weight: trusting in reverse. With `k < 0` (one of the two below chance, that is, in
  systematic error) it appears at a positive weight, `w0 = 1/|k|`.

**Estimators.** Signed sensitivity: `d_a = sqrt(2) z(AUC)`, negative when the AUC is
below 0.5. Extended grid: the 49 weights of § 3 also with the minus sign. Measured null:
the weight at which the signed turing crosses zero, by linear interpolation between the
two neighbouring grid points. Measured width: the interval in which `|Tu| < 1/2`, by
interpolation. The predictions use the measured `k` and `rho`.

Hypotheses and criteria:

- **H5a, the null is there and where it should be.** In all the pairs of the bench,
  opposite and equal, the signed turing crosses zero at a negative weight, and the
  product `w0 k` has a median between -1.1 and -0.9.
- **H5b, it is deep.** At the predicted weight `-1/k`, computed on half of the images,
  the absolute value of the turing measured on the other half has a median below 0.1.
- **H5c, the width follows the angle, the opposite of the peak.** The logarithm of the
  measured width has a median within 30 per cent of the predicted one, and the width
  grows with `phi` (rank correlation at least 0.7 over all the pairs of the bench).
- **H5d, the null at positive trust.** In the opposite pairs with `mu >= 1` we take the
  presentations in which the frame points against the truth for the machine (the
  "minus" network) beyond one standard deviation of `a`; the direction is derived from
  the sign of the slope of the machine's cue on `a`, it is not assumed. The cues are
  standardized on the validation set with the same selection. Where the sensitivity of
  the machine is negative, with an interval that excludes zero, the signed turing
  crosses zero at a positive weight, and the ratio between the measured weight and the
  predicted one, `1/|k|`, has a median between 0.8 and 1.25.
- **H5e, the whole curve.** In the pairs of the bench the mean absolute error between
  the measured and the predicted signed turing, on the extended grid, divided by
  `Tu_app`, has a median below 10 per cent.

## 9. What would count against the idea

- If H1c does not pass, the formula does not hold beyond 90 degrees even on a bench
  built to prove it right: resonance stays in the model.
- If H1d does not pass, the tuning curve is not that of a circuit: the identity holds
  for Gaussian cues and not for networks.
- If H2a does not pass, antiphase cannot be learned from errors: it can only be planted.
- If H3a does not pass, with real people we do not know how to build it.
- If H5a or H5b do not pass, the null is a property of the model and not of the networks.

## 10. Order, times, cleanup

Order of priority: E0, E1, E2, E3. One L4 GPU with 24 GB; a ceiling of two and a half
hours of computing in all. If the timing pilot says that E1 exceeds 60 minutes we go
down to two seeds, then to the first three tasks, and we write it in the Deviations. On
the server we work in a single folder, which is deleted at the end.

## 11. Raw data: the contract

The GPU scripts write only case-by-case cues; all the measures are made by the analysis,
afterwards, from these files. One `.npz` file per unit, in `results/raw/`:

- `bench/{task}_mu{mu}_s{seed}.npz`: `y` (0/1), `image` (index of the test image), `a`
  (frame shown), `z_plus`, `z_minus`, `z_plus2` (log-odds toward class 1); the same with
  the prefix `val_` on the validation set.
- `rephase/{task}_s{seed}.npz`: `y`, `image`, `a`, `z_H`, `z_H2`, and `z_M` of shape
  (number of lambdas, N) with `lambdas`; plus `val_`.
- `people/fold{f}_s{seed}.npz`: `image` (index in the CIFAR-10 test set), `y` (0-9),
  `logits` of shape (number of lambdas, N, 10) with `lambdas`; and
  `people/base_s{seed}.npz` with the logits of the base machine on all 10,000.
- Each folder has a `manifest.json`: hyperparameters, times, library versions, control
  accuracies.

## Deviations

None. Version 1.1 is from the same day as version 1 and comes before any run: it
specifies the computations of E0 and of the bootstrap, which in version 1 could not have
been run in the time available. Version 1.2, from the same day, adds question Q6 and
experiment E5 on the null (§ 8 bis): it was written with the GPU runs started, but
before any analysis of the raw data, and it changes nothing of what was there.
