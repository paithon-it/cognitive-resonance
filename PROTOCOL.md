# Cognitive resonance: protocol

Version 2.1, 19 November 2026. Written and recorded in the repository **before any run**:
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
call this condition **cognitive resonance**, and there are ten questions.

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
- **Q7.** At equal value of the pair, does a machine built on the person move the weight at
  which the pair works? And can that weight be decided by training it? (Added in version
  2.0: see § 12.)
- **Q8.** If the person sees something the machine does not see, does the built partner
  beat the best soloist? (Added in version 2.0: see § 13.)
- **Q9.** Can the value of the built partner be predicted before building it, from what the
  judge knows and the machine does not? And when does a machine that knows only the judge
  beat a machine that knows the task? (Added in version 2.1: see § 14.)
- **Q10.** How much of the agreement between measure and formula seen in E1 depends on the
  shape of the noise we injected? (Added in version 2.1: see § 15.)

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

## 6 bis. E2 bis. The partner trained for the pair (exploratory)

Added with version 1.3, **after** seeing the data of E2 and E3: it is not a test
registered beforehand, and it must be read as such. The expectations below are written
before the run.

**Why.** In E2 the penalty on the correlation was satisfied in the cheapest way: the
machine flipped its own cue (`k < 0`). It is a fake antiphase, because the pair
`(k, rho)` and the pair `(-k, -rho)` are the same pair: it is enough to believe the
machine in reverse. So we try the recipe that does not allow the shortcut: the machine
learns from the error of the **sum**.

- **H and `M_0`** as in E2: same tasks, same seeds, same halves, same frame.
- **`M_pair`.** The same network as `M_0`, half B, a frame that does not depend on the
  class. The loss is the cross-entropy of the summed cue `z_H + z_M`; H is frozen and
  looks at the same presentation.
- **`M_frame`.** Like `M_pair`, but it sees only the frame: the inside of the image is
  set to zero (in normalized units). On its own it cannot know anything about the class:
  it is the case `k = 0`, for which the model predicts `Tu = 1 / sin(phi)`.
- **Measures** as in E2, with the same estimators and the same bootstrap.

Expectations:

- **B1.** `M_pair` and `M_frame` do not fall below chance (for `M_pair` the interval of
  `d_a` lies above zero; for `M_frame` it contains it) and the slope of their cue on `a`
  has the opposite sign to that of H, in every unit.
- **B2.** `phi(H, M_pair) - phi(H, M_0) > 0`, with an interval that excludes zero in
  every seed, in at least 3 tasks out of 5.
- **B3, the best partner is not the best soloist.** `M_pair` alone is less sensitive than
  `M_0`, and the pair `(H, M_pair)` is more sensitive than the pair `(H, M_0)`, with
  intervals of the two differences that exclude zero in every seed, in at least 3 tasks
  out of 5.
- **B4, the machine that knows nothing.** The tuned turing of the pair `(H, M_frame)` has
  its interval above 1 in every seed in at least 3 tasks out of 5, and its median is
  within 10% of `1 / sin(phi)`.
- **B5.** For `M_pair` the tuned turing is, in median, within 10% of the apparent turing
  computed from `k` and `phi`.

Raw data: `rephase_bis/{task}_s{seed}.npz` with `y, image, a, z_H, z_M` (one row per
machine, in the order of `machines` = `M0, Mpair, Mframe`) and the same with the prefix
`val_`; `manifest.json`.

## 6 ter. E2 ter. The soloist that learns the person (exploratory)

Added with version 1.4, on 11 November 2026, **after** seeing all the results and after a
review of the draft of the article: it is not a test registered beforehand. The
expectations below are written before the run.

**Why.** In E2 bis the machine trained from scratch on the error of the sum learned almost
nothing but to remove the bias of H, and of the task it knows almost nothing: the best
soloist remained the best partner (B3 does not pass). What is left is to start from the
soloist, so that it does not have to choose between the task and the correction.

- **H and `M_0`** as in E2 and E2 bis: same tasks, same seeds, same halves, same frame.
- **`M_0+`, the control.** `M_0` fine-tuned on half B with its normal loss: 10 epochs, a
  learning rate one fifth of the pilot's, cosine decay to zero. It is there so as not to
  mistake ten more epochs for an effect of the recipe.
- **`M_due`** (Italian for "two": two losses). `M_0` fine-tuned in the same way, with the
  mixed loss: cross-entropy of its own cue plus cross-entropy of the summed cue
  `z_H + z_M`; H is frozen and looks at the same presentation.
- **`M_ramo`** (Italian for "branch"). `M_0` frozen, plus a branch: a new network that sees
  only the frame (like `M_frame`), trained with the pilot's hyperparameters on the
  cross-entropy of `z_H + z_M0 + z_ramo`. The cue of the machine is `z_M0 + z_ramo`.
- **Measures** as in E2 bis, with the same estimators and the same bootstrap; in addition,
  the sensitivity of the tuned pair against that of `M_0+` alone.

Expectations:

- **T1, it does not forget.** `d_a(M_due)` is at least 0.9 times `d_a(M_0+)`, in median
  over the units, and the slope of the cue of `M_due` on `a` has the opposite sign to
  that of H in every unit.
- **T2, the angle opens.** `phi(H, M) - phi(H, M_0+) > 0`, with an interval that excludes
  zero in every seed, in at least 3 tasks out of 5; it is judged for `M_due` and for
  `M_ramo`.
- **T3, the partner beats the soloist.** The tuned pair `(H, M)` is more sensitive than
  the pair `(H, M_0+)`, with an interval of the difference that excludes zero in every
  seed, in at least 3 tasks out of 5; it is judged for `M_due` and for `M_ramo`. Stated
  expectation: it passes, narrowly. The ceiling is the pair without bias of E1
  (`mu = 0`).
- **T4.** The tuned turing is, in median, within 10% of the apparent turing computed
  from `k` and `phi`, for both machines.
- **T5, the pair against the machine.** The tuned pair `(H, M_due)` is more sensitive
  than `M_0+` alone, with an interval of the difference that excludes zero in every
  seed, in at least 3 tasks out of 5. Stated expectation: uncertain.

Raw data: `rephase_ter/{task}_s{seed}.npz`, like `rephase_bis/`, with `machines` =
`M0, M0plus, Mdue, Mramo`; `manifest.json`.

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

## 7 bis. E3 bis. The recipe of the sum with real people (exploratory)

Added with version 1.4, under the same conditions as E2 ter: after seeing all the
results, with the expectations written before the run.

**Why.** E3 tried on people the recipe of punishment, the one that did not work on the
bench. The recipe of the sum came later, and it has not been tried with people.

- **Data, base machine, halves and fine-tuning** as in E3: same seeds, same two halves, 8
  epochs at learning rate 0.01 with a cosine decay, starting from the base machine.
- **The people's cue**, over ten classes: `u_c = log((v_c + 0.5) / (n + 5))`, with `v_c`
  the votes for class `c` out of `n`.
- **Three recipes**, with their Italian names. `controllo` (control): cross-entropy of
  the logits of the machine (it is `lambda = 0` of E3, done again). `somma` (sum):
  cross-entropy of `u + z_M`. `mista` (mixed): the sum of the two.
- **Measures** as in E3 (angle, accuracy of the machine, apparent turings, turing of the
  group with the combiner), on the half not used for the fine-tuning. In addition, the
  accuracy of `argmax(u + z_M)`, next to that of the majority of the votes.

Expectations:

- **P1, it does not collapse.** With `mista` the accuracy of the machine alone does not
  fall more than 2 points below that of the control, in every half and in every seed.
  For `somma` no expectation: it is reported.
- **P2, the angle opens.** For at least one of the two recipes the angle on the unseen
  half is more open than in the control, with an interval of the difference (paired
  bootstrap over images) that excludes zero, the same recipe in all seeds.
- **P3, the apparent turings rise.** For the same recipe as P2 the apparent turings
  exceed those of the control, with an interval of the difference that excludes zero in
  all seeds. It is the point where punishment failed. Stated expectation: uncertain,
  with five thousand images and one person in twenty getting it wrong.
- **P4.** The angle stays below 90 degrees for both recipes (point estimate).
- **P5, the group.** The turing of the group with the combiner, for `mista`, exceeds that
  of the control with an interval of the difference that excludes zero in all seeds.
  Stated expectation: uncertain.

Raw data: `people_bis/fold{f}_s{seed}.npz` with `image`, `y`, `logits` of shape
(3, N, 10) and `recipes` = `controllo, somma, mista`; `people_bis/base_s{seed}.npz`;
`manifest.json`.

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
- If F1 does not pass, the advantage of the partner at equal weight seen in E2 ter was an
  accident of those ten units.
- If F5 does not pass, the weight at which the pair works cannot be decided by training
  the machine: it stays a choice of whoever uses it.
- If V1 does not pass, the built partner does not beat the soloist even when the person
  knows something of its own: as a design goal, at free trust, antiphase is of no use.
- If V4 does not pass, the reading by which the machine trained on the sum learns what the
  person lacks is wrong.
- If K2 does not pass, the explanation given in hindsight in the first version of the
  article (the judge of the bench had nothing left to give) is wrong or not enough.
- If K4 does not pass, the additive model of the bias does not describe the networks.
- If S2 and S3 do not pass, the agreement of E1 does not depend on the shape of the
  injected noise, and the objection that E1 measures Gaussianity falls.

## 10. Order, times, cleanup

Order of priority: E0, E1, E2, E3. One L4 GPU with 24 GB; a ceiling of two and a half
hours of computing in all. If the timing pilot says that E1 exceeds 60 minutes we go
down to two seeds, then to the first three tasks, and we write it in the Deviations. On
the server we work in a single folder, which is deleted at the end.

E6 to E9 (version 2.1): five seeds (2 to 6) and an estimate of two hours on the same GPU,
in a sub-folder of their own, deleted at the end. Order: E6, E8, E7, E9. If time is not
enough we go down to three seeds (2, 3 and 4), and we write it in the Deviations.

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
- `trust/{task}_s{seed}.npz` (E6): like `rephase_ter/`, with seven machines and
  `built_for_weight`, the weights the four branches are built for.
- `views/{task}_o{overlap}_s{seed}.npz` (E7): like `rephase_bis/`, with `machines` =
  `M0, Mramo, Mpair` and `overlap` (100, 50 or 0).
- `chart/{task}_v{v}_s{seed}.npz` (E8): like `rephase_bis/`, with `z_H0`, `machines` =
  `M, Mramo, Mblind` and `v` (100, 50 or 25).
- `shape/{task}_mu{mu}_s{seed}.npz` (E9): `a`, `z_plus`, `z_minus` of shape (4, N), one row
  per shape of the test frame, with `shapes`.
- Each folder has a `manifest.json`: hyperparameters, times, library versions, control
  accuracies.

## 12. E6. Trust made to measure

Added with version 2.0, on 19 November 2026, after the article came out and was reviewed.

**Where it comes from.** In E2 ter the soloist, the fine-tuned soloist and the soloist with
the branch gave the tuned pair the same value, and T3 did not pass. On 19 November the same
files were read again at a fixed weight instead of the best one. The three machines differ
in the weight at which the pair reaches its maximum: about 7 with the soloist, about 1.2
with the branch. At weight 1 the pair with the branch is worth more in 10 units out of 10
(median 1.47 turings against 1.38); at weight 8 it is worth less (1.28 against 1.47). And
multiplying by a factor `c` the part of the cue that comes from the branch moves the
maximum to a weight close to `1 / c`. That reading is in
`results/e6_trust_from_e2ter.json` (`analysis/trust.py --ter`). **It is exploratory**: the
hypotheses below come from it, so it cannot confirm them. The test is made on new
trainings, with seeds never used before, and with branches *trained* for different
weights, not rescaled afterwards.

**The idea.** If the person's cue carries a bias `b a` and the machine adds `-c a`, the sum
`z_H + w z_M` cancels it at the weight `w = b / c`. A branch trained on the sum with weight
`w0` should then learn `c = b / w0`, and the pair should peak near `w0`. The weight at which
a pair works, which in § 2 is a choice of whoever uses the machine, would become a property
of the machine.

- **H, `M_0`, `M_0+`, `M_due`** as in E2 ter, with seeds 2 to 6. E1 to E5 used 0 and 1.
- **`M_ramo(w)`**, for `w` = 0.5, 1, 2 and 4. `M_0` frozen plus a branch that sees only the
  frame, trained with the pilot's hyperparameters on the cross-entropy of
  `z_H + w (z_M0 + z_ramo)`, with the log-odds as the networks give them. The four branches
  have the same seed, batch order, crops and frames: only `w` changes. The cue of the
  machine is `z_M0 + z_ramo`. `M_ramo(1)` is the `M_ramo` of E2 ter.
- **The pair at a fixed weight** (`analysis/fixed.py`). On two scales. Standardized, that
  of § 3: `d_a(z_H + w z_M)` at the weights 0.5, 1, 2 and 8. And "at face value": the sum of
  the raw log-odds, `z_H + z_M`, the two opinions taken for what they say.
- **The weight of the peak.** On the curve `Tu(w) = d_a(z_H + w z_M) / d_a(z_H)` with the
  raw log-odds, on the 49 weights of § 3: the geometric mean of the two weights at which
  `Tu^2 - 1` crosses 90% of its maximum, by linear interpolation in `ln w`. The maximum of
  a flat curve is a noisy place, its centre is not. Where a crossing falls outside the
  grid, the weight of the maximum is used.
- **Differences** paired on the same resamples of the images (§ 3), 1,000 resamples. The
  control is `M_0+`. `M_due` is reported without a criterion.

Hypotheses. The primary ones are F1 and F5.

- **F1, at equal weight the partner wins.** At standardized weight 1 the pair
  `(H, M_ramo(1))` is more sensitive than the pair `(H, M_0+)`, with the interval of the
  difference above zero in every seed, in at least 3 tasks out of 5.
- **F2, at face value too.** The same with the sum of the raw log-odds.
- **F3, the price.** At standardized weight 8 the pair `(H, M_0+)` is more sensitive than
  the pair `(H, M_ramo(1))`, with the interval below zero in every seed, in at least 3
  tasks out of 5: whoever trusts the machine a lot is better off with the soloist.
- **F4, who has to step aside.** At face value the pair `(H, M_0+)` is less sensitive than
  `M_0+` alone (interval below zero in every seed, in at least 3 tasks out of 5), and the
  pair `(H, M_ramo(1))` is not: the median over the units of its difference from `M_0+`
  alone is at least -0.05.
- **F5, the knob is in the machine.** The weight of the peak of the pair `(H, M_ramo(w))`
  is, in median over the units, within a factor 1.5 of `w` for `w` = 0.5, 1 and 2; and the
  rank correlation between `w` and the weight of the peak, over the four branches, is at
  least 0.9 in median over the units. At `w = 4` the place of the peak is not judged: there
  the curve is almost flat, as with the soloist.
- **F6, the same peak.** The tuned turing of `(H, M_ramo(w))` divided by that of
  `(H, M_0+)` is, in median over the units, between 0.97 and 1.03 for each of the four
  weights.
- **F7, what it costs alone.** The lower the weight it is built for, the less the machine
  is worth alone: the rank correlation between `w` and `d_a(M_ramo(w))` is at least 0.9 in
  median, and `M_ramo(0.5)` alone is less sensitive than `M_0+`, with the interval below
  zero in every seed, in at least 3 tasks out of 5.

Raw data: `trust/{task}_s{seed}.npz`, like `rephase_ter/`, with `machines` =
`M0, M0plus, Mdue, Mramo_w0.5, Mramo_w1, Mramo_w2, Mramo_w4` and `built_for_weight`;
`manifest.json`.

## 13. E7. Different views

Added with version 2.0, on 19 November 2026. No data exist for this test: the expectations
come from the model and from a toy bench (`synthetic.write_views`), on which the analysis
was tried.

**Why.** In E1 to E6 the machine sees everything the person sees. Then every way of putting
the two cues together is a function of what the machine has in front of it, and the best
possible machine, alone, would do at least as well. That on that bench the best soloist
remained the best partner (B3 and T3 do not pass) is therefore a consequence of the design,
not a discovery. For the removal of a bias to free something, the person must know
something the machine does not see.

- **The views.** Each judge sees half of the inside of the image and the whole frame; the
  rest is set to zero (normalised units), after the frame has been drawn. H always sees
  the left half (columns 2 to 15). The machine sees the left half (overlap 100%), the top
  half (rows 2 to 15: it shares the top-left quarter with H, 50%) or the right half
  (columns 16 to 29, 0%). Every half goes through the centre of the image: a central strip
  would be worth more than a lateral one, and the overlap would be confused with skill.
- **H**: a "plus" network, `mu = 1`, trained on half A of the training images. It is the
  same network at the three overlaps.
- **`M_0`**, the soloist: trained on half B on its view, frame independent of the class.
- **`M_ramo`**: `M_0` frozen plus a branch that sees only the frame, trained on the
  cross-entropy of `z_H + z_M0 + z_ramo`.
- **`M_pair`**: a new network on the machine's view, trained from scratch on the
  cross-entropy of `z_H + z_M` (the recipe of E2 bis), with the same seed as `M_0`.
- Seeds 2 to 6; `M_0`, the branch and `M_pair` have the same seeds at the three
  overlaps. The test frames are the same at the three overlaps.
- **Measures** of § 3 and of E2 bis, plus the pair at face value (§ 12). The differences
  are from the `M_0` of the same overlap, paired.

Hypotheses. The primary ones are V1 and V4.

- **V1, with something of its own the partner wins.** At overlap 0 the tuned pair
  `(H, M_ramo)` is more sensitive than the tuned pair `(H, M_0)`, with the interval of the
  difference above zero in every seed, in at least 3 tasks out of 5.
- **V2, the dose.** The median over the units of that difference grows as the overlap
  falls (0% > 50% > 100%), and at 100% it is at most one third of what it is at 0%.
- **V3, worse alone, better in the pair.** At overlap 0 `M_ramo` alone is less sensitive
  than `M_0` alone (interval below zero in every seed, in at least 3 tasks out of 5), and
  V1 passes.
- **V4, the machine trained on the sum learns what the person lacks.**
  `d_a(M_pair) / d_a(M_0)`, in median over the units, is at least 0.7 at overlap 0, at
  most 0.3 at 100% (in E2 bis, with the whole view, it was 0.03), and falls as the overlap
  grows.
- **V5, the pair beats the machine.** At overlap 0 the tuned pair `(H, M_ramo)` is more
  sensitive than `M_0` alone, with the interval above zero in every seed, in at least 3
  tasks out of 5.
- **V6.** At overlap 0, at face value, the pair `(H, M_ramo)` is more sensitive than the
  pair `(H, M_0)`: interval above zero in every seed, in at least 3 tasks out of 5.

A limit written beforehand: at 50% the machine's half is horizontal and H's is vertical, so
between 0%, 50% and 100% the overlap is not the only thing that changes.

Raw data: `views/{task}_o{overlap}_s{seed}.npz`, like `rephase_bis/`, with `machines` =
`M0, Mramo, Mpair` and `overlap`; `manifest.json`.

## 14. E8. The bench of the chart

Added with version 2.1, on 19 November 2026, before any run of E6 to E9. The design, the
theorem and the expectations take up those proposed by a review of the article; here they
are rewritten on the networks and the estimators of this protocol.

**The theorem.** The judge's cue is `h + beta a`: a clean cue `h`, with sensitivity `d_h`
and noise of unit variance, plus a bias that does not depend on the class, of variance `B`.
The machine's cue `m` has sensitivity `d_m`, noise of unit variance and correlation `r` with
the noise of `h`. For the best linear combination, with weights of any sign:

    D1^2 = (d_h^2 - 2 r d_h d_m + (1 + B) d_m^2) / (1 + B - r^2)      the judge with the soloist m
    D2^2 = (d_h^2 - 2 r d_h d_m + d_m^2) / (1 - r^2)                  the judge with m and whoever reads a
    D2^2 - D1^2 = B / (1 - r^2 + B) * P,     P = (d_h - r d_m)^2 / (1 - r^2)

`P = D2^2 - d_m^2` is what the clean judge knows and the machine does not, and it vanishes
at the useful angle, `r = d_h / d_m`. The judge with a blind corrector, which reads only
`a`, is worth `d_h`, and beats the pair with the soloist when `d_m / d_h` lies between
`(r - sqrt(B (1 + B - r^2))) / (1 + B)` and `(r + sqrt(B (1 + B - r^2))) / (1 + B)`. The
threshold that matters is the second; the first is above zero only if `r^2 > B`. The code is
in `src/cognitive_resonance/chart.py`, checked against the direct inversion of the matrices.

**On the old data.** Applied unit by unit to the files of E2 bis and E2 ter
(`analysis/chart.py --e2`, `results/chart_from_e2.json`), the theorem gives back the
measured pairs: with the soloist 3.05 and 3.07 (measured 3.04 and 3.06), with the blind
machine 2.86 (measured 2.85), with the soloist plus the branch 3.13 (measured 3.10). It also
says why T3 did not pass: on that bench the built partner could give 1.4% more. It is a
reading of old data, not a test.

**The bench.** In E1 to E6 the machine sees everything the judge sees, and `P` is almost
zero. Here the judge sees the whole image; the machine sees the frame and only a part of the
inside: all of it (`v = 100`, the bench of E2 ter), the left half (`v = 50`, columns 2 to
15) or the left quarter (`v = 25`, columns 2 to 8).

- **H**: a "plus" network, `mu = 1`, half A, whole image. **H0**: the same network without
  the bias (same seed, frame independent of the class).
- **`M_blind`**: sees only the frame, trained on the cross-entropy of `z_H + z_blind` (the
  `M_frame` of E2 bis). It is the same at the three values of `v`.
- For every `v`: **`M`**, the soloist on its part of the image; **`M_ramo`**, `M` frozen plus
  a branch that sees only the frame, trained on the cross-entropy of `z_H + z_M + z_ramo`.
- Seeds 2 to 6, five tasks. `M` and the branch have the same seeds at the three values of `v`.
- **The quantities of the theorem, measured unit by unit.** The slope of H's cue on the
  frame, within the class, gives the bias; what is left of the cue without the bias is `h`.
  From `h` and `M`: `d_h`, `d_m` (with `d_a`), `r` and `B`.
- **The best linear use of the three cues** `(z_H, z_M, a)`: the weights are estimated on
  half of the images and the sensitivity is read on the other half, then they are swapped.
  It is `D2` measured without going through a network.

Expectations. Δ is the sensitivity of the tuned pair `(H, M_ramo)` minus that of `(H, M)`,
paired. The primary ones are K2 and K4.

- **K1, the null result of E2 ter.** At `v = 100`, `|Δ|` over the sensitivity of the pair
  `(H, M)` is at most 0.03, in median over the units.
- **K2, with something of its own the partner wins.** At `v = 25`, Δ > 0 with an interval
  that excludes zero in every seed, in at least 3 tasks out of 5.
- **K3, the dose.** The median of Δ grows as the machine sees less: `v = 25 > 50 > 100`.
- **K4, the law.** At `v = 50` and `v = 25`, `D2^2 - D1^2` measured with the best linear use
  of the three cues, over the gain predicted by the theorem with the quantities of the same
  unit, is between 0.8 and 1.25 in median.
- **K5, worse alone, better in the pair.** At `v = 25`, `M_ramo` alone is less sensitive
  than `M` and the pair with it is more sensitive, with intervals that exclude zero in every
  seed, in at least 3 tasks out of 5.
- **K6, who knows the judge against who knows the task.** Which of `(H, M_blind)` and
  `(H, M)` is the more sensitive pair is what the threshold of the theorem says, computed
  with the quantities of the unit, in at least 80% of the units (all the `v`).
- **K7, nothing is created.** The pair `(H, M_ramo)` over the pair `(H0, M)` is at most
  1.05, in median, at every `v`.

A limit written beforehand: `M_ramo` is built for the weight 1, and a single weight has to
cancel the bias and to weigh the machine. With a weak machine it can be worth less than the
best use of the three cues: this is why K4 uses the latter, and K2 the network.

Raw data: `chart/{task}_v{v}_s{seed}.npz`, like `rephase_bis/`, with `z_H0`, `machines` =
`M, Mramo, Mblind` and `v`; `manifest.json`.

## 15. E9. The shape of the noise

Added with version 2.1, on 19 November 2026, before any run of E6 to E9.

**Why.** With sensitivity measured as a signal-to-noise ratio the formula of § 2 is an
identity (`analysis/identity.py`): no data can refute it. With `d_a`, which comes from the
AUC, it holds for Gaussian cues. And in E1, beyond 90 degrees, almost all the noise is the
frame, which we drew ourselves from a Gaussian. The question is how far the measure moves
from the formula when that part of the noise has another shape.

**What was known.** A preliminary trial made during a review, on networks with a third of
the channels, four tasks and one seed, gave for the ratio between the measured and the
predicted pair 0.29 with a two-valued frame, 0.90 with a uniform one and 0.93 with a
Student's t, with expectations wrong in direction for two shapes out of three. The
expectations below come from a model: Gaussian cues plus a frame of the given shape, with
the median parameters of E1 (§ 5). The model gives, at `mu = 1.5`, 1.09 (two values), 1.04
(uniform), 0.86 (t); at `mu = 2`, 0.79, 1.05 and 0.80.

- **The networks** are those of E1 (`gpu/bench.py`): "plus" and "minus", 9,000 images,
  Gaussian training frame. `mu` = 1, 1.5 and 2; seeds 2 to 6; five tasks.
- **At test** the frame is drawn from four distributions with zero mean and the same
  variance `1 + mu^2`: `gauss` (it is E1, on new seeds), `two` (two values,
  `±sqrt(1 + mu^2)`), `uniform`, `t3` (Student's t with 3 degrees of freedom, rescaled).
  Standardization uses the validation set with the same shape.
- **Measures** of § 3 for every shape, plus the signal-to-noise ratio next to `d_a`. The
  ratio is that of H1c: tuned turing over `Tu_app` from `k` and `rho` of the same unit and
  shape.

Expectations.

- **S1, replication.** With the Gaussian frame, at `mu` = 1.5 and 2: the angle is above 90
  degrees in every unit and the ratio is between 0.9 and 1.1 in median.
- **S2, the formula promises too much.** At `mu = 2` the median ratio is below 0.9 both with
  the two-valued frame and with the t.
- **S3, the direction depends on the shape.** At `mu = 1.5` the median ratio is above 1 with
  the two-valued frame and below 0.95 with the t.
- **S4.** With the uniform frame the median ratio stays between 0.95 and 1.10 at `mu` = 1.5
  and 2.
- **S5, the pair is the same.** The sensitivity of the tuned pair with a shape, over that
  with the Gaussian frame, is between 0.95 and 1.05 in median for every shape at `mu` = 1.5
  and 2: in the pair the frame cancels, and what changes is what the single judges promise.
- Without a criterion: the gap of the identity with the signal-to-noise ratio, for every
  shape.

Raw data: `shape/{task}_mu{mu}_s{seed}.npz`: `a`, `z_plus` and `z_minus` of shape (4, N),
one row per shape in the order of `shapes`; `manifest.json`.

## Deviations

All dated 10 November 2026. The first come from the runs; the others are choices of the
analysis where the protocol left room, fixed before reading the real data.

**The runs.**

- **E1 with two seeds** (0 and 1) instead of three. With 30 epochs one network costs about
  12.5 seconds, and E1 with three seeds was estimated at about 70 minutes, over the
  60-minute cap: it is the first reduction of § 10. With two seeds the estimate was about
  47 minutes.
- **The pilot** fixed 30 epochs, learning rate 0.1, batch 256 and `delta = 1`; the linear
  probe recovers `a` with `R^2 = 1.00`.
- **The frame** is *set* to the value `a · delta` on the 2-pixel border of the three
  channels, in normalized units, after cropping and flipping: it is not added to the
  image.
- **Seeds.** Each judge has its own seed, the same at every `mu`; the numbers `eps` of the
  validation and test frames are the same for the three judges of a unit and for every
  `mu`.
- **E3** without a pilot: hyperparameters fixed beforehand, 40 epochs at learning rate
  0.2 for the base machine, 8 epochs at learning rate 0.01 with a cosine decay to zero
  for the fine-tuning.
- **The manifests** written by the runs (that of `rephase/` too) cite version 1.1: it is
  the version written in the GPU scripts. Versions 1.2 and 1.3 change nothing of what E1,
  E2 and E3 do.

**The analysis.**

- **E0.** The protocol does not fix `dH`: we used `dH = 1`. Half of the noise variance
  belongs to the image (shared by its 5 presentations), with the same `rho` at both
  levels; validation of 1,000 images times 5 presentations. Added, without a criterion,
  the same computation at `dH = 2`.
- **AUC equal to 0 or 1**: moved half a pair from the edge, so that `d_a` stays finite.
  How often it happens is reported.
- **Bootstrap**: multinomial resampling of the images (turing-meter uses Poisson
  weights). The two halves of the tuned turing are drawn class by class, with a fixed
  seed.
- **H1a.** A task counts if the interval of `phi` lies above 90 degrees in every seed, at
  both of the two highest values of `mu`.
- **H1b.** The lower end of the tuned turing is compared with `sqrt(1 + k^2)` computed
  with the point estimate of `k`.
- **H1c.** The relative error is measured with respect to the tuned turing.
- **H1e.** "`rho` rises with `mu`" has no threshold: we use that of H1a and H2a, a median
  rank correlation of at least 0.9.
- **H1f.** The reference is, per task, the mean over seeds of the tuned opposite pair at
  `mu = 0`; at each `mu` the median of the ratio decides.
- **H2a, H2b, H3a.** A task (for H3a, the experiment) counts if the condition holds at
  the same `lambda` in every seed. **H3b**: the point estimate of the angle decides.
- **E3.** The combiner is a conditional logit on `logit((c + 0.5) / (n + 1))` of each
  class's votes and on the log probability of the machine, with 5-fold cross-validation
  over images. The group alone is the plurality vote; sensitivities with `m = 10`; the
  turing of the group is the sensitivity of the combiner over that of the plurality. The
  paired difference of the angles uses the formulas and the seed of `compare_votes`.
- **E5.** The width of the null is the ratio between the far and the near edge, the
  convention of the band: with the difference of weights the logarithm changes sign near
  1 and the criterion of H5c cannot be read (the difference is reported anyway). Null and
  edges by linear interpolation in `w`. H5e divides by
  `sqrt(1 + k^2 - 2 k cos(phi)) / sin(phi)` without the restriction to non-negative
  weights (the peak of the Fano profile). In H5d the standard deviation of `a` is
  estimated on test and validation separately, and the direction comes from the sign of
  the slope of the "minus" network on the test set.

**E2 ter and E3 bis.** Choices of the analysis fixed on 11 November 2026, with the code
written and before the runs.

- **T1.** Median over the units of the ratio `d_a(M_due) / d_a(M_0+)`; the signs of the
  slopes are point estimates, as in B1; if `d_a(M_0+)` is not positive in a unit, T1
  does not pass.
- **T2, T3.** Four verdicts, one per machine. A task counts if the lower end of the
  interval is above zero in every seed; the differences are taken against `M_0+` and
  paired on the same resamples. Those against `M_0` are reported without a criterion.
- **T4.** A single verdict, which passes if both machines pass; apparent turing as in B5
  (maximum over the non-negative weights, for `k` of any sign).
- **T5.** Sensitivity of the tuned pair `(H, M_due)` minus that of `M_0+` alone, paired;
  `M_ramo` is reported without a criterion.
- **The ceiling of T3** (the pair of E1 at `mu = 0`) is reported as context, without a
  criterion: those networks are trained on 9,000 images, not on a half.
- **Fine-tuning of E2 ter.** A copy of the weights of `M_0`, normalisation averages
  included; step 0.02; `M_0+` and `M_due` share seed, batch order, crops and frames.
- **E3 bis.** Measures on the union of the two unseen halves, per seed, as in E3; the
  majority is the most frequent vote, with ties split. P1: point estimates per half and
  per seed, and a drop of exactly 2 points passes. P3: it passes only if a recipe passes
  P2 and also has the difference of the apparent turings with the lower end above zero
  in every seed; the apparent turings are those of `compare_votes` (non-negative
  weights), and the difference is paired on the same resamples as the angles. P4: if a
  machine is at chance level the angle is not defined, and P4 cannot be evaluated. P5:
  `mista` only.
- **The manifests** of the two runs cite version 1.4.

**The versions.** Version 1.1 is from the same day as version 1 and comes before any run: it
specifies the computations of E0 and of the bootstrap, which in version 1 could not have
been run in the time available. Version 1.2, from the same day, adds question Q6 and
experiment E5 on the null (§ 8 bis): it was written with the GPU runs started, but
before any analysis of the raw data, and it changes nothing of what was there. Version
1.3 adds E2 bis (§ 6 bis), exploratory: it is written after seeing the data of E2 and E3
and before its run, and it touches neither the criteria nor the results of the other
tests. Version 1.4, of 11 November, adds E2 ter (§ 6 ter) and E3 bis (§ 7 bis),
exploratory: they come from a review of the draft of the article, they are written
after seeing all the results and before their runs, and they touch neither the criteria
nor the results of the other tests. On the server the two runs use a sub-folder of their
own, which is deleted at the end.

Version 2.0, of 19 November 2026, comes after the article was published and from a review
of it. It adds questions Q7 and Q8 and the tests E6 (§ 12) and E7 (§ 13). It is written
before the two runs, together with the training code (`gpu/trust.py`, `gpu/views.py`), the
analysis code (`analysis/fixed.py`, `analysis/trust.py`, `analysis/views.py`) and their
tests. E6 tests, on new trainings, what was seen by reading the files of E2 ter again:
that reading is in the repository, declared exploratory. E7 has no earlier data. It
touches neither the criteria nor the results of the other tests. A note on the bench of
E1 and E5, added with this version without changing their outcomes: if sensitivity is
measured as the difference of the means divided by the within-class standard deviation,
the formula of § 2 is an identity of the sample moments, true for any data. With `d_a`,
which comes from the area under the ROC curve, the criteria H1c, H1d and H5a to H5e
therefore measure how far the cues of the networks are from the equal-variance Gaussian
model, not the formula.

Version 2.1, of the same day, comes before any run of E6 and E7 and from two more reviews
of the article. It takes the seeds of E6 and E7 from three to five (2 to 6), without
touching their hypotheses and criteria, and adds questions Q9 and Q10 and the tests E8
(§ 14, the bench of the chart with its theorem) and E9 (§ 15, the shape of the noise), with
the training code (`gpu/chart.py`, `gpu/shape.py`), the analysis code
(`analysis/chart.py`, `analysis/shape.py`) and the tests. "In every seed" now means in all
five. The theorem applied to the files of E2 bis and E2 ter is a reading of old data,
declared as such. The shares of right answers were added to `analysis/fixed.py`, without a
criterion.
