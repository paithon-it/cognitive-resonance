# The resonance identity, in one page

**Model.** Two judges, a person and a machine, decide between two classes `s = +1` and
`s = -1`. Their cues are `x_H = s dH/2 + n_H` and `x_M = s dM/2 + n_M`; the noises have
unit variance and correlation `rho = cos(phi)`. The pair reads `x_H + w x_M`.

**The turing at weight w.** The pair's mean separation is `dH + w dM`, its noise variance
`1 + w^2 + 2 rho w`. Dividing by the person's `dH`, with `k = dM / dH`:

    Tu(w) = (1 + w k) / sqrt(1 + w^2 + 2 rho w)

Setting the derivative of `ln Tu` to zero, `k (1 + w^2 + 2 rho w) = (1 + w k)(w + rho)`,
gives `w* = (k - rho) / (1 - rho k)` and the peak

    Tu_app^2 = (1 + k^2 - 2 k rho) / (1 - rho^2),

the apparent turings of turing-meter. Both weights are positive only when
`rho < min(k, 1/k)`: below the useful angle `arccos(min(k, 1/k))` the non-negative optimum
is the stronger judge alone.

**k = 1.** Then `Tu^2 - 1 = 2 w (1 - rho) / (1 + w^2 + 2 rho w)`. Divide numerator and
denominator by `w`, and write `w + 1/w = (Omega - 1/Omega)^2 + 2` with `Omega = sqrt(w)`:

    Tu^2 - 1 = 2 (1 - rho) / ((Omega - 1/Omega)^2 + 2 (1 + rho))
             = [(1 - rho)/(1 + rho)] / [1 + (Omega - 1/Omega)^2 / (2 (1 + rho))]

With `(1 - rho)/(1 + rho) = tan^2(phi/2)` and `2 (1 + rho) = 4 cos^2(phi/2) = 1/Q^2`:

    Tu(w)^2 - 1 = tan^2(phi/2) / (1 + Q^2 (Omega - 1/Omega)^2),      Q = 1 / (2 cos(phi/2))

**The circuit.** In a series RLC circuit driven at frequency `omega`, the power delivered
to the resistor, over its value at resonance, is `1 / (1 + Q^2 (omega/omega0 - omega0/omega)^2)`
with `Q = omega0 L / R`. The identity is the same curve with `Omega = sqrt(w)` as the
normalized frequency, resonance at `w = 1`, and height `tan^2(phi/2)`. At the peak
`Tu^2 = 1 + tan^2(phi/2) = 1/cos^2(phi/2)`, so `Tu = 2Q`.

| phi (degrees) | 30 | 60 | 90 | 120 | 150 | 170 |
|---|---|---|---|---|---|---|
| Q | 0.52 | 0.58 | 0.71 | 1.00 | 1.93 | 5.74 |
| Tu at the peak, 2Q | 1.04 | 1.15 | 1.41 | 2.00 | 3.86 | 11.47 |

**The band.** `Tu^2 - 1` falls to half its peak where `Q^2 (Omega - 1/Omega)^2 = 1`, that
is `Omega - 1/Omega = +-1/Q`. The two solutions are reciprocal, `Omega_low = 1/Omega_high`,
so `Omega_high - Omega_low = 1/Q`: the relative bandwidth of a resonator. In weights the
width is `w_high / w_low = Omega_high^4`, with `Omega_high = (1/Q + sqrt(1/Q^2 + 4)) / 2`.
As the angle opens, `Q` grows and the band narrows: the pair is worth more, and only near
the right weight. For `k != 1` the edges are found numerically (`model.band_edges`); when
`k > 1` the high edge can be at infinity, because the machine alone may already stay above
half the peak.

**Weights of either sign (protocol v1.2).** Write `c = cos(phi)`, `s = sin(phi)` and
`eps = (w + c)/s`. Then `1 + w^2 + 2 c w = s^2 (1 + eps^2)` and `1 + w k = k s (eps + q)`
with `q = (1 - k c)/(k s)`, so

    Tu(w) = k (q + eps) / sqrt(1 + eps^2),      Tu(w)^2 = k^2 (q + eps)^2 / (1 + eps^2),

a Fano profile. It is zero at `eps = -q`, that is `w0 = -1/k`, at every angle: the two
signals cancel and the noise is left. Its extremum is at `eps = 1/q`, where
`Tu^2 = k^2 (1 + q^2) = Tu_app^2`. At the useful angle (`cos phi = 1/k`) `q = 0`: the peak
goes to infinite weight and only the null is left. With `k < 0` (one judge below chance,
systematically wrong) the null moves to a positive weight, `1/|k|`: trusting a judge that
is wrong in a consistent way cancels the other. The width of the null, the interval where
`|Tu| < 1/2`, grows with the angle; for `k = 1`, in weights, it is
`sqrt(4 t^2 + 24 t) / 3` with `t = 1 - cos(phi)`.

The formulas of this page are checked numerically in `tests/test_model.py`.
