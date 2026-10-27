# Decision process

Let S be the set of modalities already acquired for one physical sample, A the
available action set and p_S a probability vector over K >= 2 ordered classes.
The remaining candidates are A minus S.

## Uncertainty

Natural-log entropy is

```text
H(p) = -sum_i p_i log(p_i), with 0 log(0) = 0.
```

Entropy lies in [0, log(K)]. Normalized entropy is H(p)/log(K). Inputs are finite,
nonnegative probabilities whose rows sum to one within absolute 1e-6 tolerance.
Only accepted roundoff is renormalized; arbitrary scores are not probabilities.

## Ideal expected utility and implemented approximation

A possible ideal one-step utility is

```text
U(m | S) = E[H(p_S) - H(p_(S union {m})) | current evidence] - lambda C(m).
```

Here C(m) is a nonnegative acquisition cost in declared units, lambda >= 0 is
an entropy-per-cost penalty, and the expectation is over possible measurement
outcomes conditional on current evidence. This expression does not assert that
a suitable conditional distribution is available.

Version 0.1.0 instead takes one externally supplied deterministic future belief
q_m(S) and computes

```text
U_det(m | S) = H(p_S) - H(q_m(S)) - lambda C(m).
```

The lookup simulator supplies q_m(S) from the subset S union {m}. No learned
forecast or integration over outcomes is implemented. In particular,
H(E[p_future]) generally differs from E[H(p_future)]; calling a representative
future vector an expectation does not make U_det true expected information gain.
Forecasts may increase entropy, producing negative reduction or utility.

For lambda = 0, the scorer ranks the supplied entropy reductions without a cost
penalty. Hard budget eligibility still applies independently. A zero-cost action
is valid but does not automatically have positive utility.

## Eligibility and selection

For a budget B and already incurred cost c, an action is affordable when

```text
c + C(m) <= B.
```

The budget is a hard feasibility constraint, distinct from lambda's soft penalty.
The comparison is strict floating-point <=, with no hidden overspending tolerance.
Use suitable units, such as integer-valued cost units, when exact boundary
arithmetic matters. An initial state already above a budget is invalid.
Only eligible unacquired actions are scored. Greedy selection maximizes U_det;
exact ties are resolved by lexical modality name. The default minimum utility
is zero and selection requires a strict improvement above that threshold.
This is a local one-step rule, not a globally optimal sequence planner.

## Transition and termination

After acquiring m, update S to S union {m}, replace the predictive belief with
the provider's prediction, and add C(m) to total cost. Prior snapshots remain
unchanged. Initially known evidence is not acquired again; any initial incurred
cost must already be represented in the initial state.

The episode stops when there are no modalities left, a supplied confidence
threshold is met, no candidate fits the budget, or no eligible utility passes
the policy threshold. Confidence threshold means max_i p_S[i] >= threshold;
threshold is in (0,1]. Confidence alone does not guarantee correctness.
Modality exhaustion has first terminal priority in the runner. Otherwise,
composed stopping rules use their declared order for reason priority. All budget
eligibility constraints are checked even when a confidence rule is already met.

## Evaluation units

Entropy reduction is H(p_initial) - H(p_final). Cost-normalized reduction divides
that quantity by the sum of new acquisition-record costs in the episode. It is
None when additional cost is zero; it may be negative and is not an accuracy
measure. A nonfinite ratio caused by a tiny positive denominator raises
ValueError instead of being reported as a meaningful efficiency.
Correctness compares argmax(p_final) with a supplied ordered-class index.
Batch summaries average distinct physical samples. Partition membership,
calibration and uncertainty intervals require separate scientific validation.
