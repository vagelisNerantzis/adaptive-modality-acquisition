# Architecture

## Physical sample and action space

One episode concerns one physical sample. Its immutable sample_id follows every
state, acquisition event and result. A modality is an action identifier, not a
hard-coded instrument type. Available modalities define the action universe;
acquired modalities identify known evidence, and remaining candidates exclude
all previously acquired actions. Duplicate acquisition is invalid.

## State and predictions

AcquisitionState holds probability values, known evidence, accumulated cost,
step index and acquisition history. A transition returns a new snapshot rather
than mutating earlier states. Initial evidence and its incurred cost are supplied
explicitly; initial observations are not recharged by the episode runner.

Probability vectors have an externally defined, stable class order. Values must
be finite, nonnegative and sum to one within absolute 1e-6 tolerance. Accepted
roundoff is renormalized. Entropy uses natural logarithms; normalized entropy
divides by log(number of classes). Confidence is max probability, not a claim of
calibration or correctness.

## Prediction provider

LookupPredictor binds one physical sample to a deterministic mapping from
acquired-modality subsets to beliefs. It checks class dimensions and rejects
missing subset states. Its sample binding prevents accidentally using another
specimen's lookup. A real adapter can replace this mechanism without changing
policy logic, but it must distinguish a forecast of candidate utility from an
actually measured future modality. The synthetic table is a simulation device.

## Cost and utility

FixedCostModel performs deterministic nonnegative cost lookup. Unknown actions
fail explicitly. EntropyReductionScorer compares current entropy with each
supplied candidate forecast, then subtracts cost_weight times acquisition cost.
Its interface makes predictions and costs explicit, allowing a future scorer to
use a different utility while retaining the same state and decision mechanics.

A single forecast vector is not a distribution over possible post-measurement
beliefs; the scorer is not an implemented Bayesian information-gain estimator.

## Policy and stopping

GreedyPolicy chooses the eligible candidate with maximum utility. Equal scores
use lexical modality order. By default a score must be strictly positive; a
configurable threshold can stop further acquisition. A known stop reason causes
selection to return no action.

ConfidenceThresholdStop, BudgetStop and NoModalitiesStop are composable through
AnyStop. AnyStop returns the first applicable reason in supplied rule order and
intersects candidate eligibility across rules. The runner gives modality
exhaustion first terminal priority; otherwise the configured rule order decides
the reported reason. Every budget is validated even if another rule has fired.
Budget eligibility is enforced before scoring: an unaffordable high-score action
cannot displace an affordable one. Affordability uses strict floating-point <=
without an overspending tolerance. Integer-valued cost units avoid unexpected
roundoff at exact mathematical boundaries. No global planning or optimality over
later action sequences is claimed.

## Simulation and history

run_episode validates initial state/provider consistency, applies stopping rules,
filters candidates, scores, selects, queries the updated belief and creates the
next immutable state. Each step adds one new modality, so the finite action set
bounds the episode length. Acquisition history records actions, beliefs and costs;
decision history also retains considered utilities and the terminal stop.

No model fitting, dataset access or network call is part of simulation.

## Evaluation

Episode metrics include cost, additional step count, total known modalities,
final confidence, entropy and optional correctness. Entropy reduction per cost
uses additional episode cost, summed from newly added event records rather than
subtracting potentially large cumulative totals. It is undefined for a zero-cost
episode. A positive but tiny cost that produces a nonfinite ratio fails clearly.
Batch summaries require distinct physical sample IDs and report mean total cost,
mean steps, mean confidence and accuracy when labels are supplied. Means divide
terms before compensated summation to avoid overflow of finite large costs. Repeated seeds or
views of one specimen must not be treated as independent physical samples.

## Difference from ordinary fusion

Fusion asks, "How should available modalities be combined?" Acquisition asks,
"Which modality should be acquired next, if any?" Predictive encoders or fusion
models may provide beliefs, but acquisition cost and stopping are separate
scientific decisions. This package isolates those decisions without depending on
a particular encoder, domain, training pipeline or another portfolio package.
