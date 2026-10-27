# Adaptive Modality Acquisition

[![Tests](https://github.com/vagelisNerantzis/adaptive-modality-acquisition/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/vagelisNerantzis/adaptive-modality-acquisition/actions/workflows/tests.yml)

A compact NumPy framework for cost-aware sequential measurement decisions on
physical samples. Version **0.1.0** implements deterministic acquisition and
stopping logic with explicit predictions, costs and decision histories.

## Scientific motivation

Conventional multimodal pipelines often assume every modality is already
available. Scientific measurements can require equipment, operator time,
substantial energy, destructive preparation or an acquisition budget. The
relevant decision is therefore both **which measurement to acquire next** and
**when further acquisition is unnecessary or unaffordable**.

The framework is domain-independent; textile characterization is only one
motivating use case. No textile classes, instruments or modalities are built
into the package. No private data, historical result or trained model is bundled.

## Core decision loop

```text
current evidence -> predictive uncertainty -> candidate utility
    -> acquire next modality -> update belief -> stop
```

The physical sample is the unit of decision. Its identity remains attached to
immutable state snapshots, acquisition records and episode summaries.

## Features

- Validated state with acquired and remaining modalities, probability vectors,
  accumulated cost, step count and acquisition history.
- Fixed nonnegative modality costs and natural-log entropy.
- Supplied post-acquisition forecasts scored by entropy reduction minus cost.
- Greedy selection with lexical tie-breaking and a strict utility threshold.
- Composable confidence, budget and modality-exhaustion stopping rules.
- Affordable candidates filtered before scoring or acquisition.
- A deterministic, sample-bound subset lookup for synthetic simulation.
- Episode and batch metrics with explicit physical-sample units.
- Synthetic CPU-only tests and a quick educational example.

## Installation

Python 3.10+ is required. Runtime dependencies are limited to NumPy; pytest is
an optional development dependency. No GPU, PyTorch or notebook is required.

```bash
python -m pip install -e ".[dev]"
```

## Minimal usage

```python
from adaptive_acquisition.state.acquisition_state import AcquisitionState
from adaptive_acquisition.costs.models import FixedCostModel
from adaptive_acquisition.utility.scoring import EntropyReductionScorer
from adaptive_acquisition.stopping.rules import AnyStop, BudgetStop, ConfidenceThresholdStop
from adaptive_acquisition.simulation.lookup import LookupPredictor
from adaptive_acquisition.simulation.episode import run_episode

state = AcquisitionState(
    sample_id="synthetic_001",
    available_modalities=("a", "b", "c"),
    acquired_modalities=("a",),
    probabilities=(0.55, 0.45),
)
costs = FixedCostModel({"a": 0.0, "b": 1.0, "c": 2.0})
predictor = LookupPredictor("synthetic_001", {
    frozenset({"a"}): (0.55, 0.45),
    frozenset({"a", "b"}): (0.80, 0.20),
    frozenset({"a", "c"}): (0.95, 0.05),
    frozenset({"a", "b", "c"}): (0.97, 0.03),
})
result = run_episode(
    state, predictor, costs,
    scorer=EntropyReductionScorer(cost_weight=0.1),
    stopping=AnyStop((ConfidenceThresholdStop(0.90), BudgetStop(3.0))),
)
print(result.selected_modalities, result.total_cost, result.stop_reason)
```

The prediction table supplies artificial counterfactual beliefs. It is not a
learned acquisition model and does not establish real expected information gain.
Initial evidence is supplied by the caller; its cost is included only if set in
`total_cost`. It is never charged again as an acquisition action.

## Synthetic demo

```bash
python examples/synthetic_acquisition_demo.py
```

The example prints candidate scores, selected measurements, updated confidence,
cost and the final stopping reason. It uses generic modality names and fixed
synthetic predictions. Its cost weight is 0.03 to make the sequential trade-off
visible. `configs/example.yaml` records a separate illustrative weight of 0.15;
it is not loaded automatically. YAML parsing is not a required runtime dependency.

## Testing and CI

```bash
python -m compileall src tests examples
pytest -q
```

Tests require no network, GPU, dataset or secret. The included GitHub Actions
workflow targets pushes and pull requests to `main` with Python 3.12 on Ubuntu.
It installs development dependencies, compiles, tests, runs the demo and checks
version consistency. **Hosted CI is pending the first push**; a badge is not a
claim that a hosted run has passed. Actual local results are recorded in
[the validation report](docs/validation_report.md).

## Repository structure

```text
src/adaptive_acquisition/
  state/          immutable physical-sample state and transition records
  costs/          fixed modality acquisition costs
  utility/        entropy and explicit candidate scoring
  policies/       greedy decisions and utility thresholds
  stopping/       composable confidence, budget and exhaustion rules
  simulation/     synthetic prediction lookup and episode runner
  evaluation/     episode and physical-sample batch metrics
  validation/     shared identity, cost and probability contracts
tests/            deterministic synthetic pytest suite
examples/         quick CPU acquisition demo
configs/          illustrative parameters
docs/             architecture, mathematical decisions, actual validation
assets/           public asset policy
.github/workflows/tests.yml
```

## Scope of v0.1.0

The package implements auditable sequential decision mechanics, not a neural
training pipeline, reinforcement learning system, active learning system or
Bayesian-optimal controller. A supplied candidate scorer can be replaced by a
future model, but no learned acquisition policy is claimed here. Greedy utility
maximization is local, not globally optimal over future measurement sequences.

## Metrics

Episode metrics distinguish total accumulated cost, additional acquisition
steps and the total number of known modalities. Final confidence is the largest
class probability, not an estimate that a decision is scientifically correct.
Entropy uses natural logarithms. Correctness is evaluated only when a true
class index is provided; class ordering is the caller's responsibility.

Efficiency is the initial-to-final entropy reduction divided by the additional
cost incurred during the episode, summed from its new acquisition records.
It is undefined (`None`) when that cost is zero. A positive cost so tiny that
the ratio is nonfinite raises `ValueError` rather than reporting an infinity.
A negative value is possible if uncertainty increases. It is not an
accuracy-per-cost claim. Batch summaries average one result per distinct
physical sample; duplicate sample IDs are rejected rather than treated as
independent observations.

## Reproducibility principles

Preserve physical identity across all views, modalities and partitions. Fit
predictive models on training samples and develop policies on validation samples;
do not use held-out errors to select acquisition utilities or stopping points.
Record the initial evidence and cost, ordered class schema, forecast provider,
modality costs, utility weight, stopping rules and complete decision history.

Tie-breaking is lexical. Probability rows must sum to one within absolute
1e-6 tolerance and are renormalized only for accepted roundoff; invalid vectors
fail. Natural logarithms and float64 calculations define the uncertainty scale.
Budgets use strict floating-point `<=` comparisons without an overspending
tolerance. Choose suitable cost units (integer-valued units when exact boundary
arithmetic matters) and retain initial acquisition cost explicitly. Exhaustion
is the first terminal reason; other stopping rules retain their configured order.
No hidden stochastic policy is used.

## Relationship to multimodal learning

Ordinary fusion asks how to combine measurements that are available. Adaptive
acquisition asks which measurement to acquire next, if any. Encoders and fusion
models can supply beliefs to this package, but neither is implemented or required.
The other portfolio repositories are not runtime dependencies.

## Limitations

- Future beliefs come from an external forecast or synthetic lookup. Accessing
  actual unacquired measurements to rank candidates in a real system would
  create privileged-information leakage.
- Entropy of a representative future belief is generally not expected posterior
  entropy. The current scorer is an explicit deterministic approximation.
- Costs are fixed per modality. Resource scheduling, destructive interactions,
  correlated measurement noise and varying operational costs are not modeled.
- Confidence thresholds do not provide calibration, coverage or error guarantees.
- State alone cannot validate a dataset split or detect aliases of one specimen.
- Only short synthetic episodes are demonstrated; no scientific performance or
  state-of-the-art claim is made.

## License

[MIT](LICENSE) covers the software. No rights to datasets, checkpoints or
third-party institutional assets are granted.
