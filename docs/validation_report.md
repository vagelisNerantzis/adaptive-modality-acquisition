# Validation report

Validation date: 2026-10-27. Package version: **0.1.0**.

## Environment and actual commands

- Python **3.11.15**, NumPy **2.4.6**, pytest **9.1.1**.
- A new repository-local environment was created with
  `python -m venv .venv`, without inheriting system site packages. Runtime
  dependencies consist only of NumPy; no PyTorch or GPU is needed.
- `python -m pip install -e ".[dev]" --disable-pip-version-check` passed,
  exit 0, in that environment. Build dependencies, NumPy and pytest were
  installed there; caches and temporary files stayed in ignored local storage.
- `python -m compileall src tests examples`: passed, exit 0.
- `pytest -q`: **68 passed, 0 failed, 0 skipped**, with no warnings, exit 0.
- `python examples/synthetic_acquisition_demo.py`: passed, exit 0. It was run
  twice in separate processes and produced identical output.
- The README's complete minimal Python usage block executed successfully,
  selecting c at total cost 2.0 and stopping for confidence.
- A version assertion checked project, imported package and installed
  distribution metadata: all **0.1.0**. The package import succeeded.

All application checks used synthetic data on CPU without network calls or
external files. Package installation can access a package index; runtime,
tests and the demonstration do not require one. No other project environment
or repository was used as a runtime dependency or modified.

## Test coverage

The suite covers initialization, immutable snapshots, sample identity,
acquisition history/cost updates, duplicate/unknown modalities, fixed costs,
zero costs, entropy and probability validation, deterministic utility scores,
cost-weight behavior, greedy maxima, lexical ties, utility stopping, confidence
thresholds, budget filters, composed rules, complete episodes, missing forecasts,
lookup class dimensions, per-episode metrics, batch accuracy and duplicate
physical-sample rejection, and package/distribution version consistency.

Three additional regression cases beyond the initial 65-test suite address
concrete review findings: unrepresentable positive-cost efficiency is rejected,
externally constructed episode results cannot decrease cumulative cost, and
large finite batch costs average without intermediate overflow. These tests
account for the final count of 68.

## Actual synthetic demonstration

The hand-authored table is a forecast mechanism, not a learned predictive model
or evidence of real acquisition performance.

- Initial evidence: macro, confidence 0.55, caller-supplied initial cost 1.0.
- Greedy scorer: entropy reduction minus 0.03 times modality cost.
- Confidence threshold: 0.90; maximum cumulative budget: 10.0.
- Additional sequence: spectral, microscopy_high.
- Additional acquisition steps: 2; total known modalities: 3.
- Final class index: 0; final confidence: **0.98**.
- Final cumulative cost: **9.0**, including initial evidence cost.
- Stop reason: **confidence**.
- Final entropy: 0.09803911327973205 nats.
- Net episode entropy reduction: 0.5900997004338564 nats.
- Entropy reduction per newly spent cost unit: 0.07376246255423205.

These are constructed logic checks, not held-out metrics, scientific results,
or globally optimal policy claims. The configuration YAML is illustrative;
it is not automatically loaded by the demo.

## CI status

`.github/workflows/tests.yml` exists. It uses official checkout/setup-python
major v7, ubuntu-latest and Python **3.12**, with pushes and pull requests
restricted to main. Named steps install development dependencies, compile,
run pytest, execute the synthetic demo and verify version 0.1.0.

The full YAML parsed successfully with duplicate mapping keys rejected; event
branches, runner, Python version and named steps were checked. Its exact inline
version block compiled and executed locally. No secrets, GPU, external dataset
or post-install application network access are needed.

**GitHub-hosted CI has not run.** The hosted Python 3.12 / Ubuntu job must run
after publication. Local validation used Python 3.11.15 and is not a claim of
hosted success. The README badge points to the intended workflow and remains
pending the first public push.

## Public-content and scope audit

The public files below were scanned for Windows absolute paths, user-home
paths, local file URLs and credential assignments: no private paths, credentials,
API keys, access tokens or secrets were found. No dataset, model checkpoint,
measured spectrum, embedding table or archive is bundled. Every fixture and
forecast is synthetic. Ignored environment/build/log files can contain local
installation paths; they are excluded from the public inventory and Git.

Only this new repository was written. The scientific source project and both
previous portfolio repositories were untouched. No GitHub remote is configured,
and no push, tag or release is performed.

## Complete final public repository tree

Local virtual environment, logs, build metadata, bytecode and Git internals
are intentionally excluded.

```text
.github/workflows/tests.yml
.gitignore
LICENSE
README.md
assets/README.md
configs/example.yaml
docs/architecture.md
docs/decision_process.md
docs/validation_report.md
examples/synthetic_acquisition_demo.py
pyproject.toml
src/adaptive_acquisition/__init__.py
src/adaptive_acquisition/costs/__init__.py
src/adaptive_acquisition/costs/models.py
src/adaptive_acquisition/evaluation/__init__.py
src/adaptive_acquisition/evaluation/metrics.py
src/adaptive_acquisition/policies/__init__.py
src/adaptive_acquisition/policies/greedy.py
src/adaptive_acquisition/policies/threshold.py
src/adaptive_acquisition/simulation/__init__.py
src/adaptive_acquisition/simulation/episode.py
src/adaptive_acquisition/simulation/lookup.py
src/adaptive_acquisition/state/__init__.py
src/adaptive_acquisition/state/acquisition_state.py
src/adaptive_acquisition/stopping/__init__.py
src/adaptive_acquisition/stopping/rules.py
src/adaptive_acquisition/utility/__init__.py
src/adaptive_acquisition/utility/entropy.py
src/adaptive_acquisition/utility/scoring.py
src/adaptive_acquisition/validation/__init__.py
src/adaptive_acquisition/validation/inputs.py
tests/conftest.py
tests/test_costs_entropy.py
tests/test_scoring_policy.py
tests/test_simulation_metrics.py
tests/test_state.py
tests/test_stopping.py
```

## Known limitations and scientific contracts

- Greedy selection is one-step and may miss globally better acquisition sequences.
  There is no RL, active-learning trainer, learned router or Bayesian optimality.
- A supplied post-acquisition probability vector is a point forecast. Its entropy
  is not generally the expected posterior entropy over measurement outcomes.
  No actual unacquired measurement should be exposed to a deployed policy.
- Probabilities use fixed caller-declared class order and are not calibrated by
  this package. High confidence can accompany an incorrect prediction.
- Physical IDs are explicit, but split membership, lineage aliases and model
  training remain external responsibilities. Batch summaries reject duplicate
  sample IDs; they do not treat repeated seeds as independent physical entities.
- Cost units must be common and declared. Initial evidence cost is supplied by
  the caller. Budgets use strict floating <= comparisons; integer cost units
  are preferable when exact decimal boundaries matter.
- Zero additional cost gives an undefined efficiency ratio, represented by None.
  A positive denominator producing a nonfinite ratio raises ValueError. Efficiency
  measures net entropy change per new cost, not scientific correctness.
- Lookups bind one sample ID to immutable subset forecasts and fail on missing
  states. Real models require a compatible provider/scorer and validated forecasts.
- Exhaustion takes first terminal priority; configured rules otherwise retain
  AnyStop order. All budget filters apply before any candidate lookahead.
- Stopping/decision trace and state acquisition history are explicit. Initial
  evidence is not counted as a new episode step or charged twice.

## Git handoff

Implementation and validation completed before Git initialization. Inspect
`git status`, the staged file list and `git diff --cached --check` before the
single local initial commit. The final response records its actual hash and
resulting status. This report cannot embed its own commit hash without changing
that hash. No remote, push, tag or release is part of this task.
