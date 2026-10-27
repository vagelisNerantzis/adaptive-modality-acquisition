"""Deterministic synthetic acquisition: forecasts illustrate logic, not learned skill."""
from itertools import combinations
from adaptive_acquisition.state.acquisition_state import AcquisitionState
from adaptive_acquisition.costs.models import FixedCostModel
from adaptive_acquisition.utility.scoring import EntropyReductionScorer
from adaptive_acquisition.policies.greedy import GreedyPolicy
from adaptive_acquisition.stopping.rules import AnyStop, ConfidenceThresholdStop, BudgetStop
from adaptive_acquisition.simulation.lookup import LookupPredictor
from adaptive_acquisition.simulation.episode import run_episode
from adaptive_acquisition.evaluation.metrics import episode_metrics


def main() -> None:
    """Print candidates, decisions, confidence and costs for one synthetic specimen."""
    sample_id = 'synthetic_specimen_001'
    costs = FixedCostModel({'macro': 1., 'spectral': 2., 'microscopy_low': 3., 'microscopy_high': 6.})
    optional = ('spectral', 'microscopy_low', 'microscopy_high')
    # Construct every synthetic evidence subset; no measurements or labels are read.
    table = {}
    for size in range(4):
        for subset in combinations(optional, size):
            confidence = min(.98, .55 + .20 * ('spectral' in subset)
                             + .12 * ('microscopy_low' in subset) + .28 * ('microscopy_high' in subset))
            table[frozenset(('macro', *subset))] = (confidence, 1-confidence)
    predictor = LookupPredictor(sample_id, table)
    initial = AcquisitionState(sample_id, tuple(costs.costs), table[frozenset({'macro'})], ('macro',), total_cost=1.)
    result = run_episode(initial, predictor, costs, EntropyReductionScorer(cost_weight=.03),
                         GreedyPolicy(), AnyStop((ConfidenceThresholdStop(.90), BudgetStop(10.))))
    print('Synthetic forecasts only; no learned or optimal-policy performance claim.')
    print(f'Initial evidence: {initial.acquired_modalities}; confidence={max(initial.probabilities):.3f}; cost={initial.total_cost:.2f}')
    for decision in result.decision_history:
        if decision.candidate_utilities:
            print('Candidate utilities:', ', '.join(f'{name}={score:.6f}' for name,score in decision.candidate_utilities))
        if decision.selected_modality is not None:
            event = result.final_state.history[decision.state.step_index]
            print(f'Selected modality: {event.modality}; updated confidence={max(event.probabilities):.3f}; accumulated cost={event.total_cost:.2f}')
    print('Final sequence:', result.selected_modalities)
    print('Final prediction (class index):', max(range(len(result.final_probabilities)), key=result.final_probabilities.__getitem__))
    print(f'Final confidence: {result.final_confidence:.3f}')
    print(f'Total cost: {result.total_cost:.2f}; added steps: {result.n_steps}; stop reason: {result.stop_reason}')
    print('Defined acquisition metrics:', episode_metrics(result, true_label=0))


if __name__ == '__main__':
    main()
