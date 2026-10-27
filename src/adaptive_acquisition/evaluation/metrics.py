"""Precisely defined per-episode and unique-physical-sample summary metrics."""
from collections.abc import Sequence
from numbers import Integral
import math
import numpy as np
from ..simulation.episode import EpisodeResult
from ..utility.entropy import entropy


def episode_metrics(result: EpisodeResult, true_label: int | None = None) -> dict[str, float | int | bool | None]:
    """Compute cumulative cost, added steps, final uncertainty and optional correctness.

    Entropy reduction per cost divides net episode entropy reduction by NEW
    acquisition cost; zero additional cost yields None, never fabricated infinity.
    """
    if not isinstance(result, EpisodeResult):
        raise TypeError('result must be an EpisodeResult')
    correct = None
    if true_label is not None:
        if isinstance(true_label, (bool, np.bool_)) or not isinstance(true_label, Integral):
            raise TypeError('true_label must be an integer class index')
        if not 0 <= true_label < len(result.final_probabilities):
            raise ValueError('true_label is outside the declared class dimension')
        correct = bool(int(np.argmax(result.final_probabilities)) == int(true_label))
    final_entropy = float(entropy(result.final_probabilities))
    reduction = float(entropy(result.initial_state.probabilities)) - final_entropy
    efficiency = reduction / result.additional_cost if result.additional_cost > 0 else None
    if efficiency is not None and not math.isfinite(efficiency):
        raise ValueError('Entropy reduction per cost is not representable as a finite float')
    return {'total_cost': result.total_cost, 'acquisition_steps': result.n_steps,
            'n_acquired_modalities': len(result.final_state.acquired_modalities),
            'final_confidence': result.final_confidence, 'final_entropy': final_entropy,
            'correct': correct, 'entropy_reduction': reduction,
            'entropy_reduction_per_cost': efficiency}


def batch_summary(results: Sequence[EpisodeResult], true_labels: Sequence[int] | None = None) -> dict[str, float | int | None]:
    """Average episodes over distinct physical IDs; accuracy requires all labels.

    Repeated seeds/episodes for one sample must be handled separately instead of
    being treated here as independent physical samples. Labels share class order.
    """
    results = tuple(results)
    if not results:
        raise ValueError('Cannot summarize an empty episode batch')
    if not all(isinstance(result, EpisodeResult) for result in results):
        raise TypeError('Every batch item must be an EpisodeResult')
    ids = [result.final_state.sample_id for result in results]
    if len(set(ids)) != len(ids):
        raise ValueError('Batch contains duplicate physical sample IDs')
    if len({len(result.final_probabilities) for result in results}) != 1:
        raise ValueError('Batch results have inconsistent class dimensions')
    if true_labels is not None and len(true_labels) != len(results):
        raise ValueError('true_labels length must match episode count')
    metrics = [episode_metrics(result, None if true_labels is None else true_labels[i]) for i,result in enumerate(results)]
    return {'n_samples': len(results),
            'mean_total_cost': math.fsum(m['total_cost'] / len(metrics) for m in metrics),
            'mean_acquisition_steps': math.fsum(m['acquisition_steps'] / len(metrics) for m in metrics),
            'mean_final_confidence': math.fsum(m['final_confidence'] / len(metrics) for m in metrics),
            'accuracy': math.fsum(m['correct'] / len(metrics) for m in metrics) if true_labels is not None else None}
