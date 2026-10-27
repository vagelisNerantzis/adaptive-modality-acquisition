"""Shared artificial classification beliefs; no external observations are read."""
import pytest
from adaptive_acquisition.state.acquisition_state import AcquisitionState
from adaptive_acquisition.costs.models import FixedCostModel
from adaptive_acquisition.simulation.lookup import LookupPredictor


@pytest.fixture
def case_factory():
    """Build an independent synthetic physical sample and its belief lookup."""
    def build(sample_id="sample_a"):
        state = AcquisitionState(sample_id, ("a", "b", "c"), (.55, .45),
                                 acquired_modalities=("a",))
        costs = FixedCostModel({"a": 0., "b": 1., "c": 2.})
        predictor = LookupPredictor(sample_id, {
            frozenset({"a"}): (.55, .45),
            frozenset({"a", "b"}): (.80, .20),
            frozenset({"a", "c"}): (.70, .30),
            frozenset({"a", "b", "c"}): (.96, .04),
        })
        return state, costs, predictor
    return build
