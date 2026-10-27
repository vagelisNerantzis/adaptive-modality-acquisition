"""Finite nonnegative costs and explicit probability/entropy contracts."""
import math
import numpy as np
import pytest
from adaptive_acquisition.costs.models import FixedCostModel
from adaptive_acquisition.utility.entropy import entropy


def test_fixed_cost_lookup_is_deterministic():
    model = FixedCostModel({"a": 1., "b": 2.5})
    assert model.cost("b") == 2.5
    assert model.cost("b") == model.cost("b")


def test_zero_acquisition_cost_is_supported():
    assert FixedCostModel({"a": 0}).cost("a") == 0


def test_unknown_cost_request_fails_clearly():
    with pytest.raises((ValueError, KeyError)) as error:
        FixedCostModel({"a": 1}).cost("b")
    assert str(error.value)


@pytest.mark.parametrize("bad_cost", [-1., float("nan"), float("inf")])
def test_invalid_cost_is_rejected(bad_cost):
    with pytest.raises(ValueError):
        FixedCostModel({"a": bad_cost})


def test_deterministic_distribution_has_zero_entropy():
    assert entropy([1., 0.]) == pytest.approx(0, abs=1e-15)


def test_uniform_entropy_and_batch_shape():
    assert entropy([.25] * 4) == pytest.approx(math.log(4))
    assert entropy([.25] * 4, normalized=True) == pytest.approx(1.)
    result = entropy([[1., 0.], [.5, .5]])
    np.testing.assert_allclose(result, [0., math.log(2)])


def test_probability_roundoff_is_normalized_within_documented_tolerance():
    result = entropy([.5, .5000001])
    assert result == pytest.approx(math.log(2), abs=1e-12)


@pytest.mark.parametrize("probabilities", [
    [.2, .2], [-.1, 1.1], [float("nan"), 0.],
    [float("inf"), 0.], [], [[.4, .6], [.5, .4]],
])
def test_invalid_probability_vectors_are_rejected(probabilities):
    with pytest.raises((ValueError, TypeError)):
        entropy(probabilities)
