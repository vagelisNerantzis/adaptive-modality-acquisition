"""Fixed acquisition costs in one user-declared common unit."""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from ..validation.inputs import identifier, nonnegative_number


@dataclass(frozen=True)
class FixedCostModel:
    """Immutable modality->cost snapshot, including valid zero-cost modalities."""
    costs: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.costs, Mapping):
            raise TypeError('costs must be a mapping')
        values = {identifier(name, name='modality'): nonnegative_number(cost, name='cost')
                  for name, cost in self.costs.items()}
        object.__setattr__(self, 'costs', MappingProxyType(values))

    def cost(self, modality: str) -> float:
        """Return a fixed nonnegative cost; unknown modalities fail explicitly."""
        identifier(modality, name='modality')
        if modality not in self.costs:
            raise ValueError(f'Unknown cost for modality {modality}')
        return self.costs[modality]
