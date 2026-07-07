import torch
from tensordict import TensorDict
from dataclasses import dataclass
from typing import Any, Iterator


class TDView:
    def __init__(self, base: TensorDict):
        self._base: TensorDict = base

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self._base.keys())})"

    def __getattr__(self, name: str | tuple[str, ...]) -> Any:
        value: torch.Tensor = self._base[name]
        if isinstance(value, TensorDict):
            return TDView(value)
        return value


class NextTransitionTD(TDView):
    observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class TransitionTD(TDView):
    observation: torch.Tensor
    action: torch.Tensor
    next: NextTransitionTD
    index: torch.Tensor | None
    gamma: torch.Tensor | None
    priority_weight: torch.Tensor | None


@dataclass(slots=True)
class Transition:
    td: TensorDict

    @property
    def raw(self) -> TransitionTD:
        return TransitionTD(self.td)


def to_transition(td: TensorDict) -> Transition:
    accepted_keys: tuple[str, tuple[str, ...]] = ("observation",
                                                  "action",
                                                  "gamma",
                                                  "priority_weight",
                                                  "index",
                                                  ("next", "reward"),
                                                  ("next", "done"),
                                                  ("next", "observation"))
    iterator: Iterator[tuple[Any, torch.Tensor]] = td.items(include_nested=True)
    filtered: dict[str, torch.Tensor] = {key: tensor for key, tensor in iterator if key in accepted_keys}
    tensordict: TensorDict = TensorDict(filtered, batch_size=td.batch_size).to(td.device)
    return Transition(tensordict)
