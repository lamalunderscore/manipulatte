"""Utility functions for dejavu implementation."""

from typing import Literal

import torch


class TensorRecorder(torch.nn.Module):
    """A class that implements the recording of Tensors, either in a Module, or a Model.

    Attributes:
        record_mode (Literal["first", "last", "all"]): Which record mode is used.
        data (List[torch.Tensor | Any] | torch.Tensor | None): The recorded data.

    """

    def __init__(self, name: str, record_mode: Literal["first", "last", "all"] = "first"):
        super().__init__()
        self.name = name
        self.record_mode = record_mode

        self._clear("both")

        # Map enum to forward methods
        self._forward_methods = {
            "all": self._forward_all,
            "last": self._forward_last,
            "first": self._forward_first,
        }

    def _clear(self, clear_mode: Literal["gen", "prefill", "both"] = "both"):
        # Initialize data based on mode
        clear_value = lambda: list() if self.record_mode == "all" else None  # noqa
        if clear_mode in ["gen", "both"]:
            self._gen_data = clear_value()
        if clear_mode in ["prefill", "both"]:
            self._prefill_data = clear_value()

    def _safe_get(self, some_data) -> list[torch.Tensor | None] | torch.Tensor | None:
        if isinstance(some_data, list):
            return [tensor.clone() for tensor in some_data]
        if isinstance(some_data, torch.Tensor):
            return some_data.clone()
        if some_data is None:
            print(
                f"Warning: recorder data of layer {self.name} is empty, returning None. (in TensorRecorder._safe_get)"
            )
            return None
        raise TypeError(
            f"Expected self._data to be of type list | torch.Tensor | None. Got {type(some_data)} instead."
        )

    def get_clear(
        self, get_mode: Literal["gen", "prefill"]
    ) -> list[torch.Tensor | None] | torch.Tensor | None:
        """Get and clear recorder data.

        Raises:
            TypeError: If self._data is of an unknown type.

        Returns:
            list[torch.Tensor | None] | torch.Tensor | None: A copy of the stored data.

        """
        assert get_mode in ["gen", "prefill"], "Invalid get_mode."
        if get_mode == "gen":
            return_data = self._safe_get(self._gen_data)
        else:
            return_data = self._safe_get(self._prefill_data)

        if isinstance(return_data, list):
            for tensor in return_data:
                assert isinstance(tensor, torch.Tensor), "Tensor not of type Tensor."
        else:
            assert isinstance(return_data, torch.Tensor) or (return_data is None), (
                "Tensor not of type Tensor | None."
            )

        self._clear(clear_mode=get_mode)
        return return_data

    def forward(self, x):
        if x.shape[2] == 1:
            attr_name = "_gen_data"
        else:
            attr_name = "_prefill_data"
        self._forward_methods[self.record_mode](x, attr_name)

    def _forward_all(self, x, attr_name):
        """Record all inputs in a list."""
        self_data = getattr(self, attr_name)
        assert isinstance(self_data, list) and isinstance(x, torch.Tensor), (
            f"Need self._data to be `list` and x to be `torch.Tensor`, but got {type(self_data)} and {type(x)}."
        )
        self_data.append(x.detach().cpu())

    def _forward_first(self, x, attr_name):
        if getattr(self, attr_name) is None:
            setattr(self, attr_name, x.detach().cpu())

    def _forward_last(self, x, attr_name):
        setattr(self, attr_name, x.detach().cpu())


__all__ = ("TensorRecorder",)
