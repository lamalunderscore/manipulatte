"""Import structure for recorder package."""

from . import attention


TensorRecorder = attention.TensorRecorder


__all__ = ("TensorRecorder",)

# Prevents from accessing anything except the exported symbols
try:
    del torch, attention  # type: ignore
except NameError:
    pass
