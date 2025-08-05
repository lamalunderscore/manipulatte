"""Implementing the BackEnd class."""

from collections.abc import Callable
from gc import collect as cpu_clear

import torch
from psutil import virtual_memory as cpu_memory


class BackEnd:
    """Backend monitor for different devices.

    Attributes:
        allocated_memory (Union[Dict[str, float], float]): The currently allocated memory.
        device (str): The device that is used.

    """

    def __init__(self, byte_size: int = 1024**3):
        self._byte_size = byte_size
        if torch.cuda.is_available():
            self._allocated_memory = lambda: {
                str(i): round(torch.cuda.memory_allocated(torch.device(f"cuda:{i}")) / self._byte_size, 2)
                for i in range(torch.cuda.device_count())
            }
            self._peak_memory_reserved = lambda: {
                str(i): round(torch.cuda.max_memory_reserved(torch.device(f"cuda:{i}")) / self._byte_size, 2)
                for i in range(torch.cuda.device_count())
            }
            self._reset_peak = lambda: [
                torch.cuda.reset_peak_memory_stats(torch.device(f"cuda:{i}")) for i in range(torch.cuda.device_count())
            ]
            self._empty_cache: Callable[[], None] = torch.cuda.empty_cache
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            print("Warning, MPS Peak measures not implemented.")
            self._allocated_memory = lambda: {"0": round(torch.mps.current_allocated_memory() / self._byte_size, 2)}
            self._peak_memory_reserved = lambda: {"0": 0.0}
            self._reset_peak = lambda: None
            self._empty_cache = torch.mps.empty_cache
            self._device = "mps"
        else:
            print("Warning, CPU memory monitoring not implemented.")
            self._allocated_memory = lambda: {}
            self._peak_memory_reserved = lambda: {"0": 0.0}
            self._reset_peak = lambda: None
            self._empty_cache = lambda: None  # garbage collect is run every time
            self._device = "cpu"

    @property
    def allocated_memory(self) -> dict[str, float] | float:
        allocated_memory = self._allocated_memory()
        allocated_memory["cpu"] = round(cpu_memory().used / self._byte_size, 2)
        return allocated_memory

    @property
    def device(self) -> str:
        return self._device

    @property
    def peak_memory_reserved(self) -> dict[str, float]:
        return self._peak_memory_reserved()

    def empty_cache(self):
        self._empty_cache()
        cpu_clear()

    def reset_peak(self):
        self._reset_peak()
