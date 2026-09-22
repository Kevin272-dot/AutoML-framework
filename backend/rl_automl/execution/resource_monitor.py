"""Resource monitoring (spec §27).

Samples CPU, RAM and GPU usage on a background thread while a pipeline trains, then
reports peaks. Must work on CPU-only machines: every GPU probe is guarded.

The monitor also enforces a memory ceiling, because a pipeline that quietly eats all
available RAM would take the whole run down with it. On breach the executor marks that
single experiment as failed and continues (spec §29).
"""

from __future__ import annotations

import contextlib
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from rl_automl.core.logging import get_logger

logger = get_logger("execution.resource_monitor")


@dataclass
class ResourceSnapshot:
    """Peak and mean usage observed over one measured section."""

    duration_s: float = 0.0
    peak_memory_mb: float = 0.0
    mean_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    mean_cpu_percent: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    peak_gpu_utilization: float = 0.0
    n_samples: int = 0
    memory_limit_exceeded: bool = False
    available_memory_mb: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "mean_memory_mb": round(self.mean_memory_mb, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "mean_cpu_percent": round(self.mean_cpu_percent, 2),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 2),
            "peak_gpu_utilization": round(self.peak_gpu_utilization, 2),
            "n_samples": self.n_samples,
            "memory_limit_exceeded": self.memory_limit_exceeded,
        }


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - no torch or no driver
        return False


def total_memory_mb() -> float:
    return float(psutil.virtual_memory().total / 1024 / 1024)


class ResourceMonitor:
    """Thread-based sampler. Use as a context manager around the section to measure."""

    def __init__(
        self,
        memory_limit_mb: float | None = None,
        interval_s: float = 0.1,
        include_children: bool = True,
    ) -> None:
        self.memory_limit_mb = memory_limit_mb
        self.interval_s = max(0.01, interval_s)
        self.include_children = include_children

        self._process = psutil.Process()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._memory_samples: list[float] = []
        self._cpu_samples: list[float] = []
        self._gpu_memory_peak = 0.0
        self._gpu_util_peak = 0.0
        self._start_time = 0.0
        self._limit_exceeded = False
        self._warnings: list[str] = []

        # Establish a baseline so the first cpu_percent read is meaningful.
        with contextlib.suppress(Exception):  # pragma: no cover - platform dependent
            self._process.cpu_percent(interval=None)

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> ResourceMonitor:
        self._start_time = time.perf_counter()
        self._stop.clear()
        # Take one sample immediately: a fast fit can finish before the sampler thread
        # wakes, which would otherwise report a peak of 0 MB for a model that clearly
        # used memory.
        with contextlib.suppress(Exception):  # pragma: no cover - best effort
            self._collect_once()
        self._thread = threading.Thread(
            target=self._sample_loop, name="resource-monitor", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> ResourceSnapshot:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.snapshot()

    def __enter__(self) -> ResourceMonitor:
        return self.start()

    def __exit__(self, *exc_info: object) -> bool:
        self.stop()
        return False

    # -- sampling ----------------------------------------------------------------

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._collect_once()
            except Exception as exc:  # pragma: no cover - never kill the run
                with self._lock:
                    self._warnings.append(f"monitor sampling error: {exc}")
                return

    def _collect_once(self) -> None:
        memory_mb = self._process_memory_mb()
        try:
            cpu_percent = float(self._process.cpu_percent(interval=None))
        except Exception:  # pragma: no cover
            cpu_percent = 0.0
        gpu_mem, gpu_util = self._gpu_usage()

        with self._lock:
            self._memory_samples.append(memory_mb)
            self._cpu_samples.append(cpu_percent)
            self._gpu_memory_peak = max(self._gpu_memory_peak, gpu_mem)
            self._gpu_util_peak = max(self._gpu_util_peak, gpu_util)
            if self.memory_limit_mb and memory_mb > self.memory_limit_mb:
                self._limit_exceeded = True

    def _process_memory_mb(self) -> float:
        # RSS of this process plus children: matplotlib/BLAS spawn workers that count
        # towards the run's real footprint.
        total = 0
        try:
            total += self._process.memory_info().rss
            if self.include_children:
                for child in self._process.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except psutil.Error:  # pragma: no cover
            return 0.0
        return total / 1024 / 1024

    @staticmethod
    def _gpu_usage() -> tuple[float, float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0, 0.0
            memory_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        except Exception:  # pragma: no cover - CPU only
            return 0.0, 0.0

        utilization = 0.0
        try:  # optional, richer GPU telemetry
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            utilization = 0.0
        return memory_mb, utilization

    # -- reporting ---------------------------------------------------------------

    def snapshot(self) -> ResourceSnapshot:
        with self._lock:
            memory = list(self._memory_samples)
            cpu = list(self._cpu_samples)
            limit_exceeded = self._limit_exceeded
            warnings = list(self._warnings)
            gpu_mem = self._gpu_memory_peak
            gpu_util = self._gpu_util_peak

        duration = time.perf_counter() - self._start_time if self._start_time else 0.0
        return ResourceSnapshot(
            duration_s=duration,
            peak_memory_mb=max(memory) if memory else 0.0,
            mean_memory_mb=sum(memory) / len(memory) if memory else 0.0,
            peak_cpu_percent=max(cpu) if cpu else 0.0,
            mean_cpu_percent=sum(cpu) / len(cpu) if cpu else 0.0,
            peak_gpu_memory_mb=gpu_mem,
            peak_gpu_utilization=gpu_util,
            n_samples=len(memory),
            memory_limit_exceeded=limit_exceeded,
            available_memory_mb=total_memory_mb(),
            warnings=warnings,
        )


def measure_memory_mb() -> float:
    """One-shot RSS reading of the current process tree."""
    monitor = ResourceMonitor(interval_s=0.05)
    return monitor._process_memory_mb()


def describe_environment() -> dict[str, Any]:
    """Environment facts worth recording on a run for reproducibility."""
    info: dict[str, Any] = {
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_mb": round(total_memory_mb(), 1),
        "disk_free_gb": round(shutil.disk_usage(".").free / 1024**3, 2),
        "gpu_available": gpu_available(),
    }
    if info["gpu_available"]:
        try:
            import torch

            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        except Exception:  # pragma: no cover
            pass
    return info


__all__ = [
    "ResourceMonitor",
    "ResourceSnapshot",
    "describe_environment",
    "gpu_available",
    "measure_memory_mb",
    "total_memory_mb",
]
