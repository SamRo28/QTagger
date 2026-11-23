"""Background sampling of process RAM and GPU temperature during a measurement.

A daemon thread samples the resident memory of the process (alone and with its children,
the ``DataLoader`` workers) every ``ram_interval`` seconds and, on CUDA, reads the GPU
temperature with ``nvidia-smi`` every ``temperature_interval`` seconds (30 s by default).
"""

import threading
import time
from dataclasses import dataclass, field

import psutil

from qcclass.medicion.equipo import nvidia_smi

RAM_INTERVAL = 0.25  # seconds
TEMPERATURE_INTERVAL = 30.0  # seconds


@dataclass
class MonitorReading:
    """Peaks and samples collected since the last :meth:`ResourceMonitor.reset`."""

    ram_max_mb: float = 0.0
    ram_max_con_workers_mb: float = 0.0
    temperaturas: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        """RAM peaks and GPU temperature (max and mean, ``None`` without readings)."""
        temperatures = self.temperaturas
        return {
            "ram_max_mb": round(self.ram_max_mb, 1),
            "ram_max_con_workers_mb": round(self.ram_max_con_workers_mb, 1),
            "temperatura_gpu_max": max(temperatures) if temperatures else None,
            "temperatura_gpu_media": (
                round(sum(temperatures) / len(temperatures), 1) if temperatures else None
            ),
            "lecturas_temperatura": len(temperatures),
        }


class ResourceMonitor:
    """Context manager running the sampling thread (see the module docstring)."""

    def __init__(
        self,
        *,
        gpu_temperature: bool,
        ram_interval: float = RAM_INTERVAL,
        temperature_interval: float = TEMPERATURE_INTERVAL,
    ) -> None:
        self.gpu_temperature = gpu_temperature
        self.ram_interval = ram_interval
        self.temperature_interval = temperature_interval
        self.process = psutil.Process()
        self.reading = MonitorReading()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="monitor", daemon=True)

    def __enter__(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()

    def reset(self) -> MonitorReading:
        """Return the reading so far and start a new one (one per epoch); samples RAM now."""
        self.sample_ram()
        with self._lock:
            reading, self.reading = self.reading, MonitorReading()
        return reading

    def sample_ram(self) -> None:
        """Update the RAM peaks with the current resident memory."""
        try:
            own = self.process.memory_info().rss
            children = sum(
                child.memory_info().rss for child in self.process.children(recursive=True)
            )
        except psutil.Error:  # a worker may exit while being read
            return
        with self._lock:
            self.reading.ram_max_mb = max(self.reading.ram_max_mb, own / 2**20)
            self.reading.ram_max_con_workers_mb = max(
                self.reading.ram_max_con_workers_mb, (own + children) / 2**20
            )

    def _sample_temperature(self) -> None:
        values = nvidia_smi("temperature.gpu")
        if values and values[0].replace(".", "", 1).isdigit():
            with self._lock:
                self.reading.temperaturas.append(float(values[0]))

    def _run(self) -> None:
        next_temperature = time.monotonic()
        while not self._stop.is_set():
            self.sample_ram()
            if self.gpu_temperature and time.monotonic() >= next_temperature:
                self._sample_temperature()
                next_temperature = time.monotonic() + self.temperature_interval
            self._stop.wait(self.ram_interval)
