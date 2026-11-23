"""Description of the machine a measurement runs on (system, CPU, RAM, GPU, versions)."""

import importlib.metadata
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path

import psutil
import torch

NVIDIA_SMI_TIMEOUT = 10  # seconds


def machine_name() -> str:
    """Host name reduced to ``[a-z0-9-]`` (used in result file names)."""
    name = socket.gethostname().split(".")[0].lower()
    return re.sub(r"[^a-z0-9-]+", "-", name).strip("-") or "equipo"


def resolve_device(requested: str) -> str:
    """``auto`` -> ``cuda`` if available, else ``mps``, else ``cpu``; checks availability.

    Raises:
        ValueError: if the requested device is unknown or not available.
    """
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        return next(device for device, ok in available.items() if ok)
    if requested not in available:
        raise ValueError(f"unknown device '{requested}' (auto, cuda, mps, cpu)")
    if not available[requested]:
        raise ValueError(f"device '{requested}' is not available on this machine")
    return requested


def nvidia_smi(fields: str) -> list[str] | None:
    """Values of ``nvidia-smi --query-gpu=<fields>`` for the first GPU, or ``None``."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=NVIDIA_SMI_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    return [value.strip() for value in lines[0].split(",")] if lines else None


def _cpu_model() -> str:
    """Commercial CPU name (macOS ``sysctl``, Linux ``/proc/cpuinfo``)."""
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _gpu_info(device: str) -> dict:
    """GPU name, memory and driver of the measured device."""
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        driver = nvidia_smi("driver_version")
        return {
            "gpu": properties.name,
            "vram_gb": round(properties.total_memory / 2**30, 2),
            "capacidad_computo": f"{properties.major}.{properties.minor}",
            "driver_nvidia": driver[0] if driver else None,
        }
    if device == "mps":
        return {
            "gpu": f"Apple MPS ({_cpu_model()})",
            "vram_gb": None,  # unified memory: see ram_gb
            "memoria_recomendada_mps_gb": round(torch.mps.recommended_max_memory() / 2**30, 2),
        }
    return {"gpu": None, "vram_gb": None}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def machine_info(device: str, name: str) -> dict:
    """System, CPU, RAM, GPU and library versions of the machine called ``name``."""
    release = platform.release()
    return {
        "nombre": name,
        "sistema": platform.system(),
        "version_sistema": release,
        "plataforma": platform.platform(),
        "wsl": "microsoft" in release.lower(),
        "cpu": _cpu_model(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
        **_gpu_info(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": _version("torch-geometric"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
