from __future__ import annotations
import importlib.metadata
import platform
import ssl
import sys


def inspect_hardware(requested):
    import psutil
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "ssl": ssl.OPENSSL_VERSION, "cpu_threads": psutil.cpu_count(),
              "physical_ram_gib": psutil.virtual_memory().total / 1024**3,
              "requested_device": requested, "preparation_device": "cpu", "torch_version": None,
              "cuda_available": False, "gpus": [], "torch_error": None}
    try:
        report["torch_version"] = importlib.metadata.version("torch")
        import torch
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                report["gpus"].append({"index": i, "name": prop.name, "memory_gib": prop.total_memory / 1024**3})
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    if requested in ("cuda", "gpu") and not report["cuda_available"]:
        raise RuntimeError("CUDA was requested but is unavailable. Check your CUDA PyTorch environment, or use --device auto for Step 1 preparation.")
    report["future_model_device"] = "cuda" if requested != "cpu" and report["cuda_available"] else "cpu"
    report["model_assets"] = "Large encoder/native-model assets are resolved and pinned before their Step 2/3 scored pilots. Step 1 performs no downloads or training."
    return report
