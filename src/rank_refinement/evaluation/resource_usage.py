"""CPU RSS is distinct from GPU allocated memory; no GPU work in preparation."""
import threading
import time


class ResourceMonitor:
    def __init__(self):
        import psutil
        import os
        # Managed Linux PID namespaces can expose a different process through
        # /proc/<pid>. RUSAGE_SELF identifies this process without a PID lookup.
        if os.name == "nt":
            self.process = psutil.Process()
            self.process.memory_info()
        else:
            self.process = None
        self.measurement_source = "psutil_rss" if self.process is not None else "resource_self_peak_rss"
        self.peak = self._rss()
        self.started = time.perf_counter()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.done.wait(0.1):
            self.peak = max(self.peak, self._rss())

    def finish(self):
        self.peak = max(self.peak, self._rss())
        self.done.set()
        self.thread.join(timeout=1)
        return {"wall_seconds": time.perf_counter() - self.started, "peak_cpu_rss_gib": self.peak / 1024**3,
                "gpu_peak_allocated_gib": 0.0, "gpu_work_in_step01": False, "sampling_interval_seconds": 0.1,
                "measurement_source": self.measurement_source}
    def _rss(self):
        if self.process is not None:
            return self.process.memory_info().rss
        import resource
        import sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)

