"""Append-only timestamped history and live console/file diagnostics."""
from __future__ import annotations
import datetime as dt
import logging
import sys
import time
import uuid
from pathlib import Path
from .artifacts import stable_json


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class RunLog:
    def __init__(self, root, step):
        self.root = Path(root)
        self.step = step
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
        self.started = time.perf_counter()
        stage_name = {1: "step01_prepare", 2: "step02_native_refinement", 3: "step03_neural_pilot",
                      4: "step04_competitor_pilot", 5: "step05_replication", 6: "step06_frozen_test"}.get(step, f"step{step:02d}_unimplemented")
        self.folder = self.root / "outputs" / "runs" / self.run_id / stage_name
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.root / "logs").mkdir(exist_ok=True)
        self.log_path = self.root / "logs" / (self.run_id + ".log")
        self.logger = logging.getLogger("rank_refinement." + self.run_id)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        fmt.converter = time.gmtime
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(self.log_path, encoding="utf-8")):
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
        self.history("start", {"step": step})

    def history(self, event, details):
        record = {"utc": utc_now(), "local": dt.datetime.now().astimezone().isoformat(),
                  "run_id": self.run_id, "step": self.step, "event": event,
                  "elapsed_seconds": round(time.perf_counter() - self.started, 3), **details}
        with (self.root / "run_history.txt").open("a", encoding="utf-8", newline="\n") as f:
            f.write(stable_json(record) + "\n")

    def info(self, message, *args):
        self.logger.info(message, *args)

    def close(self):
        for h in list(self.logger.handlers):
            h.flush()
            h.close()
            self.logger.removeHandler(h)
