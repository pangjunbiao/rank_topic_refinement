"""Single portable entrypoint. Only the delivered stage may execute."""
from __future__ import annotations
import argparse
import io
import os
import sys
import threading
import traceback
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rank-refinement staged experiments")
    parser.add_argument("--step", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    parser.add_argument("--self-check", action="store_true", help="Run independent delivered-stage checks without fitting real data")
    parser.add_argument("--doctor-only", action="store_true", help="Record dependency/hardware checks only")
    parser.add_argument("--finalize", action="store_true", help="Step 2 only: verify and finalize an existing completed pilot without model training")
    parser.add_argument("--finish-reports", action="store_true", help="Step 6 only: finish reports from the already saved frozen test predictions")
    parser.add_argument("--repair", action="store_true", help="Step 3 only: repair the saved readout pilot with zero neural fits or encoder calls")
    parser.add_argument("--complete", action="store_true", help="Step 3 only: revised no-moment saved pilot and bounded classical competitors; no neural training")
    parser.add_argument("--improve", action="store_true", help="Step 3 only: fitting-only complete TOP10 acquisition on the selected saved neural base; no model retraining")
    parser.add_argument("--recover", action="store_true", help="Step 5 only: named saved-parent recovery, then complete locked competitors if its gate passes")
    parser.add_argument("--prepare-only", action="store_true", help="Step 7 only: inspect and prepare the external corpus without model fitting")
    parser.add_argument("--external-input", default=None, help="Step 7 only: path to the inspected transit CSV (default: data/raw/external)")
    parser.add_argument("--offline", action="store_true", help="Step 7 only: require locally cached pinned English encoder assets")
    parser.add_argument("--compact-results", action="store_true", help="Step 8 only: compact paper exhibits and native-versus-Proposed divergence/affinity analysis")
    parser.add_argument("--primary-affinity", action="store_true", help="Step 8 only: reproducible English primary affinity figure and affinity divergence table")
    parser.add_argument("--sensitivity", action="store_true", help="Step 8 only: primary support and rank-margin saved-array diagnostics")
    parser.add_argument("--sensitivity-budget", action="store_true", help="Step 8 only: replay support, rank-margin and replacement-limit diagnostics with paper-style figures")
    parser.add_argument("--sensitivity-recompute", action="store_true", help="With --sensitivity: independently solve alternative settings in a new cache")
    args = parser.parse_args()
    if args.sensitivity_budget and (args.step != 8 or any([args.sensitivity, args.sensitivity_recompute, args.primary_affinity, args.compact_results, args.doctor_only, args.finalize, args.finish_reports, args.repair, args.complete, args.improve, args.recover, args.prepare_only, args.external_input, args.offline])):
        parser.error("Use --sensitivity-budget only with --step 8; --self-check and --device are allowed.")
    if args.sensitivity_recompute and (not args.sensitivity or args.self_check):
        parser.error("Use --sensitivity-recompute with --step 8 --sensitivity, without --self-check.")
    if args.sensitivity and (args.step != 8 or any([args.primary_affinity, args.compact_results, args.doctor_only, args.finalize, args.finish_reports, args.repair, args.complete, args.improve, args.recover, args.prepare_only, args.external_input, args.offline])):
        parser.error("Use --sensitivity only with --step 8; --self-check and --device are allowed.")
    if args.primary_affinity and (args.step != 8 or any([args.compact_results, args.doctor_only, args.finalize, args.finish_reports, args.repair, args.complete, args.improve, args.recover, args.prepare_only, args.external_input, args.offline])):
        parser.error("Use --primary-affinity only with --step 8; --self-check and --device are allowed.")
    if args.compact_results and args.step != 8:
        parser.error("Use --compact-results only with --step 8.")
    if (args.prepare_only or args.external_input or args.offline) and args.step != 7:
        parser.error("External execution flags are available only with --step 7.")
    if args.step == 7 and args.prepare_only and (args.self_check or args.doctor_only):
        parser.error("Use --prepare-only, --self-check or --doctor-only separately.")
    if args.finish_reports and (args.step != 6 or args.self_check or args.doctor_only or args.finalize or args.repair or args.complete or args.improve or args.recover):
        parser.error("Use --finish-reports only with --step 6, without other execution/revision flags.")
    if args.recover and (args.step != 5 or args.improve or args.complete or args.repair or args.finalize):
        parser.error("Use --recover only with --step 5, without other stage revision flags.")
    if args.self_check and args.doctor_only:
        parser.error("Choose --self-check or --doctor-only, not both.")
    if args.finalize and (args.step != 2 or args.self_check or args.doctor_only):
        parser.error("Use --finalize only with --step 2, without --self-check or --doctor-only.")
    if args.complete and (args.step != 3 or args.repair or args.finalize):
        parser.error("Use --complete only with --step 3, without --repair or --finalize.")
    if args.repair and (args.step != 3 or args.finalize):
        parser.error("Use --repair only with --step 3, without --finalize.")
    if args.improve and (args.step != 3 or args.complete or args.repair or args.finalize):
        parser.error("Use --improve only with --step 3, without --complete, --repair or --finalize.")
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        print("Use Python 3.10, 3.11 or 3.12. Check the PyCharm interpreter; Python 3.8 is unsupported.", file=sys.stderr)
        return 2
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    root = Path(__file__).resolve().parent
    # Stage 2 already limits numerical workers to one. Avoid obsolete Windows
    # physical-core probing by loky; honor any explicit user override.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(variable, "1")
    sys.path.insert(0, str(root / "src"))
    if args.step == 8 and args.sensitivity_budget:
        from sensitivity_analysis.budget import cli
        return cli(args, root)
    if args.step == 8 and args.sensitivity:
        from sensitivity_analysis.runner import cli
        return cli(args, root)
    if args.step == 8 and args.primary_affinity:
        from primary_affinity_analysis.runner import cli
        return cli(args, root)
    if args.step == 7:
        from step07_external.runner import cli
        return cli(args, root)
    if args.step == 8 and args.compact_results:
        from step08_compact.runner import cli
        return cli(args, root)
    if args.step == 8:
        from step08_assembly.runner import cli
        return cli(args, root)
    from rank_refinement.runtime.paths import prepare_output_directories
    from rank_refinement.runtime.logging import RunLog
    from rank_refinement.runtime.artifacts import digest, write_json, write_text
    from rank_refinement.reporting.export import make_inspection_zip
    prepare_output_directories(root)
    log = RunLog(root, args.step)
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(30):
            log.info("RUNNING | step=%d | run=%s | detailed phase log=%s", args.step, log.run_id, log.log_path)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    processed, monitor, result = None, None, None
    report_only_mode = False
    exit_code = 1
    try:
        log.info("START | Step %d | run=%s | requested_device=%s", args.step, log.run_id, args.device)
        from rank_refinement.runtime.config import check_environment, load_configs
        if args.improve:
            from rank_refinement.runtime.config import PINS
            from rank_refinement.runtime.step02_contract import CORE_PINS, dependency_versions
            dependency_versions({**PINS, **CORE_PINS, "osqp": "1.0.4"}, "requirements-step03-improve.txt")
        versions = check_environment()
        if args.step in [4, 5, 6]:
            from rank_refinement.stages.step04_primary_benchmark import dependencies
            versions.update(dependencies())
        from rank_refinement.evaluation.resource_usage import ResourceMonitor
        monitor = ResourceMonitor()
        from rank_refinement.runtime.hardware import inspect_hardware
        hardware = inspect_hardware(args.device)
        configs = load_configs(root)
        write_json(log.folder / "hardware.json", hardware)
        write_json(log.folder / "versions.json", versions)
        log.info("DEVICE | models/encoder=%s | preparation and graph=CPU | CUDA=%s", hardware["future_model_device"], hardware["cuda_available"])
        for gpu in hardware["gpus"]:
            log.info("GPU | %s | VRAM=%.2f GiB", gpu["name"], gpu["memory_gib"])
        log.info("ENVIRONMENT | Python=%s | RAM=%.2f GiB | core dependency pins=PASS", sys.version.split()[0], hardware["physical_ram_gib"])
        if args.step not in [1, 2, 3, 4, 5, 6]:
            raise RuntimeError("Steps 1–7 are delivered. Review the saved primary and external results before Step 8 paper assembly.")
        if args.step == 6:
            from rank_refinement.stages.step06_frozen_test import validate_protocol
            from rank_refinement.models.neural_sources import resolve_sources as resolve_native_sources
            from rank_refinement.models.competitor_sources import resolve_sources as resolve_competitor_sources
            validate_protocol(configs, root)
            if args.self_check:
                resolve_native_sources(root, log)
                resolve_competitor_sources(root, log)
            os.environ["RANK_STEP03_REQUIRED_CHECKS"] = "1"
            os.environ["RANK_STEP04_REQUIRED_CHECKS"] = "1"
            os.environ["RANK_STEP06_REQUIRED_CHECKS"] = "1"
            write_json(log.folder / "versions.json", versions)
        if args.step == 5:
            if args.recover:
                from rank_refinement.stages.step05_recovery import validate_protocol
            else:
                from rank_refinement.stages.step05_ablations import validate_protocol
            from rank_refinement.models.neural_sources import resolve_sources as resolve_native_sources
            from rank_refinement.models.competitor_sources import resolve_sources as resolve_competitor_sources
            validate_protocol(configs, root)
            if args.self_check:
                resolve_native_sources(root, log)
                resolve_competitor_sources(root, log)
            os.environ["RANK_STEP03_REQUIRED_CHECKS"] = "1"
            os.environ["RANK_STEP04_REQUIRED_CHECKS"] = "1"
        if args.step == 4:
            from rank_refinement.stages.step04_primary_benchmark import validate_protocol
            from rank_refinement.models.competitor_sources import resolve_sources as resolve_competitor_sources
            validate_protocol(configs, root)
            if args.self_check:
                resolve_competitor_sources(root, log)
            os.environ["RANK_STEP04_REQUIRED_CHECKS"] = "1"
            write_json(log.folder / "versions.json", versions)
        if args.step == 2:
            from rank_refinement.runtime.step02_contract import CORE_PINS, dependency_versions
            from rank_refinement.stages.step02_native_refinement import validate_protocol
            versions.update(dependency_versions(CORE_PINS, "requirements-step02.txt"))
            validate_protocol(configs)
            write_json(log.folder / "versions.json", versions)
        if args.step == 3 and args.improve:
            from rank_refinement.runtime.step02_contract import CORE_PINS, dependency_versions
            from rank_refinement.stages.step03_improvement import validate_improvement
            versions.update(dependency_versions({**CORE_PINS, "osqp": "1.0.4"}, "requirements-step03-improve.txt"))
            validate_improvement(configs)
            os.environ["RANK_STEP03_REPLAY_CHECKS"] = "1"
            write_json(log.folder / "versions.json", versions)
        elif args.step == 3 and args.complete:
            from rank_refinement.runtime.step02_contract import CORE_PINS, dependency_versions
            from rank_refinement.stages.step03_completion import validate_completion
            versions.update(dependency_versions(CORE_PINS, "requirements-step02.txt"))
            validate_completion(configs)
            os.environ["RANK_STEP03_REPLAY_CHECKS"] = "1"
            write_json(log.folder / "versions.json", versions)
        elif args.step == 3 and args.repair:
            from rank_refinement.runtime.step02_contract import CORE_PINS, dependency_versions
            from rank_refinement.stages.step03_saved_replay import validate_repair
            versions.update(dependency_versions(CORE_PINS, "requirements-step02.txt"))
            validate_repair(configs)
            # Repair operates on saved arrays; it never imports/downloads author
            # modules or the encoder, and does not require a Torch runtime.
            os.environ["RANK_STEP03_REPLAY_CHECKS"] = "1"
            write_json(log.folder / "versions.json", versions)
        elif args.step == 3:
            from rank_refinement.runtime.step02_contract import CORE_PINS, dependency_versions
            from rank_refinement.stages.step03_neural_pilot import dependencies, validate_protocol
            from rank_refinement.models.neural_sources import resolve_sources
            versions.update(dependency_versions(CORE_PINS, "requirements-step02.txt"))
            versions.update(dependencies())
            validate_protocol(configs)
            # Seven small pinned author modules only. Self-check never downloads
            # the Chinese encoder or fits the primary corpus.
            resolve_sources(root, log)
            os.environ["RANK_STEP03_REQUIRED_CHECKS"] = "1"
            write_json(log.folder / "versions.json", versions)
        if args.doctor_only:
            result = {"status": "DOCTOR_COMPLETE", "engineering_passed": True, "preparation_gate_passed": False, "model_quality_established": False}
            exit_code = 0
        elif args.self_check:
            suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
            stream = io.StringIO()
            checked = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            write_text(log.folder / "self_checks.txt", stream.getvalue())
            log.info("SELF CHECK | tests=%d | failures=%d | errors=%d | skipped_optional=%d", checked.testsRun, len(checked.failures), len(checked.errors), len(checked.skipped))
            if checked.testsRun == 0 or not checked.wasSuccessful():
                log.info("%s", stream.getvalue())
                raise AssertionError("Independent checks failed; see self_checks.txt in the inspection ZIP.")
            result = {"status": "SELF_CHECK_PASSED", "tests": checked.testsRun, "skipped_optional": len(checked.skipped), "engineering_passed": True,
                      "preparation_gate_passed": False, "model_quality_established": False}
            exit_code = 0
        else:
            from rank_refinement.runtime.state import ProjectLock
            if args.step == 1:
                from rank_refinement.stages.step01_prepare import run
            elif args.step == 6:
                if args.finish_reports or (root / "state/step06_test_exposure.json").is_file():
                    from step06_finalize import run
                    report_only_mode = True
                else:
                    from rank_refinement.stages.step06_frozen_test import run
            elif args.step == 5:
                if args.recover:
                    from rank_refinement.stages.step05_recovery import run
                else:
                    from rank_refinement.stages.step05_ablations import run
            elif args.step == 4:
                from rank_refinement.stages.step04_primary_benchmark import run
            elif args.step == 3 and args.improve:
                from rank_refinement.stages.step03_improvement import run
            elif args.step == 3 and args.complete:
                from rank_refinement.stages.step03_completion import run
            elif args.step == 3 and args.repair:
                from rank_refinement.stages.step03_saved_replay import run
            elif args.step == 3:
                from rank_refinement.stages.step03_neural_pilot import run
            elif args.finalize:
                from rank_refinement.stages.step02_completion import run
            else:
                from rank_refinement.stages.step02_native_refinement import run
            with ProjectLock(root):
                result, processed = run(root, configs, versions, hardware, log)
            # A completed scientific diagnostic may reveal failure/NA. Reserve
            # software error codes for broken execution; print all gates explicitly.
            gate = "preparation_gate_passed" if args.step == 1 else "engineering_passed"
            exit_code = 0 if result[gate] else 3
        from rank_refinement.reporting.tables import export_table
        repro = [{"item": "python", "value": sys.version.split()[0]}, {"item": "requested_device", "value": args.device},
                 {"item": "future_model_device", "value": hardware["future_model_device"]},
                 {"item": "preparation_device", "value": "cpu"}, {"item": "configurations_sha256", "value": digest(configs)}]
        repro += [{"item": name, "value": version} for name, version in sorted(versions.items())]
        export_table(root / "outputs/paper/supplementary/tables", f"S1_step{args.step:02d}_reproducibility", repro, ["item", "value"])
        export_table(log.folder, f"S1_step{args.step:02d}_reproducibility", repro, ["item", "value"])
        log.info("VERDICT | %s | engineering=%s | preparation_gate=%s | model_quality=NOT_ESTABLISHED", result["status"], result["engineering_passed"], result["preparation_gate_passed"])
        if args.step == 2 and not args.self_check and not args.doctor_only:
            log.info("MECHANISM CHECK | promising=%s | complete reference=%s | accepted edits=%d/%d | full experiment ready=%s",
                result["mechanism_promising"], result["complete_coherence_reference_gate_passed"],
                result["fitting_accepted_cases"], result["completed_cases"], result["full_experiment_gate_passed"])
            if args.finalize:
                log.info("STEP 2 COMPLETE | bounded neural pilot allowed=%s | final competitor benchmark ready=False | new model fits=0",
                    result["bounded_neural_pilot_allowed"])
    except (Exception, KeyboardInterrupt) as exc:
        result = {"status": "FAILED" if not isinstance(exc, KeyboardInterrupt) else "INTERRUPTED",
                  "engineering_passed": False, "preparation_gate_passed": False, "model_quality_established": False,
                  "error_type": type(exc).__name__, "error": str(exc)}
        write_text(log.folder / "failure_traceback.txt", traceback.format_exc())
        log.logger.error("FAILED | %s: %s", type(exc).__name__, exc)
        working = log.folder / "preparation_working_path.json"
        if working.is_file():
            from rank_refinement.runtime.artifacts import read_json
            candidate = (root / read_json(working)["relative_path"]).resolve()
            if candidate.is_relative_to(root) and candidate.is_dir():
                processed = candidate
        exit_code = 2
    finally:
        stop.set()
        thread.join(timeout=1)
        if monitor is not None:
            resources = monitor.finish()
            if args.step == 2 and processed is not None and (processed / "encoder_embeddings_manifest.json").is_file():
                from rank_refinement.runtime.artifacts import read_json
                embedding_usage = read_json(processed / "encoder_embeddings_manifest.json")
                resources["gpu_peak_allocated_gib"] = embedding_usage["current_run_peak_cuda_allocated_gib"]
                resources["encoder_device"] = embedding_usage["device"]
                resources["gpu_work_in_step02"] = embedding_usage["device"] == "cuda" and not embedding_usage["reused_embeddings"]
            else:
                resources["gpu_work_in_step02"] = False
            write_json(log.folder / "resource_usage.json", resources)
            if args.step in [3, 4, 5, 6]:
                try:
                    import torch
                    resources["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.
                except ImportError:
                    resources["gpu_peak_allocated_gib"] = 0.
                resources["neural_device"] = hardware["future_model_device"] if "hardware" in locals() else None
                if args.step == 6:
                    resources.update(new_neural_fits=0,new_classical_fits=getattr(log,"new_classical_fits",0),
                                     test_parameters_selected=False,frozen_checkpoint_inference=not report_only_mode,
                                     report_only_completion=report_only_mode,new_encoder_calls=0 if report_only_mode else None,
                                     new_model_predictions=0 if report_only_mode else None)
                    log.info("RESOURCES | wall=%.2f sec | peak_CPU_RSS=%.3f GiB | frozen inference device=%s | GPU peak=%.3f GiB | neural fits=0 | classical fitting-only replays=%d",
                        resources["wall_seconds"],resources["peak_cpu_rss_gib"],resources["neural_device"],resources["gpu_peak_allocated_gib"],resources["new_classical_fits"])
                elif args.step == 5:
                    resources.update(new_neural_fits=getattr(log, "new_neural_fits", 0),
                                     new_classical_fits=getattr(log, "new_classical_fits", 0),
                                     proposed_shares_native_parent_fit=True)
                    log.info("RESOURCES | wall=%.2f sec | peak_CPU_RSS=%.3f GiB | neural device=%s | GPU peak=%.3f GiB | new neural fits=%d | new classical fits=%d",
                        resources["wall_seconds"], resources["peak_cpu_rss_gib"], resources["neural_device"], resources["gpu_peak_allocated_gib"],
                        resources["new_neural_fits"],resources["new_classical_fits"])
                elif args.step == 4:
                    resources.update(new_proposed_fits=0, new_neural_fits=getattr(log, "new_neural_fits", 0),
                                     new_classical_fits=getattr(log, "new_classical_fits", 0))
                    log.info("RESOURCES | wall=%.2f sec | peak_CPU_RSS=%.3f GiB | competitor neural device=%s | GPU peak=%.3f GiB | Proposed fits=0",
                        resources["wall_seconds"], resources["peak_cpu_rss_gib"], resources["neural_device"], resources["gpu_peak_allocated_gib"])
                elif args.repair or args.complete or args.improve:
                    # Completion can fit small classical models; replay/improve do not.
                    classical_fits = getattr(log, "new_classical_fits", 0)
                    resources.update(actual_compute_device="cpu", new_model_fits=classical_fits,
                                     new_neural_fits=0, new_classical_fits=classical_fits, new_encoder_calls=0,
                                     gpu_compute_in_repair=False)
                    log.info("RESOURCES | wall=%.2f sec | peak_CPU_RSS=%.3f GiB | saved readout/classical compute=CPU | new neural fits=0",
                        resources["wall_seconds"], resources["peak_cpu_rss_gib"])
                else:
                    log.info("RESOURCES | wall=%.2f sec | peak_CPU_RSS=%.3f GiB | neural/encoder device=%s | GPU peak=%.3f GiB",
                        resources["wall_seconds"], resources["peak_cpu_rss_gib"], resources["neural_device"], resources["gpu_peak_allocated_gib"])
                write_json(log.folder / "resource_usage.json", resources)
            else:
                log.info("RESOURCES | wall=%.2f sec | peak_CPU_RSS=%.3f GiB | preparation/readout pilot compute=CPU", resources["wall_seconds"], resources["peak_cpu_rss_gib"])
        write_json(log.folder / "run_result.json", result)
        log.history("inspection_snapshot", {"status": result["status"], "requested_device": args.device,
                    "preparation_gate_passed": result.get("preparation_gate_passed", False),
                    "preparation_id": result.get("preparation_id"), "model_quality_established": False})
        try:
            kind = ("selfcheck_" if args.self_check else "doctor_" if args.doctor_only else "finalized_" if args.finalize else "") + ("recovered_" if args.recover else "improved_r3_" if args.improve else "completed_r2_" if args.complete else "repaired_" if args.repair else "") + f"step{args.step:02d}"
            if args.step == 6:
                from step06_finalize import make_completion_inspection
                if report_only_mode:kind = "completed_reports_step06"
                inspection = make_completion_inspection(root, log, processed, kind=kind)
            else:
                inspection = make_inspection_zip(root, log, processed, kind=kind)
            log.info("INSPECTION ZIP | %s", inspection)
            log.info("REPORT | %s", log.folder / "run_result.json")
            log.history("end", {"status": result["status"], "exit_code": exit_code,
                                "inspection_zip": inspection.relative_to(root).as_posix()})
        except Exception as exc:
            log.logger.error("Inspection export failed: %s", exc)
            log.history("end", {"status": "INSPECTION_EXPORT_FAILED", "error": str(exc), "exit_code": 2})
            exit_code = 2
        log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
