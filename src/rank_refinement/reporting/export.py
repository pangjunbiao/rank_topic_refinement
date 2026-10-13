from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from ..runtime.artifacts import file_hash, write_json


def make_inspection_zip(root, run, processed_folder=None, kind="step01"):
    root = Path(root)
    destination = root / "outputs" / "inspection" / (kind + "_" + run.run_id + "_inspection.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = list(run.folder.rglob("*"))
    if processed_folder is not None:
        files += list(Path(processed_folder).rglob("*"))
        recovered_step05 = Path(processed_folder) / "reviewed_step05_recovery_reference.json"
        if recovered_step05.is_file():
            from ..runtime.artifacts import read_json
            from ..stages.step04_primary_benchmark import safe_step_folder
            info = read_json(recovered_step05)
            original = safe_step_folder(root, info["relative_path"], "step05_recovery")
            if file_hash(original / "artifact_manifest.json") != info["artifact_manifest_sha256"]:
                raise ValueError("Reviewed Step 5 R2 source changed before test inspection export.")
            files += list(original.rglob("*"))
        completed_step05 = Path(processed_folder) / "completed_step05_reference.json"
        if completed_step05.is_file():
            from ..runtime.artifacts import read_json
            from ..stages.step04_primary_benchmark import safe_step_folder
            info = read_json(completed_step05)
            original = safe_step_folder(root, info["relative_path"], "step05_replication")
            if file_hash(original / "artifact_manifest.json") != info["artifact_manifest_sha256"]:
                raise ValueError("Original Step 5 R1 source changed before recovery export.")
            files += list(original.rglob("*"))
        completed_panel = Path(processed_folder) / "completed_step04_reference.json"
        if completed_panel.is_file():
            from ..runtime.artifacts import read_json
            from ..stages.step04_primary_benchmark import safe_step_folder
            info = read_json(completed_panel)
            original = safe_step_folder(root, info["relative_path"], "step04_competitor_pilot")
            if file_hash(original / "artifact_manifest.json") != info["artifact_manifest_sha256"]:
                raise ValueError("Reviewed Step 4 source changed before inspection export.")
            files += list(original.rglob("*"))
        reference = Path(processed_folder) / "source_result_reference.json"
        if reference.is_file():
            from ..runtime.artifacts import read_json
            from ..stages.step02_completion import safe_source_path
            original = safe_source_path(root.resolve(), read_json(reference)["relative_path"])
            files += list(original.rglob("*"))
        frozen_proposed = Path(processed_folder) / "frozen_proposed_reference.json"
        if frozen_proposed.is_file():
            from ..runtime.artifacts import read_json
            from ..stages.step04_primary_benchmark import safe_step_folder
            info = read_json(frozen_proposed)
            original = safe_step_folder(root, info["relative_path"], "step03_coherent_refinement")
            if file_hash(original / "artifact_manifest.json") != info["artifact_manifest_sha256"]:
                raise ValueError("Frozen Proposed inspection reference changed.")
            files += list(original.rglob("*"))
        saved_neural = Path(processed_folder) / "source_neural_pilot_reference.json"
        if saved_neural.is_file():
            from ..runtime.artifacts import read_json
            info = read_json(saved_neural)
            rel = Path(info["relative_path"])
            if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 4 or rel.parts[:3] != ("outputs", "steps", "step03_neural_pilot"):
                raise ValueError("Unsafe saved neural-pilot inspection reference.")
            original = (root / rel).resolve()
            if not original.is_relative_to(root.resolve()) or file_hash(original / "artifact_manifest.json") != info["artifact_manifest_sha256"]:
                raise ValueError("Saved neural-pilot inspection source differs from its repair record.")
            files += list(original.rglob("*"))
        saved_repair = Path(processed_folder) / "repaired_saved_source_reference.json"
        if saved_repair.is_file():
            from ..runtime.artifacts import read_json
            info = read_json(saved_repair)
            rel = Path(info["relative_path"])
            if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 4 or rel.parts[:3] != ("outputs", "steps", "step03_saved_replay"):
                raise ValueError("Unsafe repaired-pilot inspection reference.")
            original = (root / rel).resolve()
            if not original.is_relative_to(root.resolve()) or file_hash(original / "artifact_manifest.json") != info["artifact_manifest_sha256"]:
                raise ValueError("Repaired-pilot inspection source differs from the completion record.")
            files += list(original.rglob("*"))
        saved_completion = Path(processed_folder) / "completed_pilot_reference.json"
        if saved_completion.is_file():
            from ..runtime.artifacts import read_json
            info = read_json(saved_completion)
            rel = Path(info["relative_path"])
            if (rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 4
                    or rel.parts[:3] != ("outputs", "steps", "step03_completion")):
                raise ValueError("Unsafe completed-pilot inspection reference.")
            original = (root / rel).resolve()
            if (not original.is_relative_to(root.resolve())
                    or file_hash(original / "artifact_manifest.json") != info["artifact_manifest_sha256"]):
                raise ValueError("Completed-pilot inspection source differs from the full-set record.")
            files += list(original.rglob("*"))
    files += [run.log_path, root / "run_history.txt"]
    files += list((root / "configs").glob("*.yaml"))
    files += list((root / "configs").glob("*.json"))
    if run.step in [2, 3, 4, 5, 6]:
        # The review bundle includes executed project code as well as outcomes;
        # it is not an installation archive and never embeds an older ZIP.
        files += [root / "main.py"] + list(root.glob("requirements*.txt"))
        files += list((root / "src").rglob("*.py"))
        files += list((root / "src/rank_refinement/vendor").glob("*.json"))
        files += list((root / "tests").glob("test_*.py"))
    if processed_folder is not None:
        files += [p for p in (root / "state").glob("step0[123456]*.json") if ".running." not in p.name]
        files += list((root / "outputs" / "paper" / "main" / "tables").glob("T1_data_protocol.*"))
        files += list((root / "outputs" / "paper" / "supplementary" / "tables").glob("S3_step01*"))
        if run.step in [2, 3, 4, 5, 6]:
            # Step 2R3 summaries remain in their own stage folder. Previous R2
            # paper exports do not belong to the current inspection snapshot.
            from ..runtime.artifacts import read_json
            accepted = root / "state/step01_accepted.json"
            if accepted.is_file():
                preparation = root / read_json(accepted)["processed_relative_path"]
                files += list(preparation.rglob("*"))
    selected = sorted({p.resolve() for p in files if p.is_file()})
    manifest = [{"path": p.relative_to(root.resolve()).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in selected]
    tmp = destination.with_suffix(".tmp.zip")
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED) as z:
        for path, info in zip(selected, manifest):
            z.write(path, info["path"])
        import json
        z.writestr("INSPECTION_CONTENTS.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(tmp) as z:
        if z.testzip() is not None:
            raise RuntimeError("Inspection ZIP failed its integrity check.")
    tmp.replace(destination)
    return destination
