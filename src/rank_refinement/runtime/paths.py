from pathlib import Path


def relative_to_root(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def prepare_output_directories(root):
    for rel in ["logs", "state", "outputs/inspection", "outputs/steps/step01_prepare",
                "outputs/paper/main/tables", "outputs/paper/supplementary/tables",
                "outputs/paper/supplementary/details", "data/processed/primary", "data/cache/tokenization"]:
        (Path(root) / rel).mkdir(parents=True, exist_ok=True)
