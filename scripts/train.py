r"""Train the final model and save the deployable artifact to models/model.joblib.

Prerequisite: processed splits exist (python scripts\build_datasets.py).
Run (venv):   python scripts\train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retail_repeat import config, training  # noqa: E402


def main() -> int:
    if not (config.DATA_PROCESSED / "train.parquet").exists():
        print("Processed splits not found. Run:  python scripts/build_datasets.py", file=sys.stderr)
        return 1
    bundle = training.train_final()
    print("Saved model bundle ->", config.MODEL_PATH)
    for k in ("model_version", "params", "threshold", "trained_at", "n_train_rows"):
        print(f"  {k:<14} {bundle[k]}")
    print(f"  features       {bundle['features']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
