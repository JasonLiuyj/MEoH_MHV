from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


RESULT_RE = re.compile(
    r"^MHV_Step_(?P<algorithm>.+)_(?P<problem>DTLZ[1-4]|WFG[4-9])_Set(?P<setting>\d+)_Tau(?P<tau>\d+)\.mat$",
    re.IGNORECASE,
)


def load_mhv_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(path)
    if "MHV_Data" not in data:
        raise ValueError(f"{path.name}: missing MHV_Data")

    cell = data["MHV_Data"]
    runs = cell.ravel()
    if runs.size == 0:
        raise ValueError(f"{path.name}: empty MHV_Data")

    histories = []
    for idx, run in enumerate(runs, start=1):
        arr = np.asarray(run, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{path.name}: run {idx} has invalid shape {arr.shape}")
        histories.append(arr)

    target_gens = histories[0][:, 0]
    trigger_gens = np.vstack([arr[:, 1] for arr in histories])
    mhv = np.vstack([arr[:, 2] for arr in histories])

    for idx, arr in enumerate(histories[1:], start=2):
        if not np.array_equal(arr[:, 0], target_gens):
            raise ValueError(f"{path.name}: run {idx} target generations differ")

    return target_gens, trigger_gens, mhv


def summarize_file(path: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    match = RESULT_RE.match(path.name)
    if not match:
        return [], None

    target_gens, trigger_gens, mhv = load_mhv_file(path)
    meta = match.groupdict()
    rows = []
    for stage_idx in range(mhv.shape[1]):
        values = mhv[:, stage_idx]
        rows.append(
            {
                "algorithm": meta["algorithm"],
                "problem": meta["problem"].upper(),
                "setting": int(meta["setting"]),
                "tau": int(meta["tau"]),
                "stage": stage_idx + 1,
                "target_gen": int(target_gens[stage_idx]),
                "runs": int(mhv.shape[0]),
                "mean_mhv": float(np.mean(values)),
                "std_mhv": float(np.std(values, ddof=1)) if mhv.shape[0] > 1 else 0.0,
                "min_mhv": float(np.min(values)),
                "max_mhv": float(np.max(values)),
                "mean_trigger_gen": float(np.mean(trigger_gens[:, stage_idx])),
            }
        )

    run_mean_mhv = np.mean(mhv, axis=1)
    overall = {
        "algorithm": meta["algorithm"],
        "problem": meta["problem"].upper(),
        "setting": int(meta["setting"]),
        "tau": int(meta["tau"]),
        "runs": int(mhv.shape[0]),
        "stages": int(mhv.shape[1]),
        "mean_of_run_mean_mhv": float(np.mean(run_mean_mhv)),
        "std_of_run_mean_mhv": float(np.std(run_mean_mhv, ddof=1)) if mhv.shape[0] > 1 else 0.0,
        "min_run_mean_mhv": float(np.min(run_mean_mhv)),
        "max_run_mean_mhv": float(np.max(run_mean_mhv)),
    }
    return rows, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MHV .mat result files.")
    parser.add_argument("--input-dir", default="Results_StepMHV_Python")
    parser.add_argument("--output", default="mhv_summary.csv")
    parser.add_argument("--overall-output", default="mhv_overall_summary.csv")
    parser.add_argument("--strict-runs", type=int, default=31)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for path in sorted(input_dir.glob("MHV_Step_*.mat")):
        try:
            rows, overall = summarize_file(path)
            if rows and rows[0]["runs"] != args.strict_runs:
                warnings.append(f"{path.name}: runs={rows[0]['runs']} expected={args.strict_runs}")
            all_rows.extend(rows)
            if overall is not None:
                overall_rows.append(overall)
        except Exception as exc:
            warnings.append(f"{path.name}: {exc}")

    if not all_rows:
        raise RuntimeError(f"No valid MHV result files found in {input_dir}")

    fieldnames = [
        "algorithm",
        "problem",
        "setting",
        "tau",
        "stage",
        "target_gen",
        "runs",
        "mean_mhv",
        "std_mhv",
        "min_mhv",
        "max_mhv",
        "mean_trigger_gen",
    ]
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} summary rows to {output}")

    overall_fieldnames = [
        "algorithm",
        "problem",
        "setting",
        "tau",
        "runs",
        "stages",
        "mean_of_run_mean_mhv",
        "std_of_run_mean_mhv",
        "min_run_mean_mhv",
        "max_run_mean_mhv",
    ]
    overall_output = Path(args.overall_output)
    with overall_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fieldnames)
        writer.writeheader()
        writer.writerows(overall_rows)

    print(f"Wrote {len(overall_rows)} overall rows to {overall_output}")
    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
