"""Cloud-only independent-process resume acceptance and bounded cost estimate."""

import argparse
import json
from pathlib import Path

from tools.cloud_guard import require_cloud_workspace


def main():
    require_cloud_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = json.loads(args.first.read_text(encoding="utf-8"))
    resumed = json.loads(args.resumed.read_text(encoding="utf-8"))
    checks = {
        "both_runs_passed": first["passed"] and resumed["passed"],
        "same_identity": first["identity"] == resumed["identity"],
        "same_config": first["config"] == resumed["config"],
        "checkpoint_identity": first["saved_checkpoint"]["checkpoint_sha256"] == resumed["restore"]["checkpoint_sha256"],
        "validation_exact_after_process_restart": first["final_validation"] == resumed["initial_validation"],
        "next_batch_exact_after_restart": first["saved_next_batch_sha256"] == resumed["next_batch_sha256"],
        "counter_continued": resumed["updates"][0]["step"] == first["updates"][-1]["step"] + 1,
        "full_model": all(r["parameter_tensors"] == 916 and r["parameter_elements"] == 27580719 for r in (first, resumed)),
        "every_gradient_finite": all(u["all_gradients_finite"] and u["gradient_tensors"] == 916 for r in (first, resumed) for u in r["updates"]),
        "every_step_changed_parameters": all(u["changed_parameter_tensors"] > 0 for r in (first, resumed) for u in r["updates"]),
        "split_disjoint": set(first["train_ids"]).isdisjoint(first["valid_ids"]),
    }
    # One warmed observation is an initial planning measurement, not a sustained benchmark.
    warmed = first["updates"][1:]
    seconds = sum(u["audited_update_seconds"] for u in warmed) / len(warmed) if warmed else None
    report = {"checks": checks, "passed": all(checks.values()), "completed_updates": resumed["updates"][-1]["step"],
              "warm_observations": len(warmed), "audited_warm_step_seconds": seconds,
              "nominal_yuan_per_hour": 20.828,
              "estimate_excludes": "initialization, data loading, per-parameter update audit, validation, checkpoints, idle/storage charges; not a bill or balance",
              "first_run_training_wall_seconds": first["training_wall_seconds"],
              "peak_allocated_bytes": max(u["peak_allocated_bytes"] for r in (first, resumed) for u in r["updates"]),
              "peak_reserved_bytes": max(u["peak_reserved_bytes"] for r in (first, resumed) for u in r["updates"])}
    if seconds:
        report["planning_only"] = {str(n): {"hours": n * seconds / 3600, "yuan": n * seconds / 3600 * 20.828}
                                   for n in (100, 1000, 5000, 250000)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("Stage4 independent resume acceptance failed")


if __name__ == "__main__":
    main()
