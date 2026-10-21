"""Run the connected-network A20 batch from its frozen trial manifest."""

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import networkx as nx
import numpy as np

from ic_model import simulate_ic
from make_a20_manifest import derive_seed
from threshold_model import simulate_threshold2


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("lattice/ic", "lattice/threshold2", "random/ic", "random/threshold2")


def utc_now():
    """Use actual UTC offsets for fields named *_utc."""
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    """Create an immutable artifact without overwriting an earlier file."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def git_read(*args):
    """Record unavailable metadata explicitly rather than inventing a value."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return {"value": result.stdout.strip(), "error": None}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"value": None, "error": str(error)}


def graph_snapshot(graph):
    """Store exact labeled topology, its fingerprint and measured properties."""
    topology = {
        "nodes": sorted(graph.nodes()),
        "edges": sorted(tuple(sorted(edge)) for edge in graph.edges()),
    }
    digest = sha256(json.dumps(topology, sort_keys=True).encode()).hexdigest()
    sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    return {
        **topology,
        "topology_sha256": digest,
        "checks": {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "degree_values": sorted(set(dict(graph.degree()).values())),
            "self_loop_count": nx.number_of_selfloops(graph),
            "connected": len(sizes) == 1,
            "component_count": len(sizes),
            "component_sizes": sizes,
            "average_clustering": nx.average_clustering(graph),
        },
    }


def load_inputs(manifest_path):
    """Reject unsupported settings before creating a batch of mismatched runs."""
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    cfg = manifest["config"]
    expected = {
        "experiment_id": "a20_batch_v2",
        "n_trials": 50,
        "update_mode": "synchronous",
        "stop_condition": "no_new_adopters",
        "history": {"include_initial_state": True, "include_terminal_no_growth_round": False},
        "ic": {"p": 0.2},
        "threshold": {"adopted_neighbors_required": 2},
        "initial_adopters": {
            "strategy": "seed_neighborhood",
            "center_sampling": "uniform_with_replacement",
            "pair_center_across_networks": True,
        },
        "rewiring": {
            "swaps_per_edge": 10,
            "max_tries_per_swap": 100,
            "connectivity_policy": "require_connected_with_logged_rejection",
            "max_graph_candidates": 100,
        },
        "random_streams": {
            "purposes": ["graph", "center", "ic_lattice", "ic_random"],
            "seed_derivation": "sha256_first16_hex",
            "graph_retry_purpose_template": "graph_attempt_{attempt}",
        },
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unsupported v2 setting: {key}")
    live_cfg = json.loads((ROOT / "configs/a20_batch_v2.json").read_text(encoding="utf-8"))
    if cfg != live_cfg:
        raise ValueError("Manifest and current configuration differ; review design version.")
    trials = manifest["trials"]
    if [row["trial_id"] for row in trials] != list(range(50)):
        raise ValueError("Manifest must contain trial IDs 0 through 49 in order.")
    for row in trials:
        for purpose in cfg["random_streams"]["purposes"]:
            if row[purpose] != derive_seed(cfg["master_seed"], row["trial_id"], purpose):
                raise ValueError("Manifest seed derivation mismatch.")
    lattice_path = ROOT / cfg["base_lattice_snapshot"]
    lattice_raw = lattice_path.read_bytes()
    lattice_hash = sha256(lattice_raw).hexdigest()
    if lattice_path.stem != f"lattice_{lattice_hash}":
        raise ValueError("Lattice checksum mismatch.")
    saved = json.loads(lattice_raw.decode("utf-8"))
    lattice = nx.Graph()
    lattice.add_nodes_from(tuple(node) for node in saved["nodes"])
    lattice.add_edges_from((tuple(a), tuple(b)) for a, b in saved["edges"])
    snapshot = graph_snapshot(lattice)
    checks = snapshot["checks"]
    if not (
        checks["node_count"] == 128 and checks["edge_count"] == 384
        and checks["degree_values"] == [6] and checks["self_loop_count"] == 0
        and checks["connected"] and abs(checks["average_clustering"] - 0.4) < 1e-10
    ):
        raise ValueError("Source lattice does not satisfy the A20 specification.")
    return manifest, sha256(raw).hexdigest(), lattice, snapshot, lattice_hash


def connected_counterpart(lattice, trial, cfg, trial_dir, lattice_hash):
    """Accept the first connected candidate and preserve every rejected candidate."""
    swaps = cfg["rewiring"]["swaps_per_edge"] * lattice.number_of_edges()
    max_tries = cfg["rewiring"]["max_tries_per_swap"] * swaps
    for attempt in range(cfg["rewiring"]["max_graph_candidates"]):
        purpose = "graph" if attempt == 0 else cfg["random_streams"][
            "graph_retry_purpose_template"
        ].format(attempt=attempt)
        seed = trial["graph"] if attempt == 0 else derive_seed(
            cfg["master_seed"], trial["trial_id"], purpose
        )
        candidate = lattice.copy()
        started = utc_now()
        clock = time.perf_counter()
        error = None
        try:
            nx.double_edge_swap(candidate, nswap=swaps, max_tries=max_tries, seed=seed)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        snapshot = graph_snapshot(candidate)
        checks = snapshot["checks"]
        checks["node_set_preserved"] = set(candidate) == set(lattice)
        checks["degrees_preserved"] = dict(candidate.degree()) == dict(lattice.degree())
        checks["topology_changed"] = snapshot["topology_sha256"] != lattice_hash
        structural_ok = (
            checks["node_count"] == 128 and checks["edge_count"] == 384
            and checks["node_set_preserved"] and checks["degrees_preserved"]
            and checks["self_loop_count"] == 0 and checks["topology_changed"]
        )
        status = (
            "swap_error" if error else "structural_failure" if not structural_ok
            else "accepted_connected" if checks["connected"] else "rejected_disconnected"
        )
        candidate_record = {
            **snapshot, "attempt": attempt, "purpose": purpose, "seed": seed,
            "requested_swaps": swaps, "max_tries": max_tries,
            "status": status, "error": error, "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "generation_and_check_seconds": time.perf_counter() - clock,
        }
        save_json(trial_dir / f"candidate_{attempt:03d}.json", candidate_record)
        if error or not structural_ok:
            raise RuntimeError(f"Trial {trial['trial_id']} candidate {attempt}: {status}")
        if checks["connected"]:
            return candidate, candidate_record
    raise RuntimeError(f"Trial {trial['trial_id']}: connected-candidate limit exhausted.")


def summarize(records, cfg, lattice_snapshot, batch_dir):
    """Summarize complete trial blocks without treating time points as replicates."""
    finals = np.array([
        [row["results"][key]["adoption_fractions"][-1] for key in CONDITIONS]
        for row in records
    ])
    bootstrap_seed = derive_seed(cfg["master_seed"], -1, "bootstrap")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(records), size=(2000, len(records)))
    boot_means = finals[indices].mean(axis=1)
    estimates = {}
    for col, key in enumerate(CONDITIONS):
        low, high = np.percentile(boot_means[:, col], [2.5, 97.5])
        curves = [tuple(row["results"][key]["adoption_counts"]) for row in records]
        estimates[key] = {
            "mean_final_fraction": float(finals[:, col].mean()),
            "sd_final_fraction": float(finals[:, col].std(ddof=1)),
            "ci95_mean_final_fraction": [float(low), float(high)],
            "unique_trajectory_count": len(set(curves)),
            "invariant_observed_trajectory": len(set(curves)) == 1,
        }
    differences = {}
    for rule, left, right in (("ic", 0, 2), ("threshold2", 1, 3)):
        low, high = np.percentile(boot_means[:, left] - boot_means[:, right], [2.5, 97.5])
        differences[rule] = {
            "mean_lattice_minus_random": float((finals[:, left] - finals[:, right]).mean()),
            "ci95": [float(low), float(high)],
        }
    clustering = [row["random_graph_checks"]["average_clustering"] for row in records]
    summary = {
        "completed_trials": len(records), "trajectory_count": 4 * len(records),
        "conditions": estimates, "paired_final_fraction_differences": differences,
        "inference": {
            "method": "percentile bootstrap of whole paired trial IDs",
            "resamples": 2000, "confidence_level": 0.95, "seed": bootstrap_seed,
            "scope": "Fixed p and lattice; connected outputs of the finite-swap generator.",
            "limitations": [
                "Observed deterministic duplicates are not independent confirmations.",
                "Collapsed intervals do not establish certainty beyond this design.",
                "Graph, center and IC variation are combined, not separately estimated.",
                "No human or LLM hypothesis is tested by this classical batch.",
            ],
        },
        "networks": {
            "lattice_clustering": lattice_snapshot["checks"]["average_clustering"],
            "random_clustering_mean": float(np.mean(clustering)),
            "random_clustering_min": min(clustering), "random_clustering_max": max(clustering),
            "accepted_connected_count": sum(row["random_graph_checks"]["connected"] for row in records),
            "unique_random_topologies": len({row["random_graph_sha256"] for row in records}),
            "unique_centers": len({tuple(row["center"]) for row in records}),
            "candidate_count": sum(row["accepted_attempt"] + 1 for row in records),
            "disconnected_rejections": sum(row["accepted_attempt"] for row in records),
        },
    }
    save_json(batch_dir / "summary.json", summary)

    # Export only actually recorded rounds; carry-forward is a plotting operation.
    with (batch_dir / "trajectories.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "condition", "round", "adopted_count", "adopted_fraction"])
        for row in records:
            for key in CONDITIONS:
                result = row["results"][key]
                for t, (count, fraction) in enumerate(zip(result["adoption_counts"], result["adoption_fractions"])):
                    writer.writerow([row["trial_id"], key, t, count, fraction])
    with (batch_dir / "final_adoption.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "condition", "final_count", "final_fraction", "last_growth_round"])
        for row in records:
            for key in CONDITIONS:
                result = row["results"][key]
                writer.writerow([row["trial_id"], key, result["adoption_counts"][-1],
                                 result["adoption_fractions"][-1], len(result["adoption_counts"]) - 1])

    # Faint curves show realizations; bold curves show means, not confidence bands.
    length = max(len(row["results"][key]["adoption_fractions"]) for row in records for key in CONDITIONS)
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color in zip(CONDITIONS, ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        padded = []
        for row in records:
            values = row["results"][key]["adoption_fractions"]
            values = values + [values[-1]] * (length - len(values))
            padded.append(values)
            ax.plot(range(length), values, color=color, alpha=0.06, linewidth=0.6)
        ax.plot(range(length), np.mean(padded, axis=0), color=color, linewidth=2.2, label=key)
    ax.set(xlabel="Round (0 = initial state)", ylabel="Adopted fraction", ylim=(0, 1.05),
           title=f"A20: {len(records)} trials per condition, connected networks, IC p={cfg['ic']['p']}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(0.5, 0.02, "Faint: individual trajectories. Bold: means. Terminal states carried forward.\n"
             "Final-fraction bootstrap intervals are in summary.json; no curve confidence bands shown.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(batch_dir / "adoption_curves.png", dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/a20_trial_manifest_v2.json")
    parser.add_argument("--output-root", default="runs/a20_batches")
    args = parser.parse_args()
    clock = time.perf_counter()
    started = utc_now()
    batch_dir = ROOT / args.output_root / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex
    )
    batch_dir.mkdir(parents=True, exist_ok=False)
    print("Batch directory:", batch_dir, flush=True)
    completed_ids = []
    try:
        manifest, manifest_hash, lattice, lattice_snapshot, lattice_hash = load_inputs(ROOT / args.manifest)
        cfg = manifest["config"]
        source_files = {}
        for name in ("run_a20_batch.py", "make_a20_manifest.py", "ic_model.py", "threshold_model.py"):
            data = (ROOT / "src" / name).read_bytes()
            source_files[name] = {"sha256": sha256(data).hexdigest(), "text": data.decode("utf-8")}
        save_json(batch_dir / "batch_started.json", {
            "status": "started", "started_at_utc": started, "manifest_sha256": manifest_hash,
            "manifest": manifest, "source_files": source_files,
            "lattice_file_sha256": lattice_hash, "git_commit": git_read("rev-parse", "HEAD"),
            "git_status": git_read("status", "--porcelain"),
            "environment": {"python": sys.version, "executable": sys.executable,
                            "platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor() or None, "networkx": nx.__version__,
                            "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            "usage": {"model_version": None, "prompt_version": None, "calls": 0,
                      "input_tokens": 0, "output_tokens": 0, "gpu_hours": 0, "api_cost": 0,
                      "reason": "Classical CPU-only simulation; no LLM or API used."},
        })
        save_json(batch_dir / "lattice.json", lattice_snapshot)
        records = []
        for trial in manifest["trials"]:
            trial_dir = batch_dir / f"trial_{trial['trial_id']:03d}"
            trial_dir.mkdir()
            trial_clock = time.perf_counter()
            save_json(trial_dir / "started.json", {"started_at_utc": utc_now(), "seeds": trial})
            try:
                # Center selection is independent of how many graph candidates are rejected.
                center = random.Random(trial["center"]).choice(sorted(lattice.nodes()))
                randomized, accepted = connected_counterpart(
                    lattice, trial, cfg, trial_dir, lattice_snapshot["topology_sha256"]
                )
                results, initial = {}, {}
                for name, graph in (("lattice", lattice), ("random", randomized)):
                    adopters = {center} | set(graph.neighbors(center))
                    if len(adopters) != 7:
                        raise ValueError("Seed neighborhood must contain seven nodes.")
                    initial[name] = sorted(adopters)
                    for rule in ("ic", "threshold2"):
                        sim_clock = time.perf_counter()
                        counts = (simulate_ic(graph, adopters, cfg["ic"]["p"], trial[f"ic_{name}"])
                                  if rule == "ic" else simulate_threshold2(graph, adopters))
                        if not (counts and counts[0] == 7 and all(7 <= c <= 128 for c in counts)
                                and all(a < b for a, b in zip(counts, counts[1:]))):
                            raise ValueError("Invalid adoption history.")
                        results[f"{name}/{rule}"] = {
                            "adoption_counts": counts, "adoption_fractions": [c / 128 for c in counts],
                            "simulation_seconds": time.perf_counter() - sim_clock,
                        }
                record = {
                    "status": "completed", "trial_id": trial["trial_id"], "seeds": trial,
                    "center": center, "initial_adopters": initial,
                    "accepted_attempt": accepted["attempt"], "accepted_graph_seed": accepted["seed"],
                    "random_graph_sha256": accepted["topology_sha256"],
                    "random_graph_checks": accepted["checks"], "results": results,
                    "finished_at_utc": utc_now(), "trial_wall_clock_seconds": time.perf_counter() - trial_clock,
                }
                save_json(trial_dir / "completed.json", record)
                records.append(record)
                completed_ids.append(trial["trial_id"])
                print(f"Completed trials: {len(records)}/{len(manifest['trials'])}", flush=True)
            except BaseException as exc:
                save_json(trial_dir / "failed.json", {"status": "failed", "at_utc": utc_now(),
                    "trial_id": trial["trial_id"], "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})
                raise
        summary = summarize(records, cfg, lattice_snapshot, batch_dir)
        save_json(batch_dir / "batch_completed.json", {
            "status": "completed", "started_at_utc": started, "finished_at_utc": utc_now(),
            "wall_clock_seconds_before_final_log": time.perf_counter() - clock,
            "completed_trials": len(records), "trajectory_count": len(records) * 4,
            "output_directory": str(batch_dir),
        })
        print("Batch status: COMPLETED")
        print("Trajectory count:", summary["trajectory_count"])
        print("Accepted connected networks:", summary["networks"]["accepted_connected_count"])
        print("Disconnected candidates rejected:", summary["networks"]["disconnected_rejections"])
        for key, values in summary["conditions"].items():
            print(f"{key}: mean final fraction={values['mean_final_fraction']:.6f}, "
                  f"95% CI={values['ci95_mean_final_fraction']}")
        print("Saved summary:", batch_dir / "summary.json")
        print("Saved figure:", batch_dir / "adoption_curves.png")
    except BaseException as exc:
        save_json(batch_dir / "batch_failed.json", {
            "status": "failed", "started_at_utc": started, "finished_at_utc": utc_now(),
            "completed_trial_ids": completed_ids, "error": f"{type(exc).__name__}: {exc}",
            "wall_clock_seconds": time.perf_counter() - clock, "traceback": traceback.format_exc(),
        })
        print("Batch status: FAILED; records preserved in", batch_dir, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
