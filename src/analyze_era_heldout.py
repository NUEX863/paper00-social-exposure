"""Compare frozen held-out amounts with matched core personas, entirely offline."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from statsmodels.stats.multitest import multipletests

from run_era_formal import ROOT, read_json, sha, rows, formal_token_score, payload_for
from response_readout import classify_response

MODELS = ["qwen3_8b", "mistral7b_v03"]
ITEMS = ["s1_holdout_low", "s1_main", "s1_holdout_high"]
AMOUNTS = [100, 500, 2000]
CELLS = ["NO_INFO", "R1_0", "R1_1", "R1_2", "R1_3"]
POLICIES = ["semantic_valid", "strict_valid", "invalid_as_no", "invalid_as_yes"]
B, SEED = 2000, 2026101902


def save(path, value):
    """Turn numerical missing values into JSON null."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def ci(x):
    return np.quantile(x, [.025, .975]) if np.isfinite(x).all() else (np.nan, np.nan)


def pvalue(point, samples):
    if not np.isfinite(point) or not np.isfinite(samples).all() or np.ptp(samples) < 1e-12:
        return np.nan
    return (1+np.count_nonzero(np.abs(samples-point) >= abs(point)))/(B+1)


def diagnostics(p):
    """Use exactly the core IC and ratio guards; do not force a mechanism label."""
    q = p[..., 1:]
    result = {"P_no_information": p[..., 0], "known_zero_minus_noinfo": q[..., 0]-p[..., 0]}
    for k in range(4):
        result[f"P{k}"] = q[..., k]
    for k in range(1, 4):
        result[f"increment{k}"] = q[..., k]-q[..., k-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (q[..., 1]-q[..., 0])/(1-q[..., 0])
        valid = (q[..., 0] < 1-1e-6) & (beta >= 0) & (beta <= 1)
        for k in (2, 3):
            result[f"S{k}"] = np.where(valid, q[..., k]-(1-(1-q[..., 0])*(1-beta)**k), np.nan)
        result["rho"] = np.where(np.abs(q[..., 1]-q[..., 0]) >= .02,
                                  (q[..., 2]-q[..., 1])/(q[..., 1]-q[..., 0]), np.nan)
    return result


def extract(run, out, runtime, plan):
    """Verify new responses against the frozen held-out manifest and raw files."""
    snapshot = run / "sources"
    meta = read_json(run / "run.json")
    for name, digest in meta["source_sha256"].items():
        if sha(snapshot / name) != digest:
            raise ValueError("Snapshot mismatch: " + name)
    design = read_json(snapshot / runtime["design"])
    policy = read_json(snapshot / runtime["readout_policy"])
    prompts = {p["prompt_id"]: p for p in rows(snapshot / design["files"]["prompts"])}
    manifest = {r["request_id"]: r for r in rows(snapshot / design["files"]["manifest"]) if r["module"] == "heldout_items"}
    found, observations, checks = set(), [], []
    for path in sorted((run / "attempts").glob("*/result.json")):
        result = read_json(path)
        if result["outcome"] != "response":
            continue
        m = result["manifest"]
        rid = m["request_id"]
        if rid in found or m != manifest[rid]:
            raise ValueError("Duplicate or changed request: " + rid)
        found.add(rid)
        prompt = prompts[m["prompt_id"]]
        if prompt != result["prompt_metadata"] or result["request"] != payload_for(m, prompt, runtime["models"][m["model_slot"]], design["generation"]):
            raise ValueError("Prompt/settings mismatch")
        if result["backend_version"] != runtime["server_version"] or result["model_digest"] != runtime["models"][m["model_slot"]]["digest"]:
            raise ValueError("Identity mismatch")
        raw_path = path.parent / "raw_response.json"
        response = read_json(raw_path)["response"]
        if response != result["response"]:
            raise ValueError("Saved response mismatch")
        classified = classify_response(response, policy)
        score = formal_token_score(response, m["model_slot"], policy) if m["mode"] == "logprob" else None
        if classified != result["readout"] or score != result["token_score"]:
            raise ValueError("Readout mismatch")
        cell = "NO_INFO" if prompt["representation"] == "NO_INFO" else f"R1_{prompt['k']}"
        observations.append({**m, "persona_id": prompt["persona_id"], "item_id": prompt["item_id"],
            "wording": prompt["wording"], "representation": prompt["representation"], "condition": cell,
            **classified, "token_score": score["native_label_yes_score"] if score else None,
            "token_mass": score["native_label_mass"] if score else None,
            "input_tokens": response.get("prompt_eval_count"), "output_tokens": response.get("eval_count"),
            "wall_seconds": result["wall_seconds"], "result_path": path.relative_to(ROOT).as_posix()})
        checks.append({"request_id": rid, "path": path.relative_to(ROOT).as_posix(),
                       "result_sha256": sha(path), "raw_sha256": sha(raw_path)})
    if found != set(manifest):
        raise ValueError("Collection is incomplete; do not analyze a partial batch")
    df = pd.DataFrame(observations)
    df.to_csv(out / "heldout_observations.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "input_hashes.csv", index=False)
    core_file = snapshot / plan["core_observations"]
    if sha(core_file) != plan["core_observations_sha256"]:
        raise ValueError("Core analysis input changed")
    core = pd.read_csv(core_file)
    core = core[core.persona_id.isin(plan["persona_ids"]) & core.wording.eq("w1") & core.condition.isin(CELLS)].copy()
    if len(core) != 1200 or core.item_id.unique().tolist() != ["s1_main"]:
        raise ValueError("Matched core subset must contain 1,200 existing records")
    core.to_csv(out / "matched_500mb_observations.csv", index=False)
    return pd.concat([core, df], ignore_index=True), df


def analyze(run):
    runtime = read_json(run / "run.json")["runtime"]
    snapshot = run / "sources"
    plan = read_json(snapshot / runtime["analysis_plan"])
    # Refuse silent changes to the prospective analysis script after collection starts.
    if sha(Path(__file__)) != sha(snapshot / "src/analyze_era_heldout.py"):
        raise ValueError("Analysis source changed since launch")
    out = run / "analysis"
    if (out / "summary.json").exists():
        print("Existing completed analysis retained:", out, flush=True)
        return
    out.mkdir(exist_ok=True)
    all_df, new_df = extract(run, out, runtime, plan)
    sampling = all_df[all_df["mode"].eq("sampling")].copy()
    personas = sorted(plan["persona_ids"])
    count_rows, quality = [], []
    for (model, item, cell), group in sampling.groupby(["model_slot", "item_id", "condition"]):
        yes = int(group.semantic_class.eq("Yes").sum())
        no = int(group.semantic_class.eq("No").sum())
        n = len(group)
        record = {"model": model, "item": item, "condition": cell, "planned": n, "yes": yes, "no": no,
                  "invalid": n-yes-no, "ambiguity_low": yes/n, "ambiguity_high": (n-no)/n}
        if yes+no == n and yes in (0, n):
            record.update(boundary_ci_low=0 if yes == 0 else beta_dist.ppf(.025, n, 1),
                          boundary_ci_high=1 if yes == n else beta_dist.ppf(.975, 1, n))
        count_rows.append(record)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(out / "counts_and_boundary_uncertainty.csv", index=False)
    for (model, mode), group in new_df.groupby(["model_slot", "mode"]):
        quality.append({"model": model, "mode": mode, "completed": len(group),
            "invalid": int(group.semantic_class.eq("invalid").sum()),
            "available_token_scores": int(group.token_score.notna().sum()),
            "input_tokens": int(group.input_tokens.sum()), "output_tokens": int(group.output_tokens.sum())})
    if new_df.semantic_class.eq("invalid").mean() > .01:
        save(out / "quality_stop.json", {"status": "INFERENCE_PAUSED_INVALID_GT_1_PERCENT", "quality": quality})
        raise ValueError("More than 1% invalid replies: retain data and pause inferential claims")
    weights = np.random.default_rng(SEED).multinomial(20, np.full(20, .05), size=B)/20
    curve_rows, metric_rows, contrast_rows = [], [], []
    for policy in POLICIES:
        rates = np.empty((2, 3, 20, 5))
        for m, model in enumerate(MODELS):
            for j, item in enumerate(ITEMS):
                for p, person in enumerate(personas):
                    for c, cell in enumerate(CELLS):
                        group = sampling[sampling.model_slot.eq(model) & sampling.item_id.eq(item) &
                                         sampling.persona_id.eq(person) & sampling.condition.eq(cell)]
                        if len(group) != 5:
                            raise ValueError("Unbalanced matched sampling cell")
                        labels = group.strict_class if policy == "strict_valid" else group.semantic_class
                        y, n = labels.eq("Yes").sum(), labels.ne("invalid").sum()
                        if policy == "invalid_as_yes":
                            y, n = y+5-n, 5
                        elif policy == "invalid_as_no":
                            n = 5
                        if not n:
                            raise ValueError("All-invalid cell; do not silently drop a persona")
                        rates[m, j, p, c] = y/n
        mean = rates.mean(axis=2)
        boot = np.einsum("bp,mipc->mibc", weights, rates)
        for m, model in enumerate(MODELS):
            for j, item in enumerate(ITEMS):
                point, draws = diagnostics(mean[m, j]), diagnostics(boot[m, j])
                for c, cell in enumerate(CELLS):
                    lo, hi = ci(boot[m, j, :, c])
                    curve_rows.append({"model": model, "item": item, "data_mb": AMOUNTS[j], "policy": policy,
                        "condition": cell, "estimate": mean[m, j, c], "ci_low": lo, "ci_high": hi,
                        "degenerate_bootstrap": np.ptp(boot[m, j, :, c]) < 1e-12})
                for metric, value in point.items():
                    samples = draws[metric].copy()
                    if metric == "rho":
                        lo, hi = ci(draws["increment1"])
                        if lo <= 0 <= hi:
                            value, samples[:] = np.nan, np.nan
                    lo, hi = ci(samples)
                    metric_rows.append({"model": model, "item": item, "policy": policy, "metric": metric,
                        "estimate": value, "ci_low": lo, "ci_high": hi,
                        "finite_bootstrap_fraction": np.isfinite(samples).mean(),
                        "flag": "undefined_or_unstable" if not np.isfinite(value) else
                                "bootstrap_boundary_or_admissibility" if not np.isfinite(lo) or np.ptp(samples) < 1e-12 else "OK"})
            for j in (0, 2):
                for c, cell in enumerate(CELLS):
                    effect = mean[m, j, c]-mean[m, 1, c]
                    samples = boot[m, j, :, c]-boot[m, 1, :, c]
                    lo, hi = ci(samples)
                    contrast_rows.append({"model": model, "item": ITEMS[j], "policy": policy, "condition": cell,
                        "amount_minus_500mb": effect, "ci_low": lo, "ci_high": hi,
                        "p_centered_exploratory": pvalue(effect, samples),
                        "degenerate_bootstrap": np.ptp(samples) < 1e-12})
        if policy == "semantic_valid":
            np.savez_compressed(out / "paired_bootstrap.npz", weights=weights, means=mean, bootstrap=boot)
    curves = pd.DataFrame(curve_rows)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_holm_exploratory"] = np.nan
    for policy in POLICIES:
        ids = contrasts.index[contrasts.policy.eq(policy)]
        p = contrasts.loc[ids, "p_centered_exploratory"].to_numpy()
        if len(ids) != 20:
            raise ValueError("Exploratory contrast family must have 20 members")
        adj = multipletests(np.where(np.isfinite(p), p, 1.), method="holm")[1]
        contrasts.loc[ids, "p_holm_exploratory"] = np.where(np.isfinite(p), adj, np.nan)
    curves.to_csv(out / "matched_amount_curves.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "micro_metrics.csv", index=False)
    contrasts.to_csv(out / "paired_amount_contrasts.csv", index=False)
    all_df[all_df["mode"].eq("logprob")].to_csv(out / "secondary_token_scores.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, model in zip(axes, MODELS):
        for item, amount, color in zip(ITEMS, AMOUNTS, ["#1565c0", "#666666", "#c45100"]):
            part = curves[curves.model.eq(model) & curves.item.eq(item) & curves.policy.eq("semantic_valid")].set_index("condition").loc[CELLS[1:]]
            ax.plot(range(4), part.estimate, "o-", color=color, label=f"{amount} MB")
            ax.fill_between(range(4), part.ci_low, part.ci_high, color=color, alpha=.13)
        ax.set(title=model, xlabel="Adopting friends k (of 6)", xticks=range(4), ylim=(-.025, 1.025))
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel("Yes rate conditional on valid answers")
    fig.suptitle("Frozen held-out amounts: matched 20 personas, R1/w1 only")
    fig.text(.5, .01, "Shading: paired persona-bootstrap 95% CI; boundary collapse is not certainty.\n500 MB reuses matched core responses. This is within-scenario transfer, not new-persona or cross-domain validation.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .94))
    fig.savefig(out / "heldout_amount_curves.png", dpi=180)
    plt.close(fig)
    save(out / "summary.json", {"status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": run.relative_to(ROOT).as_posix(), "new_completed_calls": len(new_df),
        "matched_existing_core_calls": len(all_df)-len(new_df), "quality": quality,
        "bootstrap_resamples": B, "analysis_seed": SEED,
        "curves": curves[curves.policy.eq("semantic_valid")].to_dict("records"),
        "limits": plan["limits"],

        "new_calls_in_analysis": 0})
    save(out / "artifact_hashes.json", {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "artifact_hashes.json"})
    print("HELDOUT ANALYSIS COMPLETE:", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    analyze(parser.parse_args().run_dir.resolve())
