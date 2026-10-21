"""Analyze a saved formal ERA batch offline; never contact a model endpoint."""



import argparse

import hashlib

import json

import math

import platform

import shutil

import sys

import time

import uuid

from collections import Counter

from datetime import datetime, timezone

from functools import lru_cache

from pathlib import Path



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy

import statsmodels

from scipy.optimize import minimize

from scipy.special import expit, xlogy

from scipy.stats import beta as beta_distribution

from statsmodels.stats.multitest import multipletests



from response_readout import classify_response

from run_era_formal import formal_token_score, payload_for



ROOT = Path(__file__).resolve().parents[1]

MODELS = ["qwen3_8b", "mistral7b_v03"]

LABELS = {"qwen3_8b": "Qwen3-8B", "mistral7b_v03": "Mistral-7B-v0.3"}

CELLS = ["NO_INFO"] + [f"{r}_{k}" for r in ("R1", "R2") for k in range(4)] + ["R2_2_tied", "R2_2_untied"]

WORDS = ["w1", "w2", "w3"]

POLICIES = ["semantic_valid", "strict_valid", "invalid_as_no", "invalid_as_yes"]

B = 2000

BOOT_SEED = 2026101901

EPS = 1e-9





def read(path):

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))





def sha(path):

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()





def clean(value):

    """Serialize missing/nonfinite statistics as null, not nonstandard JSON NaN."""

    if isinstance(value, dict):

        return {str(k): clean(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, np.ndarray)):

        return [clean(v) for v in value]

    if isinstance(value, (np.integer,)):

        return int(value)

    if isinstance(value, (float, np.floating)):

        return float(value) if np.isfinite(value) else None

    if isinstance(value, np.bool_):

        return bool(value)

    return value





def write(path, value):

    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")





def jsonlines(path):

    with Path(path).open(encoding="utf-8-sig") as handle:

        for line in handle:

            if line.strip():

                yield json.loads(line)





def condition(p):

    if p["representation"] == "NO_INFO":

        return "NO_INFO"

    base = f"{p['representation']}_{p['k']}"

    return base if p["relation"] == "unspecified" else base + "_" + p["relation"]





def extract(run, out):

    """Join every durable response to its frozen request; preserve invalid answers."""

    meta = read(run / "run.json")

    runtime = meta["runtime"]

    snapshot = run / "sources"

    for name, digest in meta["source_sha256"].items():

        if sha(snapshot / name) != digest:

            raise ValueError("Collection snapshot hash mismatch: " + name)

    # Reclassification uses the exact same implementation, not a repaired parser.

    for name in ["src/response_readout.py", "src/run_era_formal.py", "src/probe_logprobs.py", "src/llm_client.py"]:

        if sha(ROOT / name) != sha(snapshot / name):

            raise ValueError("Parser source drift: " + name)

    design = read(snapshot / "configs/era_submission_v1.json")

    policy = read(snapshot / runtime["readout_policy"])

    prompts = {r["prompt_id"]: r for r in jsonlines(snapshot / design["files"]["prompts"])}

    manifest = {r["request_id"]: r for r in jsonlines(snapshot / design["files"]["manifest"]) if r["module"] == "core"}

    records, invalid, provenance = [], [], []

    completed, counter, versions = set(), Counter(), set()

    input_tokens = output_tokens = 0

    call_wall = 0.0

    directories = sorted(p for p in (run / "attempts").iterdir() if p.is_dir())

    for index, directory in enumerate(directories):

        result_path = directory / "result.json"

        if not result_path.exists():

            raise ValueError("Unresolved attempt: " + str(directory))

        raw_bytes = result_path.read_bytes()

        result = json.loads(raw_bytes)

        row = result["manifest"]

        rid = row["request_id"]

        if row != manifest[rid]:

            raise ValueError("Manifest mismatch: " + rid)

        if result["outcome"] != "response":

            counter["failed_attempts"] += 1

            continue

        if rid in completed:

            raise ValueError("Duplicate response: " + rid)

        completed.add(rid)

        prompt = prompts[row["prompt_id"]]

        if prompt != result["prompt_metadata"]:

            raise ValueError("Prompt metadata mismatch: " + rid)

        expected = payload_for(row, prompt, runtime["models"][row["model_slot"]], design["generation"])

        if result["request"] != expected:

            raise ValueError("Generation settings mismatch: " + rid)

        if result["backend_version"] != runtime["server_version"] or result["model_digest"] != runtime["models"][row["model_slot"]]["digest"]:

            raise ValueError("Model/backend identity mismatch: " + rid)

        response = result["response"]

        raw_path = directory / "raw_response.json"

        raw_blob = raw_path.read_bytes()

        if json.loads(raw_blob)["response"] != response:

            raise ValueError("Raw/result response disagreement: " + rid)

        classification = classify_response(response, policy)

        if classification != result["readout"]:

            raise ValueError("Classification mismatch: " + rid)

        score = formal_token_score(response, row["model_slot"], policy) if row["mode"] == "logprob" else None

        if score != result["token_score"]:

            raise ValueError("Token-score mismatch: " + rid)

        versions.add(result["backend_version"])

        counter[f"{row['model_slot']}/{row['mode']}"] += 1

        input_tokens += response.get("prompt_eval_count", 0)

        output_tokens += response.get("eval_count", 0)

        call_wall += result["wall_seconds"]

        slim = {**row, **{k: v for k, v in prompt.items() if k not in ("prompt", "prompt_id", "module")},

                "condition": condition(prompt), **classification,

                "token_score": score["native_label_yes_score"] if score else None,

                "token_mass": score["native_label_mass"] if score else None,

                "input_tokens": response.get("prompt_eval_count"), "output_tokens": response.get("eval_count"),

                "wall_seconds": result["wall_seconds"], "result_path": result_path.relative_to(ROOT).as_posix()}

        records.append(slim)

        provenance.append({"request_id": rid, "result_path": slim["result_path"],

                           "result_sha256": hashlib.sha256(raw_bytes).hexdigest(),

                           "raw_sha256": hashlib.sha256(raw_blob).hexdigest()})

        if classification["semantic_class"] == "invalid" or classification["strict_class"] == "invalid":

            invalid.append({**slim, "content": response.get("message", {}).get("content"),

                            "done_reason": response.get("done_reason")})

        if (index + 1) % 10000 == 0:

            print("Offline records checked:", index + 1, flush=True)

    if completed != set(manifest):

        raise ValueError("Missing/unexpected requests in completed batch")

    df = pd.DataFrame(records)

    df.to_csv(out / "observations.csv", index=False)

    pd.DataFrame(provenance).to_csv(out / "input_hashes.csv", index=False)

    pd.DataFrame(invalid).to_csv(out / "invalid_and_strict_exceptions.csv", index=False)

    audit = {"source_run": run.relative_to(ROOT).as_posix(), "status": "PASS", "expected": len(manifest),

             "completed_unique": len(completed), "attempt_directories": len(directories), "counts": counter,

             "backend_versions": sorted(versions), "input_tokens": input_tokens, "output_tokens": output_tokens,

             "call_wall_seconds": call_wall, "collection_status": read(run / "status.json"),

             "snapshot_hashes_verified": len(meta["source_sha256"]),

             "checks": "All request IDs, seeds, prompts, settings, model/backend identity, raw/result consistency, parser and score recomputation checked.",

             "observations_sha256": sha(out / "observations.csv"), "input_hashes_sha256": sha(out / "input_hashes.csv"),

             "new_model_calls": 0, "analysis_seed": BOOT_SEED}

    write(out / "audit.json", audit)

    return df





def interval(values):

    values = np.asarray(values)

    if not np.all(np.isfinite(values)):

        return np.nan, np.nan

    return tuple(np.quantile(values, [0.025, 0.975]))





def centered_p(point, samples):

    """Use a centered two-sided bootstrap test, with a finite-resample correction."""

    samples = np.asarray(samples)

    if not np.isfinite(point) or not np.all(np.isfinite(samples)) or np.ptp(samples) < 1e-12:

        return np.nan

    return (1 + np.count_nonzero(np.abs(samples - point) >= abs(point))) / (len(samples) + 1)





def values_for(group, policy):

    labels = group["strict_class"] if policy == "strict_valid" else group["semantic_class"]

    y = labels.eq("Yes").to_numpy(dtype=float)

    n = labels.ne("invalid").to_numpy(dtype=float)

    if policy == "invalid_as_no":

        n[:] = 1

    if policy == "invalid_as_yes":

        y += 1 - n

        n[:] = 1

    return y, n





def arrays_for(df, policy):

    """Retain equal persona/wording weights; P0 is one control, not 100 personas."""

    personas = sorted(df.loc[df.persona_group.eq("P1"), "persona_id"].unique())

    lookup = {p: i for i, p in enumerate(personas)}

    counts = np.zeros((2, 100, 3, 11, 2))

    p0 = np.full((2, 3, 11, 100, 2), np.nan)

    for keys, group in df.groupby(["model_slot", "persona_id", "wording", "condition"], sort=False):

        model, person, word, cell = keys

        m, w, c = MODELS.index(model), WORDS.index(word), CELLS.index(cell)

        y, n = values_for(group.sort_values("repeat"), policy)

        if person in lookup:

            if len(group) != 5:

                raise ValueError("Incomplete P1 sampling cell")

            counts[m, lookup[person], w, c] = [y.sum(), n.sum()]

        else:

            if len(group) != 100:

                raise ValueError("Incomplete P0 sampling cell")

            p0[m, w, c, :, 0], p0[m, w, c, :, 1] = y, n

    if np.any(counts[..., 1] == 0):

        raise ValueError("All-invalid P1 cell requires explicit missing-cell analysis")

    rates = counts[..., 0] / counts[..., 1]

    return counts, rates, p0, personas





def metrics(p, control):

    """Compute signed contrasts without clipping inadmissible IC diagnostics."""

    output = {"P_no_information": p[..., 0], "D": p[..., 10] - p[..., 9]}

    for rep, start in [("R1", 1), ("R2", 5)]:

        q = p[..., start:start + 4]

        for k in range(4):

            output[f"{rep}_P{k}"] = q[..., k]

        for k in range(1, 4):

            output[f"{rep}_increment{k}"] = q[..., k] - q[..., k-1]

        output[f"{rep}_known_zero_minus_noinfo"] = q[..., 0] - p[..., 0]

        with np.errstate(divide="ignore", invalid="ignore"):

            beta = (q[..., 1] - q[..., 0]) / (1 - q[..., 0])

            admissible = (q[..., 0] < 1 - 1e-6) & (beta >= 0) & (beta <= 1)

            output[f"{rep}_beta"] = np.where(admissible, beta, np.nan)

            for k in (2, 3):

                independent = 1 - (1 - q[..., 0]) * (1 - beta) ** k

                output[f"{rep}_S{k}"] = np.where(admissible, q[..., k] - independent, np.nan)

            output[f"{rep}_rho"] = np.where(np.abs(q[..., 1] - q[..., 0]) >= 0.02,

                (q[..., 2] - q[..., 1]) / (q[..., 1] - q[..., 0]), np.nan)

    output["Delta_R"] = output["R2_S2"] - output["R1_S2"]

    for k in range(4):

        output[f"P1_minus_P0_R1_P{k}"] = p[..., 1+k] - control[..., 1+k]

        output[f"R2_minus_R1_P{k}"] = p[..., 5+k] - p[..., 1+k]

    output["persona_increment_interaction"] = (p[..., 2]-p[..., 1]) - (control[..., 2]-control[..., 1])

    return output





def analyze_metrics(sampling, out):

    """Bootstrap persona blocks jointly across all conditions and both models."""

    rng = np.random.default_rng(BOOT_SEED)

    weights = rng.multinomial(100, np.full(100, 0.01), size=B) / 100

    # The same P0 resampling indices are reused across readout sensitivities.

    p0_draws = rng.integers(0, 100, size=(2, 3, 11, B, 100), dtype=np.int16)

    scopes = {"all": [0, 1, 2], "w1": [0], "w2": [1], "w3": [2],

              "without_w1": [1, 2], "without_w2": [0, 2], "without_w3": [0, 1]}

    curve_rows, metric_rows, heterogeneity = [], [], []

    main_cache = {}

    for policy in POLICIES:

        counts, rates, p0, personas = arrays_for(sampling, policy)

        p0_sum = np.nansum(p0, axis=3)

        p0_mean = np.divide(p0_sum[..., 0], p0_sum[..., 1], out=np.full((2, 3, 11), np.nan), where=p0_sum[..., 1] > 0)

        p0_boot = np.full((2, B, 3, 11), np.nan)

        for m in range(2):

            for w in range(3):

                for c in range(11):

                    if np.isnan(p0[m, w, c]).all():

                        continue

                    drawn = p0[m, w, c][p0_draws[m, w, c]].sum(axis=1)

                    p0_boot[m, :, w, c] = drawn[:, 0] / drawn[:, 1]

        for scope, word_indices in scopes.items():

            pp = rates[:, :, word_indices, :].mean(axis=2)

            mean = pp.mean(axis=1)

            boot = np.einsum("bp,mpc->mbc", weights, pp)

            cp = p0_mean[:, word_indices, :].mean(axis=1)

            cb = p0_boot[:, :, word_indices, :].mean(axis=2)

            point, draws = metrics(mean, cp), metrics(boot, cb)

            for m, model in enumerate(MODELS):

                for name in point:

                    value, sample = point[name][m], draws[name][m].copy()

                    flag = "OK"

                    if name.endswith("rho"):

                        inc = name.split("_")[0] + "_increment1"

                        low, high = interval(draws[inc][m])

                        if not np.isfinite(value) or not np.isfinite(low) or low <= 0 <= high:

                            value, sample[:], flag = np.nan, np.nan, "unstable_denominator"

                    if not np.isfinite(value) and flag == "OK":

                        flag = "nonadmissible_IC_or_undefined"

                    lo, hi = interval(sample)

                    if np.isfinite(value) and not np.isfinite(lo):

                        flag = "bootstrap_crosses_admissibility_boundary"

                    if np.all(np.isfinite(sample)) and np.ptp(sample) < 1e-12:

                        flag = "degenerate_bootstrap_no_inferential_p"

                    metric_rows.append({"model": model, "policy": policy, "wording_scope": scope,

                        "metric": name, "estimate": value, "ci_low": lo, "ci_high": hi,

                        "finite_bootstrap_fraction": np.isfinite(sample).mean(), "flag": flag,

                        "p_centered": centered_p(value, sample), "primary_family": name in ("R1_S2", "R2_S2", "Delta_R", "persona_increment_interaction")})

                for group, means, boots in [("P1", mean, boot), ("P0", cp, cb)]:

                    for c, cell in enumerate(CELLS):

                        if not np.isfinite(means[m, c]):

                            continue

                        lo, hi = interval(boots[m, :, c])

                        curve_rows.append({"model": model, "policy": policy, "wording_scope": scope,

                            "persona_group": group, "condition": cell, "estimate": means[m, c],

                            "ci_low": lo, "ci_high": hi,

                            "degenerate_bootstrap": bool(np.ptp(boots[m, :, c]) < 1e-12)})

            if policy == "semantic_valid" and scope == "all":

                main_cache = {"means": mean, "boot": boot, "control": cp, "point": point}

                profiles = []

                for m, model in enumerate(MODELS):

                    individual = metrics(pp[m], np.broadcast_to(cp[m], pp[m].shape))

                    for person_index, person in enumerate(personas):

                        profiles.append({"model": model, "persona_id": person,

                            **{cell: pp[m, person_index, c] for c, cell in enumerate(CELLS)},

                            **{name: individual[name][person_index] for name in ("R1_S2", "R2_S2", "D")},

                            "limit": "Descriptive finite-sample profile; not an identified individual threshold or demographic causal effect."})

                    for c, cell in enumerate(CELLS):

                        sd = float(pp[m, :, c].std(ddof=1))

                        bs = np.sqrt(np.maximum(0, (weights @ (pp[m, :, c]**2) - (weights @ pp[m, :, c])**2) * 100/99))

                        lo, hi = interval(bs)

                        heterogeneity.append({"model": model, "condition": cell, "observed_persona_rate_sd": sd,

                            "ci_low": lo, "ci_high": hi, "interpretation": "Descriptive variation includes finite sampling noise; not human or latent threshold variance."})

                pd.DataFrame(profiles).to_csv(out / "persona_response_profiles.csv", index=False)

                np.savez_compressed(out / "paired_bootstrap.npz", weights=weights, estimates=mean,

                                    probability_bootstrap=boot, p0_probability_bootstrap=cb)

    curves, stats = pd.DataFrame(curve_rows), pd.DataFrame(metric_rows)

    # Keep the prespecified family size eight, including non-estimable contrasts.

    stats["p_holm"] = np.nan

    for policy in POLICIES:

        indices = stats.index[stats.policy.eq(policy) & stats.wording_scope.eq("all") & stats.primary_family]

        p = stats.loc[indices, "p_centered"].to_numpy()

        if len(indices) != 8:

            raise ValueError("Primary family must have eight contrasts")

        adjusted = multipletests(np.where(np.isfinite(p), p, 1.0), method="holm")[1]

        stats.loc[indices, "p_holm"] = np.where(np.isfinite(p), adjusted, np.nan)

    curves.to_csv(out / "probability_curves.csv", index=False)

    stats.to_csv(out / "metrics_and_sensitivity.csv", index=False)

    pd.DataFrame(heterogeneity).to_csv(out / "persona_heterogeneity.csv", index=False)

    return curves, stats, main_cache





def count_tables(df, out):

    """Expose planned-denominator bounds separately from conditional valid-only rates."""

    rows = []

    keys = ["model_slot", "mode", "persona_group", "wording", "condition"]

    for group, data in df.groupby(keys, sort=True):

        n = len(data)

        yes, no = data.semantic_class.eq("Yes").sum(), data.semantic_class.eq("No").sum()

        unknown = n - yes - no

        row = dict(zip(keys, group))

        row.update(planned=n, yes=yes, no=no, invalid=unknown,

                   strict_invalid=data.strict_class.eq("invalid").sum(),

                   yes_per_planned=yes/n, ambiguity_low=yes/n, ambiguity_high=(yes+unknown)/n,

                   valid_only_yes=yes/(yes+no) if yes+no else np.nan,

                   token_scores_available=data.token_score.notna().sum())

        # At all-identical boundaries report a conditional exact-binomial interval.

        # This is not a CI for a population of humans or new synthetic personas.

        if unknown == 0 and yes in (0, n):

            row.update(boundary_ci_low=0 if yes == 0 else beta_distribution.ppf(.025, n, 1),

                       boundary_ci_high=1 if yes == n else beta_distribution.ppf(.975, 1, n),

                       boundary_note="Conditional independent draws at fixed prompts; boundary uncertainty only")

        rows.append(row)

    table = pd.DataFrame(rows)

    table.to_csv(out / "condition_counts_and_bounds.csv", index=False)

    summary = []

    for (model, mode), data in df.groupby(["model_slot", "mode"]):

        summary.append({"model": model, "mode": mode, "planned": len(data),

                        "valid": int(data.semantic_class.ne("invalid").sum()),

                        "invalid": int(data.semantic_class.eq("invalid").sum()),

                        "strict_invalid": int(data.strict_class.eq("invalid").sum()),

                        "token_scores_available": int(data.token_score.notna().sum()),

                        "wall_seconds": data.wall_seconds.sum(), "input_tokens": data.input_tokens.sum(),

                        "output_tokens": data.output_tokens.sum()})

    pd.DataFrame(summary).to_csv(out / "readout_and_cost.csv", index=False)

    return summary





def missingness_bounds(curves, out):

    """Bound contrasts over all assignments of invalid replies, without imputing them."""

    records = []

    for model in MODELS:

        for scope in curves.wording_scope.unique():

            base = curves[curves.model.eq(model) & curves.wording_scope.eq(scope)]

            lower = base[base.policy.eq("invalid_as_no")].set_index(["persona_group", "condition"]).estimate

            upper = base[base.policy.eq("invalid_as_yes")].set_index(["persona_group", "condition"]).estimate

            result = {}

            for group in ("P1", "P0"):

                cells = [c for c in CELLS if (group, c) in lower.index]

                for cell in cells:

                    result[f"{group}_{cell}"] = (lower.loc[group, cell], upper.loc[group, cell])

            def difference(a, b):

                return a[0]-b[1], a[1]-b[0]

            for rep in ("R1", "R2"):

                p = [result[f"P1_{rep}_{k}"] for k in range(4)]

                for k in range(1, 4):

                    result[f"{rep}_increment{k}"] = difference(p[k], p[k-1])

                # S_k increases with Pk and P0 and decreases with P1.

                # Only apply the rectangle bound if every allowed IC baseline is admissible.

                admissible = p[0][1] < 1-1e-6 and p[1][0] >= p[0][1]

                for k in (2, 3):

                    if admissible:

                        lo = p[k][0] - 1 + (1-p[1][1])**k/(1-p[0][0])**(k-1)

                        hi = p[k][1] - 1 + (1-p[1][0])**k/(1-p[0][1])**(k-1)

                        result[f"{rep}_S{k}"] = (lo, hi)

                    else:

                        result[f"{rep}_S{k}"] = (np.nan, np.nan)

            result["Delta_R"] = difference(result["R2_S2"], result["R1_S2"])

            result["D"] = difference(result["P1_R2_2_untied"], result["P1_R2_2_tied"])

            p0_increment = difference(result["P0_R1_1"], result["P0_R1_0"])

            result["persona_increment_interaction"] = difference(result["R1_increment1"], p0_increment)

            for metric, (lo, hi) in result.items():

                records.append({"model": model, "wording_scope": scope, "metric": metric,

                    "identification_low": lo, "identification_high": hi,

                    "note": "Invalid-answer assignment bounds at observed frequencies, not confidence intervals; undefined IC cases remain unavailable."})

    pd.DataFrame(records).to_csv(out / "invalid_assignment_bounds.csv", index=False)





def prediction(kind, parameters):

    k = np.arange(4)

    if kind == "M_CAT":

        return np.asarray(parameters)

    p0 = parameters[0]

    if kind == "M_IC":

        return p0 + (1-p0) * (1-(1-parameters[1])**k)

    if kind == "M_LIN":

        return p0 + (1-p0) * parameters[1] * k / 6

    tau, scale = parameters[1:]

    # Stable normalized logistic CDF, exactly zero at k=0 before adding p0.

    reinforcement = -np.expm1(-k/scale) * expit((k-tau)/scale)

    return p0 + (1-p0) * reinforcement





def negloglik(p, y, n):

    p = np.clip(p, EPS, 1-EPS)

    return float(-np.sum(xlogy(y, p) + xlogy(n-y, 1-p)))





def fit(kind, y, n):

    # Sensitivity tables often share identical sufficient statistics; reuse only exact matches.

    return fit_cached(kind, tuple(y), tuple(n))





@lru_cache(maxsize=4096)

def fit_cached(kind, y_tuple, n_tuple):

    y, n = np.asarray(y_tuple), np.asarray(n_tuple)

    rates = y/n

    if kind == "M_CAT":

        return rates, rates, negloglik(rates, y, n), True, bool(np.any((rates == 0) | (rates == 1)))

    if kind == "M_THR":

        bounds = [(0, 1), (0, 6), (.1, 5)]

        starts = [[rates[0], t, s] for t in (0, 2, 4, 6) for s in (.2, 1, 4)]

    else:

        bounds = [(0, 1), (0, 1)]

        starts = [[rates[0], v] for v in (0, .05, .3, .8, 1)]

    candidates = [minimize(lambda x: negloglik(prediction(kind, x), y, n), x,

                           method="L-BFGS-B", bounds=bounds,

                           options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 500}) for x in starts]

    best = min(candidates, key=lambda r: r.fun)

    # Retry with a derivative-free method only if no successful fit reaches the optimum.

    successful = [r for r in candidates if r.success and r.fun <= best.fun + 1e-5]

    if not successful:

        retry = minimize(lambda x: negloglik(prediction(kind, x), y, n), best.x,

                         method="Powell", bounds=bounds, options={"ftol": 1e-11, "maxiter": 1000})

        if retry.success and retry.fun <= best.fun + 1e-5:

            best = retry

    else:

        best = min(successful, key=lambda r: r.fun)

    boundary = any(abs(x-lo) < 1e-4 or abs(x-hi) < 1e-4 for x, (lo, hi) in zip(best.x, bounds))

    return best.x, prediction(kind, best.x), float(best.fun), bool(best.success), boundary





def fit_models(sampling, out):

    rows, predictions, folds = [], [], []

    for policy in POLICIES:

        for model in MODELS:

            for rep in ("R1", "R2"):

                for scope in ("all", *WORDS):

                    subset = sampling[sampling.model_slot.eq(model) & sampling.persona_group.eq("P1") &

                                      sampling.representation.eq(rep) & sampling.relation.eq("unspecified")].copy()

                    if scope != "all":

                        subset = subset[subset.wording.eq(scope)].copy()

                    subset["y"], subset["n_fit"] = values_for(subset, policy)

                    total = subset.groupby("k")[["y", "n_fit"]].sum().reindex(range(4))

                    y, n = total.y.to_numpy(), total.n_fit.to_numpy()

                    for kind, dims in [("M_IC", 2), ("M_LIN", 2), ("M_THR", 3), ("M_CAT", 4)]:

                        parameters, pred, nll, success, boundary = fit(kind, y, n)

                        fold_loss, fold_n, all_success = [], [], success

                        for fold in range(5):

                            test = subset[subset.cv_fold.eq(fold)].groupby("k")[["y", "n_fit"]].sum().reindex(range(4))

                            yt, nt = test.y.to_numpy(), test.n_fit.to_numpy()

                            _, pp, _, ok, _ = fit(kind, y-yt, n-nt)

                            loss = negloglik(pp, yt, nt)

                            fold_loss.append(loss)

                            fold_n.append(nt.sum())

                            all_success &= ok

                            folds.append({"model": model, "policy": policy, "representation": rep, "wording_scope": scope,

                                          "candidate": kind, "fold": fold, "test_n": nt.sum(), "log_loss": loss/nt.sum(), "optimizer_success": ok})

                        rows.append({"model": model, "policy": policy, "representation": rep, "wording_scope": scope,

                            "candidate": kind, "n_binary": n.sum(), "n_parameters": dims,

                            "parameters": json.dumps(parameters.tolist()), "nll": nll,

                            "bic": 2*nll + dims*np.log(n.sum()), "cv_log_loss": sum(fold_loss)/sum(fold_n),

                            "optimizer_success_all": all_success, "parameter_boundary": boundary,

                            "caveat": "Working conditional Bernoulli BIC; persona-grouped CV is primary predictive check; boundary THR parameters can be weakly identified."})

                        for k in range(4):

                            predictions.append({"model": model, "policy": policy, "representation": rep,

                                "wording_scope": scope, "candidate": kind, "k": k, "observed": y[k]/n[k],

                                "predicted": pred[k], "residual": y[k]/n[k]-pred[k]})

                print("Fitted:", policy, model, rep, flush=True)

    table = pd.DataFrame(rows)

    grouping = ["model", "policy", "representation", "wording_scope"]

    table["delta_bic"] = table.bic - table.groupby(grouping).bic.transform("min")

    table["delta_cv_log_loss"] = table.cv_log_loss - table.groupby(grouping).cv_log_loss.transform("min")

    table.to_csv(out / "model_comparison.csv", index=False)

    pd.DataFrame(predictions).to_csv(out / "model_predictions.csv", index=False)

    pd.DataFrame(folds).to_csv(out / "grouped_cv_folds.csv", index=False)

    return table





def secondary_scores(df, sampling, out):

    """Keep T=1 token-pair scores separate from T=.7 sampled decisions."""

    lp = df[df["mode"].eq("logprob")].copy()

    sampled = sampling.groupby(["model_slot", "prompt_id"]).semantic_class.agg(

        lambda x: (x == "Yes").sum() / (x != "invalid").sum()).rename("sampling_valid_rate")

    lp = lp.merge(sampled, on=["model_slot", "prompt_id"], how="left", validate="one_to_one")

    lp["score_minus_sampling"] = lp.token_score - lp.sampling_valid_rate

    lp.to_csv(out / "secondary_token_scores.csv", index=False)

    summaries = []

    weights = np.random.default_rng(BOOT_SEED).multinomial(100, np.full(100, .01), size=B)/100

    for keys, group in lp.groupby(["model_slot", "persona_group", "wording", "condition"]):

        model, pg, word, cell = keys

        finite = group.token_score.notna()

        lo = hi = np.nan

        if pg == "P1" and finite.all():

            score = group.sort_values("persona_id").token_score.to_numpy()

            lo, hi = interval(weights @ score)

        summaries.append({"model": model, "persona_group": pg, "wording": word, "condition": cell,

            "n_planned": len(group), "n_available": finite.sum(), "coverage": finite.mean(),

            "mean_available_score": group.token_score.mean(), "ci_low": lo, "ci_high": hi,

            "score_ambiguity_low": group.token_score.fillna(0).mean(),

            "score_ambiguity_high": group.token_score.fillna(1).mean(),

            "mean_available_label_mass": group.token_mass.mean(),

            "fraction_planned_with_mass_ge_095": (group.token_mass >= .95).sum()/len(group),

            "mean_score_minus_sampling_on_available_pairs": group.score_minus_sampling.mean()})

    pd.DataFrame(summaries).to_csv(out / "secondary_token_summary.csv", index=False)





def plots(curves, stats, models, out):

    """Draw scientific figures with fixed axes, uncertainty, and explicit estimands."""

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    for ax, model in zip(axes, MODELS):

        for rep, color in [("R1", "#1565c0"), ("R2", "#c45100")]:

            selection = curves[curves.model.eq(model) & curves.policy.eq("semantic_valid") &

                               curves.wording_scope.eq("all") & curves.persona_group.eq("P1")].set_index("condition")

            part = selection.loc[[f"{rep}_{k}" for k in range(4)]]

            ax.plot(range(4), part.estimate, "o-", label=rep, color=color)

            ax.fill_between(range(4), part.ci_low, part.ci_high, color=color, alpha=.16)

            bounds = []

            for policy in ("invalid_as_no", "invalid_as_yes"):

                b = curves[curves.model.eq(model) & curves.policy.eq(policy) & curves.wording_scope.eq("all") &

                           curves.persona_group.eq("P1")].set_index("condition").loc[[f"{rep}_{k}" for k in range(4)]]

                bounds.append(b.estimate.to_numpy())

            ax.vlines(np.arange(4)+.045, bounds[0], bounds[1], color=color, lw=3, alpha=.8)

        noinfo = selection.loc["NO_INFO", "estimate"]

        ax.axhline(noinfo, color="gray", ls=":", label="No-information (P1)")

        ax.set(title=LABELS[model], xlabel="Adopting friends k (of 6)", xticks=range(4), ylim=(-.025, 1.025))

        ax.grid(alpha=.2)

        ax.legend(loc="best", fontsize=9)

    axes[0].set_ylabel("Yes rate conditional on a valid answer")

    fig.suptitle("ERA core: fixed coupon, 100 synthetic personas, three wordings")

    fig.text(.5, .01, "Shading: paired persona-bootstrap 95% CI. Offset bars: invalid-answer ambiguity.\nBoundary bootstrap CIs can collapse; consult boundary uncertainty table. No human calibration claim.", ha="center", fontsize=8)

    fig.tight_layout(rect=(0, .065, 1, .95))

    fig.savefig(out / "era_core_curves.png", dpi=180)

    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)

    for row, model in enumerate(MODELS):

        for col, rep in enumerate(("R1", "R2")):

            ax = axes[row, col]

            for word in WORDS:

                part = curves[curves.model.eq(model) & curves.policy.eq("semantic_valid") & curves.wording_scope.eq(word) &

                              curves.persona_group.eq("P1")].set_index("condition").loc[[f"{rep}_{k}" for k in range(4)]]

                ax.plot(range(4), part.estimate, "o-", label=word)

                ax.fill_between(range(4), part.ci_low, part.ci_high, alpha=.1)

            ax.set(title=f"{LABELS[model]} / {rep}", xticks=range(4), ylim=(-.025, 1.025), xlabel="Adopting friends k", ylabel="Valid-answer Yes rate")

            ax.legend()

            ax.grid(alpha=.2)

    fig.suptitle("Wording sensitivity (three fixed wordings, not a language population)")

    fig.tight_layout()

    fig.savefig(out / "wording_sensitivity.png", dpi=180)

    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for ax, model in zip(axes, MODELS):

        chosen = stats[stats.model.eq(model) & stats.policy.eq("semantic_valid") & stats.wording_scope.eq("all")]

        names = ["R1_S2", "R2_S2", "Delta_R", "persona_increment_interaction", "D"]

        for i, name in enumerate(names):

            r = chosen[chosen.metric.eq(name)].iloc[0]

            if np.isfinite(r.estimate):

                ax.plot(r.estimate, i, "o", color="#1565c0")

                if np.isfinite(r.ci_low):

                    ax.plot([r.ci_low, r.ci_high], [i, i], color="#1565c0")

                else:

                    ax.annotate("CI unavailable", (r.estimate, i), xytext=(7, 5), textcoords="offset points", fontsize=8)

            else:

                ax.text(.02, i, "not admissible / undefined", fontsize=8, va="center")

        ax.axvline(0, color="gray", ls=":")

        ax.set(yticks=range(len(names)), yticklabels=names, title=LABELS[model], xlabel="Effect (probability units)", ylim=(-.6, 4.6))

        ax.grid(axis="x", alpha=.2)

    fig.suptitle("ERA contrasts: pointwise 95% CIs; D is exploratory")

    fig.tight_layout()

    fig.savefig(out / "era_contrasts.png", dpi=180)

    plt.close(fig)





def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run-dir", type=Path, required=True)

    parser.add_argument("--analysis-dir", type=Path)

    parser.add_argument("--extract-only", action="store_true")

    args = parser.parse_args()

    run = args.run_dir.resolve()

    out = args.analysis_dir or ROOT / "runs/era_analysis" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12])

    out = out.resolve()

    out.mkdir(parents=True, exist_ok=True)

    if (out / "summary.json").exists():

        raise ValueError("Completed analysis is immutable; choose a new output directory")



    started = time.perf_counter()

    if (out / "audit.json").exists():

        audit = read(out / "audit.json")

        if audit["source_run"] != run.relative_to(ROOT).as_posix() or audit["observations_sha256"] != sha(out / "observations.csv"):

            raise ValueError("Offline extraction cache mismatch")

        df = pd.read_csv(out / "observations.csv")

    else:

        df = extract(run, out)

    print("Analysis output:", out, flush=True)

    if args.extract_only:

        return

    sampling = df[df["mode"].eq("sampling")].copy()

    quality = count_tables(df, out)

    if df.semantic_class.eq("invalid").mean() > .01:

        raise ValueError("More than 1% invalid core replies; inferential analysis paused")

    curves, stats, cache = analyze_metrics(sampling, out)

    missingness_bounds(curves, out)

    print("Paired bootstrap complete", flush=True)

    comparison = fit_models(sampling, out)

    secondary_scores(df, sampling, out)

    plots(curves, stats, comparison, out)

    primary = stats[stats.policy.eq("semantic_valid") & stats.wording_scope.eq("all") & stats.primary_family]

    main_models = comparison[comparison.policy.eq("semantic_valid") & comparison.wording_scope.eq("all")]

    audit = read(out / "audit.json")


    shutil.copy2(__file__, out / "analysis_source.py")


    metadata = {"status": "COMPLETE", "source_run": audit["source_run"], "analysis_directory": out.relative_to(ROOT).as_posix(),

        "created_utc": datetime.now(timezone.utc).isoformat(), "new_model_calls": 0,

        "analysis_wall_seconds": time.perf_counter()-started, "bootstrap_replicates": B, "bootstrap_seed": BOOT_SEED,

        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,

        "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform(),

        "analysis_sha256": sha(__file__),

        "quality": quality, "primary_tests": primary.to_dict("records"), "model_comparison": main_models.to_dict("records"),

        "interpretation_limits": ["Finite synthetic coupon study; no matched human probability target.",

            "Valid-answer curves are conditional estimates; planned-denominator ambiguity bounds are separately retained.",

            "All-invalid-as-Yes/No analyses are sensitivity scenarios, not sharp bounds on nonlinear contrasts.",

            "Undefined IC baselines and unstable ratios have no inferential p-value.",

            "BIC uses a working conditional Bernoulli likelihood; grouped CV and boundary warnings must accompany interpretation.",

            "P1 bootstrap conditions on three wordings and includes response sampling in observed persona variation; no human-population inference.",

            "Auxiliary T=1 token-pair scores are not T=.7 adoption frequencies or full-answer probabilities.",

            "Bridge-support, held-out items, full-LLM macro, human calibration and RCA remain unexecuted."]}

    write(out / "summary.json", metadata)

    artifact_hashes = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "artifact_hashes.json"}

    write(out / "artifact_hashes.json", artifact_hashes)

    print(json.dumps(clean({"status": "COMPLETE", "output": str(out), "quality": quality,

                            "primary": primary.to_dict("records")}), ensure_ascii=False, indent=2), flush=True)





if __name__ == "__main__":

    main()

