"""Collect the frozen ERA core design with durable raw responses and resumability."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import uuid

from llm_client import ROOT, OllamaClient, git_output
from probe_logprobs import inspect_logprobs
from response_readout import classify_response

DESIGN = "configs/era_submission_v1.json"
RUNTIME = "configs/era_submission_runtime_v3.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    """Replace only operational metadata; raw response filenames are unique."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@contextmanager
def run_lock(run):
    """An OS lock is released on process exit, including an unexpected crash."""
    with (run / "process.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_design():
    design = read_json(ROOT / DESIGN)
    # These checks prevent collecting data against silently edited materials.
    for name, expected in {**design["artifact_sha256"], **design["source_sha256"]}.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    prompts = {row["prompt_id"]: row for row in rows(ROOT / design["files"]["prompts"])}
    manifest = list(rows(ROOT / design["files"]["manifest"]))
    return design, prompts, manifest


def payload_for(row, prompt, binding, generation):
    options = dict(generation["options_common"])
    options.update(seed=row["seed"], temperature=generation[
        "temperature_logprob" if row["mode"] == "logprob" else "temperature_sampling"])
    payload = {"model": binding["tag"], "messages": [{"role": "user", "content": prompt["prompt"]}],
               "stream": False, "keep_alive": "30m", "options": options}
    if binding["thinking"] is not None:
        payload["think"] = binding["thinking"]
    if row["mode"] == "logprob":
        payload.update(logprobs=True, top_logprobs=generation["top_logprobs"])
    return payload


def token_score(response):
    """Keep missing labels unknown and distinguish token scores from decisions."""
    checks = inspect_logprobs(response)
    result = {"exact_label_yes_score": None, "exact_label_mass": None,
              "returned_candidate_mass": None, "checks": checks}
    rc = checks["response_checks"]
    usable = (checks["both_exact_labels_available"] and checks["first_token_matches_content_prefix"]
              and checks["first_recorded_token"] in ("Yes", "No")
              and rc["no_visible_thinking"] and rc["finished_without_truncation"])
    if usable:
        yes = checks["exact_label_entries"]["Yes"][0]["logprob"]
        no = checks["exact_label_entries"]["No"][0]["logprob"]
        peak = max(yes, no)
        result.update(logp_yes=yes, logp_no=no,
                      exact_label_yes_score=math.exp(yes-peak)/(math.exp(yes-peak)+math.exp(no-peak)),
                      exact_label_mass=math.exp(yes)+math.exp(no))
    if checks["first_position_alternatives_valid"]:
        result["returned_candidate_mass"] = sum(math.exp(r["logprob"]) for r in checks["first_position_alternatives"])
    return result


def formal_token_score(response, slot, policy):
    """Use a model-specific, frozen exact token pair; never strip candidate tokens."""
    legacy = token_score(response)
    pair = policy["token_score"]["exact_pair_by_model_slot"][slot]
    checks = legacy["checks"]
    alternatives = checks["first_position_alternatives"]
    labels = {name: [r for r in alternatives if r["token"] == token]
              for name, token in pair.items()} if checks["first_position_alternatives_valid"] else {}
    eligible = classify_response(response, policy)["response_eligible"]
    usable = (eligible and checks["first_token_matches_content_prefix"]
              and checks["first_recorded_token"] in pair.values()
              and all(len(labels.get(label, [])) == 1 for label in ("Yes", "No")))
    result = {"policy_version": policy["version"], "exact_token_pair": pair,
              "native_label_yes_score": None, "native_label_mass": None,
              "legacy_exact_no_space": legacy,
              "limit": "Conditional first-token pair score, not full-answer or human probability; model-specific label conventions."}
    if usable:
        yes, no = labels["Yes"][0]["logprob"], labels["No"][0]["logprob"]
        peak = max(yes, no)
        result.update(logp_yes=yes, logp_no=no,
                      native_label_yes_score=math.exp(yes-peak)/(math.exp(yes-peak)+math.exp(no-peak)),
                      native_label_mass=math.exp(yes)+math.exp(no))
    return result


def identity(client, runtime, slot):
    """Stop rather than pool responses from different backends or weights."""
    version = client.request("/api/version")
    if version.get("version") != runtime["server_version"]:
        raise ValueError("Ollama server version changed from the runtime binding.")
    bound = runtime["models"][slot]
    matches = [m for m in client.request("/api/tags")["models"] if m["name"] == bound["tag"]]
    if len(matches) != 1 or matches[0]["digest"] != bound["digest"]:
        raise ValueError("Model digest changed: " + slot)
    info = client.request("/api/show", {"model": bound["tag"]})
    if hashlib.sha256(info.get("template", "").encode()).hexdigest() != bound["template_sha256"]:
        raise ValueError("Model template changed: " + slot)
    supported = (info.get("thinking") or {}).get("values")
    if bound["thinking"] is False and supported is not None and False not in supported:
        raise ValueError("Model does not support the specified thinking setting.")
    return {"checked_at": utc(), "version": version, "model": matches[0], "show": info}


def initialize(run, runtime, design):
    if (run / "run.json").exists():
        meta = read_json(run / "run.json")
        for name, expected in meta["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError("Resume source mismatch; an explicit amendment is required: " + name)
        return meta
    sources = set(design["artifact_sha256"]) | set(design["source_sha256"])
    sources.update([DESIGN, RUNTIME, "src/run_era_formal.py", "src/llm_client.py",
                    "src/probe_logprobs.py", "src/response_readout.py"])
    sources.update([runtime["readout_policy"]])
    for bound in runtime["models"].values():
        sources.add(bound["show_file"])
    hashes = {}
    for name in sorted(sources):
        destination = run / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = sha(ROOT / name)
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as error:
        gpu = {"unavailable": str(error)}
    meta = {"created_utc": utc(), "runtime": runtime, "source_sha256": hashes,
            "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
            "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "hardware": gpu, "gpu_hours": None, "api_cost": None,
            "cost_note": "GPU utilization time and billing not measured; wall time is recorded separately.",

            "purpose": "Formal core collection; no viability pilot and no interim scientific decision."}
    atomic_json(run / "run.json", meta)
    return meta


def collect(run, runtime, design, prompts, manifest):
    selected = [r for slot in runtime["model_order"] for r in manifest
                if r["model_slot"] == slot and r["module"] in runtime["modules"]]
    if runtime["profile"] != "full" or runtime["modules"] != ["core"] or len(selected) != 42630:
        raise ValueError("This runner is bound to the authorized full core design only.")
    if len({r["request_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate request IDs.")
    initialize(run, runtime, design)
    (run / "attempts").mkdir(exist_ok=True)
    policy = read_json(ROOT / runtime["readout_policy"])
    completed, counts, totals = set(), Counter(), Counter()
    # Recover a saved raw reply after a crash between receipt and classification.
    # A model response that already exists must never cause a second model call.
    for raw_path in (run / "attempts").glob("*/raw_response.json"):
        destination = raw_path.parent / "result.json"
        if destination.exists():
            continue
        request_record = read_json(raw_path.parent / "request.json")
        envelope = read_json(raw_path)
        response = envelope["response"]
        score = formal_token_score(response, request_record["manifest"]["model_slot"], policy) if request_record["manifest"]["mode"] == "logprob" else None
        atomic_json(destination, {**request_record, **envelope, "outcome": "response",
                                  "readout": classify_response(response, policy), "token_score": score,
                                  "recovered_from_saved_raw_reply": True})
    # Finished invalid answers are still completed observations; never retry them.
    for path in sorted((run / "attempts").glob("*/result.json")):
        result = read_json(path)
        totals["attempts"] += 1
        totals["call_wall_seconds"] += result["wall_seconds"]
        if result["outcome"] == "response":
            request_id = result["manifest"]["request_id"]
            if request_id in completed:
                raise ValueError("Duplicate completed observation: " + request_id)
            completed.add(request_id)
            counts[result["manifest"]["model_slot"] + "/" + result["manifest"]["mode"]] += 1
            totals["input_tokens"] += result["response"].get("prompt_eval_count") or 0
            totals["output_tokens"] += result["response"].get("eval_count") or 0
            totals["invalid_responses"] += result["readout"]["semantic_class"] == "invalid"
            if result["manifest"]["mode"] == "logprob":
                totals["unavailable_token_scores"] += result["token_score"]["native_label_yes_score"] is None
        else:
            totals["transport_or_api_errors"] += 1
    orphaned = [p.parent.name for p in (run / "attempts").glob("*/request.json")
                if not (p.parent / "result.json").exists()]
    # Count durable request attempts even when their response was lost on interruption.
    totals["attempts"] += len(orphaned)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    atomic_json(run / ("session_" + session + ".json"),
                {"started_utc": utc(), "pid": os.getpid(), "completed_before": len(completed),
                 "unresolved_prior_attempts": orphaned,
                 "note": "An interrupted request may have run remotely without a saved reply; its reattempt is separately recorded."})
    start, previous_slot, consecutive_errors = time.perf_counter(), None, 0
    client = OllamaClient(runtime["base_url"], timeout=600)

    def status(state, **extra):
        value = {"state": state, "updated_utc": utc(), "pid": os.getpid(), "planned": len(selected),
                 "completed_responses": len(completed), "counts": dict(counts), "totals": dict(totals),
                 "session_wall_seconds": time.perf_counter()-start,
                 "orphaned_prior_attempts": len(orphaned),
                 "inference": "Not analyzed; collection progress is not a research conclusion.", **extra}
        atomic_json(run / "status.json", value)

    def progress(slot):
        # The visible terminal reports counts only; responses remain on disk.
        print(f"\rModel: {slot:<14} | Completed: {len(completed):>5} / {len(selected)} ({100*len(completed)/len(selected):6.2f}%)", end="", flush=True)

    status("RUNNING")
    print("Formal run:", run, "planned:", len(selected), flush=True)
    progress("starting")
    try:
        for row in selected:
            if row["request_id"] in completed:
                continue
            if (run / "STOP_REQUESTED").exists():
                status("PAUSED_BY_STOP_FILE")
                print("\nPAUSED. Saved responses are retained.", flush=True)
                return
            slot = row["model_slot"]
            if slot != previous_slot:
                if previous_slot is not None:
                    # Free the previous model before loading the next on a small GPU.
                    unloaded = client.request("/api/generate", {"model": runtime["models"][previous_slot]["tag"], "keep_alive": 0})
                    atomic_json(run / ("unload_" + previous_slot + "_" + session + ".json"), unloaded)
                atomic_json(run / ("identity_" + slot + "_" + session + ".json"), identity(client, runtime, slot))
                previous_slot = slot
            elif len(completed) % 500 == 0:
                identity(client, runtime, slot)
            prompt = prompts[row["prompt_id"]]
            payload = payload_for(row, prompt, runtime["models"][slot], design["generation"])
            attempt = run / "attempts" / (str(row["manifest_index"]).zfill(6) + "_" + uuid.uuid4().hex)
            attempt.mkdir(exist_ok=False)
            record = {"started_utc": utc(), "manifest": row, "prompt_metadata": prompt,
                      "request": payload, "model_digest": runtime["models"][slot]["digest"],
                      "backend_version": runtime["server_version"]}
            atomic_json(attempt / "request.json", record)
            began = time.perf_counter()
            totals["attempts"] += 1
            try:
                response = client.request("/api/chat", payload)
                if response.get("error"):
                    raise RuntimeError(json.dumps(response))
            except Exception as error:
                detail = {"type": type(error).__name__, "message": str(error)}
                if isinstance(error, urllib.error.HTTPError):
                    detail["body"] = error.read().decode("utf-8", errors="replace")
                elapsed = time.perf_counter()-began
                atomic_json(attempt / "result.json", {**record, "outcome": "error", "error": detail,
                            "ended_utc": utc(), "wall_seconds": elapsed})
                totals["transport_or_api_errors"] += 1
                totals["call_wall_seconds"] += elapsed
                consecutive_errors += 1
                print("\nRequest error recorded; see saved logs.", flush=True)
                if consecutive_errors >= 3:
                    raise RuntimeError("Paused after three consecutive API/transport errors; repair before resume.")
                status("RUNNING", last_error=detail)
                continue
            elapsed = time.perf_counter()-began
            # Save the raw answer before parsing, so parser failures cannot lose it.
            envelope = {"response": response, "wall_seconds": elapsed, "ended_utc": utc()}
            atomic_json(attempt / "raw_response.json", envelope)
            readout = classify_response(response, policy)
            score = formal_token_score(response, slot, policy) if row["mode"] == "logprob" else None
            atomic_json(attempt / "result.json", {**record, "outcome": "response", "response": response,
                        "readout": readout, "token_score": score, "ended_utc": utc(), "wall_seconds": elapsed})
            completed.add(row["request_id"])
            counts[slot + "/" + row["mode"]] += 1
            totals["call_wall_seconds"] += elapsed
            totals["input_tokens"] += response.get("prompt_eval_count") or 0
            totals["output_tokens"] += response.get("eval_count") or 0
            totals["invalid_responses"] += readout["semantic_class"] == "invalid"
            if score is not None:
                totals["unavailable_token_scores"] += score["native_label_yes_score"] is None
            consecutive_errors = 0
            if counts[slot + "/sampling"] + counts[slot + "/logprob"] == 1 or len(completed) % 500 == 0:
                atomic_json(run / ("loaded_models_" + str(len(completed)) + ".json"), client.request("/api/ps"))
            if len(completed) == 1 or len(completed) % 25 == 0:
                status("RUNNING", current_model=slot)
                progress(slot)
        status("COLLECTION_COMPLETE" if len(completed) == len(selected) else "INCOMPLETE_ERRORS_NEED_REVIEW")
        progress(previous_slot or "finished")
        print("\nCOLLECTION COMPLETE" if len(completed) == len(selected) else "\nINCOMPLETE: requests with errors require review.", flush=True)
    except BaseException:
        status("PAUSED_ERROR", error=traceback.format_exc())
        raise
    finally:
        print("\nStatus:", run / "status.json", flush=True)
        if previous_slot:
            try:
                atomic_json(run / ("final_unload_" + session + ".json"), client.request("/api/generate", {"model": runtime["models"][previous_slot]["tag"], "keep_alive": 0}))
            except Exception as error:
                atomic_json(run / ("unload_error_" + session + ".json"), {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Existing run resumes missing observations; new directory starts a run.")
    parser.add_argument("--validate-design", action="store_true", help="Local consistency only; sends no model requests.")
    args = parser.parse_args()
    design, prompts, manifest = load_design()
    if args.validate_design:
        core = [r for r in manifest if r["module"] == "core"]
        print("Frozen design verified; core calls:", len(core), "; prompts:", len(prompts), "; no generation.")
        return
    runtime = read_json(ROOT / RUNTIME)
    if sha(ROOT / runtime["readout_policy"]) != runtime["readout_policy_sha256"]:
        raise ValueError("Frozen readout policy changed.")
    run = args.run_dir or ROOT / "runs/era_formal" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex)
    if not run.is_absolute():
        run = ROOT / run
    run.mkdir(parents=True, exist_ok=True)
    with run_lock(run):
        collect(run, runtime, design, prompts, manifest)


if __name__ == "__main__":
    main()
