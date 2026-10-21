"""Collect the approved, bounded micro support and synchronous Full-LLM runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import traceback
import networkx as nx
from llm_client import OllamaClient, git_output
from random_streams import seed
from run_era_formal import identity, payload_for, formal_token_score, atomic_json
from response_readout import classify_response

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sprint_v4_20261019.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    config = load(CONFIG)
    auth = load(ROOT / args.authorization)
    if auth.get("approved") is not True or auth.get("config_sha256") != sha(CONFIG):
        raise ValueError("Explicit approval of this exact frozen budget is required.")
    run = ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    (run / "calls").mkdir()
    (run / "sources").mkdir()
    runtime = load(ROOT / config["runtime"])
    policy = load(ROOT / runtime["readout_policy"])
    client = OllamaClient(runtime["base_url"], timeout=90)
    prompts = {p["prompt_id"]: p for p in config["prompts"]}
    sources = ["configs/sprint_v4_20261019.json", config["runtime"], runtime["readout_policy"],
               "src/run_sprint_v4.py", "src/random_streams.py", "src/run_era_formal.py",
               "src/response_readout.py", "src/llm_client.py", args.authorization]
    for name in sources:
        target = run / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    started = time.perf_counter()
    counts = {"attempted": 0, "completed": 0, "support": 0, "full_llm": 0, "invalid": 0,
              "input_tokens": 0, "output_tokens": 0, "request_wall_seconds": 0.0}
    meta = {"start_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha(CONFIG),
            "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
            "python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": {s: sha(ROOT/s) for s in sources}, "gpu_hours": None, "api_cost": None,
            "authorization": auth}
    atomic_json(run / "run.json", meta)

    def status(state, **extra):
        atomic_json(run / "status.json", {"status": state, **counts, "elapsed_seconds": time.perf_counter()-started, **extra})

    def call(row, prompt, kind, context=None):
        # Never replace an old response or retry an invalid answer.
        if counts["attempted"] >= config["budget"]["total_max_calls"]:
            raise RuntimeError("Frozen call cap reached.")
        if time.perf_counter()-started > config["budget"]["hard_wall_seconds"]:
            raise TimeoutError("Frozen 45-minute collection limit reached.")
        payload = payload_for(row, prompt, runtime["models"]["qwen3_8b"], config["generation"])
        index = counts["attempted"]
        counts["attempted"] += 1
        folder = run / "calls" / f"{index:04d}"
        folder.mkdir()
        atomic_json(folder / "request.json", {"row": row, "prompt": prompt, "kind": kind, "context": context, "payload": payload})
        before = time.perf_counter()
        try:
            response = client.request("/api/chat", payload)
        except Exception as error:
            atomic_json(folder / "failure.json", {"error": repr(error), "wall_seconds": time.perf_counter()-before})
            raise
        wall = time.perf_counter()-before
        # Save the untouched response before interpretation.
        atomic_json(folder / "raw.json", {"response": response, "wall_seconds": wall})
        readout = classify_response(response, policy)
        token = formal_token_score(response, "qwen3_8b", policy) if row["mode"] == "logprob" else None
        result = {"row": row, "prompt_id": prompt["prompt_id"], "persona_id": prompt["persona_id"],
                  "representation": prompt["representation"], "k": prompt["k"], "kind": kind,
                  "readout": readout, "token_score": token, "wall_seconds": wall, "context": context}
        atomic_json(folder / "result.json", result)
        counts["completed"] += 1
        counts[kind] += 1
        counts["invalid"] += int(readout["semantic_class"] == "invalid")
        counts["input_tokens"] += response.get("prompt_eval_count", 0)
        counts["output_tokens"] += response.get("eval_count", 0)
        counts["request_wall_seconds"] += wall
        status("RUNNING")
        if counts["completed"] % 20 == 0:
            print(f"Completed {counts['completed']}/1800 maximum | support {counts['support']}/432 | Full-LLM {counts['full_llm']}", flush=True)
        return result

    try:
        atomic_json(run / "identity.json", identity(client, runtime, "qwen3_8b"))
        for row in config["support_requests"]:
            call(row, prompts[row["prompt_id"]], "support")
        atomic_json(run / "support_complete.json", {"completed": counts["support"], "expected": 432})
        trajectories = []
        for rep in config["representations"]:
            for spec in config["networks"]:
                graph = nx.Graph()
                graph.add_nodes_from(range(64))
                graph.add_edges_from(spec["edges"])
                adopted = set(spec["initial_adopters"])
                history = [{"round": 0, "adopted": sorted(adopted), "count": len(adopted), "calls": 0}]
                for round_id in range(1, 4):
                    newly = set()
                    decisions = []
                    # All k values use the previous round's state, never within-round additions.
                    for node in sorted(set(graph)-adopted):
                        k = len(set(graph.neighbors(node)) & adopted)
                        if k == 0:
                            continue
                        key = f"full:{spec['trial']}:{spec['network']}:{rep}:{round_id}:{node}"
                        prompt_id = f"{spec['persona_by_node'][node]}_{rep}_{k}"
                        row = {"request_id": key, "prompt_id": prompt_id, "seed": seed(key), "mode": "sampling"}
                        context = {"trial": spec["trial"], "network": spec["network"], "round": round_id, "node": node, "k": k}
                        result = call(row, prompts[prompt_id], "full_llm", context)
                        if result["readout"]["semantic_class"] == "Yes":
                            newly.add(node)
                        decisions.append({**context, "request_id": key, "class": result["readout"]["semantic_class"]})
                    adopted |= newly
                    history.append({"round": round_id, "adopted": sorted(adopted), "count": len(adopted), "calls": len(decisions), "decisions": decisions})
                    atomic_json(run / f"full_{rep}_{spec['trial']}_{spec['network']}.json", {"representation": rep, "trial": spec["trial"], "network": spec["network"], "history": history, "complete": round_id == 3})
                trajectories.append({"representation": rep, "trial": spec["trial"], "network": spec["network"], "history": history})
        atomic_json(run / "full_trajectories.json", trajectories)
        status("COMPLETED")
        print(json.dumps({"status": "COMPLETED", "run": str(run), **counts}, indent=2), flush=True)
    except Exception as error:
        status("STOPPED_PARTIAL", error=repr(error), traceback=traceback.format_exc())
        print(f"STOPPED_PARTIAL: {error}; all received raw responses retained in {run}", flush=True)
        raise


if __name__ == "__main__":
    main()
