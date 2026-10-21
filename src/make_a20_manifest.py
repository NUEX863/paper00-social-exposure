"""Create the v2 trial seed manifest without running simulations."""

import json
from hashlib import sha256
from pathlib import Path


def derive_seed(master_seed, trial_id, purpose):
    """Derive a stable integer seed for one trial and one purpose."""
    label = f"{master_seed}|{trial_id}|{purpose}"
    digest = sha256(label.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def main():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "a20_batch_v2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Reject a configuration that does not match the connected-network design.
    if (
        config["experiment_id"] != "a20_batch_v2"
        or config["n_trials"] != 50
        or config["rewiring"]["connectivity_policy"]
        != "require_connected_with_logged_rejection"
        or config["initial_adopters"]["pair_center_across_networks"] is not True
        or config["ic"]["p"] != 0.2
    ):
        raise ValueError("Configuration does not match the v2 batch design.")

    purposes = ("graph", "center", "ic_lattice", "ic_random")
    if config["random_streams"]["purposes"] != list(purposes):
        raise ValueError("Unexpected random-stream purposes.")

    # Assign four base seeds to each trial.
    trials = []
    for trial_id in range(config["n_trials"]):
        trial = {"trial_id": trial_id}
        for purpose in purposes:
            trial[purpose] = derive_seed(
                config["master_seed"], trial_id, purpose
            )
        trials.append(trial)

    # Check the base seeds before saving the manifest.
    seed_values = [
        trial[purpose]
        for trial in trials
        for purpose in purposes
    ]
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("Duplicate base seeds detected.")

    manifest = {
        "kind": "a20_trial_manifest",
        "config": config,
        "seed_label_format": "{master_seed}|{trial_id}|{purpose}",
        "trials": trials,
    }
    payload = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")

    # Reuse identical content and refuse to overwrite a changed manifest.
    output_path = project_root / "configs" / "a20_trial_manifest_v2.json"
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise ValueError(
                "Existing manifest differs; review changes before versioning."
            )
        print("Reused trial manifest:", output_path)
    else:
        with output_path.open("xb") as file:
            file.write(payload)
        print("Saved trial manifest:", output_path)

    print("Trial count:", len(trials))
    print("Unique base seed count:", len(set(seed_values)))
    print("First trial:", trials[0])
    print("Last trial ID:", trials[-1]["trial_id"])
    print("Manifest checks: PASS")


if __name__ == "__main__":
    main()
