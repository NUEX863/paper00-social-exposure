"""Offline micro-to-macro and RCA-inspired analysis of the frozen V4 extension."""

import argparse

from datetime import datetime, timezone

import hashlib

import json

from pathlib import Path

import shutil

import numpy as np

import pandas as pd

from scipy.special import expit, logit

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from random_streams import seed



ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs/sprint_v4_20261019.json"





def read(path):

    return json.loads(path.read_text(encoding="utf-8"))





def clean(value):

    """JSON null denotes unavailable statistics; never invent a numeric zero."""

    if isinstance(value, dict):

        return {str(k): clean(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):

        return [clean(v) for v in value]

    if isinstance(value, np.ndarray):

        return clean(value.tolist())

    if isinstance(value, (np.integer, np.floating)):

        value = value.item()

    if isinstance(value, float) and not np.isfinite(value):

        return None

    return value





def save(path, value):

    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")





def micro_metrics(curve):

    p0, p1, p2 = curve[:3]

    beta = (p1-p0)/(1-p0) if p0 < 1-1e-6 else np.nan

    admissible = bool(np.isfinite(beta) and 0 <= beta <= 1)

    return {"P0": p0, "P1": p1, "P2": p2, "S2_excess_IC": p2-(1-(1-p0)*(1-beta)**2) if admissible else np.nan,

            "S2_status": "estimable_descriptive" if admissible else "unavailable_ceiling_or_inadmissible_beta",

            "increment1": p1-p0, "increment2": p2-p1, "second_difference": p2-2*p1+p0,

            "min_P": float(min(curve)), "max_P": float(max(curve)), "doses_at_least_099": int(sum(curve >= .99))}





def simulate(probabilities, config, stage, output_rows):

    """Vectorized Bernoulli simulations share random streams across policies and paired graphs."""

    results = {}

    ids = config["persona_ids"]

    repeats = config["projection"]["mc_repetitions"]

    for policy, matrix in probabilities.items():

        if not np.isfinite(matrix).all():

            continue

        for spec in config["networks"]:

            trial, network = spec["trial"], spec["network"]

            adjacency = np.zeros((64,64), dtype=np.int16)

            for a, b in spec["edges"]:

                adjacency[a,b] = adjacency[b,a] = 1

            persona_index = np.array([ids.index(p) for p in spec["persona_by_node"]])

            uniforms = np.random.default_rng(seed(f"projection:{trial}")).random((repeats,3,64))

            adopted = np.zeros((repeats,64), dtype=bool)

            adopted[:,spec["initial_adopters"]] = True

            trajectory = [adopted.mean(axis=1)]

            for round_id in range(3):

                exposure = adopted.astype(np.int16) @ adjacency

                chance = matrix[persona_index[None,:], exposure]

                # Eligibility gate exactly matches the Full-LLM state update.

                newly = (~adopted) & (exposure >= 1) & (uniforms[:,round_id,:] < chance)

                adopted |= newly

                trajectory.append(adopted.mean(axis=1))

            trajectory = np.stack(trajectory, axis=1)

            results[(policy,trial,network)] = trajectory

            for mc in range(repeats):

                for round_id in range(4):

                    output_rows.append({"stage":stage,"policy":policy,"trial":trial,"network":network,"mc":mc,"round":round_id,"fraction":trajectory[mc,round_id]})

    return results





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--run-dir", default="runs/sprint_v4/20261019_authorized")

    parser.add_argument("--analysis-dir", help="New output directory; never overwrite released results")

    args = parser.parse_args()

    run = ROOT / args.run_dir

    config = read(CONFIG)

    analysis = ROOT / args.analysis_dir if args.analysis_dir else run / "analysis"

    analysis.mkdir(parents=True, exist_ok=False)

    shutil.copy2(__file__, analysis / "analysis_source.py")

    core = pd.read_csv(ROOT / config["core_observations"])

    assert hashlib.sha256((ROOT/config["core_observations"]).read_bytes()).hexdigest() == config["core_observations_sha256"]

    ids = config["persona_ids"]

    subset = core[(core.model_slot == "qwen3_8b") & core.persona_id.isin(ids) & (core.wording == "w1")]

    calls = [read(p) for p in sorted((run/"calls").glob("*/result.json"))]

    supported = [r for r in calls if r["kind"] == "support"]

    matrices = {mode: {rep: np.full((12,7), np.nan) for rep in ["R1","R2"]} for mode in ["sampling","token"]}

    records, sources = [], []

    appeal = np.full(12, np.nan)

    for i, persona in enumerate(ids):

        base = subset[(subset.persona_id == persona) & (subset.condition == "NO_INFO") & (subset["mode"] == "logprob")]

        assert len(base) == 1

        appeal[i] = base.iloc[0].token_score

        sources.append({"role":"no_information_appeal", "persona":persona,"request_id":base.iloc[0].request_id,"result_path":base.iloc[0].result_path})

        for rep in ["R1","R2"]:

            for k in range(7):

                if k <= 3:

                    rows = subset[(subset.persona_id == persona) & (subset.condition == f"{rep}_{k}")]

                    samples = rows[rows["mode"] == "sampling"]

                    score_rows = rows[rows["mode"] == "logprob"]

                    classes = samples.semantic_class.tolist()

                    score = score_rows.iloc[0].token_score if len(score_rows) == 1 else np.nan

                    for r in rows.itertuples():

                        sources.append({"role":"reused_core", "persona":persona,"request_id":r.request_id,"result_path":r.result_path})

                else:

                    rows = [r for r in supported if r["persona_id"] == persona and r["representation"] == rep and r["k"] == k]

                    classes = [r["readout"]["semantic_class"] for r in rows if r["row"]["mode"] == "sampling"]

                    scores = [r["token_score"]["native_label_yes_score"] for r in rows if r["row"]["mode"] == "logprob"]

                    score = scores[0] if len(scores) == 1 and scores[0] is not None else np.nan

                p = classes.count("Yes")/5 if len(classes) == 5 else np.nan

                matrices["sampling"][rep][i,k] = p

                matrices["token"][rep][i,k] = score

                records.append({"persona_id":persona,"representation":rep,"k":k,"sampling_n":len(classes),"yes":classes.count("Yes"),"invalid":classes.count("invalid"),"sampling_probability":p,"token_score":score})

    assert np.isfinite(appeal).all()

    settings = config["rca"]

    clipped = np.clip(appeal, settings["epsilon"], 1-settings["epsilon"])

    dose = np.arange(7)

    rca = expit(logit(clipped[:,None])+settings["delta1"]*(dose>=1)+settings["delta2"]*(dose>=2))

    pd.DataFrame(records).to_csv(analysis/"micro_response_curves.csv",index=False)

    pd.DataFrame(sources).to_csv(analysis/"reused_record_index.csv",index=False)

    pd.DataFrame({"persona_id":ids,"a_no_information":appeal,"a_numerically_clipped":clipped,**{f"P{k}":rca[:,k] for k in range(7)}}).to_csv(analysis/"rca_persona_curves.csv",index=False)

    metrics = []

    for mode in matrices:

        for rep, matrix in matrices[mode].items():

            if np.isfinite(matrix).all():

                metrics.append({"readout":mode,"policy":rep,**micro_metrics(matrix.mean(axis=0))})

    metrics.append({"readout":"token","policy":"RCA",**micro_metrics(rca.mean(axis=0))})

    pd.DataFrame(metrics).to_csv(analysis/"micro_metrics.csv",index=False)

    all_rows = []

    projection = simulate(matrices["sampling"], config, "M1_sampling_projection", all_rows)

    score_sim = simulate({**matrices["token"],"RCA":rca},config,"M2_score_policy",all_rows)

    pd.DataFrame(all_rows).to_csv(analysis/"macro_trajectories.csv",index=False)

    macro, differences, intervention = [], [], []

    for stage, results in [("M1_sampling_projection",projection),("M2_score_policy",score_sim)]:

        policies = sorted({key[0] for key in results})

        for policy in policies:

            # Pair both graph seeds at each Monte Carlo index; uncertainty is conditional on these graphs.

            by_network = {n: np.mean([results[(policy,t,n)] for t in range(2)],axis=0) for n in ["lattice","random"]}

            for n, curves in by_network.items():

                final = curves[:,-1]

                speed = curves[:,1:].mean(axis=1)

                macro.append({"stage":stage,"policy":policy,"network":n,"final_mean":final.mean(),"final_sim_low":np.quantile(final,.025),"final_sim_high":np.quantile(final,.975),

                              "speed_mean_rounds_1_to_3":speed.mean(),"between_graph_trial_final_mean_range":float(np.ptp([results[(policy,t,n)][:,-1].mean() for t in range(2)]))})

            delta = by_network["lattice"][:,-1]-by_network["random"][:,-1]

            differences.append({"stage":stage,"policy":policy,"delta_net":delta.mean(),"sim_low":np.quantile(delta,.025),"sim_high":np.quantile(delta,.975)})

        if stage == "M2_score_policy":

            for default in ["R1","R2"]:

                if default not in policies:

                    continue

                for network in ["lattice","random"]:

                    change = np.mean([results[("RCA",t,network)]-results[(default,t,network)] for t in range(2)],axis=0)

                    intervention.append({"default":default,"network":network,"final_change":change[:,-1].mean(),"change_sim_low":np.quantile(change[:,-1],.025),"change_sim_high":np.quantile(change[:,-1],.975),"speed_change":change[:,1:].mean()})

    errors, full_summary = [], []

    full = read(run/"full_trajectories.json") if (run/"full_trajectories.json").exists() else []

    for result in full:

        rep, trial, network = result["representation"],result["trial"],result["network"]

        observed = np.array([h["count"]/64 for h in result["history"]])

        full_summary.append({"representation":rep,"trial":trial,"network":network,"counts":[h["count"] for h in result["history"]],"calls":sum(h["calls"] for h in result["history"]),"speed_mean_rounds_1_to_3":observed[1:].mean()})

        if (rep,trial,network) in projection:

            predicted = projection[(rep,trial,network)].mean(axis=0)

            lo, hi = np.quantile(projection[(rep,trial,network)],[.025,.975],axis=0)

            errors.append({"representation":rep,"trial":trial,"network":network,"full_final":observed[-1],"predicted_final":predicted[-1],"final_signed_error":observed[-1]-predicted[-1],

                           "final_absolute_error":abs(observed[-1]-predicted[-1]),"trajectory_mae_rounds_1_to_3":abs(observed[1:]-predicted[1:]).mean(),"full_final_within_conditional_sim_range":bool(lo[-1]<=observed[-1]<=hi[-1])})

    for name, data in [("macro_summary",macro),("network_differences",differences),("intervention_changes",intervention),("prediction_errors",errors)]:

        pd.DataFrame(data).to_csv(analysis/f"{name}.csv",index=False)

    # Existing amounts are a retrospective sensitivity check, not a fresh holdout.

    amount_path = ROOT/"runs/era_readout_alignment/20261019T080301Z_ec8ffdefaa6c/aligned_token_scores.csv"

    amounts = pd.read_csv(amount_path)

    amounts = amounts[(amounts.model_slot=="qwen3_8b") & amounts.persona_id.isin(ids) & (amounts.condition=="NO_INFO")]

    content=[]

    for amount, frame in amounts.groupby("data_mb"):

        a = frame.token_score.to_numpy(float)

        assert len(a)==12

        c = expit(logit(np.clip(a,settings["epsilon"],1-settings["epsilon"]))[:,None]+settings["delta1"]*(dose>=1)+settings["delta2"]*(dose>=2))

        content.append({"data_mb":int(amount),"n_personas":12,"appeal_mean":a.mean(),"appeal_min":a.min(),"appeal_max":a.max(),"rca_P1":c[:,1].mean(),"rca_P2":c[:,2].mean(),"status":"retrospective_token_score_sensitivity_only"})

    pd.DataFrame(content).to_csv(analysis/"content_sensitivity.csv",index=False)

    summary={"created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":hashlib.sha256(CONFIG.read_bytes()).hexdigest(),"run_status":read(run/"status.json"),

             "coverage_complete":{mode:{rep:bool(np.isfinite(a).all()) for rep,a in mats.items()} for mode,mats in matrices.items()},

             "full_llm_runs_complete":len(full),"full_llm":full_summary,"prediction_errors":errors,"macro":macro,"delta_net":differences,"intervention_changes":intervention,

             "micro_metrics":metrics,"appeal":{"mean":appeal.mean(),"min":appeal.min(),"max":appeal.max(),"sd":appeal.std(ddof=1),"clipped_count":int(np.sum(appeal!=clipped)),"max_clip_change":float(np.max(abs(appeal-clipped)))},

             "content_sensitivity":content,"uncertainty":"All supplementary macro ranges are conditional Monte Carlo ranges for two selected graph pairs and fixed estimated response tables; not CIs for prediction accuracy or humans.",

             "limitations":["Exploratory subset and R2/w1 selection after ERA.","Five draws per response cell; probability estimation error omitted from primary simulation bands.","Two network pairs, one genuine trajectory per condition; no confirmatory macro significance test.","Memoryless canonical status slots; no longitudinal friend identity.","RCA uses T=1 score policies, not T=.7 sampling probabilities.","RCA replaces the whole social response; changes cannot be assigned to delta2 alone.","No human calibration; no parameter search."]}

    save(analysis/"summary.json",summary)

    figs=ROOT/"outputs/figures"

    figs.mkdir(parents=True,exist_ok=True)

    plt.rcParams.update({"font.family":"Times New Roman","font.size":8,"axes.titlesize":8,"axes.labelsize":8,"legend.fontsize":7})

    colors={"lattice":"#1764ab","random":"#ba4c00"}

    if len(errors)==8:

        fig,axs=plt.subplots(2,1,figsize=(3.48,3.3),sharex=True,sharey=True)

        for ax,rep in zip(axs,["R1","R2"]):

            for n,color in colors.items():

                projected=np.mean([projection[(rep,t,n)] for t in range(2)],axis=0)

                observed=np.mean([[h["count"]/64 for h in r["history"]] for r in full if r["representation"]==rep and r["network"]==n],axis=0)

                lo,hi=np.quantile(projected,[.025,.975],axis=0)

                ax.plot(range(4),projected.mean(axis=0),color=color,linestyle="--",label=n+" projection",linewidth=1)

                ax.fill_between(range(4),lo,hi,color=color,alpha=.15,linewidth=0)

                ax.plot(range(4),observed,color=color,marker="o",markersize=3,label=n+" Full-LLM",linewidth=1)

            ax.set_title(rep+" / w1",loc="left",pad=3)

            ax.set_ylabel("Adopted fraction")

            ax.set_ylim(0,1.04)

            ax.grid(alpha=.18)

        axs[0].legend(loc="upper left",fontsize=6,ncol=1,frameon=False)

        axs[1].set_xlabel("Synchronous round (0 = initial)")

        axs[1].set_xticks(range(4))

        fig.tight_layout(pad=.6)

        fig.savefig(figs/"figure2_macro.png",dpi=350)

        plt.close(fig)

    fig,axs=plt.subplots(2,1,figsize=(3.48,3.3))

    for rep,color in [("R1","#1764ab"),("R2","#ba4c00")]:

        if np.isfinite(matrices["token"][rep]).all():

            axs[0].plot(dose,matrices["token"][rep].mean(axis=0),label="Default "+rep,color=color,marker="o",markersize=2.5,linewidth=1)

    axs[0].plot(dose,rca.mean(axis=0),label="RCA-inspired",color="#238443",linestyle="--",linewidth=1.2)

    axs[0].set_ylabel("Conditional Yes score")

    axs[0].set_xlabel("Adopting friends k")

    axs[0].set_ylim(0,1.04)

    axs[0].legend(loc="lower right",frameon=False,fontsize=6)

    axs[0].set_xticks(range(7))

    policies=[p for p in ["R1","R2","RCA"] if (p,0,"lattice") in score_sim]

    x=np.arange(len(policies))

    for offset,(n,color) in zip([-.18,.18],colors.items()):

        vals=[np.mean([score_sim[(p,t,n)][:,-1].mean() for t in range(2)]) for p in policies]

        axs[1].bar(x+offset,vals,width=.35,color=color,label=n)

    axs[1].set_xticks(x,policies)

    axs[1].set_ylim(0,1.04)

    axs[1].set_ylabel("Round-3 adopted fraction")

    axs[1].set_xlabel("Offline score policy")

    axs[1].legend(frameon=False,fontsize=7,loc="lower right")

    for ax in axs: ax.grid(axis="y",alpha=.18)

    fig.tight_layout(pad=.6)

    fig.savefig(figs/"figure3_rca.png",dpi=350)

    plt.close(fig)

    print(json.dumps(clean({"analysis":str(analysis),"coverage":summary["coverage_complete"],"full_llm":full_summary,"errors":errors,"macro":macro,"delta_net":differences,"intervention_changes":intervention,"appeal":summary["appeal"],"micro_metrics":metrics}),indent=2))





if __name__=="__main__":

    main()

