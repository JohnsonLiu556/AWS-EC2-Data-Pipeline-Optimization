import argparse
import yaml
import pandas as pd
import numpy as np
import cvxpy as cp

from pathlib import Path


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def solve_performance_aware(jobs, inst, use_effective_price=True):
    """Solve your main optimization and return (assignments_df, total_cost)."""

    inst = inst.copy().reset_index(drop=True)

    if use_effective_price:
        inst["effective_price"] = inst["avg_spot_price"].fillna(inst["on_demand_price"])
    else:
        inst["effective_price"] = inst["on_demand_price"]

    def get_family(itype: str) -> str:
        return itype.split(".")[0]

    families = inst["instance_type"].apply(get_family)
    perf_tier_map = {"c5": 3, "c5a": 3, "m5": 2, "r5": 2, "r5a": 2}
    inst["perf_tier"] = families.map(perf_tier_map).fillna(1)

    J, I = len(jobs), len(inst)

    cpu_hours = jobs["CPU_hours"].values
    job_mem   = jobs["memory_GB"].values
    job_perf  = jobs["performance_score"].values

    inst_mem  = inst["memory_gb"].values
    price     = inst["effective_price"].values
    inst_perf = inst["perf_tier"].values

    x = cp.Variable((J, I))

    constraints = [x >= 0, x <= 1, cp.sum(x, axis=1) == 1]

    for j in range(J):
        for i in range(I):
            if job_mem[j] > inst_mem[i]:
                constraints.append(x[j, i] == 0)
                continue
            if job_perf[j] == 3 and inst_perf[i] < 3:
                constraints.append(x[j, i] == 0)
                continue
            if job_perf[j] == 2 and inst_perf[i] < 2:
                constraints.append(x[j, i] == 0)
                continue

    cost_matrix = np.outer(cpu_hours, price)
    total_cost = cp.sum(cp.multiply(cost_matrix, x))
    prob = cp.Problem(cp.Minimize(total_cost), constraints)
    prob.solve(solver=cp.ECOS)

    x_val = x.value
    assignments = []
    for j in range(J):
        i_chosen = int(np.argmax(x_val[j, :]))
        assignments.append({
            "job_id": jobs.loc[j, "job_id"],
            "CPU_hours": cpu_hours[j],
            "memory_GB": job_mem[j],
            "performance_score": job_perf[j],
            "instance_type": inst.loc[i_chosen, "instance_type"],
            "instance_memory": inst.loc[i_chosen, "memory_gb"],
            "instance_perf_tier": inst.loc[i_chosen, "perf_tier"],
            "price_per_hour": inst.loc[i_chosen, "effective_price"],
        })

    result_df = pd.DataFrame(assignments)
    result_df["job_cost"] = result_df["CPU_hours"] * result_df["price_per_hour"]
    return result_df, float(total_cost.value)



def run_base(config):
    jobs = pd.read_csv(config["paths"]["jobs"])
    inst = pd.read_csv(config["paths"]["instance_prices"])

    assign, total_cost = solve_performance_aware(jobs, inst, use_effective_price=True)
    out_path = Path("optimal_job_assignments_from_runner.csv")
    assign.to_csv(out_path, index=False)

    print(f"[base] Total cost: {total_cost:.3f}")
    print(f"[base] Saved {out_path}")


def run_scenario(config):
    from scenario_analysis import solve_single_scenario  

    jobs = pd.read_csv(config["paths"]["jobs"])
    inst = pd.read_csv(config["paths"]["instance_prices"])
    inst["effective_price"] = inst["avg_spot_price"].fillna(inst["on_demand_price"])

    num = config["scenario"]["num_scenarios"]
    noise = config["scenario"]["noise_level"]

    np.random.seed(42)
    costs = []

    for s in range(num):
        inst_perturbed = inst.copy()
        mult = np.random.uniform(1 - noise, 1 + noise, size=len(inst_perturbed))
        inst_perturbed["effective_price"] = inst_perturbed["effective_price"] * mult
        total_cost = solve_single_scenario(jobs, inst_perturbed)
        costs.append(total_cost)
        print(f"[scenario] {s+1}/{num}: cost={total_cost:.3f}")

    df = pd.DataFrame({"scenario_id": range(num), "total_cost": costs})
    df.to_csv("scenario_costs_from_runner.csv", index=False)
    print("[scenario] Saved scenario_costs_from_runner.csv")


def run_risk(config, lam):
    from risk_tradeoff import solve_with_lambda

    jobs = pd.read_csv(config["paths"]["jobs"])
    inst = pd.read_csv(config["paths"]["instance_prices"])

    cost, risk = solve_with_lambda(jobs, inst, lam)
    print(f"[risk] lambda={lam}  cost={cost:.3f}  risk={risk:.3f}")


def run_baseline(config):
    from baseline_compare import get_family, pick_baseline_instance  
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Pipeline Cost Optimizer Runner")
    parser.add_argument(
        "--mode",
        choices=["base", "scenario", "risk"],
        required=True,
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.0,
        help="Risk aversion lambda for --mode risk.",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.mode == "base":
        run_base(cfg)
    elif args.mode == "scenario":
        run_scenario(cfg)
    elif args.mode == "risk":
        run_risk(cfg, args.lam)
