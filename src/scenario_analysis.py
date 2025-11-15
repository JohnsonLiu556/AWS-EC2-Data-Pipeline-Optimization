import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def solve_single_scenario(jobs_df, inst_df):
    """
    Run the same optimization as optimize_pipeline.py
    but using the given instance price table (inst_df).
    Returns the optimal total cost.
    """

    inst = inst_df.copy().reset_index(drop=True)

    def get_family(itype: str) -> str:
        return itype.split(".")[0]

    families = inst["instance_type"].apply(get_family)

    perf_tier_map = {
        "c5": 3,
        "c5a": 3,
        "m5": 2,
        "r5": 2,
        "r5a": 2,
    }

    inst["perf_tier"] = families.map(perf_tier_map).fillna(1)

    J = len(jobs_df)
    I = len(inst)

    cpu_hours = jobs_df["CPU_hours"].values
    job_mem   = jobs_df["memory_GB"].values
    job_perf  = jobs_df["performance_score"].values

    inst_mem  = inst["memory_gb"].values
    price     = inst["effective_price"].values
    inst_perf = inst["perf_tier"].values

    x = cp.Variable((J, I))

    constraints = []
    constraints.append(x >= 0)
    constraints.append(x <= 1)
    constraints.append(cp.sum(x, axis=1) == 1)  

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
    objective = cp.Minimize(cp.sum(cp.multiply(cost_matrix, x)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver failed with status {problem.status}")

    return problem.value



if __name__ == "__main__":
    np.random.seed(42)

    jobs = pd.read_csv("jobs.csv")
    inst_base = pd.read_csv("merged_instance_prices.csv")

    inst_base["effective_price"] = inst_base["avg_spot_price"].fillna(
        inst_base["on_demand_price"]
    )

    NUM_SCENARIOS = 50      # try 50 different spot-price worlds
    NOISE_LEVEL   = 0.30    # ±30% variation

    scenario_costs = []

    for s in range(NUM_SCENARIOS):
        inst_perturbed = inst_base.copy()

        multipliers = np.random.uniform(
            1 - NOISE_LEVEL, 1 + NOISE_LEVEL, size=len(inst_perturbed)
        )
        inst_perturbed["effective_price"] = (
            inst_perturbed["effective_price"] * multipliers
        )

        total_cost = solve_single_scenario(jobs, inst_perturbed)
        scenario_costs.append(total_cost)
        print(f"Scenario {s+1}/{NUM_SCENARIOS}: total cost = {total_cost:.3f}")

    costs_df = pd.DataFrame({"scenario_id": range(NUM_SCENARIOS),
                             "total_cost": scenario_costs})
    costs_df.to_csv("scenario_costs.csv", index=False)
    print("\nSaved scenario_costs.csv")
    print(costs_df.describe())

    plt.figure()
    plt.hist(scenario_costs, bins=10, edgecolor="black")
    plt.xlabel("Total cost ($)")
    plt.ylabel("Frequency")
    plt.title(f"Total Cost Distribution over {NUM_SCENARIOS} Spot-Price Scenarios")
    plt.tight_layout()
    plt.show()
