import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def solve_with_lambda(jobs_df, inst_df, lam):
    """
    Solve optimization with cost + lam * risk objective.
    Returns (total_cost, total_risk).
    """

    inst = inst_df.copy().reset_index(drop=True)
    inst["effective_price"] = inst["avg_spot_price"].fillna(inst["on_demand_price"])

 
    disc = (inst["on_demand_price"] - inst["effective_price"]) / inst["on_demand_price"]
    inst["risk_score"] = disc.clip(lower=0.0)

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
    inst_risk = inst["risk_score"].values

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
    risk_matrix = np.outer(cpu_hours, inst_risk)

    total_cost = cp.sum(cp.multiply(cost_matrix, x))
    total_risk = cp.sum(cp.multiply(risk_matrix, x))

    objective = cp.Minimize(total_cost + lam * total_risk)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver failed with status {prob.status}")

    x_val = x.value
    cost_value = float(total_cost.value)
    risk_value = float(total_risk.value)

    return cost_value, risk_value


if __name__ == "__main__":
    jobs = pd.read_csv("jobs.csv")
    inst = pd.read_csv("merged_instance_prices.csv")

    lambdas = [0.0, 0.2, 0.5, 1.0, 2.0]

    results = []
    for lam in lambdas:
        cost, risk = solve_with_lambda(jobs, inst, lam)
        results.append({"lambda": lam, "cost": cost, "risk": risk})
        print(f"lambda={lam:.2f}  cost={cost:.3f}  risk={risk:.3f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv("cost_risk_tradeoff.csv", index=False)
    print("\nSaved cost_risk_tradeoff.csv")
    print(res_df)

    plt.figure(figsize=(10, 6))
    plt.plot(res_df["risk"], res_df["cost"], marker="o")

    for idx, row in res_df.iterrows():
        plt.annotate(
            f"λ={row['lambda']}",
            xy=(row["risk"], row["cost"]),          
            xytext=(10, 10 * (idx - 2)),            
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.7, color="black"),
        )

    plt.xlabel("Total interruption risk (proxy units)", fontsize=12)
    plt.ylabel("Total cost ($)", fontsize=12)
    plt.title("Cost–Risk Tradeoff (varying λ)", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


