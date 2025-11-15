import pandas as pd
import numpy as np
import cvxpy as cp

jobs = pd.read_csv("jobs.csv")
inst = pd.read_csv("merged_instance_prices.csv")

inst["effective_price"] = inst["avg_spot_price"].fillna(inst["on_demand_price"])

inst = inst.dropna(subset=["effective_price"]).reset_index(drop=True)

def get_family(itype: str) -> str:
    # "c5.large" -> "c5"
    return itype.split(".")[0]

families = inst["instance_type"].apply(get_family)

perf_tier_map = {
    "c5": 3,
    "c5a": 3,
    "m5": 2,
    "r5": 2,
    "r5a": 2,
}

inst["perf_tier"] = families.map(perf_tier_map).fillna(1)  # default to 1 if unknown


capacity_by_family = {
    "c5": 5,
    "c5a": 5,
    "m5": 10,
    "r5": 8,
    "r5a": 8,
}

inst["capacity_jobs"] = families.map(capacity_by_family).fillna(100)

J = len(jobs)   
I = len(inst)   

print(f"{J} jobs, {I} instance types")

cpu_hours = jobs["CPU_hours"].values            # shape (J,)
job_mem    = jobs["memory_GB"].values           # shape (J,)
job_perf   = jobs["performance_score"].values   # shape (J,)

inst_mem   = inst["memory_gb"].values           # shape (I,)
price      = inst["effective_price"].values     # shape (I,)
inst_perf  = inst["perf_tier"].values           # shape (I,)
inst_cap   = inst["capacity_jobs"].values       # shape (I,)

x = cp.Variable((J, I))

constraints = []

constraints.append(x >= 0)
constraints.append(x <= 1)

constraints.append(cp.sum(x, axis=1) == 1)

for j in range(J):
    for i in range(I):
        # Memory constraint
        if job_mem[j] > inst_mem[i]:
            constraints.append(x[j, i] == 0)
            continue

        # Performance constraints:
        if job_perf[j] == 3 and inst_perf[i] < 3:
            constraints.append(x[j, i] == 0)
            continue

        if job_perf[j] == 2 and inst_perf[i] < 2:
            constraints.append(x[j, i] == 0)
            continue
        # score 1: no extra restriction

for i in range(I):
    constraints.append(cp.sum(x[:, i]) <= inst_cap[i])

cost_matrix = np.outer(cpu_hours, price)  # shape (J, I)
objective = cp.Minimize(cp.sum(cp.multiply(cost_matrix, x)))

problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.ECOS)

print("Status:", problem.status)
print("Optimal cost:", problem.value)

x_val = x.value  # (J, I) array

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
        "instance_capacity_jobs": inst.loc[i_chosen, "capacity_jobs"],
        "price_per_hour": inst.loc[i_chosen, "effective_price"],
    })

result_df = pd.DataFrame(assignments)
result_df["job_cost"] = result_df["CPU_hours"] * result_df["price_per_hour"]

print("\nOptimal assignment (first few jobs):")
print(result_df.head())

result_df.to_csv("optimal_job_assignments_with_capacity.csv", index=False)
print("\nSaved optimal_job_assignments_with_capacity.csv")
