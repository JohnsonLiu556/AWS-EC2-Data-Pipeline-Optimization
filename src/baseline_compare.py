import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_family(itype: str) -> str:
    return itype.split(".")[0]


def pick_baseline_instance(job_mem, job_perf, inst):
    """
    Baseline policy:
    - Prefer m5 family (general purpose, on-demand pricing).
    - If no m5 instance fits the memory/perf, fall back to r5 family.
    - Use ON-DEMAND price only (ignore spot discounts).
    """

    inst = inst.copy()

    families = inst["instance_type"].apply(get_family)
    perf_tier_map = {
        "c5": 3,
        "c5a": 3,
        "m5": 2,
        "r5": 2,
        "r5a": 2,
    }
    inst["perf_tier"] = families.map(perf_tier_map).fillna(1)

    # Required perf tier by job_perf
    if job_perf == 3:
        required_tier = 3
    elif job_perf == 2:
        required_tier = 2
    else:
        required_tier = 1

    # 1) Try m5 family first
    m5_candidates = inst[
        (families == "m5")
        & (inst["memory_gb"] >= job_mem)
        & (inst["perf_tier"] >= required_tier)
    ]

    # 2) If no m5 fits, try r5
    r5_candidates = inst[
        (families.str.startswith("r5"))
        & (inst["memory_gb"] >= job_mem)
        & (inst["perf_tier"] >= required_tier)
    ]

    # Decide which candidate pool to use
    if not m5_candidates.empty:
        pool = m5_candidates
    elif not r5_candidates.empty:
        pool = r5_candidates
    else:
        # Fall back to any instance that fits memory & perf
        pool = inst[
            (inst["memory_gb"] >= job_mem)
            & (inst["perf_tier"] >= required_tier)
        ]

    # Among the pool, choose cheapest *on-demand* instance
    chosen = pool.loc[pool["on_demand_price"].idxmin()]
    return chosen


if __name__ == "__main__":
    jobs = pd.read_csv("jobs.csv")
    inst = pd.read_csv("merged_instance_prices.csv")
    opt = pd.read_csv("optimal_job_assignments.csv")

    optimized_total_cost = opt["job_cost"].sum()

    baseline_rows = []
    for _, row in jobs.iterrows():
        job_id = row["job_id"]
        cpu_hours = row["CPU_hours"]
        job_mem = row["memory_GB"]
        job_perf = row["performance_score"]

        chosen = pick_baseline_instance(job_mem, job_perf, inst)
        inst_type = chosen["instance_type"]
        price_on_demand = chosen["on_demand_price"]

        job_cost = cpu_hours * price_on_demand

        baseline_rows.append(
            {
                "job_id": job_id,
                "CPU_hours": cpu_hours,
                "memory_GB": job_mem,
                "performance_score": job_perf,
                "instance_type": inst_type,
                "on_demand_price": price_on_demand,
                "job_cost": job_cost,
            }
        )

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv("baseline_on_demand_assignments.csv", index=False)

    baseline_total_cost = baseline_df["job_cost"].sum()

    # Print comparison
    print("Baseline (on-demand) total cost:   ", baseline_total_cost)
    print("Optimized (spot-aware) total cost: ", optimized_total_cost)

    savings_abs = baseline_total_cost - optimized_total_cost
    savings_pct = savings_abs / baseline_total_cost * 100.0

    print(f"\nAbsolute savings: ${savings_abs:.2f}")
    print(f"Relative savings: {savings_pct:.2f}%")

    # Simple bar chart
    labels = ["Baseline (on-demand)", "Optimized (spot-aware)"]
    costs = [baseline_total_cost, optimized_total_cost]

    plt.figure()
    plt.bar(labels, costs)
    plt.ylabel("Total cost ($)")
    plt.title("Baseline vs Optimized Total Cost")
    for i, v in enumerate(costs):
        plt.text(i, v + 0.1, f"${v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()
