import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import plotly.express as px



def get_family(itype: str) -> str:
    return itype.split(".")[0]


def solve_pipeline(jobs, inst, lam=0.0, noise_level=0.0, enforce_capacity=True):
    """
    Solve performance-aware, risk-aware optimization with optional capacity and noise.
    Returns (assignments_df, total_cost, total_risk).
    """

    inst = inst.copy().reset_index(drop=True)

    inst["effective_price"] = inst["avg_spot_price"].fillna(inst["on_demand_price"])

    if noise_level > 0:
        rng = np.random.default_rng(0)  
        multipliers = rng.uniform(1 - noise_level, 1 + noise_level, size=len(inst))
        inst["effective_price"] *= multipliers

    families = inst["instance_type"].apply(get_family)
    perf_tier_map = {"c5": 3, "c5a": 3, "m5": 2, "r5": 2, "r5a": 2}
    inst["perf_tier"] = families.map(perf_tier_map).fillna(1)

    capacity_by_family = {
        "c5": 5,
        "c5a": 5,
        "m5": 10,
        "r5": 8,
        "r5a": 8,
    }
    inst["capacity_jobs"] = families.map(capacity_by_family).fillna(100)

    # Risk score based on discount
    disc = (inst["on_demand_price"] - inst["effective_price"]) / inst["on_demand_price"]
    inst["risk_score"] = disc.clip(lower=0.0)

    J, I = len(jobs), len(inst)

    cpu_hours = jobs["CPU_hours"].values
    job_mem   = jobs["memory_GB"].values
    job_perf  = jobs["performance_score"].values

    inst_mem  = inst["memory_gb"].values
    price     = inst["effective_price"].values
    inst_perf = inst["perf_tier"].values
    inst_cap  = inst["capacity_jobs"].values
    inst_risk = inst["risk_score"].values

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

    if enforce_capacity:
        for i in range(I):
            constraints.append(cp.sum(x[:, i]) <= inst_cap[i])

    cost_matrix = np.outer(cpu_hours, price)
    risk_matrix = np.outer(cpu_hours, inst_risk)

    total_cost = cp.sum(cp.multiply(cost_matrix, x))
    total_risk = cp.sum(cp.multiply(risk_matrix, x))

    objective = cp.Minimize(total_cost + lam * total_risk)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

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
            "instance_capacity_jobs": inst.loc[i_chosen, "capacity_jobs"],
            "price_per_hour": inst.loc[i_chosen, "effective_price"],
            "instance_risk": inst.loc[i_chosen, "risk_score"],
        })

    result_df = pd.DataFrame(assignments)
    result_df["job_cost"] = result_df["CPU_hours"] * result_df["price_per_hour"]

    total_cost_val = float(total_cost.value)
    total_risk_val = float(total_risk.value)

    return result_df, total_cost_val, total_risk_val



st.set_page_config(page_title="Data Pipeline Cost Optimizer", layout="wide")

st.title("Data Pipeline Cost Optimizer (AWS EC2)")

st.markdown(
    """
This dashboard assigns pipeline jobs to EC2 instance types to minimize **cost** 
while respecting **memory**, **performance tiers**, and optional **capacity constraints**. 
You can also tune **risk aversion** and **spot price volatility**.
"""
)

jobs = pd.read_csv("jobs.csv")
inst = pd.read_csv("merged_instance_prices.csv")

st.sidebar.header("Controls")

lam = st.sidebar.slider("Risk aversion λ", 0.0, 3.0, 0.5, 0.1)
noise_level = st.sidebar.slider("Spot price noise (±%)", 0.0, 0.5, 0.0, 0.05)
enforce_capacity = st.sidebar.checkbox("Enforce capacity constraints", value=True)

if st.sidebar.button("Run optimization"):
    with st.spinner("Solving optimization problem..."):
        result_df, total_cost, total_risk = solve_pipeline(
            jobs, inst, lam=lam, noise_level=noise_level, enforce_capacity=enforce_capacity
        )

    st.success("Optimization complete.")

    col1, col2 = st.columns(2)
    col1.metric("Total cost ($)", f"{total_cost:.3f}")
    col2.metric("Total risk (proxy units)", f"{total_risk:.3f}")

    st.subheader("Optimal job assignments")
    st.dataframe(result_df)

    cost_by_inst = (
        result_df.groupby("instance_type")["job_cost"]
        .sum()
        .reset_index()
        .sort_values("job_cost", ascending=False)
    )

    fig_cost = px.bar(
        cost_by_inst,
        x="instance_type",
        y="job_cost",
        title="Total cost per instance type",
        labels={"job_cost": "Total cost ($)", "instance_type": "Instance type"},
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    fig_scatter = px.scatter(
        result_df,
        x="performance_score",
        y="job_cost",
        color="instance_type",
        title="Job cost vs performance score",
        labels={"performance_score": "Performance score", "job_cost": "Job cost ($)"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("Set parameters in the sidebar and click **Run optimization** to see results.")
