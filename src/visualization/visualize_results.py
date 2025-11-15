import pandas as pd
import matplotlib.pyplot as plt

assign = pd.read_csv("optimal_job_assignments.csv")

cost_by_inst = (
    assign
    .groupby("instance_type")["job_cost"]
    .sum()
    .reset_index()
    .sort_values("job_cost", ascending=False)
)

plt.figure()
plt.bar(cost_by_inst["instance_type"], cost_by_inst["job_cost"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total cost ($)")
plt.title("Total cost per instance type")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(assign["performance_score"], assign["job_cost"])
plt.xlabel("Job performance_score (1 = low, 3 = high)")
plt.ylabel("Job cost ($)")
plt.title("Job cost vs performance requirement")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(assign["performance_score"], assign["job_cost"], s=80)
plt.xlabel("Performance Score (1=low, 3=high)")
plt.ylabel("Job Cost ($)")
plt.title("Job Cost vs Performance Requirement")
plt.grid(True)
plt.show()

cost_by_perf = (
    assign.groupby("instance_perf_tier")["job_cost"]
    .sum()
    .reset_index()
)

plt.figure()
plt.bar(cost_by_perf["instance_perf_tier"], cost_by_perf["job_cost"])
plt.xlabel("Instance Performance Tier")
plt.ylabel("Total Cost ($)")
plt.title("Cost Distribution by Performance Tier")
plt.show()
