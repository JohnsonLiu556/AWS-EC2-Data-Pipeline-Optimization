import pandas as pd
import numpy as np

np.random.seed(42)

N_JOBS = 15  # you can change this

job_ids = [f"job_{i+1:03d}" for i in range(N_JOBS)]

cpu_hours = np.round(np.random.uniform(1, 20, size=N_JOBS), 2)  # 1–20 hours
memory_gb = np.random.choice([4, 8, 16, 32, 64, 128], size=N_JOBS)

# Performance importance score: 1 (low) to 3 (high)
performance_score = np.random.choice([1, 2, 3], size=N_JOBS, p=[0.4, 0.4, 0.2])

df = pd.DataFrame({
    "job_id": job_ids,
    "CPU_hours": cpu_hours,
    "memory_GB": memory_gb,
    "performance_score": performance_score,
})

df.to_csv("jobs.csv", index=False)
print("Saved jobs.csv")
print(df.head())
