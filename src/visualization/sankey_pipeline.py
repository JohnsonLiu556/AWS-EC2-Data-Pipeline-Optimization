import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("optimal_job_assignments.csv")

df["job_cost"] = df["job_cost"].round(3)

job_ids = df["job_id"].tolist()
instance_types = df["instance_type"].unique().tolist()

nodes = job_ids + instance_types

node_index = {name: i for i, name in enumerate(nodes)}

sources = [node_index[j] for j in df["job_id"]]
targets = [node_index[it] for it in df["instance_type"]]
values  = df["job_cost"].tolist()

link_labels = [
    f"{j} → {it}<br>Cost: ${c:.2f}"
    for j, it, c in zip(df["job_id"], df["instance_type"], df["job_cost"])
]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=15,
        label=nodes
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        label=link_labels
    )
)])

fig.update_layout(
    title_text="Data Pipeline Cost Flow: Jobs → EC2 Instance Types",
    font=dict(size=12)
)

fig.show()
