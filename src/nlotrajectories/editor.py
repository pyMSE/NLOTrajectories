import streamlit as st
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

st.set_page_config(layout="wide")
st.title("Trajectory Config Editor")

# Persistent session state
if "obstacles" not in st.session_state:
    st.session_state.obstacles = []

# Start/Goal position
start_x = st.slider("Start X", 0.0, 1.0, 0.1)
start_y = st.slider("Start Y", 0.0, 1.0, 0.1)
goal_x = st.slider("Goal X", 0.0, 1.0, 0.9)
goal_y = st.slider("Goal Y", 0.0, 1.0, 0.9)

# Add new obstacle
with st.expander("➕ Add Obstacle"):
    col1, col2 = st.columns(2)
    with col1:
        kind = st.selectbox("Type", ["circle", "square"])
        x = st.slider("X", 0.0, 1.0, 0.5)
        y = st.slider("Y", 0.0, 1.0, 0.5)
    with col2:
        size = st.slider("Size (radius or square half-width)", 0.01, 0.5, 0.1)
        margin = st.slider("Margin", 0.0, 0.2, 0.05)
        if st.button("Add"):
            st.session_state.obstacles.append({
                "type": kind,
                "center": [x, y],
                "radius": size if kind == "circle" else None,
                "size": size if kind == "square" else None,
                "margin": margin
            })

# Remove obstacle
if st.session_state.obstacles:
    idx_to_remove = st.selectbox("🗑️ Remove Obstacle", range(len(st.session_state.obstacles)))
    if st.button("Remove Selected"):
        st.session_state.obstacles.pop(idx_to_remove)

# Show preview
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

for obs in st.session_state.obstacles:
    if obs["type"] == "circle":
        ax.add_patch(Circle(obs["center"], obs["radius"], color="red", alpha=0.5))
    else:
        cx, cy = obs["center"]
        size = obs["size"]
        ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size, color="blue", alpha=0.5))

ax.plot(start_x, start_y, "go", label="Start")
ax.plot(goal_x, goal_y, "ro", label="Goal")
ax.legend()
st.pyplot(fig)

# YAML generation
config = {
    "body": {
        "shape": "dot",
        "dynamic": "point_1st",
        "goal_mode": "center",
        "start_state": [start_x, start_y, 0.0, 0.0],
        "goal_state": [goal_x, goal_y, 0.0, 0.0],
        "control_bounds": [-1.0, 1.0]
    },
    "obstacles": [
        {k: v for k, v in obs.items() if v is not None}
        for obs in st.session_state.obstacles
    ],
    "solver": {
        "N": 40,
        "dt": 0.1,
        "mode": "casadi"
    }
}

yaml_str = yaml.dump(config, sort_keys=False)
st.code(yaml_str, language="yaml")

# Save button
if st.button("💾 Save YAML to benchmarks/generated.yaml"):
    with open("src/nlotrajectories/benchmarks/generated.yaml", "w") as f:
        f.write(yaml_str)
    st.success("Saved to benchmarks/generated.yaml")

# Run button
if st.button("🚀 Run Optimizer"):
    import subprocess
    result = subprocess.run(["poetry", "run", "run-benchmark", "--config", "src/nlotrajectories/benchmarks/generated.yaml"])
    if result.returncode == 0:
        st.success("Optimization done")
        st.image("result/generated.png", caption="Result")
    else:
        st.error("Error running optimizer.")
