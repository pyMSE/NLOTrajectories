import argparse

import casadi as ca
import yaml

from nlotrajectories.core.config import BodyConfig
from nlotrajectories.core.runner import RunBenchmark
from nlotrajectories.core.visualizer import plot_trajectory


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_sdf_circle(center, radius):
    def sdf(x, y):
        dx = x - center[0]
        dy = y - center[1]
        return ca.sqrt(dx**2 + dy**2) - radius

    return sdf


def draw_obstacle(ax, cfg):
    from matplotlib.patches import Circle

    c = Circle(cfg["obstacles"]["center"], cfg["obstacles"]["radius"], color="r", alpha=0.5)
    ax.add_patch(c)


def main(config_path):
    cfg = load_config(config_path)

    body_config = BodyConfig(**cfg["body"])
    sdf_func = make_sdf_circle(cfg["obstacles"]["center"], cfg["obstacles"]["radius"])

    x0 = ca.MX(cfg["body"]["start_state"])
    x_goal = ca.MX(cfg["body"]["goal_state"])
    geometry = body_config.create_geometry()

    runner = RunBenchmark(
        dynamics=body_config.create_dynamics(),
        geometry=geometry,
        x0=x0,
        x_goal=x_goal,
        N=cfg["solver"]["N"],
        dt=cfg["solver"]["dt"],
        sdf_func=sdf_func,
        control_bounds=tuple(cfg["body"]["control_bounds"]),
    )

    X_opt, U_opt, _ = runner.run()
    plot_trajectory(
        X_opt, geometry, title="Benchmark 1", goal=cfg["body"]["goal_state"], obstacle_draw_fn=draw_obstacle, cfg=cfg
    )

    print("Optimization complete.")
    print("Final state:", X_opt[:, -1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark YAML config")
    args = parser.parse_args()

    main(args.config)
