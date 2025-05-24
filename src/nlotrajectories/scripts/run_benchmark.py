import argparse
import numpy as np
from pathlib import Path

import casadi as ca
import l4casadi as l4c
import yaml

from nlotrajectories.core.config import Config
from nlotrajectories.core.runner import RunBenchmark
from nlotrajectories.core.sdf.l4casadi import NNObstacle, NNObstacleTrainer
from nlotrajectories.core.visualizer import plot_levels, plot_trajectory
from nlotrajectories.core.metrics import mse, iou, hausdorf, chamfer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def compute_metrics(obstacles, x_range=(-1, 2), y_range=(-1, 2), n_samples=500):
    x = np.linspace(x_range[0], x_range[1], n_samples)
    y = np.linspace(y_range[0], y_range[1], n_samples)
    X, Y = np.meshgrid(x, y)
    sdf_pred = obstacles.approximated_sdf(X, Y)
    sdf_target = obstacles.sdf(X, Y)
    mse_value = mse(sdf_target, sdf_pred)
    iou_value = iou(sdf_target, sdf_pred, threshold = 0.0)
    hausdorf_value = hausdorf(sdf_target, sdf_pred)
    chamfer_value = chamfer(sdf_pred, sdf_target, X, Y, eps = 1e-2)

    return mse_value, iou_value, hausdorf_value, chamfer_value


def run_benchmark(config_path: Path):
    config = Config(**load_config(config_path))
    obstacles = config.get_obstacles()
    if config.solver.mode == "l4casadi":
        model = l4c.naive.MultiLayerPerceptron(2, 128, 1, 2, "Tanh")
        trainer = NNObstacleTrainer(obstacles, model)
        trainer.train((-0.5, 1.5), (-0.5, 1.5))
        obstacles = NNObstacle(obstacles, trainer.model)

    x0 = ca.MX(config.body.start_state)
    x_goal = ca.MX(config.body.goal_state)
    geometry = config.body.create_geometry()

    runner = RunBenchmark(
        dynamics=config.body.create_dynamics(),
        geometry=geometry,
        x0=x0,
        x_goal=x_goal,
        N=config.solver.N,
        dt=config.solver.dt,
        sdf_func=obstacles.approximated_sdf,
        control_bounds=tuple(config.body.control_bounds),
        use_slack=config.solver.use_slack,
        slack_penalty=config.solver.slack_penalty,
    )

    X_opt, U_opt, _ = runner.run()
    plot_trajectory(X_opt, geometry, obstacles, title=config_path.stem, goal=config.body.goal_state)
    plot_levels(obstacles.sdf, title=str(config_path.stem) + "_sdf")
    if config.solver.mode == "l4casadi":
        plot_levels(obstacles.approximated_sdf, title=str(config_path.stem) + "_nn_sdf")
    else:
        plot_levels(obstacles.approximated_sdf, title=str(config_path.stem) + "_math_sdf")

    # TODO: print metrics and save .csv with the result
    mse_value, iou_value, hausdorf_value, chamfer_value = compute_metrics(obstacles) 
    print("MSE:", mse_value)
    print("IoU:", iou_value)
    print("Hausdorf:", hausdorf_value)
    print("Chamfer:", chamfer_value)

    print("Optimization complete.")
    print("Final state:", X_opt[:, -1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark YAML config")
    args = parser.parse_args()

    run_benchmark(Path(args.config))


if __name__ == "__main__":
    main()
