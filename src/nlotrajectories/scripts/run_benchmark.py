import argparse
import time
from pathlib import Path

import casadi as ca
import l4casadi as l4c
import numpy as np
import yaml

from nlotrajectories.core.config import Config
from nlotrajectories.core.metrics import chamfer, hausdorff, iou, mse, surface_loss
from nlotrajectories.core.nn_architectures import SIREN, FourierMLP
from nlotrajectories.core.runner import RunBenchmark
from nlotrajectories.core.sdf.l4casadi import NNObstacle, NNObstacleTrainer
from nlotrajectories.core.trajectory_initialization import (
    LinearInitializer,
    RRTInitializer,
)
from nlotrajectories.core.visualizer import (
    animation_plot,
    plot_control,
    plot_levels,
    plot_trajectory,
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(obstacles, x_range=(-1, 2), y_range=(-1, 2), n_samples=1000):
    x = np.linspace(x_range[0], x_range[1], n_samples)
    y = np.linspace(y_range[0], y_range[1], n_samples)
    X, Y = np.meshgrid(x, y)
    sdf_pred = obstacles.approximated_sdf(X, Y)
    sdf_target = obstacles.sdf(X, Y)
    mse_value = mse(sdf_target, sdf_pred)
    iou_value = iou(sdf_target, sdf_pred, threshold=0.0)
    hausdorff_value = hausdorff(sdf_pred, sdf_target, X, Y, eps=1e-2)
    chamfer_value = chamfer(sdf_pred, sdf_target, X, Y, eps=1e-2)
    surface_loss_value = surface_loss(sdf_target, sdf_pred, eps=1e-2)

    return mse_value, iou_value, hausdorff_value, chamfer_value, surface_loss_value


def run_benchmark(config_path: Path):
    config = Config(**load_config(config_path))
    obstacles = config.get_obstacles()

    if config.solver.mode == "l4casadi":
        # model_cfg = getattr(config.solver, "model", {})
        model_type = config.model.type
        print(f"Using model type: {model_type}")
        hidden_dim = config.model.hidden_dim
        num_hidden_layers = config.model.num_hidden_layers
        activation_function = config.model.activation_function
        omega_0 = config.model.omega_0

        if model_type == "mlp":
            model = l4c.naive.MultiLayerPerceptron(2, hidden_dim, 1, num_hidden_layers, activation_function)
        elif model_type == "fourier":
            model = FourierMLP(
                input_dim=2,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=num_hidden_layers + 2,  # FourierMLP may have an embedding layer
                activation_function=activation_function,
            )
        elif model_type == "siren":
            model = SIREN(
                input_dim=2,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=num_hidden_layers + 2,
                omega_0=omega_0,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        surface_loss_weight = 1
        eikonal_weight = 0.01
        trainer = NNObstacleTrainer(
            obstacles, model, eikonal_weight=eikonal_weight, surface_loss_weight=surface_loss_weight
        )
        trainer.train((-0.5, 1.5), (-0.5, 1.5))
        if type(model) is l4c.naive.MultiLayerPerceptron:
            obstacles = NNObstacle(obstacles, trainer.model)
        else:
            model_l4c = l4c.L4CasADi(trainer.model, device="cpu")
            obstacles = NNObstacle(obstacles, model_l4c)

    x0 = ca.MX(config.body.start_state)
    x_goal = ca.MX(config.body.goal_state)
    geometry = config.body.create_geometry()

    init_cfg = config.solver.initializer.choice
    if init_cfg.mode == "linear":
        initializer = LinearInitializer(
            N=config.solver.N, x0=np.array(config.body.start_state), x_goal=np.array(config.body.goal_state)
        )
    else:
        # init_cfg.mode == "rrt"
        initializer = RRTInitializer(
            N=config.solver.N + 1,
            x0=np.array(config.body.start_state),
            x_goal=np.array(config.body.goal_state),
            dt=config.solver.dt,
            sdf_func=obstacles.sdf,
            bounds=init_cfg.rrt_bounds,
            step_size=init_cfg.step_size,
            max_iter=init_cfg.max_iter,
            min_sdf=init_cfg.min_sdf,
        )

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
        initializer=initializer,
    )

    start_time = time.time()
    X_opt, U_opt, opti, X_init = runner.run()
    end_time = time.time()

    objective_value = float(opti.debug.value(opti.f))
    solver_time = end_time - start_time
    print("Objective value:", objective_value)
    print("Computation time for the solver:", solver_time)
    plot_trajectory(X_opt, geometry, obstacles, X_init=X_init, title=config_path.stem, goal=config.body.goal_state)
    plot_levels(obstacles.sdf, title=str(config_path.stem) + "_sdf")
    plot_control(U_opt, config.solver.dt, title=str(config_path.stem) + "_control")
    animation_plot(
        X_opt, U_opt, geometry, obstacles, title=str(config_path.stem) + "_animation", goal=config.body.goal_state
    )

    suffix = "_nn_sdf" if config.solver.mode == "l4casadi" else "_math_sdf"
    plot_levels(obstacles.approximated_sdf, title=f"{config_path.stem}{suffix}")

    # TODO: print metrics and save .csv with the result
    mse_value, iou_value, hausdorff_value, chamfer_value, surface_loss_value = compute_metrics(obstacles)
    print("MSE:", mse_value)
    print("IoU:", iou_value)
    print("Hausdorff:", hausdorff_value)
    print("Chamfer:", chamfer_value)
    print("Surface loss:", surface_loss_value)

    print("Optimization complete.")
    print("Final state:", X_opt[:, -1])

    # save results to a csv file
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / f"{config_path.stem}_results.csv"

    file_exists = results_file.exists()
    with open(results_file, "a") as f:
        # If the file doesn't exist, write the header first
        if not file_exists:
            f.write(
                "solver_mode,model_type,num_hidden_layers,hidden_dim,activation_function,"
                "omega_0,surface_loss_weight,eikonal_weight,num_steps,objective_value,"
                "solver_time,mse,iou,hausdorff,chamfer,surface_loss\n"
            )
        # Append the results
        if config.solver.mode == "l4casadi":
            f.write(
                f"{config.solver.mode},{model_type},{num_hidden_layers},{hidden_dim},{activation_function},"
                f"{omega_0},{surface_loss_weight},{eikonal_weight},{config.solver.N},{objective_value:3f},"
                f"{solver_time:2f},{mse_value:6f},{iou_value:6f},{hausdorff_value:6f},{chamfer_value:6f},"
                f"{surface_loss_value:6f}\n"
            )

        else:
            f.write(
                f"{config.solver.mode},None,None,None,None,None,None,None,{config.solver.N},"
                f"{objective_value:3f},{solver_time:2f},{mse_value:6f},{iou_value:6f},{hausdorff_value:6f},"
                f"{chamfer_value:6f},{surface_loss_value:6f}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark YAML config")
    args = parser.parse_args()

    run_benchmark(Path(args.config))


if __name__ == "__main__":
    main()
