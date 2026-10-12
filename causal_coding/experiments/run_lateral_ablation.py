"""Runner script for Lateral Connection Ablation Experiment (OFF vs ON).

Supports:
- --lateral-mode {on, off}
- --seed {int}
- --epochs {int} (default 60, allows small values for smoke testing)
- --smoke-test flag (sets 1 task, 2 epochs)

Outputs are saved in isolated results directories under results/causal_coding_lat_{mode}_seed_{seed}.
"""

import argparse
import json
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Ensure paths are set
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.data import load_mnist, split_into_tasks
from benchmarker import CLBenchmark
from src.model import CausalCodingModel
from src.method import CausalCodingMethod


def compute_lateral_diagnostics(params):
    """Compute summary norms of lateral parameters for diagnostics."""
    lateral_U_list = params.get("lateral_U", [])
    lateral_log_alpha_list = params.get("lateral_log_alpha", [])
    
    layer_diagnostics = []
    for l_idx, (U, rho) in enumerate(zip(lateral_U_list, lateral_log_alpha_list)):
        alpha_val = float(jax.nn.softplus(rho))
        U_frob = float(jnp.linalg.norm(U))
        UtU = U.T @ U
        eigs = jnp.linalg.eigvalsh(UtU)
        lambda_max = float(jnp.max(eigs)) if len(eigs) > 0 else 0.0
        # Effective lateral matrix F-norm: ||alpha * U U^T||_F = alpha * ||U^T U||_F
        lambda_frob = float(alpha_val * jnp.linalg.norm(UtU))
        
        layer_diagnostics.append({
            "layer": l_idx,
            "alpha": alpha_val,
            "U_frob": U_frob,
            "lambda_max": lambda_max,
            "lambda_frob": lambda_frob,
        })
    return layer_diagnostics


def main():
    parser = argparse.ArgumentParser(description="Lateral Connection Ablation Runner")
    parser.add_argument(
        "--lateral-mode",
        choices=["on", "off"],
        required=True,
        help="Whether lateral connections are 'on' or 'off'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for model initialization and PRNG",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of epochs per task (default 60)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test (2 tasks, 2 epochs) to verify pipeline",
    )
    args = parser.parse_args()

    lateral_enabled = (args.lateral_mode == "on")
    epochs = 2 if args.smoke_test else args.epochs

    run_name = f"causal_coding_lat_{args.lateral_mode}_seed_{args.seed}"
    if args.smoke_test:
        run_name += "_smoke"

    print(f"============================================================")
    print(f"Starting run: {run_name}")
    print(f"Lateral Enabled: {lateral_enabled}")
    print(f"Seed: {args.seed} | Epochs per task: {epochs} | Smoke test: {args.smoke_test}")
    print(f"============================================================")

    # Load dataset & create tasks
    X, y, test_X, test_y = load_mnist()
    if args.smoke_test:
        # Use first 2 tasks for smoke test
        class_pairs = [[0, 1], [2, 3]]
    else:
        class_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    tasks = split_into_tasks(X, y, test_X, test_y, class_pairs)

    # Initialize model
    model = CausalCodingModel(
        [784, 512, 512, 10],
        lateral_rank=32,
        lateral_init_scale=1e-3,
        alpha_init=0.01,
    )
    key = jax.random.PRNGKey(args.seed)
    params = model.init_params(key)

    # Initialize method
    method = CausalCodingMethod(
        lr_z=0.05,
        lr_w=0.05,
        num_inference_steps=30,
        gate_p=2.0,
        gate_kappa=1e-3,
        ridge=1e-4,
        lambda_s=0.0,
        batch_size=128,
        epochs=epochs,
        beta_pi=0.99,
        k_probe=10,

        # Lateral precision settings
        lateral_enabled=lateral_enabled,
        lr_lat=1e-3,
        beta_cov=0.99,
        eps_lat=1e-2,
        lat_warmup_epochs=5,
        lat_ramp_epochs=10,
        lambda_max_cap=1.0,
        beta_logdet=0.1,
        lambda_fro=1e-2,
        lambda_U=1e-5,

        # Lateral clarity enabled
        lambda_d=1e-3,
        clarity_t=1.0,
        clarity_eps=1e-4,

        # Tight-clipped structured diagonal residual precision
        pi0=2.718281828459045,
        rho_v=0.1,
        delta_abs=1e-12,
        d_min=0.75,
        d_max=1.5,

        # Soft hidden-only vertical pruning gates
        vertical_pruning_enabled=True,
        vertical_layer_scales=(1.0, 1.0, 0.0),
        vertical_alpha_g=8.0,
        lambda_vert_match=1e-3,
        lambda_vert_sparse=1e-5,
        vertical_eps=1e-2,
        lr_vert=3e-3,
        vertical_warmup_epochs=30 if not args.smoke_test else 1,
        vertical_ramp_epochs=15 if not args.smoke_test else 1,
        vertical_importance_update_epochs=5 if not args.smoke_test else 1,
        vertical_importance_batch_size=256,
        vertical_prune_threshold=0.2,
        vertical_importance_lambda_dyn=1e-3,
        vertical_importance_window=15,
        vertical_importance_num_iters=50,
        vertical_importance_standardize_x=False,
    )

    # Check initial lateral diagnostics
    init_diag = compute_lateral_diagnostics(params)
    print(f"\nInitial Lateral Diagnostics:")
    for d in init_diag:
        print(f"  Layer {d['layer']}: alpha={d['alpha']:.5f}, ||U||_F={d['U_frob']:.5f}, ||Lambda||_F={d['lambda_frob']:.5e}")

    # Run CLBenchmark
    benchmark = CLBenchmark(method=method, model=model, tasks=tasks, name=run_name)
    results = benchmark.run(params, None)

    # Compute final lateral diagnostics
    # Re-evaluate final params from benchmark if accessible or run diagnostic
    out_dir = os.path.join("results", run_name)
    diag_file = os.path.join(out_dir, "lateral_diagnostics.json")
    
    # Extract final metrics of interest
    m = results.metrics
    summary_report = {
        "run_name": run_name,
        "lateral_enabled": lateral_enabled,
        "seed": args.seed,
        "task_il_accuracy": m["task_il"]["average_accuracy"],
        "task_il_bwt": m["task_il"]["backward_transfer"],
        "task_il_fwt": m["task_il"]["forward_transfer"],
        "task_il_forgetting": m["task_il"]["forgetting"],
        "class_il_nll": m["class_il"]["average_nll"],
        "initial_diagnostics": init_diag,
    }
    
    with open(os.path.join(out_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary_report, f, indent=4)

    print("\n============================================================")
    print(f"Run {run_name} Complete!")
    print(f"Task-IL Average Accuracy : {m['task_il']['average_accuracy'] * 100:.2f}%")
    print(f"Task-IL BWT              : {m['task_il']['backward_transfer'] * 100:.2f}%")
    print(f"Task-IL FWT              : {m['task_il']['forward_transfer'] * 100:.2f}%")
    print(f"Class-IL NLL             : {m['class_il']['average_nll']:.4f}")
    print(f"Summary saved to {os.path.join(out_dir, 'experiment_summary.json')}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
