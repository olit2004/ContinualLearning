import argparse
import csv
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


def attach_lateral_trace(method):
    """Trace lateral parameter dynamics across tasks without polluting method config."""
    rows = []
    original = method.train_task

    def traced(model, params, state, task, task_idx, diagnostics_hook=None):
        U_in = [np.asarray(U) for U in params.get("lateral_U", [])]
        out = original(model, params, state, task, task_idx, diagnostics_hook)
        protocol = "task_il" if getattr(method, "task_il_training", False) else "class_il"
        
        for l, (U0, U, rho) in enumerate(zip(U_in, out[0].get("lateral_U", []), out[0].get("lateral_log_alpha", []))):
            U = np.asarray(U)
            alpha = float(jax.nn.softplus(rho))
            UtU = U.T @ U
            rows.append({
                "protocol": protocol,
                "task": task_idx + 1,
                "layer": l + 1,
                "alpha": alpha,
                "U_fro": float(np.linalg.norm(U)),
                "lambda_max": alpha * float(np.max(np.linalg.eigvalsh(UtU))),
                "lambda_frob": alpha * float(np.linalg.norm(UtU)),
                "rel_change_of_U": float(np.linalg.norm(U - U0) / (np.linalg.norm(U0) + 1e-12)),
            })
        return out

    method.train_task = traced
    return rows


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if it exists",
    )
    args = parser.parse_args()

    lateral_enabled = (args.lateral_mode == "on")
    epochs = 2 if args.smoke_test else args.epochs

    run_name = f"causal_coding_lat_{args.lateral_mode}_seed_{args.seed}"
    if args.smoke_test:
        run_name += "_smoke"

    out_dir = os.path.join("results", run_name)
    summary_file = os.path.join(out_dir, "experiment_summary.json")

    # Overwrite protection guard
    if os.path.exists(summary_file) and not args.force:
        raise FileExistsError(
            f"Results already exist at '{summary_file}'. "
            f"Pass --force to overwrite."
        )

    print(f"============================================================")
    print(f"Starting run: {run_name}")
    print(f"Lateral Enabled: {lateral_enabled}")
    print(f"Seed: {args.seed} | Epochs per task: {epochs} | Smoke test: {args.smoke_test}")
    print(f"============================================================")

    # Load dataset & create tasks
    X, y, test_X, test_y = load_mnist()
    if args.smoke_test:
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

    # Benchmark initialization
    benchmark = CLBenchmark(method=method, model=model, tasks=tasks, name=run_name)

    # Attach lateral trace wrapper strictly AFTER CLBenchmark init so vars(method)
    # does not capture the wrapper function into benchmark config/JSON.
    trace_rows = attach_lateral_trace(method)

    results = benchmark.run(params, None)

    os.makedirs(out_dir, exist_ok=True)

    # Write lateral trajectory trace CSV
    trace_file = os.path.join(out_dir, "lateral_trace.csv")
    if trace_rows:
        fieldnames = ["protocol", "task", "layer", "alpha", "U_fro", "lambda_max", "lambda_frob", "rel_change_of_U"]
        with open(trace_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace_rows)

    # Extract final metrics of interest
    m = results.metrics
    final_lateral_frob = trace_rows[-1]["lambda_frob"] if trace_rows else 0.0

    summary_report = {
        "run_name": run_name,
        "lateral_enabled": lateral_enabled,
        "seed": args.seed,
        "task_il_accuracy": m["task_il"]["average_accuracy"],
        "task_il_bwt": m["task_il"]["backward_transfer"],
        "task_il_fwt": m["task_il"]["forward_transfer"],
        "task_il_forgetting": m["task_il"]["forgetting"],
        "class_il_nll": m["class_il"]["average_nll"],
        "final_lambda_frob": final_lateral_frob,
    }

    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=4)

    print("\n============================================================")
    print(f"Run {run_name} Complete!")
    print(f"Task-IL Average Accuracy : {m['task_il']['average_accuracy'] * 100:.2f}%")
    print(f"Task-IL BWT              : {m['task_il']['backward_transfer'] * 100:.2f}%")
    print(f"Task-IL FWT              : {m['task_il']['forward_transfer'] * 100:.2f}%")
    print(f"Class-IL NLL             : {m['class_il']['average_nll']:.4f}")
    print(f"Final ||Lambda||_F       : {final_lateral_frob:.5e}")
    print(f"Summary saved to {summary_file}")
    print(f"Lateral trace saved to {trace_file}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
