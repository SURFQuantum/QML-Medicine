import yaml
import logging
from argparse import ArgumentParser
from typing import Dict, Any

# 1. IMPORT FRAMEWORK COMPONENTS
# Adjust path based on your clone location
from benchmark.benchmark_framework import BenchmarkFramework
from benchmark.metrics.optimality import OptimalityMetric
from benchmark.metrics.profiling import TimeProfiler
from benchmark.metrics.circuit_metric import CircuitDepthMetric
from benchmark.visualization.basic_visualization import StandardVisualizer

# 2. IMPORT THE NEWLY CREATED ADAPTER
from qcnn_wrapper import pennylane_qcnn_adapter

# Configure logging (replicated from train.py)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config_wrapper(path: str) -> dict:
    """Wrapper to load YAML config (Assumes original load_config is accessible)."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {path}. Using default configuration.")
        return {
            'dataset_type': 'pcam',
            'model': {'latent_dim': 16, 'pcam_filters': 4, 'num_classes': 1,
                      'quantum_head_type': 'angle', 'n_qubits': 4,
                      'n_quantum_layers': 3, 'entangling_layer': 'strong'},
            'training': {'epochs': 3, 'lr': 1e-3, 'batch_size': 32}
        }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    parser.add_argument('--mode', type=str, choices=['classical', 'quantum'], default='quantum')
    args = parser.parse_args()

    config = load_config_wrapper(args.config)

    # --- SETUP BENCHMARK ---
    KNOWN_MIN_LOSS = 0.05 if config['dataset_type'] == 'pcam' else 0.1
    N_QUBITS = config['model'].get('n_qubits', 4)
    RUN_LABEL = f"QCNN_Benchmark_{args.mode}_{N_QUBITS}q"

    print("=====================================================")
    print(f"  Starting Benchmark for: {RUN_LABEL}")
    print("=====================================================")

    # A. Setup Framework (Composition)
    benchmark = BenchmarkFramework(run_label=RUN_LABEL)
    benchmark.add_metric(OptimalityMetric())
    benchmark.add_metric(TimeProfiler())

    # Only add CircuitDepthMetric if running a quantum job
    if args.mode == 'quantum':
        benchmark.add_metric(CircuitDepthMetric())

    # B. Set Context
    benchmark.set_context({
        'known_minimum': KNOWN_MIN_LOSS,
        'optimality_threshold': 0.01,
        'run_label': RUN_LABEL
    })

    # C. Run the Algorithm, passing the adapter function
    raw_results = benchmark.run_algorithm(
        pennylane_qcnn_adapter,
        config=config,
        mode=args.mode  # Pass the mode to the adapter
    )

    # D. Analyze and Report
    if raw_results:
        benchmark.analyze()
        print(benchmark.generate_full_report())
        StandardVisualizer.generate_all_plots(
            raw_results=raw_results,
            context=benchmark.context,
            run_label=RUN_LABEL
        )
        benchmark.generate_json_report()
        logger.info("\nBenchmark execution complete. Check 'benchmark_plots' for visuals.")