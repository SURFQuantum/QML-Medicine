import time
from typing import Dict, Any, List
import pennylane as qml
import torch.nn as nn
from datetime import datetime

# 1. IMPORT REFACTORED TEST CASE COMPONENT
# NOTE: We assume train.py has been refactored or its core functions
# are available. The functions below must be accessible for this script to run.
# You must ensure train_epoch, validate, get_dataloaders, etc. are available.
from src.train_extended import run_qcnn_training_job  # Assuming this is your refactored function
from src.model.models import HybridClassifier  # The QCNN model

# --- INSTRUMENTATION GLOBALS AND HOOKS ---
QUANTUM_EVAL_TIMES: List[float] = []


def qnode_timer_wrapper(dev: qml.device) -> qml.device:
    """Wraps the PennyLane device with hooks to record wall-clock time per QNode call."""

    def pre_exec(self, circuits, execute_kwargs):
        self._start_time = time.monotonic()

    def post_exec(self, results, execute_kwargs):
        end_time = time.monotonic()
        # Record the time for this single QNode execution
        QUANTUM_EVAL_TIMES.append(end_time - self._start_time)
        return results

    dev.preprocess = pre_exec.__get__(dev, qml.device)
    dev.postprocess = post_exec.__get__(dev, qml.device)
    return dev


def pennylane_qcnn_adapter(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    The main adapter function passed to the BenchmarkFramework.
    It instruments the QNode, runs the training job, and aggregates results.
    """
    global QUANTUM_EVAL_TIMES
    QUANTUM_EVAL_TIMES = []  # Reset timer for this run

    use_quantum = mode == 'quantum'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.get('dataset_type', 'pcam')}_{mode}_{timestamp}"

    # --- INSTRUMENTATION HACK ---
    original_device = qml.device

    def instrumented_device(*args, **kwargs):
        dev = original_device(*args, **kwargs)
        if use_quantum and dev.name == "default.qubit":
            # Only instrument the PennyLane device when in quantum mode
            return qnode_timer_wrapper(dev)
        return dev

    qml.device = instrumented_device

    try:
        # 1. Run the Training Job (Relies on refactored function in train.py)
        # This function must return start_time, end_time, and cost_history
        print(f"Starting QCNN training job in {mode} mode...")
        raw_results = run_qcnn_training_job(config, use_quantum, run_name)

    finally:
        # 2. Restore original qml.device function
        qml.device = original_device

        # 3. Add Instrumentation and Circuit Metrics
    if use_quantum:
        n_qubits = config['model'].get('n_qubits', 4)
        n_layers = config['model'].get('n_quantum_layers', 1)

        raw_results['quantum_eval_times'] = QUANTUM_EVAL_TIMES
        raw_results['num_qubits'] = n_qubits
        # Estimate static circuit complexity
        raw_results['circuit_depth'] = int(3 * n_layers * n_qubits)
        raw_results['max_two_qubit_gates'] = int(n_layers * (n_qubits - 1))
    else:
        # Classical run returns empty quantum metrics
        raw_results['quantum_eval_times'] = []
        raw_results['num_qubits'] = None
        raw_results['circuit_depth'] = None
        raw_results['max_two_qubit_gates'] = None

    return raw_results