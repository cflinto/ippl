import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SOLVER_PATH = "./test/solver/TestCGSolver"
GRID_POWER = "7"

# Define configurations to benchmark: (Solver Name, Preconditioner, Display Label)
CONFIGS = [
    ("native", "none", "Native IPPL\n(No Prec)"),
    ("gko_mf", "none", "Ginkgo MF\n(No Prec)"),
    ("gko_csr", "none", "Ginkgo CSR\n(No Prec)"),
    ("gko_csr", "ilu", "Ginkgo CSR\n(ILU)"),
    ("gko_csr", "jacobi", "Ginkgo CSR\n(Jacobi)")
]

def run_benchmark(solver, prec):
    print(f"Running benchmark for: {solver} with {prec}...")
    cmd = [SOLVER_PATH, GRID_POWER, solver, prec, "--info", "5"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {solver} {prec}:\n{e.stderr}")
        return None

def parse_output(output):
    data = {
        "setup_matrix": 0.0, "setup_prec": 0.0, "setup_solver": 0.0,
        "solve_total": 0.0, "math": 0.0, "pack": 0.0, "solve_overhead": 0.0
    }
    
    # Robust regex for floats, including scientific notation (e.g., 5.12e-05)
    float_re = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    
    for line in output.split('\n'):
        # --- Parse Setup Timings ---
        if match := re.search(r"1a\. Setup: Matrix.*?(?:max|tot) =\s+" + float_re, line):
            data["setup_matrix"] = float(match.group(1))
        elif match := re.search(r"1b\. Setup: Precond.*?(?:max|tot) =\s+" + float_re, line):
            data["setup_prec"] = float(match.group(1))
        elif match := re.search(r"1c\. Setup: Solver.*?(?:max|tot) =\s+" + float_re, line):
            data["setup_solver"] = float(match.group(1))
            
        # --- Parse Solve Timings ---
        elif match := re.search(r"(?:1\.|2\.) SOLVE:.*?(?:max|tot) =\s+" + float_re, line):
            data["solve_total"] = float(match.group(1))
        elif match := re.search(r"(?:Ginkgo MF: Laplace|applyOp).*?(?:max|tot) =\s+" + float_re, line):
            data["math"] = float(match.group(1))
        elif match := re.search(r"Ginkgo MF: Pack/Unp.*?(?:max|tot) =\s+" + float_re, line):
            data["pack"] = float(match.group(1))

    # Calculate remaining overhead
    data["solve_overhead"] = max(0.0, data["solve_total"] - data["math"] - data["pack"])
    return data

def generate_plot(results):
    print("Generating dual bar chart...")
    labels = [cfg[2] for cfg in CONFIGS]
    x = np.arange(len(labels))
    width = 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1]})

    # --- PLOT 1: THE SETUP PHASE ---
    mat_times = [results[lbl]["setup_matrix"] for lbl in labels]
    prec_times = [results[lbl]["setup_prec"] for lbl in labels]
    sol_times = [results[lbl]["setup_solver"] for lbl in labels]

    ax1.bar(x, mat_times, width, label='Operator Setup / Assembly', color='#ff7f0e', edgecolor='black')
    ax1.bar(x, prec_times, width, bottom=mat_times, label='Preconditioner Generation (ILU/Jacobi)', color='#9467bd', edgecolor='black')
    # ax1.bar(x, sol_times, width, bottom=np.add(mat_times, prec_times), label='Solver Object Generation', color='#8c564b', edgecolor='black')

    ax1.set_ylabel('Time (Seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Setup Phase Cost', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax1.legend(loc='upper left')

    # --- PLOT 2: THE SOLVE PHASE ---
    math_times = [results[lbl]["math"] for lbl in labels]
    pack_times = [results[lbl]["pack"] for lbl in labels]
    over_times = [results[lbl]["solve_overhead"] for lbl in labels]

    ax2.bar(x, math_times, width, label='Raw Math (Stencil/SpMV)', color='#2ca02c', edgecolor='black')
    ax2.bar(x, pack_times, width, bottom=math_times, label='Layout Tax (Pack/Unpack)', color='#d62728', edgecolor='black')
    ax2.bar(x, over_times, width, bottom=np.add(math_times, pack_times), label='Solver Logic & Preconditioner Apply', color='#1f77b4', edgecolor='black')

    ax2.set_title('Iterative Solve Phase Cost', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax2.legend(loc='upper left')

    # Add numeric labels to the top of the solve bars
    for i, lbl in enumerate(labels):
        total = results[lbl]["solve_total"]
        ax2.text(i, total + 0.1, f'{total:.2f}s', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'Poisson Solver Profiling: Setup vs Solve ($128^3$ Grid)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("solver_benchmark_results.png", dpi=300)
    print("Plot saved successfully!")

if __name__ == "__main__":
    results = {}
    for solver, prec, label in CONFIGS:
        data = run_benchmark(solver, prec)
        if data:
            results[label] = data
            
    generate_plot(results)