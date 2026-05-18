<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/6149396c-2e96-4722-ae3d-d0fab7a44dad" />

# Project Overview

This is an academic Machine Learning project focused on implementing and reproducing the results of the 2017 paper **"Overcoming catastrophic forgetting in neural networks"** by Kirkpatrick et al. The project specifically implements **Elastic Weight Consolidation (EWC)**, a continual learning algorithm designed to mitigate catastrophic forgetting in neural networks by anchoring critical weights using the Fisher Information Matrix.

The repository includes:
1.  **Core EWC Implementation:** A PyTorch implementation of a Multi-Layer Perceptron (MLP) trained sequentially on the **Permuted MNIST** benchmark, comparing standard Stochastic Gradient Descent (SGD) against EWC.
2.  **Algorithmic Improvement (Online EWC):** An enhanced version of the algorithm (`Bonus Solution/improvement.py`) that implements two major upgrades:
    *   **Accuracy Improvement (Lines 13-28):** A modernized MLP architecture incorporating Dropout layers, which creates more robust data representations and outperforms the original algorithm in terms of retained accuracy (97.17% vs 95.75% on Task 1).
    *   **Memory Optimization (Lines 30-74):** Reduces memory complexity from $O(N)$ to $O(1)$ by using a running average of the Fisher Information Matrix, successfully maintaining a constant memory footprint regardless of the number of tasks.
3.  **Medical Domain Application (LifeLonger):** A conceptual simulation applying EWC to a Continual Disease Classification problem (MRI vs. CT scans) based on the *LifeLonger* benchmark.

# Directory Structure
*   `src/`: Contains the core implementation files.
    *   `model.py`: Defines the MLP architecture.
    *   `data.py`: Handles downloading MNIST and applying spatial permutations.
    *   `ewc.py`: The core EWC logic and Fisher Information Matrix computation.
    *   `main.py`: Main training loop for sequential task learning.
    *   `plot_results.py` / `plot_graph2.py`: Scripts for generating performance visualizations.
*   `Bonus Solution/`: Contains the advanced (+5 bonus) objectives.
    *   `improvement.py`: Implements $O(1)$ Online EWC and the upgraded network.
    *   `bonus_solution.py`: Medical imaging continual learning simulation.
    *   Documentation Markdown files explaining the improvements.
*   `Academic articals/`: Reference PDF papers.
*   `data/`: Automatically generated directory for downloaded datasets (ignored by git).

# Building and Running

Dependencies are listed in `requirements.txt` and should be installed within a virtual environment.

**Setup:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Running the Core Project:**
```powershell
# Run the base training loop
python src/main.py

# Generate Graph 1: Performance on Task 1 while learning new tasks
python src/plot_results.py

# Generate Graph 2: Average Performance Across Multiple Sequential Tasks
python src/plot_graph2.py
```

### Why Compare Standard SGD with EWC?
**Standard SGD (Stochastic Gradient Descent)** is the default way neural networks learn. When trained on Task 1, it tunes its weights perfectly. But when moved to Task 2, it ruthlessly overwrites all those weights to minimize the new errors, completely destroying its memory of Task 1. This is known as **Catastrophic Forgetting**.

**Elastic Weight Consolidation (EWC)** solves this by acting as an add-on to SGD. Before starting Task 2, EWC calculates the **Fisher Information Matrix** to identify which specific weights were most critical for Task 1. While learning Task 2, EWC acts like a rubber band—it allows SGD to easily change "unimportant" weights, but heavily penalizes any changes to the crucial weights needed for Task 1. 

By comparing them, we prove the algorithm works: the Standard SGD graph plummets (showing forgetting), while the EWC graph remains stable, proving it successfully protected the network's prior knowledge.

**Running the Bonus Implementations:**
```powershell
# Run the LifeLonger continual disease classification simulation
python "Bonus Solution\bonus_solution.py"

# Run the O(1) Online EWC algorithmic improvement
python "Bonus Solution\improvement.py"

# Generate Bonus Graph 1: Performance on Task 1 (Improved Online EWC)
python "Bonus Solution\plot_bonus_graph1.py"

# Generate Bonus Graph 2: Average Performance (Improved Online EWC)
python "Bonus Solution\plot_bonus_graph2.py"
```

# Development Conventions

*   **Framework:** All models and algorithms are built using PyTorch.
*   **Documentation:** This is an academic submission; hence, heavy commenting in the Python files is required to explain the mathematical logic (e.g., Fisher Information calculations). Thorough explanations and AI interaction logs are maintained in the root Markdown files (`Research_Report.md`, `AI_Logs.md`).
*   **Visualization:** `matplotlib` is used for comparing standard SGD and EWC models. Ensure all graph outputs are saved to the project root as `.png` files.
*   **Version Control:** The `.gitignore` file enforces that massive datasets (e.g., the `data/` folder), virtual environments (`venv/`), and `__pycache__` directories are never committed to the repository.
