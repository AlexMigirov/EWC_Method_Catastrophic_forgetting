# Project Overview

This is an academic Machine Learning project focused on implementing and reproducing the results of the 2017 paper **"Overcoming catastrophic forgetting in neural networks"** by Kirkpatrick et al. The project specifically implements **Elastic Weight Consolidation (EWC)**, a continual learning algorithm designed to mitigate catastrophic forgetting in neural networks by anchoring critical weights using the Fisher Information Matrix.

The repository includes:
1.  **Core EWC Implementation:** A PyTorch implementation of a Multi-Layer Perceptron (MLP) trained sequentially on the **Permuted MNIST** benchmark, comparing standard Stochastic Gradient Descent (SGD) against EWC.
2.  **Algorithmic Improvement (Online EWC):** An enhanced version of the algorithm (`Bonus Solution/improvement.py`) that reduces memory complexity from $O(N)$ to $O(1)$ by using a running average of the Fisher Information Matrix, combined with a modernized MLP architecture (Dropout).
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

**Running the Bonus Implementations:**
```powershell
# Run the LifeLonger continual disease classification simulation
python "Bonus Solution\bonus_solution.py"

# Run the O(1) Online EWC algorithmic improvement
python "Bonus Solution\improvement.py"
```

# Development Conventions

*   **Framework:** All models and algorithms are built using PyTorch.
*   **Documentation:** This is an academic submission; hence, heavy commenting in the Python files is required to explain the mathematical logic (e.g., Fisher Information calculations). Thorough explanations and AI interaction logs are maintained in the root Markdown files (`Research_Report.md`, `AI_Logs.md`).
*   **Visualization:** `matplotlib` is used for comparing standard SGD and EWC models. Ensure all graph outputs are saved to the project root as `.png` files.
*   **Version Control:** The `.gitignore` file enforces that massive datasets (e.g., the `data/` folder), virtual environments (`venv/`), and `__pycache__` directories are never committed to the repository.
