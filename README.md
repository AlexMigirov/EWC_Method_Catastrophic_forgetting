# Project Instructions: EWC Catastrophic Forgetting

This file contains the architecture, conventions, and workflows established for this academic project.

## Project Scope
- **Goal:** Reproduce the findings of the 2017 paper "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et al.
- **Bonus Goal:** Adapt the Elastic Weight Consolidation (EWC) algorithm to the medical domain using concepts from the "LifeLonger" Continual Disease Classification benchmark.

## Architecture & Tech Stack
- **Framework:** PyTorch
- **Core Model:** Multi-Layer Perceptron (MLP) defined in `src/model.py`
- **Data Handling:** 
  - Standard datasets (MNIST) are automatically downloaded via `torchvision`. 
  - Custom permutations are applied to generate sequential tasks (`src/data.py`).
  - Synthetic medical data is used for the LifeLonger bonus (`bonus_solution.py`).

## Coding Conventions
- **Comments:** All Python code must be heavily commented to serve as educational material for the oral defense.
- **Dependencies:** Managed strictly through `requirements.txt` (requires `torch`, `torchvision`, `matplotlib`, `tqdm`).
- **Visualization:** Metrics tracking forgetting must be visualized using `matplotlib` and saved as `results_graph.png`.

## Git & Version Control Policy
- Do not commit large datasets (e.g., the `data/` folder).
- Do not commit virtual environments (`venv/`).
- Do not commit cached Python bytecode (`__pycache__/`, `*.pyc`).
- Ensure `.gitignore` is utilized to enforce these rules.

## Documentation
- **Research Report:** The comprehensive breakdown of the assignment, the mathematical foundations of EWC (Fisher Information Matrix), and the analysis of the graphs are maintained in `Research_Report.md`.
- **Bonus Solution:** The code and explanation for the LifeLonger benchmark application are maintained in `bonus_solution.md` and `bonus_solution.py`.
