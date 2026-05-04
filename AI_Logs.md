# AI Interaction Logs & Documentation

This document serves as the required log of AI interactions, prompts, and workflows used to develop the Elastic Weight Consolidation (EWC) project and the LifeLonger bonus objective.

## 1. Project Initialization & Context Gathering
*   **User Action:** The user provided the project instructions in a Word document (`.docx`) and two academic papers in PDF format (the original EWC paper by Kirkpatrick et al., and the LifeLonger medical benchmark paper).
*   **AI Action:** The AI explained its workspace constraints and used a shell command (`Move-Item`) to move the project folder from the `C:\` drive to the user's Desktop to ensure proper access permissions.
*   **AI Action:** The AI explained the purpose of the `GEMINI.md` file and the difference between "Plan Mode" and "Default Mode".

## 2. Research Phase
*   **User Prompt:** *"find for me the best top 3 academic articals that discuss top EWC method in continuous learning"*
*   **AI Action:** The AI utilized the Google Web Search tool to find and summarize the seminal EWC paper, the Online EWC scalability paper, and the theoretical Bayesian foundation paper.
*   **User Prompt:** *"i want an 2 seperate agents. one that does the research, reads my PDF files... creates me a 'report'... the 2nd agent, i want him do specialized code work..."*
*   **AI Action:** The AI orchestrated two sub-agents. 
    1.  `codebase_investigator`: Attempted to map the files.
    2.  `generalist`: Successfully read the Word document and PDFs, synthesizing the requirements and academic theory into `Research_Report.md`.

## 3. Implementation Phase (Baseline & EWC)
*   **User Prompt:** *"yes, start phase 1 and 2"*
*   **AI Action:** The AI invoked the `generalist` sub-agent to write the PyTorch codebase. This included setting up the MLP (`src/model.py`), the data loaders for Permuted MNIST (`src/data.py`), the EWC mathematical logic including the Fisher Information Matrix (`src/ewc.py`), and the main training loop (`src/main.py`).
*   **AI Action:** The AI created a Python virtual environment (`venv`), installed `torch` and `torchvision`, and successfully ran the baseline model to verify that EWC prevented catastrophic forgetting across 3 tasks.

## 4. Evaluation & Visualization Phase
*   **User Prompt:** *"yes, and i want you to compare the results of the graph with the articals... update the report file every time"*
*   **AI Action:** The AI wrote a custom plotting script (`src/plot_results.py`) utilizing `matplotlib`. It trained both a standard SGD model and the EWC model, continuously tracking their accuracy on Task 1.
*   **AI Action:** Generated `results_graph.png` and updated `Research_Report.md` to document that the Standard SGD model dropped to ~56% accuracy, while the EWC model retained ~95% accuracy, successfully mirroring the original paper's findings.
*   **User Prompt:** *"is 1 graph is enough?"* -> *"yes" (to generate the second graph)*
*   **AI Action:** The AI identified that the original paper requires a second graph for "Average performance across multiple sequential tasks". The AI created a new script (`src/plot_graph2.py`), ran it, generated `results_graph_2_average.png`, and appended the findings to the Research Report.

## 5. Bonus Objective (LifeLonger Benchmark)
*   **User Prompt:** *"create a new file called 'bonus_solution' and put the code\answer\results in there."*
*   **AI Action:** The AI wrote a standalone script (`bonus_solution.py`) to simulate Continual Disease Classification. It generated synthetic medical image embeddings (MRI vs. CT). The EWC model successfully learned to detect a second disease without forgetting the first. The theoretical explanation and results were compiled into `bonus_solution.md`.

## 6. Educational Explanations & GitHub Preparation
*   **User Prompt:** *"what is a 'Git repository'?"* and *"do i really ned the .gitignore file?"*
*   **AI Action:** The AI explained Version Control, GitHub repositories, and provided a step-by-step guide for manually dragging and dropping the correct folders (emphasizing omitting the `venv` and `data` folders to avoid size limits).
*   **User Prompt:** *"what is standard SGD and why its compared with EWC method?"*
*   **AI Action:** Provided the following foundational explanation (which the user requested to attach beneath their graph on GitHub):

### Why Compare Standard SGD with EWC?
**Standard SGD (Stochastic Gradient Descent)** is the default way neural networks learn. When trained on Task 1, it tunes its weights perfectly. But when moved to Task 2, it ruthlessly overwrites all those weights to minimize the new errors, completely destroying its memory of Task 1. This is known as **Catastrophic Forgetting**.

**Elastic Weight Consolidation (EWC)** solves this by acting as an add-on to SGD. Before starting Task 2, EWC calculates the **Fisher Information Matrix** to identify which specific weights were most critical for Task 1. While learning Task 2, EWC acts like a rubber band—it allows SGD to easily change "unimportant" weights, but heavily penalizes any changes to the crucial weights needed for Task 1. 

By comparing them, we prove the algorithm works: the Standard SGD graph plummets (showing forgetting), while the EWC graph remains stable, proving it successfully protected the network's prior knowledge.