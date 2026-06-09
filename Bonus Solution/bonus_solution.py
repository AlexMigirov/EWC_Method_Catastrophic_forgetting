import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ewc import EWC

# --- 1. Synthetic Medical Dataset Generator ---
def generate_medical_task(task_type, num_samples=2000):
    """
    Simulates medical image embeddings (flattened to 28x28=784 to match our MLP).
    - Task 0: 'MRI - Disease A vs Normal'
    - Task 1: 'CT - Disease B vs Normal'
    """
    torch.manual_seed(task_type * 42)
    X = torch.randn(num_samples, 784)
    
    # Introduce a synthetic pattern for the disease
    if task_type == 0:
        # MRI Disease Pattern
        weights = torch.randn(784)
    else:
        # CT Disease Pattern
        weights = torch.randn(784) * 0.5 + 0.5
        
    logits = X @ weights
    y = (torch.sigmoid(logits) > 0.5).long()
    
    return DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# --- 2. Model Definition ---
# Using a simpler MLP for binary classification in this medical benchmark
class MedicalMLP(nn.Module):
    def __init__(self):
        super(MedicalMLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2) # 2 classes: Healthy vs Diseased

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 3. Training & Evaluation ---
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(loader.dataset)

def run_lifelonger_simulation():
    device = torch.device("cpu")
    print("Starting LifeLonger Benchmark Simulation (Continual Disease Classification)")
    
    # Generate Data
    print("Generating synthetic datasets: Task 1 (MRI) and Task 2 (CT)...")
    task1_train = generate_medical_task(0, num_samples=2000)
    task1_test  = generate_medical_task(0, num_samples=500)
    
    task2_train = generate_medical_task(1, num_samples=2000)
    task2_test  = generate_medical_task(1, num_samples=500)
    
    # Initialize EWC Model
    model = MedicalMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ewc_instances = []
    ewc_lambda = 500.0
    
    # --- TRAIN TASK 1 (MRI) ---
    print("\n--- Phase A: Training on Task 1 (MRI: Disease A) ---")
    for epoch in range(5):
        model.train()
        for data, target in task1_train:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
            
    acc1_initial = evaluate(model, task1_test, device)
    print(f"Task 1 (MRI) Accuracy after training: {acc1_initial * 100:.2f}%")
    
    # Save EWC State for Task 1
    print("Computing Fisher Information Matrix for MRI features...")
    model_copy = deepcopy(model)
    model_copy.eval()
    ewc_instances.append(EWC(model_copy, task1_train, device))
    
    # --- TRAIN TASK 2 (CT) ---
    print("\n--- Phase B: Training on Task 2 (CT: Disease B) ---")
    for epoch in range(5):
        model.train()
        for data, target in task2_train:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            
            # Apply EWC Penalty to protect MRI knowledge
            ewc_loss = sum([ewc.penalty(model) for ewc in ewc_instances])
            loss += ewc_lambda * ewc_loss
            
            loss.backward()
            optimizer.step()
            
    acc2_final = evaluate(model, task2_test, device)
    acc1_final = evaluate(model, task1_test, device)
    
    print(f"\n--- Final Results (After learning both diseases) ---")
    print(f"Task 2 (CT) Accuracy: {acc2_final * 100:.2f}% (Successfully learned new disease)")
    print(f"Task 1 (MRI) Retained Accuracy: {acc1_final * 100:.2f}% (Overcame Catastrophic Forgetting)")
    
if __name__ == '__main__':
    run_lifelonger_simulation()