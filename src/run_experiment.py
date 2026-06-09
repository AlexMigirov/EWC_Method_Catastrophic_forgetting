import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import matplotlib.pyplot as plt
import os

from model import MLP
from data import get_dataloaders, get_permutation
from ewc import EWC

def train_epoch(model, optimizer, data_loader, ewc_instances, ewc_lambda, device):
    model.train()
    total_loss = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.cross_entropy(output, target)
        
        if ewc_instances:
            ewc_loss = 0
            for ewc in ewc_instances:
                ewc_loss += ewc.penalty(model)
            loss += ewc_lambda * ewc_loss
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    return correct / len(data_loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = 3
    epochs_per_task = 5
    batch_size = 128
    lr = 1e-3
    ewc_lambda = 400.0

    print(f"Using device: {device}")

    # To ensure fairness, we initialize a single base model and copy its weights
    base_model = MLP().to(device)
    
    # Model 1: Standard SGD (No EWC)
    model_sgd = deepcopy(base_model)
    optimizer_sgd = optim.Adam(model_sgd.parameters(), lr=lr)
    
    # Model 2: EWC
    model_ewc = deepcopy(base_model)
    optimizer_ewc = optim.Adam(model_ewc.parameters(), lr=lr)
    ewc_instances = []

    # Store dataloaders
    train_loaders = []
    test_loaders = []

    for task_id in range(num_tasks):
        permutation = get_permutation() if task_id > 0 else None
        train_loader, test_loader = get_dataloaders(permutation=permutation, batch_size=batch_size, data_dir='../data')
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    task1_acc_sgd = []
    task1_acc_ewc = []
    
    epochs_x = []
    current_epoch = 0

    print("--- Training ---")
    for task_id in range(num_tasks):
        print(f"\nStarting Task {task_id + 1}/{num_tasks}")
        
        for epoch in range(epochs_per_task):
            current_epoch += 1
            epochs_x.append(current_epoch)
            
            # Train SGD
            train_epoch(model_sgd, optimizer_sgd, train_loaders[task_id], [], 0, device)
            # Train EWC
            train_epoch(model_ewc, optimizer_ewc, train_loaders[task_id], ewc_instances, ewc_lambda, device)
            
            # Evaluate Task 1
            acc_sgd = evaluate(model_sgd, test_loaders[0], device)
            acc_ewc = evaluate(model_ewc, test_loaders[0], device)
            
            task1_acc_sgd.append(acc_sgd * 100)
            task1_acc_ewc.append(acc_ewc * 100)
            
            print(f"Epoch {current_epoch} | Task 1 Acc: SGD={acc_sgd*100:.2f}%, EWC={acc_ewc*100:.2f}%")
        
        # After completing the task, compute Fisher for EWC model
        if task_id < num_tasks - 1:
            print(f"Computing Fisher for Task {task_id + 1}...")
            model_ewc_copy = deepcopy(model_ewc)
            model_ewc_copy.eval()
            ewc = EWC(model_ewc_copy, train_loaders[task_id], device)
            ewc_instances.append(ewc)

    # Final evaluation on all tasks
    print("\n--- Final Evaluation ---")
    final_acc_sgd = []
    final_acc_ewc = []
    for t_id, t_loader in enumerate(test_loaders):
        acc_sgd = evaluate(model_sgd, t_loader, device)
        acc_ewc = evaluate(model_ewc, t_loader, device)
        final_acc_sgd.append(acc_sgd * 100)
        final_acc_ewc.append(acc_ewc * 100)
        print(f"Task {t_id + 1} Final Accuracy:")
        print(f"  SGD: {acc_sgd * 100:.2f}%")
        print(f"  EWC: {acc_ewc * 100:.2f}%")
        
    print(f"\nFINAL_SGD: {final_acc_sgd}")
    print(f"FINAL_EWC: {final_acc_ewc}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_x, task1_acc_sgd, label="Standard SGD (No penalty)", marker='o', linestyle='dashed')
    plt.plot(epochs_x, task1_acc_ewc, label="EWC", marker='s')
    
    # Add vertical lines to indicate task boundaries
    for i in range(1, num_tasks):
        plt.axvline(x=i * epochs_per_task + 0.5, color='gray', linestyle='--', alpha=0.7)
        plt.text(i * epochs_per_task + 0.5, 50, f' Start Task {i+1}', rotation=90, verticalalignment='center', color='gray')

    plt.title("Catastrophic Forgetting on Task 1 (Permuted MNIST)")
    plt.xlabel("Total Training Epochs")
    plt.ylabel("Test Accuracy on Task 1 (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save in project root
    plt.savefig('../results_graph.png')
    print("Graph saved to ../results_graph.png")

if __name__ == '__main__':
    main()
