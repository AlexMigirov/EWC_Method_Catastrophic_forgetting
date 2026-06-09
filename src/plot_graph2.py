import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os
sys.path.append(os.path.dirname(__file__))

from model import MLP
from data import get_dataloaders, get_permutation
from main import train_epoch, evaluate
from ewc import EWC

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = 3
    epochs_per_task = 5
    batch_size = 128
    lr = 1e-3
    ewc_lambda = 400.0

    print(f"Using device: {device}")

    # Generate permutations for the tasks
    permutations = [None] + [get_permutation() for _ in range(1, num_tasks)]
    
    # Loaders for all tasks
    task_loaders = [get_dataloaders(permutation=p, batch_size=batch_size) for p in permutations]
    test_loaders = [loader[1] for loader in task_loaders]
    
    # Models
    model_sgd = MLP().to(device)
    opt_sgd = optim.Adam(model_sgd.parameters(), lr=lr)
    
    model_ewc = MLP().to(device)
    opt_ewc = optim.Adam(model_ewc.parameters(), lr=lr)
    ewc_instances = []

    sgd_avg_accs = []
    ewc_avg_accs = []
    
    epochs_elapsed = []
    current_epoch = 0

    for task_id in range(num_tasks):
        print(f"--- Training on Task {task_id+1}/{num_tasks} ---")
        train_loader = task_loaders[task_id][0]
        
        # We only evaluate the average accuracy over tasks seen SO FAR
        seen_test_loaders = test_loaders[:task_id+1]
        
        for epoch in range(epochs_per_task):
            current_epoch += 1
            epochs_elapsed.append(current_epoch)
            
            # Train SGD baseline
            train_epoch(model_sgd, opt_sgd, train_loader, [], ewc_lambda, device)
            # Evaluate SGD average accuracy on seen tasks
            sgd_accs = [evaluate(model_sgd, loader, device) for loader in seen_test_loaders]
            sgd_avg_accs.append(sum(sgd_accs) / len(sgd_accs))
            
            # Train EWC model
            train_epoch(model_ewc, opt_ewc, train_loader, ewc_instances, ewc_lambda, device)
            # Evaluate EWC average accuracy on seen tasks
            ewc_accs = [evaluate(model_ewc, loader, device) for loader in seen_test_loaders]
            ewc_avg_accs.append(sum(ewc_accs) / len(ewc_accs))
            
            print(f"  Epoch {epoch+1} | Avg Acc (Seen Tasks) - SGD: {sgd_avg_accs[-1]:.4f} | EWC: {ewc_avg_accs[-1]:.4f}")

        # EWC phase: compute the Fisher Information Matrix for the next task
        if task_id < num_tasks - 1:
            print("  Computing Fisher Information Matrix for EWC...")
            model_copy = deepcopy(model_ewc)
            model_copy.eval()
            ewc = EWC(model_copy, train_loader, device)
            ewc_instances.append(ewc)

    # Plotting Graph 2
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_elapsed, [acc * 100 for acc in sgd_avg_accs], label='Standard SGD (Avg)', marker='o', color='red')
    plt.plot(epochs_elapsed, [acc * 100 for acc in ewc_avg_accs], label='EWC (Avg)', marker='s', color='green')
    
    # Add vertical lines for task boundaries
    for i in range(1, num_tasks):
        plt.axvline(x=i * epochs_per_task + 0.5, color='gray', linestyle='--')
        plt.text(i * epochs_per_task + 0.6, 50, f'Start Task {i+1}', rotation=90, color='gray')

    plt.title('Average Performance Across Multiple Sequential Tasks')
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Accuracy on Seen Tasks (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig('results_graph_2_average.png')
    print("Saved graph to results_graph_2_average.png")
    
    return sgd_avg_accs[-1], ewc_avg_accs[-1]

if __name__ == '__main__':
    final_sgd, final_ewc = run_experiment()
    print(f"\nFINAL AVERAGE RESULTS (Across all {3} tasks):")
    print(f"SGD Average Accuracy: {final_sgd:.4f}")
    print(f"EWC Average Accuracy: {final_ewc:.4f}")