import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
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

    # Generate permutations for the tasks so both models get the exact same tasks
    permutations = [None] + [get_permutation() for _ in range(1, num_tasks)]
    
    # Loaders for the tasks
    task_loaders = [get_dataloaders(permutation=p, batch_size=batch_size) for p in permutations]
    
    # Model 1: SGD Baseline
    model_sgd = MLP().to(device)
    opt_sgd = optim.Adam(model_sgd.parameters(), lr=lr)
    
    # Model 2: EWC
    model_ewc = MLP().to(device)
    opt_ewc = optim.Adam(model_ewc.parameters(), lr=lr)
    ewc_instances = []

    # To track accuracy on Task 1 specifically over time
    sgd_accs = []
    ewc_accs = []
    
    epochs_elapsed = []
    current_epoch = 0

    task1_test_loader = task_loaders[0][1]

    for task_id in range(num_tasks):
        print(f"--- Training on Task {task_id+1}/{num_tasks} ---")
        train_loader = task_loaders[task_id][0]
        
        for epoch in range(epochs_per_task):
            current_epoch += 1
            epochs_elapsed.append(current_epoch)
            
            # Train SGD baseline
            train_epoch(model_sgd, opt_sgd, train_loader, [], ewc_lambda, device)
            # Evaluate SGD on Task 1
            sgd_accs.append(evaluate(model_sgd, task1_test_loader, device))
            
            # Train EWC model
            train_epoch(model_ewc, opt_ewc, train_loader, ewc_instances, ewc_lambda, device)
            # Evaluate EWC on Task 1
            ewc_accs.append(evaluate(model_ewc, task1_test_loader, device))
            
            print(f"  Epoch {epoch+1} | Task 1 Acc - SGD: {sgd_accs[-1]:.4f} | EWC: {ewc_accs[-1]:.4f}")

        # EWC phase: compute the Fisher Information Matrix for the next task
        if task_id < num_tasks - 1:
            print("  Computing Fisher Information Matrix for EWC...")
            model_copy = deepcopy(model_ewc)
            model_copy.eval()
            ewc = EWC(model_copy, train_loader, device)
            ewc_instances.append(ewc)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_elapsed, [acc * 100 for acc in sgd_accs], label='Standard SGD', marker='o')
    plt.plot(epochs_elapsed, [acc * 100 for acc in ewc_accs], label='EWC', marker='s')
    
    # Add vertical lines for task boundaries
    for i in range(1, num_tasks):
        plt.axvline(x=i * epochs_per_task + 0.5, color='gray', linestyle='--')
        plt.text(i * epochs_per_task + 0.6, 50, f'Start Task {i+1}', rotation=90, color='gray')

    plt.title('Performance on Task 1 while sequentially learning new tasks')
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy on Task 1 (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig('results_graph.png')
    print("Saved graph to results_graph.png")
    
    return sgd_accs[-1], ewc_accs[-1]

if __name__ == '__main__':
    final_sgd, final_ewc = run_experiment()
    print(f"\nFINAL RESULTS:")
    print(f"SGD Task 1 Accuracy: {final_sgd:.4f}")
    print(f"EWC Task 1 Accuracy: {final_ewc:.4f}")