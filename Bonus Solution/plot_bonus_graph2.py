import sys
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Append src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data import get_dataloaders, get_permutation
from main import evaluate
from improvement import ImprovedMLP, OnlineEWC

def plot_improved_average_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = 3
    epochs_per_task = 5
    batch_size = 128
    lr = 1e-3
    ewc_lambda = 5000.0

    print("Running experiment to generate Graph 2 (Improved Online EWC Average)...")
    
    # Generate same permutations as standard
    torch.manual_seed(42)
    permutations = [None] + [get_permutation() for _ in range(1, num_tasks)]
    task_loaders = [get_dataloaders(permutation=p, batch_size=batch_size) for p in permutations]
    test_loaders = [loader[1] for loader in task_loaders]

    # --- Train Standard SGD (Baseline for comparison) ---
    sgd_model = ImprovedMLP().to(device)
    sgd_optimizer = optim.Adam(sgd_model.parameters(), lr=lr)
    sgd_avg_accs = []

    for task_id in range(num_tasks):
        train_loader = task_loaders[task_id][0]
        for epoch in range(epochs_per_task):
            sgd_model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                sgd_optimizer.zero_grad()
                output = sgd_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                sgd_optimizer.step()
            
            # Evaluate average accuracy across all tasks seen SO FAR
            accs = [evaluate(sgd_model, test_loaders[i], device) for i in range(task_id + 1)]
            avg_acc = sum(accs) / len(accs)
            sgd_avg_accs.append(avg_acc * 100)

    # --- Train Improved Online EWC ---
    ewc_model = ImprovedMLP().to(device)
    ewc_optimizer = optim.Adam(ewc_model.parameters(), lr=lr)
    online_ewc = OnlineEWC(ewc_model, device, gamma=1.0)
    ewc_avg_accs = []
    has_previous_task = False

    for task_id in range(num_tasks):
        train_loader = task_loaders[task_id][0]
        for epoch in range(epochs_per_task):
            ewc_model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                ewc_optimizer.zero_grad()
                output = ewc_model(data)
                loss = F.cross_entropy(output, target)
                
                if has_previous_task:
                    loss += ewc_lambda * online_ewc.penalty(ewc_model)
                    
                loss.backward()
                ewc_optimizer.step()
                
            # Evaluate average accuracy across all tasks seen SO FAR
            accs = [evaluate(ewc_model, test_loaders[i], device) for i in range(task_id + 1)]
            avg_acc = sum(accs) / len(accs)
            ewc_avg_accs.append(avg_acc * 100)

        online_ewc.update_fisher(train_loader)
        has_previous_task = True

    # --- Plotting ---
    epochs = list(range(1, num_tasks * epochs_per_task + 1))
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, sgd_avg_accs, marker='o', label='Standard SGD (ImprovedMLP)')
    plt.plot(epochs, ewc_avg_accs, marker='s', label='Online EWC (ImprovedMLP)')

    plt.axvline(x=epochs_per_task + 0.5, color='gray', linestyle='--')
    plt.text(epochs_per_task + 0.6, 50, 'Start Task 2', rotation=90, color='gray')
    plt.axvline(x=2 * epochs_per_task + 0.5, color='gray', linestyle='--')
    plt.text(2 * epochs_per_task + 0.6, 50, 'Start Task 3', rotation=90, color='gray')

    plt.title('Average Performance Across Sequential Tasks (Improved)')
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bonus_results_graph_2_average.png'))
    plt.savefig(save_path)
    print(f"Graph 2 saved to: {save_path}")

if __name__ == '__main__':
    plot_improved_average_accuracy()
