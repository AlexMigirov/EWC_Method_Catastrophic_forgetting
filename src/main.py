import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from model import MLP
from data import get_dataloaders, get_permutation

from ewc import EWC

def train_epoch(model, optimizer, data_loader, ewc_instances, ewc_lambda, device):
    """
    Trains the model for one epoch. If ewc_instances is not empty, applies the EWC penalty.
    """
    model.train()
    total_loss = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Standard Cross-Entropy loss for the current task
        loss = F.cross_entropy(output, target)
        
        # Add EWC penalty for all previous tasks
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
    """
    Evaluates the model on the given data loader.
    """
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
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = 3          # E.g., train on 3 sequential tasks
    epochs_per_task = 5    # Number of epochs to train on each task
    batch_size = 128
    lr = 1e-3
    ewc_lambda = 400.0     # Importance weight for the EWC penalty

    print(f"Using device: {device}")

    # Initialize the single MLP model
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Store EWC objects from previous tasks
    ewc_instances = []
    
    # Track test loaders to evaluate forgetting on previous tasks
    test_loaders = []

    for task_id in range(num_tasks):
        print(f"\n--- Starting Task {task_id + 1}/{num_tasks} ---")
        
        # For task 0, use standard MNIST (no permutation or identity permutation)
        # For task > 0, generate a new random permutation
        permutation = get_permutation() if task_id > 0 else None
        
        train_loader, test_loader = get_dataloaders(permutation=permutation, batch_size=batch_size)
        test_loaders.append(test_loader)
        
        # Train on the current task
        for epoch in range(epochs_per_task):
            train_loss = train_epoch(model, optimizer, train_loader, ewc_instances, ewc_lambda, device)
            print(f"Task {task_id + 1} | Epoch {epoch + 1}/{epochs_per_task} | Train Loss: {train_loss:.4f}")
            
        # Evaluate on all tasks seen so far
        print("\nEvaluation after Task", task_id + 1)
        for t_id, t_loader in enumerate(test_loaders):
            accuracy = evaluate(model, t_loader, device)
            print(f"  Accuracy on Task {t_id + 1}: {accuracy * 100:.2f}%")
            
        # Phase 2: After training on the task, prepare for the next task by saving the model state
        # Compute Fisher Information Matrix and store optimal weights via the EWC class
        if task_id < num_tasks - 1:
            print(f"Computing Fisher Information Matrix for Task {task_id + 1}...")
            # We use a copy of the model to freeze the weights at this state for EWC calculation
            model_copy = deepcopy(model)
            model_copy.eval()
            ewc = EWC(model_copy, train_loader, device)
            ewc_instances.append(ewc)
            print("EWC stored. Moving to next task.")

if __name__ == '__main__':
    main()
