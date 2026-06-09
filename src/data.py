import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_permutation():
    """
    Generates a random permutation of 784 indices (for a 28x28 image).
    Each task in Permuted MNIST uses a different permutation.
    Returns:
        torch.Tensor: A tensor containing a random permutation of integers from 0 to 783.
    """
    return torch.randperm(784)

class PermutedMNISTTransform:
    """
    A custom transform that applies a specific permutation to the flattened image,
    and then reshapes it back to 28x28 so standard transforms can still chain if needed.
    """
    def __init__(self, permutation=None):
        # If no permutation is provided, use the identity permutation (standard MNIST)
        self.permutation = permutation if permutation is not None else torch.arange(784)

    def __call__(self, tensor):
        # Flatten, apply permutation, and reshape back to 1x28x28
        flat_tensor = tensor.view(-1)
        permuted_tensor = flat_tensor[self.permutation]
        return permuted_tensor.view(1, 28, 28)

def get_dataloaders(permutation=None, batch_size=128, data_dir='./data'):
    """
    Creates DataLoaders for the Permuted MNIST dataset given a specific permutation.
    
    Args:
        permutation (torch.Tensor, optional): The permutation to apply. Defaults to None.
        batch_size (int): Batch size for the dataloaders.
        data_dir (str): Directory to store/download the dataset.
        
    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the testing set.
    """
    # Define the sequence of transformations: ToTensor scales pixels to [0, 1]
    # PermutedMNISTTransform applies the task-specific permutation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # Standard MNIST normalization
        PermutedMNISTTransform(permutation)
    ])

    # Load train and test datasets
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
