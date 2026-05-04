import torch
import torch.nn.functional as F
import copy

class EWC:
    """
    Elastic Weight Consolidation (EWC) implementation.
    Calculates the Fisher Information Matrix (diagonal) and computes the penalty loss.
    """
    def __init__(self, model, dataloader, device):
        """
        Initializes EWC by calculating and storing the optimal weights (theta_A)
        and the diagonal Fisher Information Matrix.
        
        Args:
            model (nn.Module): The model trained on the previous task.
            dataloader (DataLoader): The data loader for the previous task (used to compute Fisher).
            device (torch.device): Device to perform computations on.
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        # Extract optimal parameters from the trained model
        self.params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
        # Calculate the diagonal Fisher Information Matrix
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        """
        Computes the diagonal approximation of the Fisher Information Matrix.
        
        Returns:
            dict: A dictionary mapping parameter names to their Fisher diagonal tensors.
        """
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        
        # We process the data to accumulate the squared gradients
        # Use a subset of the data if the dataset is very large, but here we use the whole loader
        num_samples = 0
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # EWC calculates Fisher using the empirical Fisher matrix or the expected Fisher.
            # Here we use the standard negative log-likelihood (CrossEntropy) gradients.
            # To compute empirical Fisher: F = sum (grad_log_likelihood)^2
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Accumulate squared gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 * data.size(0)
            
            num_samples += data.size(0)

        # Average over the number of samples
        for n in fisher.keys():
            fisher[n] /= num_samples
            
        return fisher

    def penalty(self, current_model):
        """
        Computes the EWC penalty for the current model given the saved parameters and Fisher matrix.
        
        Args:
            current_model (nn.Module): The model currently being trained on the new task.
            
        Returns:
            torch.Tensor: The calculated penalty term to be added to the loss.
        """
        loss = 0
        for n, p in current_model.named_parameters():
            if n in self.fisher:
                # EWC penalty formula: sum( (fisher_i / 2) * (theta_i - theta_A,i)^2 )
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss
