import torch


def binary_log_loss(logits, labels, normalize=True):
    """
    Binary cross entropy loss

    Args:
        logits: torch.Tensor
        labels: torch.Tensor
        normalize: bool

    Returns:
        torch.Tensor

    """
    if normalize:
        dummy_logits = torch.zeros_like(logits)
        dummy_labels = torch.zeros_like(labels)
        return torch.mean(
            binary_log_loss(logits, labels, normalize=False) /
            binary_log_loss(dummy_logits, dummy_labels, normalize=False)
        )
    else:
        return -torch.mean(labels * torch.log(logits) + (1 - labels) * torch.log(1 - logits))
    

def log_loss(logits, labels, normalize=True):
    """
    Cross entropy loss

    Args:
        logits: torch.Tensor
        labels: torch.Tensor
        normalize: bool

    Returns:
        torch.Tensor

    """
    if normalize:
        dummy_logits = torch.zeros_like(logits)
        dummy_labels = torch.zeros_like(labels)
        return torch.mean(
            log_loss(logits, labels, normalize=False) /
            log_loss(dummy_logits, dummy_labels, normalize=False)
        )
    else:
        return -torch.mean(labels * torch.log(logits))
    

def brier_score(logits, labels, normalize=True):
    """
    Brier score

    Args:
        logits: torch.Tensor
        labels: torch.Tensor
        normalize: bool

    Returns:
        torch.Tensor
        
    """
    if normalize:
        dummy_logits = torch.zeros_like(logits)
        dummy_labels = torch.zeros_like(labels)
        return torch.mean(
            brier_score(logits, labels, normalize=False) /
            brier_score(dummy_logits, dummy_labels, normalize=False)
        )
    else:
        return torch.mean((logits - labels) ** 2)

