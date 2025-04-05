# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional, Tuple


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss_ner: torch.nn.Module,
    loss_sa: torch.nn.Module,
    optimizer_ner: torch.optim.Optimizer,
    optimizer_sa: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.
    """
    model.train()
    losses_ner = []
    losses_sa = []
    
    for inputs, ner_targets, sa_targets in train_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calcular pérdida
        loss_ner_ = loss_ner(ner_targets, outputs[0])
        loss_sa_ = loss_sa(sa_targets, outputs[1])
        
        # Backward y optimización
        optimizer_ner.zero_grad()
        optimizer_sa.zero_grad()

        loss_ner_.backward()
        loss_sa_.backward()

        optimizer_ner.step()
        optimizer_sa.step()
        
        losses_ner.append(loss_ner_.item())
        losses_sa.append(loss_sa_.item())
    
    writer.add_scalar("train/loss_ner", np.mean(losses_ner), epoch)
    writer.add_scalar("train/loss_sa", np.mean(losses_sa), epoch)
    print(f"Epoch {epoch}: Train Loss NER = {np.mean(losses_ner):.4f} | Train Loss SA = {np.mean(losses_sa):.4f}")


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss_ner: torch.nn.Module,
    loss_sa: torch.nn.Module,
    #scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.
    """
    model.eval()
    losses_ner = []
    losses_sa = []
    
    for inputs, ner_targets, sa_targets in val_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calcular pérdida
        loss_ner_ = loss_ner(ner_targets, outputs[0])
        loss_sa_ = loss_sa(sa_targets, outputs[1])

        losses_ner.append(loss_ner_.item())
        losses_sa.append(loss_sa_.item())
    
    writer.add_scalar("val/loss_ner", np.mean(losses_ner), epoch)
    writer.add_scalar("val/loss_sa", np.mean(losses_sa), epoch)
    print(f"Epoch {epoch}: Train Loss NER = {np.mean(losses_ner):.4f} | Train Loss SA = {np.mean(losses_sa):.4f}")


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    This function tests the model.
    """
    model.eval()
    losses_ner = []
    losses_sa = []
    loss_ner = torch.nn.CrossEntropyLoss()
    loss_sa = torch.nn.CrossEntropyLoss()
    
    for inputs, ner_targets, sa_targets in test_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calcular pérdida
        loss_ner_ = loss_ner(ner_targets, outputs[0])
        loss_sa_ = loss_sa(sa_targets, outputs[1])

        losses_ner.append(loss_ner_.item())
        losses_sa.append(loss_sa_.item())
    
    test_ner = np.mean(losses_ner)
    test_sa = np.mean(losses_sa)

    print(f"Test NER: {test_ner:.4f} | Test SA: {test_sa:.4f}")
    return test_ner, test_sa
