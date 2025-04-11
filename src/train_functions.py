# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional, Tuple


# NER o SA
@torch.enable_grad()
def train_step(
    modo: str,
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.
    """
    model.train()
    losses = []

    for inputs, ner_targets, sa_targets, lengths in train_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs, lengths)
        
        # Calcular pérdida
        if modo == "NER":
            idx = 0
            targets = ner_targets
        else:
            idx = 1
            targets = sa_targets
        loss_ = loss(targets, outputs[idx])
        
        # Backward y optimización
        optimizer.zero_grad()

        loss_.backward()

        optimizer.step()
        
        losses.append(loss_.item())
    
    writer.add_scalar(f"train/loss_{modo}", np.mean(losses), epoch)
    print(f"Epoch {epoch}: Train Loss {modo} = {np.mean(losses):.4f}")

@torch.no_grad()
def val_step(
    modo: str,
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    #scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.
    """
    model.eval()
    losses = []

    for inputs, ner_targets, sa_targets, lengths in val_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs, lengths)
        
        # Calcular pérdida
        if modo == "NER":
            idx = 0
            targets = ner_targets
        else:
            idx = 1
            targets = sa_targets
        loss_ = loss(targets, outputs[idx])
        
        losses.append(loss_.item())
    
    writer.add_scalar(f"val/loss_{modo}", np.mean(losses), epoch)
    print(f"Epoch {epoch}: Validation Loss {modo} = {np.mean(losses):.4f}")

@torch.no_grad()
def t_step(
    modo: str,
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    This function tests the model.
    """
    model.eval()
    loss = torch.nn.CrossEntropyLoss()
    losses = []

    for inputs, ner_targets, sa_targets, lengths in test_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs, lengths)
        
        # Calcular pérdida
        if modo == "NER":
            idx = 0
            targets = ner_targets
        else:
            idx = 1
            targets = sa_targets
        loss_ = loss(targets, outputs[idx])
        
        losses.append(loss_.item())
    
    test = np.mean(losses)
    print(f"Test Loss {modo} = {test:.4f}")
    return test


# NER y SA
@torch.enable_grad()
def train_step_nersa(
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

    # -------------------------------- lengths es necesario para que el modelo sepa la longit de la cadena/frase    
    for inputs, ner_targets, sa_targets, lengths in train_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs, lengths)
        
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
def val_step_nersa(
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
def t_step_nersa(
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
