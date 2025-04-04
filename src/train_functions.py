# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional


@torch.enable_grad()
def train_step(
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

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    model.train()
    losses = []
    
    for inputs, targets in train_data:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calcular pérdida MAE
        loss_ = loss(outputs, targets)
        
        # Backward y optimización
        optimizer.zero_grad()
        loss_.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        losses.append(loss_.item())
    
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    print(f"Epoch {epoch}: Train Loss = {np.mean(losses):.4f}")


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    model.eval()
    losses = []
    
    for inputs, targets in val_data:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calcular pérdida MAE
        loss_ = loss(outputs, targets)
        losses.append(loss_.item())
    
    writer.add_scalar("val/loss", np.mean(losses), epoch)
    print(f"Epoch {epoch}: Val Loss = {np.mean(losses):.4f}")


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        mean: mean of the target.
        std: std of the target.
        device: device for running operations.

    Returns:
        mae of the test data.
    """
    model.eval()
    losses = []
    loss_fn = torch.nn.L1Loss()  # MAE
    
    for inputs, targets in test_data:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calcular pérdida MAE
        loss_ = loss_fn(outputs, targets)
        losses.append(loss_.item())
    
    test_mae = np.mean(losses)
    test_mae = test_mae  # Desnormalizar la pérdida
    print(f"Test MAE: {test_mae:.4f}")
    return test_mae
