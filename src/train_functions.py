# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Tuple

from src.utils import Accuracy


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
    This function trains the model.
    """
    model.train()
    losses = []
    acc = Accuracy(modo)

    for inputs, ner_targets, sa_targets, lengths in train_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs, _ = model(inputs, lengths)
        
        # Calcular pérdida
        if modo == "NER":
            _, _, D = outputs.size()
            loss_ = loss(outputs.view(-1, D).float(), ner_targets.view(-1).long()) # [batch_size, lenght]
            acc.update(outputs, ner_targets)
        else:
            loss_ = loss(outputs.float(), sa_targets.long())  # [batch_size,]
            acc.update(outputs, sa_targets)
        
        # Backward y optimización
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        
        losses.append(loss_.item())

    # Compute accuracy and loss
    accuracy = acc.compute()
    
    writer.add_scalar(f"train/loss_{modo}", np.mean(losses), epoch)
    writer.add_scalar(f"train/accuracy_{modo}", accuracy, epoch)
    
    print(f"Epoch {epoch}: Train Loss {modo} = {np.mean(losses):.4f} | Train Accuracy {modo} = {accuracy:.4f}")


@torch.no_grad()
def val_step(
    modo: str,
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function validates the model.
    """
    model.eval()
    losses = []
    acc = Accuracy(modo)

    for inputs, ner_targets, sa_targets, lengths in val_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs, _ = model(inputs, lengths)
        
        # Calcular pérdida
        if modo == "NER":
            _, _, D = outputs.size()
            loss_ = loss(outputs.view(-1, D).float(), ner_targets.view(-1).long()) # [batch_size, lenght]
            acc.update(outputs, ner_targets)
        else:
            loss_ = loss(outputs.float(), sa_targets.long())  # [batch_size,]
            acc.update(outputs, sa_targets)
        
        losses.append(loss_.item())

    # Compute accuracy and loss
    accuracy = acc.compute()
    
    writer.add_scalar(f"val/loss_{modo}", np.mean(losses), epoch)
    writer.add_scalar(f"val/accuracy_{modo}", accuracy, epoch)
    
    print(f"Epoch {epoch}: Validation Loss {modo} = {np.mean(losses):.4f} | Validation Accuracy {modo} = {accuracy:.4f}")


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
    losses = []
    acc = Accuracy(modo)
    loss = torch.nn.CrossEntropyLoss()

    for inputs, ner_targets, sa_targets, lengths in test_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs, _ = model(inputs, lengths)
        
        # Calcular pérdida
        if modo == "NER":
            _, _, D = outputs.size()
            loss_ = loss(outputs.view(-1, D).float(), ner_targets.view(-1).long()) # [batch_size, lenght]
            acc.update(outputs, ner_targets)
        else:
            loss_ = loss(outputs.float(), sa_targets.long())  # [batch_size,]
            acc.update(outputs, sa_targets)
        
        losses.append(loss_.item())

    # Compute accuracy and loss
    accuracy = acc.compute()
    
    test_loss = np.mean(losses)
    print(f"Test Loss {modo} = {test_loss:.4f} | Test Accuracy {modo} = {accuracy:.4f}")
    
    return test_loss, accuracy


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
    This function trains the model.
    """
    model.train()
    losses_ner = []
    losses_sa = []
    acc_ner = Accuracy("NER")
    acc_sa = Accuracy("SA")

    for inputs, ner_targets, sa_targets, lengths in train_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs, lengths)
        
        # Calcular pérdidas
        loss_ner_ = loss_ner(ner_targets.double(), outputs[0].double())
        loss_sa_ = loss_sa(sa_targets.double(), outputs[1].double())
        
        # Update accuracy
        acc_ner.update(outputs[0], ner_targets)
        acc_sa.update(outputs[1], sa_targets)
        
        # Backward y optimización
        optimizer_ner.zero_grad()
        optimizer_sa.zero_grad()

        loss_ner_.backward()
        loss_sa_.backward()

        optimizer_ner.step()
        optimizer_sa.step()
        
        losses_ner.append(loss_ner_.item())
        losses_sa.append(loss_sa_.item())

    # Compute accuracy and loss for NER and SA
    accuracy_ner = acc_ner.compute()
    accuracy_sa = acc_sa.compute()

    writer.add_scalar("train/loss_ner", np.mean(losses_ner), epoch)
    writer.add_scalar("train/loss_sa", np.mean(losses_sa), epoch)
    writer.add_scalar("train/accuracy_ner", accuracy_ner, epoch)
    writer.add_scalar("train/accuracy_sa", accuracy_sa, epoch)

    print(f"Epoch {epoch}: Train Loss NER = {np.mean(losses_ner):.4f} | Train Accuracy NER = {accuracy_ner:.4f}")
    print(f"Epoch {epoch}: Train Loss SA = {np.mean(losses_sa):.4f} | Train Accuracy SA = {accuracy_sa:.4f}")


@torch.no_grad()
def val_step_nersa(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss_ner: torch.nn.Module,
    loss_sa: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function validates the model.
    """
    model.eval()
    losses_ner = []
    losses_sa = []
    acc_ner = Accuracy("NER")
    acc_sa = Accuracy("SA")

    for inputs, ner_targets, sa_targets in val_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calcular pérdidas
        loss_ner_ = loss_ner(ner_targets, outputs[0])
        loss_sa_ = loss_sa(sa_targets, outputs[1])

        # Update accuracy
        acc_ner.update(outputs[0], ner_targets)
        acc_sa.update(outputs[1], sa_targets)

        losses_ner.append(loss_ner_.item())
        losses_sa.append(loss_sa_.item())

    # Compute accuracy and loss for NER and SA
    accuracy_ner = acc_ner.compute()
    accuracy_sa = acc_sa.compute()

    writer.add_scalar("val/loss_ner", np.mean(losses_ner), epoch)
    writer.add_scalar("val/loss_sa", np.mean(losses_sa), epoch)
    writer.add_scalar("val/accuracy_ner", accuracy_ner, epoch)
    writer.add_scalar("val/accuracy_sa", accuracy_sa, epoch)

    print(f"Epoch {epoch}: Validation Loss NER = {np.mean(losses_ner):.4f} | Validation Accuracy NER = {accuracy_ner:.4f}")
    print(f"Epoch {epoch}: Validation Loss SA = {np.mean(losses_sa):.4f} | Validation Accuracy SA = {accuracy_sa:.4f}")


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
    acc_ner = Accuracy("NER")
    acc_sa = Accuracy("SA")
    loss_ner = torch.nn.CrossEntropyLoss()
    loss_sa = torch.nn.CrossEntropyLoss()

    for inputs, ner_targets, sa_targets in test_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calcular pérdidas
        loss_ner_ = loss_ner(ner_targets, outputs[0])
        loss_sa_ = loss_sa(sa_targets, outputs[1])

        # Update accuracy
        acc_ner.update(outputs[0], ner_targets)
        acc_sa.update(outputs[1], sa_targets)

        losses_ner.append(loss_ner_.item())
        losses_sa.append(loss_sa_.item())

    # Compute accuracy and loss for NER and SA
    accuracy_ner = acc_ner.compute()
    accuracy_sa = acc_sa.compute()

    test_ner = np.mean(losses_ner)
    test_sa = np.mean(losses_sa)

    print(f"Test Loss NER: {test_ner:.4f} | Test Accuracy NER: {accuracy_ner:.4f}")
    print(f"Test Loss SA: {test_sa:.4f} | Test Accuracy SA: {accuracy_sa:.4f}")
    
    return test_ner, test_sa
