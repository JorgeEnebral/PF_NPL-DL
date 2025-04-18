# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Tuple
from src.utils import Accuracy


@torch.enable_grad()
def train_step_sa(model, train_data, loss_fn, optimizer, writer, epoch, device):
    model.train()
    losses = []
    acc = Accuracy("SA")

    for inputs, _, sa_targets, lengths in train_data:
        inputs, sa_targets = inputs.to(device), sa_targets.to(device)
        outputs, _ = model(inputs, lengths)
        loss_ = loss_fn(outputs.float(), sa_targets.long())

        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        acc.update(outputs, sa_targets)
        losses.append(loss_.item())

    acc_val, _ = acc.compute()
    writer.add_scalar("train/loss_SA", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy_SA", acc_val, epoch)
    print(f"Epoch {epoch}: Train Loss SA = {np.mean(losses):.4f} | Train Accuracy = {acc_val:.4f}")


@torch.enable_grad()
def train_step_ner(model, train_data, loss_fn_ner1, loss_fn_ner2, optimizer, writer, epoch, device):
    model.train()
    losses = []
    acc = Accuracy("NER")
    loss_ponderation = model.loss_ponderation.tolist()

    for inputs, ner_targets, _, lengths in train_data:
        inputs, ner_targets = inputs.to(device), ner_targets.to(device)
        outputs, _ = model(inputs, lengths)
        _, _, D = outputs.size()

        mask = ner_targets == 0
        mask_ent = (~mask) & (ner_targets != -1)
        loss1 = loss_fn_ner1(outputs[mask].view(-1, D).float(), ner_targets[mask].view(-1).long())
        if mask_ent.sum() > 0:
            loss2 = loss_fn_ner2(outputs[mask_ent].view(-1, D).float(), ner_targets[mask_ent].view(-1).long())
        else:
            # loss2 = loss_fn_ner2(torch.tensor([0.0]), torch.tensor([0.0]))
            loss2 = torch.tensor(0.0, device=device, requires_grad=True)
        total_loss = loss_ponderation[0] * loss1 + loss_ponderation[1] * loss2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        acc.update(outputs, ner_targets)
        losses.append(total_loss.item())

    acc_val, acc_out = acc.compute()
    writer.add_scalar("train/loss_NER", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy_NER", acc_val, epoch)
    writer.add_scalar("train/accuracy_NER_out", acc_out, epoch)
    print(f"Epoch {epoch}: Train Loss NER = {np.mean(losses):.4f} | Train Accuracy = {acc_val:.4f} | Out Accuracy = {acc_out:.4f}")


@torch.enable_grad()
def train_step_nersa(model, train_data, loss_fn_ner1, loss_fn_ner2, loss_fn_sa, opt_ner, opt_sa, writer, epoch, device):
    model.train()
    losses_ner, losses_sa = [], []
    acc_ner, acc_sa = Accuracy("NER"), Accuracy("SA")
    loss_ponderation = model.loss_ponderation.tolist()

    for inputs, ner_targets, sa_targets, lengths in train_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        out_ner, out_sa = model(inputs, lengths)

        _, _, D = out_ner.size()
        mask = ner_targets == 0
        mask_ent = (~mask) & (ner_targets != -1)
        loss1 = loss_fn_ner1(out_ner[mask].view(-1, D).float(), ner_targets[mask].view(-1).long())
        if mask_ent.sum() > 0:
            loss2 = loss_fn_ner2(out_ner[mask_ent].view(-1, D).float(), ner_targets[mask_ent].view(-1).long())
        else:
            loss2 = torch.tensor(0.0, device=device, requires_grad=True)
        loss_sa = loss_fn_sa(out_sa.float(), sa_targets.long())

        total_loss = loss_ponderation[0] * loss1 + loss_ponderation[1] * loss2 + loss_ponderation[2] * loss_sa

        opt_ner.zero_grad()
        opt_sa.zero_grad()
        total_loss.backward()
        opt_ner.step()
        opt_sa.step()

        acc_ner.update(out_ner, ner_targets)
        acc_sa.update(out_sa, sa_targets)
        losses_ner.append((loss_ponderation[0] * loss1 + loss_ponderation[1] * loss2).item())
        losses_sa.append(loss_ponderation[2] * loss_sa.item())

    acc_ner_val, acc_ner_out = acc_ner.compute()
    acc_sa_val, _ = acc_sa.compute()
    writer.add_scalar("train/loss_NER", np.mean(losses_ner), epoch)
    writer.add_scalar("train/loss_SA", np.mean(losses_sa), epoch)
    writer.add_scalar("train/accuracy_NER", acc_ner_val, epoch)
    writer.add_scalar("train/accuracy_NER_out", acc_ner_out, epoch)
    writer.add_scalar("train/accuracy_SA", acc_sa_val, epoch)
    print(f"Epoch {epoch}: Train Loss NER = {np.mean(losses_ner):.4f} | Accuracy NER = {acc_ner_val:.4f} | Out = {acc_ner_out:.4f}")
    print(f"Epoch {epoch}: Train Loss SA = {np.mean(losses_sa):.4f} | Accuracy SA = {acc_sa_val:.4f}")



@torch.no_grad()
def val_step_sa(model, val_data, loss_fn, writer, epoch, device):
    model.eval()
    losses = []
    acc = Accuracy("SA")

    for inputs, _, sa_targets, lengths in val_data:
        inputs, sa_targets = inputs.to(device), sa_targets.to(device)
        outputs, _ = model(inputs, lengths)
        loss_ = loss_fn(outputs.float(), sa_targets.long())

        acc.update(outputs, sa_targets)
        losses.append(loss_.item())

    acc_val, _ = acc.compute()
    writer.add_scalar("val/loss_SA", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy_SA", acc_val, epoch)
    print(f"Epoch {epoch}: Validation Loss SA = {np.mean(losses):.4f} | Accuracy = {acc_val:.4f}")


@torch.no_grad()
def val_step_ner(model, val_data, loss_fn_ner1, loss_fn_ner2, writer, epoch, device):
    model.eval()
    losses = []
    acc = Accuracy("NER")
    loss_ponderation = model.loss_ponderation.tolist()

    for inputs, ner_targets, _, lengths in val_data:
        inputs, ner_targets = inputs.to(device), ner_targets.to(device)
        outputs, _ = model(inputs, lengths)
        _, _, D = outputs.size()

        mask = ner_targets == 0
        mask_ent = (~mask) & (ner_targets != -1)
        loss1 = loss_fn_ner1(outputs[mask].view(-1, D).float(), ner_targets[mask].view(-1).long())
        if mask_ent.sum() > 0:
            loss2 = loss_fn_ner2(outputs[mask_ent].view(-1, D).float(), ner_targets[mask_ent].view(-1).long())
        else:
            loss2 = torch.tensor(0.0, device=device, requires_grad=True)
        total_loss = loss_ponderation[0] * loss1 + loss_ponderation[1] * loss2

        acc.update(outputs, ner_targets)
        losses.append(total_loss.item())

    acc_val, acc_out = acc.compute()
    writer.add_scalar("val/loss_NER", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy_NER", acc_val, epoch)
    writer.add_scalar("val/accuracy_NER_out", acc_out, epoch)
    print(f"Epoch {epoch}: Validation Loss NER = {np.mean(losses):.4f} | Accuracy = {acc_val:.4f} | Out = {acc_out:.4f}")


@torch.no_grad()
def val_step_nersa(model, val_data, loss_fn_ner1, loss_fn_ner2, loss_fn_sa, writer, epoch, device):
    model.eval()
    losses_ner, losses_sa = [], []
    acc_ner, acc_sa = Accuracy("NER"), Accuracy("SA")
    loss_ponderation = model.loss_ponderation.tolist()

    for inputs, ner_targets, sa_targets, lengths in val_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        out_ner, out_sa = model(inputs, lengths)
        _, _, D = out_ner.size()
        
        mask = ner_targets == 0
        mask_ent = (~mask) & (ner_targets != -1)
        loss1 = loss_fn_ner1(out_ner[mask].view(-1, D).float(), ner_targets[mask].view(-1).long())
        if mask_ent.sum() > 0:
            loss2 = loss_fn_ner2(out_ner[mask_ent].view(-1, D).float(), ner_targets[mask_ent].view(-1).long())
        else:
            loss2 = torch.tensor(0.0, device=device, requires_grad=True)
        loss_sa = loss_fn_sa(out_sa.float(), sa_targets.long())

        acc_ner.update(out_ner, ner_targets)
        acc_sa.update(out_sa, sa_targets)
        losses_ner.append((loss_ponderation[0] * loss1 + loss_ponderation[1] * loss2).item())
        losses_sa.append(loss_ponderation[2] * loss_sa.item())

    acc_ner_val, acc_ner_out = acc_ner.compute()
    acc_sa_val, _ = acc_sa.compute()
    writer.add_scalar("val/loss_NER", np.mean(losses_ner), epoch)
    writer.add_scalar("val/loss_SA", np.mean(losses_sa), epoch)
    writer.add_scalar("val/accuracy_NER", acc_ner_val, epoch)
    writer.add_scalar("val/accuracy_NER_out", acc_ner_out, epoch)
    writer.add_scalar("val/accuracy_SA", acc_sa_val, epoch)
    print(f"Epoch {epoch}: Validation Loss NER = {np.mean(losses_ner):.4f} | Accuracy = {acc_ner_val:.4f} | Out = {acc_ner_out:.4f}")
    print(f"Epoch {epoch}: Validation Loss SA = {np.mean(losses_sa):.4f} | Accuracy SA = {acc_sa_val:.4f}")
    
    

@torch.no_grad()
def t_step_sa(model, test_data, loss_fn, device):
    model.eval()
    acc = Accuracy("SA")
    losses = []
    for inputs, _, sa_targets, lengths in test_data:
        inputs, sa_targets = inputs.to(device), sa_targets.to(device)
        outputs, _ = model(inputs, lengths)
        loss_ = loss_fn(outputs.float(), sa_targets.long())
        acc.update(outputs, sa_targets)
        losses.append(loss_.item())
    acc_val, _ = acc.compute()
    print(f"[TEST] Loss SA = {np.mean(losses):.4f} | Accuracy = {acc_val:.4f}")


@torch.no_grad()
def t_step_ner(model, test_data, loss_fn_ner1, loss_fn_ner2, device):
    model.eval()
    acc = Accuracy("NER")
    loss_ponderation = model.loss_ponderation.tolist()
    losses = []
    for inputs, ner_targets, _, lengths in test_data:
        inputs, ner_targets = inputs.to(device), ner_targets.to(device)
        outputs, _ = model(inputs, lengths)
        _, _, D = outputs.size()
        
        mask = ner_targets == 0
        mask_ent = (~mask) & (ner_targets != -1)
        loss1 = loss_fn_ner1(outputs[mask].view(-1, D).float(), ner_targets[mask].view(-1).long())
        if mask_ent.sum() > 0:
            loss2 = loss_fn_ner2(outputs[mask_ent].view(-1, D).float(), ner_targets[mask_ent].view(-1).long())
        else:
            loss2 = torch.tensor(0.0, device=device, requires_grad=True)
        total_loss = loss_ponderation[0] * loss1 + loss_ponderation[1] * loss2
        
        acc.update(outputs, ner_targets)
        losses.append(total_loss.item())
    acc_val, acc_out = acc.compute()
    print(f"[TEST] Loss NER = {np.mean(losses):.4f} | Accuracy = {acc_val:.4f} | Out Accuracy = {acc_out:.4f}")


@torch.no_grad()
def t_step_nersa(model, test_data, loss_fn_ner1, loss_fn_ner2, loss_fn_sa, device):
    model.eval()
    acc_ner, acc_sa = Accuracy("NER"), Accuracy("SA")
    loss_ponderation = model.loss_ponderation.tolist()
    losses_ner, losses_sa = [], []
    for inputs, ner_targets, sa_targets, lengths in test_data:
        inputs, ner_targets, sa_targets = inputs.to(device), ner_targets.to(device), sa_targets.to(device)
        out_ner, out_sa = model(inputs, lengths)
        _, _, D = out_ner.size()
        
        mask = ner_targets == 0
        mask_ent = (~mask) & (ner_targets != -1)
        loss1 = loss_fn_ner1(out_ner[mask].view(-1, D).float(), ner_targets[mask].view(-1).long())
        if mask_ent.sum() > 0:
            loss2 = loss_fn_ner2(out_ner[mask_ent].view(-1, D).float(), ner_targets[mask_ent].view(-1).long())
        else:
            loss2 = torch.tensor(0.0, device=device, requires_grad=True)
        loss_sa = loss_fn_sa(out_sa.float(), sa_targets.long())
        
        acc_ner.update(out_ner, ner_targets)
        acc_sa.update(out_sa, sa_targets)
        losses_ner.append((loss_ponderation[0] * loss1 + loss_ponderation[1] * loss2).item())
        losses_sa.append(loss_ponderation[2] * loss_sa.item())
    acc_ner_val, acc_ner_out = acc_ner.compute()
    acc_sa_val, _ = acc_sa.compute()
    print(f"[TEST] Loss NER = {np.mean(losses_ner):.4f} | Accuracy = {acc_ner_val:.4f} | Out = {acc_ner_out:.4f}")
    print(f"[TEST] Loss SA = {np.mean(losses_sa):.4f} | Accuracy = {acc_sa_val:.4f}")
