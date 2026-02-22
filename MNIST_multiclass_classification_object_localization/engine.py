import os
import random
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm.auto import tqdm
from typing import Dict, List


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set seed for reproducibility across:
    - Python
    - NumPy
    - PyTorch (CPU, CUDA, MPS)

    Args:
        seed (int): Random seed value
        deterministic (bool): If True, forces deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to {seed} | Deterministic={deterministic}")

    
# -----------------------------------------------------------
# IoU METRIC
# -----------------------------------------------------------

def compute_iou(pred_boxes: torch.Tensor,
                true_boxes: torch.Tensor, h:int, w:int) -> torch.Tensor:
    """
    Computes IoU for boxes in (x1, y1, x2, y2) format
    pred_boxes: [B, 4]
    true_boxes: [B, 4]
    """

    pred_boxes = pred_boxes * torch.tensor([w, h, w, h], device=pred_boxes.device)  # Scale to pixel coordinates
    true_boxes = true_boxes * torch.tensor([w, h, w, h], device=true_boxes.device)  # Scale to pixel coordinates

    x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    pred_a1 = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0)
    pred_a2 = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    true_a1 = (true_boxes[:, 2] - true_boxes[:, 0]).clamp(min=0)
    true_a2 = (true_boxes[:, 3] - true_boxes[:, 1]).clamp(min=0)
    pred_area = pred_a1 * pred_a2
    true_area = true_a1 * true_a2
    union = pred_area + true_area - intersection + 1e-6

    iou = intersection / union
    return iou


# -----------------------------------------------------------
# TRAIN STEP
# -----------------------------------------------------------

def train_step(model: nn.Module,
               loader: DataLoader,
               optimizer: Optimizer,
               device: torch.device,
               loss_fns: List[nn.Module],
               loss_weights: List[float]) -> Dict[str, float]:
    
    assert len(loss_fns) == len(loss_weights) == 2, \
        "loss_fns and loss_weights must match length of 2 (classification and localization)"

    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    total_correct = 0
    total_iou = 0.0

    total_samples = 0

    cls_loss_fn, loc_loss_fn = loss_fns
    cls_loss_weight, loc_loss_weight = loss_weights

    for img_batch, label_batch, bbox_batch in tqdm(loader, desc="Training", leave=False):

        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)
        bbox_batch = bbox_batch.to(device)

        optimizer.zero_grad()

        class_logits, pred_bbox_batch = model(img_batch)

        cls_loss = cls_loss_fn(class_logits, label_batch)
        loc_loss = loc_loss_fn(pred_bbox_batch, bbox_batch)

        weighted_loss = cls_loss_weight * cls_loss + loc_loss_weight * loc_loss

        weighted_loss.backward()
        optimizer.step()

        batch_size = img_batch.size(0)

        total_loss += weighted_loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_loc_loss += loc_loss.item() * batch_size

        # Classification accuracy
        preds = torch.argmax(class_logits, dim=1)
        correct = (preds == label_batch).sum().item()

        # IoU metric
        h = img_batch.size(2)
        w = img_batch.size(3)
        iou = compute_iou(pred_bbox_batch, bbox_batch, h=h, w=w).mean().item()

        total_correct += correct
        total_iou += iou * batch_size
        total_samples += batch_size

    return {
        "total_loss": total_loss / total_samples,
        "cls_loss": total_cls_loss / total_samples,
        "loc_loss": total_loc_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "iou": total_iou / total_samples
    }


# -----------------------------------------------------------
# VALIDATION STEP
# -----------------------------------------------------------

@torch.no_grad()
def val_step(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             loss_fns: List[nn.Module],
             loss_weights: List[float]) -> Dict[str, float]:
    
    assert len(loss_fns) == len(loss_weights) == 2, \
        "loss_fns and loss_weights must match length of 2 (classification and localization)"

    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    total_correct = 0
    total_iou = 0.0

    total_samples = 0

    cls_loss_fn, loc_loss_fn = loss_fns
    cls_loss_weight, loc_loss_weight = loss_weights

    for img_batch, label_batch, bbox_batch in tqdm(loader, desc="Validation", leave=False):

        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)
        bbox_batch = bbox_batch.to(device)

        class_logits, pred_bbox_batch = model(img_batch)

        cls_loss = cls_loss_fn(class_logits, label_batch)
        loc_loss = loc_loss_fn(pred_bbox_batch, bbox_batch)

        weighted_loss = cls_loss_weight * cls_loss + loc_loss_weight * loc_loss

        batch_size = img_batch.size(0)

        total_loss += weighted_loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_loc_loss += loc_loss.item() * batch_size

        preds = torch.argmax(class_logits, dim=1)
        correct = (preds == label_batch).sum().item()

        h = img_batch.size(2)
        w = img_batch.size(3)
        iou = compute_iou(pred_bbox_batch, bbox_batch, h=h, w=w).mean().item()

        total_correct += correct
        total_iou += iou * batch_size
        total_samples += batch_size

    return {
        "val_total_loss": total_loss / total_samples,
        "val_cls_loss": total_cls_loss / total_samples,
        "val_loc_loss": total_loc_loss / total_samples,
        "val_accuracy": total_correct / total_samples,
        "val_iou": total_iou / total_samples
    }


# -----------------------------------------------------------
# FULL TRAIN LOOP
# -----------------------------------------------------------

def model_fit(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: Optimizer,
          device: torch.device,
          loss_fns: List[nn.Module],
          loss_weights: List[float],
          epochs: int) -> Dict[str, Dict[str, List[float]]]:

    results = {
        "train": {
            "total_loss": [],
            "cls_loss": [],
            "loc_loss": [],
            "accuracy": [],
            "iou": []
        },
        "val": {
            "total_loss": [],
            "cls_loss": [],
            "loc_loss": [],
            "accuracy": [],
            "iou": []
        }
    }

    best_val_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):

        print(f"\nEpoch [{epoch+1}/{epochs}]")

        train_results = train_step(model, train_loader, optimizer, device, loss_fns, loss_weights)
        val_results = val_step(model, val_loader, device, loss_fns, loss_weights)

        print(
            f"Training Step | "
            f"Loss: {train_results['total_loss']:.4f} | "
            f"Cls_loss: {train_results['cls_loss']:.4f} | "
            f"Loc_loss: {train_results['loc_loss']:.4f} | "
            f"Acc: {train_results['accuracy']:.4f} | "
            f"IoU: {train_results['iou']:.4f}"
        )

        print(
            f"Validation Step | "
            f"Loss: {val_results['val_total_loss']:.4f} | "
            f"Cls_loss: {val_results['val_cls_loss']:.4f} | "
            f"Loc_loss: {val_results['val_loc_loss']:.4f} | "
            f"Acc: {val_results['val_accuracy']:.4f} | "
            f"IoU: {val_results['val_iou']:.4f}"
        )

        for key in results["train"]:
            results["train"][key].append(train_results[key])

        for key in results["val"]:
            results["val"][key].append(val_results[f"val_{key}"])

        if val_results["val_total_loss"] < best_val_loss:
            best_val_loss = val_results["val_total_loss"]
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved with val_total_loss: {best_val_loss:.4f}")

    return results