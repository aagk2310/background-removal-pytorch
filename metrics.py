import torch
import torch.nn as nn

# IoU (Intersection over Union) for PyTorch
def iou(y_true, y_pred):
    """
    Compute the Intersection over Union (IoU) metric.
    """
    smooth = 1e-15
    intersection = (y_true * y_pred).sum(dim=(1, 2, 3))
    union = (y_true + y_pred).sum(dim=(1, 2, 3)) - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score.mean()

# Dice Coefficient for PyTorch
def dice_coef(y_true, y_pred):
    """
    Compute the Dice Coefficient metric.
    """
    smooth = 1e-15
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + smooth)

# Dice Loss for PyTorch
def dice_loss(y_true, y_pred):
    """
    Compute the Dice loss function, which is 1 - Dice Coefficient.
    """
    return 1.0 - dice_coef(y_true, y_pred)