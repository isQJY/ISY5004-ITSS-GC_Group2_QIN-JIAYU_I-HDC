import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def compute_confusion_rate(outputs, labels, target_classes=(14, 15)):  # Speed Limit 60/80
    _, preds = torch.max(outputs, 1)
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=target_classes)
    if cm.shape == (2, 2):
        confusion_rate = (cm[0, 1] + cm[1, 0]) / cm.sum()
        return confusion_rate
    return 0.0
