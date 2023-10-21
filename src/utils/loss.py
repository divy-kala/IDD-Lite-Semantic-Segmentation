import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, num_classes=8, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        batch_size, _, height, width = predictions.size()

        # Flatten the predictions and targets
        predictions = predictions.reshape(batch_size, self.num_classes, -1)
        targets = targets.reshape(batch_size, -1)

        # One-hot encode the target masks for multi-class dice loss
        targets_one_hot = torch.eye(self.num_classes, device=predictions.device)[targets].to(predictions.device)
        targets_one_hot = targets_one_hot.transpose(1,2)
        
        intersection = (predictions * targets_one_hot).sum((1,2))
        union = predictions.sum((1,2)) + targets_one_hot.sum((1,2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        
        return loss
