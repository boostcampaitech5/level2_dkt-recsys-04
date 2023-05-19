import torch
import torch.nn.functional as F


def BCEwithLL(output: torch.Tensor, target: torch.Tensor):
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    return loss(output, target)
